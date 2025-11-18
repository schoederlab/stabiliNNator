#!/usr/bin/env python3
"""
Load a trained ResidueGAT model and annotate a PDB with disulfide probabilities.

The script replicates the cysteine graph construction (including torsions,
chi1, sequential + spatial edges, and edge distance features), runs inference,
and writes probabilities (0..1) into the B-factor column for every residue.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Tuple

from dataclasses import asdict

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv
from Bio.PDB import PDBIO, PDBParser

from graph_generation import residues_to_graph, chain_residues


# ---------- Feature helpers (mirrors train_gcn.py) ----------

def torsion_to_cos_sin(angle_deg: float | None) -> Tuple[float, float]:
    if angle_deg is None:
        return 0.0, 0.0
    angle_rad = math.radians(angle_deg)
    return math.cos(angle_rad), math.sin(angle_rad)


def build_node_features(node: dict) -> List[float]:
    phi_c, phi_s = torsion_to_cos_sin(node.get("phi"))
    psi_c, psi_s = torsion_to_cos_sin(node.get("psi"))
    omg_c, omg_s = torsion_to_cos_sin(node.get("omega"))
    chi1_c, chi1_s = torsion_to_cos_sin(node.get("chi1"))
    x, y, z = node.get("ca_coord", (0.0, 0.0, 0.0))
    return [
        phi_c,
        phi_s,
        psi_c,
        psi_s,
        omg_c,
        omg_s,
        chi1_c,
        chi1_s,
        float(x),
        float(y),
        float(z),
    ]


def graph_to_data(graph_entry: dict) -> Data:
    nodes = graph_entry["nodes"]
    edges = graph_entry.get("edges", [])
    edge_attrs = graph_entry.get("edge_attrs", [])

    x = torch.tensor([build_node_features(node) for node in nodes], dtype=torch.float)
    y = torch.zeros(len(nodes), dtype=torch.float)

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float) if edge_attrs else torch.zeros((len(edges), 2), dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 2), dtype=torch.float)

    data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
    return data


class ResidueGAT(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.0, heads: int = 4):
        super().__init__()
        self.gat = GATv2Conv(
            in_channels=in_dim,
            out_channels=hidden_dim,
            heads=heads,
            concat=False,
            dropout=dropout,
            edge_dim=2,
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x = self.gat(data.x, data.edge_index, data.edge_attr)
        x = torch.relu(x)
        return self.mlp(x).squeeze(-1)


def annotate_structure(residue_refs: List, probabilities: torch.Tensor) -> None:
    for residue, prob in zip(residue_refs, probabilities.tolist()):
        for atom in residue.get_atoms():
            atom.set_bfactor(float(prob))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate a PDB with cysteine bonding probabilities from a trained GAT model."
    )
    parser.add_argument("--model-path", required=True, type=Path, help="Path to trained model (.pt).")
    parser.add_argument("--pdb-path", required=True, type=Path, help="Input PDB file.")
    parser.add_argument(
        "--output-path",
        required=True,
        type=Path,
        help="Destination PDB with probabilities in B-factor column.",
    )
    parser.add_argument("--hidden-dim", type=int, default=32, help="Hidden dimension used during training.")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    pdb_path = args.pdb_path.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB not found: {pdb_path}")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("inference", str(pdb_path))
    graph = residues_to_graph(structure, pdb_path.stem)
    graph_dict = {
        "structure_id": graph.structure_id,
        "nodes": [],
        "edges": graph.edges,
        "edge_attrs": graph.edge_attrs,
    }
    for node in graph.nodes:
        graph_dict["nodes"].append(asdict(node))

    model = next(structure.get_models())
    residue_refs: List = []
    for chain in model:
        residue_refs.extend(chain_residues(chain))

    data = graph_to_data(graph_dict)
    if data.num_nodes == 0:
        raise RuntimeError("No residues with CA atoms found in the structure.")

    device = torch.device(args.device if torch.cuda.is_available() or "cpu" not in args.device else "cpu")
    model = ResidueGAT(in_dim=data.num_features, hidden_dim=args.hidden_dim)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    data = data.to(device)
    with torch.no_grad():
        logits = model(data)
        probs = torch.sigmoid(logits).cpu()

    annotate_structure(residue_refs, probs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(output_path))
    print(f"Wrote annotated PDB to {output_path}")


if __name__ == "__main__":
    main()

