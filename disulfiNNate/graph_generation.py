#!/usr/bin/env python3
"""
Graph generation for cysteine disulfide prediction.

Each residue is a node located at the Cα position with torsion features, sidechain
counts, chi1 information, and binary labels indicating whether a cysteine is part
of a disulfide bond. Edges connect sequential residues and residue pairs whose
Cα atoms are within 6 Å; each edge stores Cα–Cα and Cβ–Cβ distances.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from Bio.PDB import MMCIFParser, PDBParser, is_aa
from Bio.PDB.vectors import Vector, calc_dihedral


@dataclass
class ResidueNode:
    """Node representation for a residue."""

    chain_id: str
    residue_id: str
    residue_name: str
    ca_coord: Tuple[float, float, float]
    cb_coord: Optional[Tuple[float, float, float]]
    phi: Optional[float]
    psi: Optional[float]
    omega: Optional[float]
    chi1: Optional[float]
    label: int


@dataclass
class ResidueGraph:
    """Graph containing nodes and peptide-bond edges."""

    structure_id: str
    nodes: List[ResidueNode]
    edges: List[Tuple[int, int]]
    edge_attrs: List[Tuple[float, float]]


BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}
CYSTEINE_NAMES = {"CYS", "CYX"}
DISULFIDE_MAX_DIST = 2.4
SPATIAL_EDGE_CUTOFF = 6.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate residue graphs with torsion features."
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        type=Path,
        help="Directory with protein structure files (PDB/mmCIF).",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        type=Path,
        help="Destination JSONL file for serialized graphs.",
    )
    parser.add_argument(
        "--max-structures",
        type=int,
        default=None,
        help="Optional upper bound on processed structures (for debugging).",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".pdb", ".ent", ".cif", ".mmcif"],
        help="File extensions to consider when scanning dataset_dir.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def iter_structure_files(root: Path, extensions: Sequence[str]) -> Iterable[Path]:
    """Yield structure files under root matching given extensions."""
    normalized = tuple(ext.lower() for ext in extensions)
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() in normalized:
            yield path


def parse_structure(path: Path, pdb_parser: PDBParser, cif_parser: MMCIFParser):
    """Parse a structure file into a Bio.PDB Structure object."""
    suffix = path.suffix.lower()
    structure_id = path.stem
    if suffix in {".cif", ".mmcif"}:
        return cif_parser.get_structure(structure_id, str(path))
    return pdb_parser.get_structure(structure_id, str(path))


def residue_id(residue) -> str:
    """Create a stable identifier for a residue."""
    hetflag, seq_id, insertion = residue.id
    insertion = insertion.strip() or "-"
    return f"{hetflag}{seq_id}{insertion}"


def get_atom_vector(residue, atom_name: str) -> Optional[Vector]:
    """Return the atom vector if present, else None."""
    if residue is None:
        return None
    if atom_name not in residue:
        return None
    return residue[atom_name].get_vector()


def dihedral_or_none(vectors: Sequence[Optional[Vector]]) -> Optional[float]:
    """Compute dihedral angle in degrees when all vectors are present."""
    if any(vec is None for vec in vectors):
        return None
    angle = calc_dihedral(*vectors)
    return math.degrees(angle)


def count_sidechain_heavy_atoms(residue) -> int:
    """Count non-hydrogen atoms that are not part of the backbone."""
    count = 0
    for atom in residue.get_atoms():
        name = atom.get_name().strip()
        if name in BACKBONE_ATOMS:
            continue
        element = (atom.element or "").strip().upper()
        if not element:
            element = name[0].upper()
        if element == "H":
            continue
        count += 1
    return count


def chain_residues(chain) -> List:
    """Filter residues in a chain to standard amino acids with CA atoms."""
    residues = []
    for residue in chain.get_residues():
        if not is_aa(residue, standard=True):
            continue
        if "CA" not in residue:
            continue
        residues.append(residue)
    return residues


def residue_key(chain_id: str, residue) -> Tuple[str, Tuple]:
    return (chain_id, residue.id)


def get_coord_tuple(atom) -> Tuple[float, float, float]:
    return tuple(float(coord) for coord in atom.coord)


def distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def find_disulfide_cysteines(model) -> Dict[Tuple[str, Tuple], bool]:
    """Return a mapping indicating whether a cysteine participates in a disulfide."""
    sg_atoms: List[Tuple[Tuple[str, Tuple], any]] = []
    for chain in model:
        for residue in chain:
            resname = residue.resname.strip().upper()
            if resname not in CYSTEINE_NAMES:
                continue
            if "SG" not in residue:
                continue
            sg_atoms.append((residue_key(chain.id, residue), residue["SG"]))

    bound: Dict[Tuple[str, Tuple], bool] = {}
    for idx in range(len(sg_atoms)):
        key_i, atom_i = sg_atoms[idx]
        for jdx in range(idx + 1, len(sg_atoms)):
            key_j, atom_j = sg_atoms[jdx]
            if (atom_i - atom_j) <= DISULFIDE_MAX_DIST:
                bound[key_i] = True
                bound[key_j] = True
    return bound


def compute_chi1(residue) -> Optional[float]:
    needed = ["N", "CA", "CB", "SG"]
    if not all(atom in residue for atom in needed):
        return None
    vectors = [residue[atom].get_vector() for atom in needed]
    return math.degrees(calc_dihedral(*vectors))


def residues_to_graph(structure, structure_id: str) -> ResidueGraph:
    """Convert a parsed structure into a residue graph."""
    nodes: List[ResidueNode] = []
    edge_pairs: Dict[Tuple[int, int], Tuple[float, float]] = {}
    ca_coords: List[Tuple[float, float, float]] = []
    cb_coords: List[Optional[Tuple[float, float, float]]] = []

    model = next(structure.get_models())
    bound_cys = find_disulfide_cysteines(model)

    for chain in model:
        residues = chain_residues(chain)
        prev_node_index: Optional[int] = None
        for idx, residue in enumerate(residues):
            prev_res = residues[idx - 1] if idx > 0 else None
            next_res = residues[idx + 1] if idx + 1 < len(residues) else None

            phi = (
                dihedral_or_none(
                    [
                        get_atom_vector(prev_res, "C"),
                        get_atom_vector(residue, "N"),
                        get_atom_vector(residue, "CA"),
                        get_atom_vector(residue, "C"),
                    ]
                )
                if prev_res is not None
                else None
            )
            psi = (
                dihedral_or_none(
                    [
                        get_atom_vector(residue, "N"),
                        get_atom_vector(residue, "CA"),
                        get_atom_vector(residue, "C"),
                        get_atom_vector(next_res, "N"),
                    ]
                )
                if next_res is not None
                else None
            )
            omega = (
                dihedral_or_none(
                    [
                        get_atom_vector(residue, "CA"),
                        get_atom_vector(residue, "C"),
                        get_atom_vector(next_res, "N"),
                        get_atom_vector(next_res, "CA"),
                    ]
                )
                if next_res is not None
                else None
            )

            chi1 = compute_chi1(residue)
            ca_atom = residue["CA"]
            cb_atom = residue["CB"] if "CB" in residue else None
            ca_coord = get_coord_tuple(ca_atom)
            cb_coord = get_coord_tuple(cb_atom) if cb_atom is not None else None

            resname = residue.resname.strip().upper()
            key = residue_key(chain.id, residue)
            node = ResidueNode(
                chain_id=chain.id,
                residue_id=residue_id(residue),
                residue_name=residue.resname,
                ca_coord=ca_coord,
                cb_coord=cb_coord,
                phi=phi,
                psi=psi,
                omega=omega,
                chi1=chi1,
                label=int(resname in CYSTEINE_NAMES and bound_cys.get(key, False)),
            )
            node_index = len(nodes)
            nodes.append(node)
            ca_coords.append(ca_coord)
            cb_coords.append(cb_coord)

            if prev_node_index is not None:
                ca_dist = distance(ca_coords[prev_node_index], ca_coord)
                cb_dist = (
                    distance(cb_coords[prev_node_index], cb_coord)
                    if cb_coords[prev_node_index] is not None and cb_coord is not None
                    else 0.0
                )
                key = tuple(sorted((prev_node_index, node_index)))
                edge_pairs[key] = (ca_dist, cb_dist)

            prev_node_index = node_index

    # Spatial edges within cutoff
    num_nodes = len(nodes)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            ca_dist = distance(ca_coords[i], ca_coords[j])
            if ca_dist <= SPATIAL_EDGE_CUTOFF:
                key = tuple(sorted((i, j)))
                if key in edge_pairs:
                    continue
                cb_dist = (
                    distance(cb_coords[i], cb_coords[j])
                    if cb_coords[i] is not None and cb_coords[j] is not None
                    else 0.0
                )
                edge_pairs[key] = (ca_dist, cb_dist)

    edges: List[Tuple[int, int]] = []
    edge_attrs: List[Tuple[float, float]] = []
    for (i, j), attr in edge_pairs.items():
        edges.append((i, j))
        edges.append((j, i))
        edge_attrs.append(attr)
        edge_attrs.append(attr)

    return ResidueGraph(structure_id=structure_id, nodes=nodes, edges=edges, edge_attrs=edge_attrs)


def serialize_graph(graph: ResidueGraph) -> str:
    """Serialize a graph dataclass as JSON."""
    return json.dumps(
        {
            "structure_id": graph.structure_id,
            "nodes": [asdict(node) for node in graph.nodes],
            "edges": graph.edges,
            "edge_attrs": graph.edge_attrs,
        }
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    dataset_dir = args.dataset_dir.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pdb_parser = PDBParser(QUIET=True)
    cif_parser = MMCIFParser(QUIET=True)

    files = iter_structure_files(dataset_dir, args.extensions)

    processed = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for structure_path in files:
            if args.max_structures is not None and processed >= args.max_structures:
                break
            try:
                structure = parse_structure(structure_path, pdb_parser, cif_parser)
                graph = residues_to_graph(structure, structure_path.stem)
            except Exception as exc:  # pragma: no cover - logging path
                logging.warning("Failed to process %s (%s)", structure_path, exc)
                continue

            if not graph.nodes:
                logging.debug("Skipping %s - no residues with CA atoms", structure_path)
                continue

            handle.write(f"{serialize_graph(graph)}\n")
            processed += 1
            if processed % 100 == 0:
                logging.info("Processed %d structures", processed)

    logging.info("Finished. Structures processed: %d", processed)


if __name__ == "__main__":
    main()

