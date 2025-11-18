# Cysteine Disulfide Graph Pipeline

End-to-end utilities to detect disulfide-bonded cysteines in protein structures, train a graph neural network, and annotate new PDBs with the predicted bonding probability.

## Components

- `graph_generation.py`: converts structures (PDB/mmCIF) into residue graphs. Each node stores backbone torsions (φ, ψ, ω), chi1, sidechain heavy-atom counts, Cα/Cβ coordinates, and binary labels indicating whether a cysteine participates in a disulfide bridge. Edges include both peptide-bond connections and spatial contacts (Cα–Cα ≤ 6 Å) with edge attributes for Cα–Cα and Cβ–Cβ distances.
- `graph_visualization.py`: renders a single graph (3D scatter) for sanity checks; label 1 residues (bound cysteines) appear in red.
- `train_gcn.py`: trains a GATv2 + MLP node classifier with class-weighted BCE, optional kernel L2 penalty, early stopping, and diagnostic plots (loss curves, PR curve, normalized confusion matrix).
- `predict_cysteine_probabilities.py`: loads a trained model, rebuilds the same feature graph for a single PDB, and writes per-residue probabilities to the B-factor column.

## Requirements

- Python 3.10+
- [Biopython](https://biopython.org/) for structure parsing
- PyTorch + PyTorch Geometric (with GATv2Conv support)
- NumPy, Matplotlib, scikit-learn

Example install:

```bash
pip install biopython torch torchvision torchaudio pyg-lib torch_geometric numpy matplotlib scikit-learn
```

## Usage

### 1. Generate Graphs

```bash
python graph_generation.py \
  --dataset-dir /media/data/jri/cath_S40/whole-dataset \
  --output-path graphs.jsonl
```

JSONL contents per graph:

- Node features: torsion cos/sin pairs for φ/ψ/ω, chi1 cos/sin, sidechain heavy-atom counts, Cα coordinates, and `is_cysteine` flag.
- Labels: `1` only for cysteines engaged in an S–S bond (SG–SG distance ≤ 2.4 Å).
- Edges: peptide neighbors + spatial contacts within 6 Å, each storing `[ca_distance, cb_distance]`.

### 2. Visualize (Optional)

```bash
python graph_visualization.py \
  --graph-file graphs.jsonl \
  --structure-id 2g3aA01 \
  --output-path example.png
```

### 3. Train the Model

```bash
python train_gcn.py \
  --graph-file graphs.jsonl \
  --hidden-dim 32 \
  --epochs 500 \
  --batch-size 8 \
  --device cuda \
  --model-out models/cys_gat.pt \
  --report-path reports/cys_training.png \
  --early-stop-patience 30 \
  --kernel-l2 1e-5
```

Key notes:

- Split defaults to 70/15/15 (train/val/test).
- `pos_weight` is computed automatically to counter class imbalance.
- Diagnostics PNG shows loss curves, PR curve, and normalized confusion matrix.

### 4. Predict on New Structures

```bash
python predict_cysteine_probabilities.py \
  --model-path models/cys_gat.pt \
  --pdb-path /media/data/jri/cath_S40/whole-dataset/2g3aA01.pdb \
  --output-path annotated/2g3aA01_probs.pdb \
  --hidden-dim 32 \
  --device cuda
```

The resulting PDB has every atom’s B-factor set to the predicted probability for its residue.

## Tips

- Re-run `graph_generation.py` whenever you change feature logic.
- Use `--max-structures` for quick debugging subsets.
- If probabilities stay near zero, try increasing `pos_weight`, raising `hidden_dim`, or loosening the disulfide detection cutoff slightly.
- Inspect `reports/*.png` after training to verify convergence and class balance.

