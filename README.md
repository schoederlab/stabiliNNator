# stabiliNNator

stabillinator contains two graph neural networks, prolinnator and disulfinnate, which can be used to predict proline mutations or engineer disulfide bonds.

## Installation

```bash
conda create --name stabilinnator python=3.10
conda activate stabilinnator
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.0+cpu.html
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.11.0+cpu.html
```

## proliNNator
### description
ProliNNator is a tool that predicts proline probabilities in protein structures using pre-trained neural networks. It takes a Protein Data Bank (PDB) file as input and outputs a PDB file with per-residue proline probabilities.

## disulfiNNate
### description
DisulfiNNate, similar to ProliNNator, utilizes Graph Convolutional Networks (GCNs) to analyze protein structures. A pre-trained model then processes these graphs to predict the likelihood of a disulfide bond forming between pairs of cysteine residues.

## How to execute it
Details can be found in the respective directories