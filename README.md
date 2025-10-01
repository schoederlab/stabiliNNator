# stabiliNNator

## proliNNator

### What it does
ProliNNator is a tool that predicts proline probabilities in protein structures using pre-trained neural networks. It takes a Protein Data Bank (PDB) file as input and outputs a CSV file with per-residue proline probabilities, and an updated PDB file where B-factors are replaced by these probabilities. It can also generate a Ramachandran plot visualizing these predictions.

### How it works
ProliNNator employs Graph Convolutional Networks (GCNs) to analyze protein structures. It uses the `menten_gcn` library to convert PDB structures into graph representations, where residues are nodes and interactions are edges. A pre-trained Keras model then processes these graphs to predict the likelihood of each residue being a proline. Optionally, a FastRelax protocol from `PyRosetta` can be applied to the input structure before analysis.

### Dependencies
The `proliNNator` package relies on the following key Python libraries:
*   `pandas`
*   `numpy`
*   `matplotlib`
*   `seaborn`
*   `argparse`
*   `pyrosetta`
*   `tensorflow` (specifically `keras`)
*   `spektral`
*   `menten_gcn`

### How to execute it
To run `proliNNator`, execute the `proliNNator.py` script from the command line with the following arguments:

```bash
python proliNNator/proliNNator.py -i <input_pdb_file> [OPTIONS]
```

**Arguments:**
*   `-i`, `--input`: Path to the input PDB file (required).
*   `-m`, `--model`: Path to the pre-trained model file (default: `3D-model-v2.5.keras`).
*   `-p`, `--pdb`: Name of the output PDB file with B-factors updated (default: `output.pdb`).
*   `--csv`: Filename to save a CSV file containing per-residue proline probabilities (default: `output.csv`).
*   `--ramachandran`: Filename to save a Ramachandran plot with probabilities as a PNG image (default: `ramachandran.png`).
*   `--fastrelax`: Include this flag to perform a FastRelax simulation on the input structure before analysis.

## disulfiNNate

### What it does
DisulfiNNate is a tool that predicts the probability of disulfide bonds in protein structures using pre-trained neural networks. It takes a Protein Data Bank (PDB) file as input and outputs a CSV file with per-pair disulfide bond probabilities.

### How it works
DisulfiNNate, similar to ProliNNator, utilizes Graph Convolutional Networks (GCNs) to analyze protein structures. It uses the `menten_gcn` library to convert PDB structures into graph representations. A pre-trained Keras model then processes these graphs to predict the likelihood of a disulfide bond forming between pairs of cysteine residues. Optionally, a FastRelax protocol from `PyRosetta` can be applied to the input structure before analysis.

### Dependencies
The `disulfiNNate` package relies on the following key Python libraries:
*   `pandas`
*   `numpy`
*   `argparse`
*   `pyrosetta`
*   `tensorflow` (specifically `keras`)
*   `spektral`
*   `menten_gcn`

### How to execute it
To run `disulfiNNate`, execute the `disulfiNNate.py` script from the command line with the following arguments:

```bash
python disulfiNNate/disulfiNNate.py -i <input_pdb_file> [OPTIONS]
```

**Arguments:**
*   `-i`, `--input`: Path to the input PDB file (required).
*   `-o`, `--output`: Name of the output CSV file with per-pair disulfide bond probabilities (default: `out.csv`).
*   `-m`, `--model`: Path to the pre-trained model file (default: `disulfiNNate-model.keras`).
*   `--fastrelax`: Include this flag to perform a FastRelax simulation on the input structure before analysis.
