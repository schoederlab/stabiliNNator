import pyrosetta
pyrosetta.init('-mute all')
from pyrosetta.rosetta.core.scoring import ScoreType

import menten_gcn as mg
import menten_gcn.decorators as decs

import numpy as np
import glob
import pandas as pd
import os
from multiprocessing import Pool, cpu_count

from utils import GraphDataGenerator, get_cys, get_negative_examples, process_chunks, merge_npz_files
import config

# Process the pdb_list in chunks
directories_to_process = {
    "train": config.TRAIN_DATA_PATH,
    "test": config.TEST_DATA_PATH,
    "val": config.VAL_DATA_PATH
}
chunk_size = 10

for name, dir_path in directories_to_process.items():
    pdb_list = glob.glob(f'{dir_path}/*.pdb')
    print(f"Found {len(pdb_list)} PDB files in {name} directory ({dir_path}).")
    process_chunks(pdb_list, chunk_size, dir_path, config.EDGE_DISTANCE_CUTOFF_A, config.MAX_RESIDUES, config.NBR_DISTANCE_CUTOFF_A)

    file_list = glob.glob(f'{dir_path}/*?.npz')
    output_file = f'{dir_path}/merged_file.npz'
    merge_npz_files(file_list, output_file)
