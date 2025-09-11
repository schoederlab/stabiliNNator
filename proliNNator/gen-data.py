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

from utils import GraphDataGenerator, get_proline_positions, get_non_proline_positions
import config

def process_and_save_chunk(chunk_index, pdb_chunk, path, edge_distance_cutoff_A, max_residues, nbr_distance_cutoff_A):
    decorators = [decs.SequenceSeparation(ln = True),
                  decs.SimpleBBGeometry(use_nm = False),
                  decs.Rosetta_Ref2015_TwoBodyEneriges(individual=True,
                                                       score_types=[ScoreType.fa_rep, ScoreType.fa_atr, ScoreType.fa_sol, ScoreType.lk_ball_wtd,
                                                                    ScoreType.fa_elec, ScoreType.hbond_sr_bb, ScoreType.hbond_lr_bb, ScoreType.hbond_bb_sc,
                                                                    ScoreType.hbond_sc])]

    data_generator = GraphDataGenerator(decorators=decorators,
                                        edge_distance_cutoff_A=edge_distance_cutoff_A,
                                        max_residues=max_residues,
                                        nbr_distance_cutoff_A=nbr_distance_cutoff_A)

    Xs_chunk = []
    As_chunk = []
    Es_chunk = []
    outs_chunk = []

    for pdb in pdb_chunk:
        pose = pyrosetta.pose_from_pdb(pdb)
        proline_residues = get_proline_positions(pose)
        non_prolines = get_non_proline_positions(pose, proline_residues)
        wrapped_pose = mg.RosettaPoseWrapper(pose)
        cache = data_generator.data_maker.make_data_cache(wrapped_pose)

        for resid in proline_residues:
            X, A, E, resids = data_generator.data_maker.generate_input_for_resid(wrapped_pose, resid, data_cache=cache)
            Xs_chunk.append(X)
            As_chunk.append(A)
            Es_chunk.append(E)
            outs_chunk.append([1.0,])

        for resid in non_prolines:
            X, A, E, resids = data_generator.data_maker.generate_input_for_resid(wrapped_pose, resid, data_cache=cache)
            Xs_chunk.append(X)
            As_chunk.append(A)
            Es_chunk.append(E)
            outs_chunk.append([0.0,])

    np.savez_compressed(f'{path}/data_chunk_{chunk_index}.npz', Xs=Xs_chunk, As=As_chunk, Es=Es_chunk, outs=outs_chunk)
    print(f'Chunk {chunk_index} saved!')

def process_chunks(pdb_list, chunk_size, path, edge_distance_cutoff_A, max_residues, nbr_distance_cutoff_A):
    chunks = [pdb_list[i:i + chunk_size] for i in range(0, len(pdb_list), chunk_size)]
    
    with Pool(processes=cpu_count()) as pool:
        pool.starmap(process_and_save_chunk, [(i + 1, chunk, path, edge_distance_cutoff_A, max_residues, nbr_distance_cutoff_A) for i, chunk in enumerate(chunks)])
    print(f"Processing and saving completed for path: {path}")

def merge_npz_files(file_list, output_file): 
    combined_arrays = {} 
    for file in file_list: 
        with np.load(file) as data: 
            for key in data: 
                if key in combined_arrays: 
                    combined_arrays[key] = np.concatenate((combined_arrays[key], data[key]), axis=0) 
                else:
                    combined_arrays[key] = data[key]
    np.savez(output_file, **combined_arrays)

directories_to_process = {
    "train": config.TRAIN_DATA_PATH,
    "test": config.TEST_DATA_PATH,
    "val": config.VAL_DATA_PATH
}
chunk_size = 200

for name, dir_path in directories_to_process.items():
    pdb_list = glob.glob(f'{dir_path}/*.pdb')
    print(f"Found {len(pdb_list)} PDB files in {name} directory ({dir_path}).")
    process_chunks(pdb_list, chunk_size, dir_path, config.EDGE_DISTANCE_CUTOFF_A, config.MAX_RESIDUES, config.NBR_DISTANCE_CUTOFF_A)
    
    file_list = glob.glob(f'{dir_path}/*?.npz')
    output_file = f'{dir_path}/merged_file.npz'
    merge_npz_files(file_list, output_file)
    
