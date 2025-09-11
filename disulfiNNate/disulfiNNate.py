import pyrosetta
pyrosetta.init('-mute all')
from pyrosetta.rosetta.core.scoring import ScoreType
from pyrosetta.rosetta.protocols.relax import FastRelax

import menten_gcn as mg
import menten_gcn.decorators as decs

from tensorflow.keras.models import load_model

import numpy as np
import pandas as pd
import argparse
import multiprocessing as mp

import config
from utils import calculate_distance, GraphDataGenerator, GCNModel

# Decorators for Menten GCN
decorators = [decs.CACA_dist(use_nm=True),
              decs.trRosettaEdges(sincos=True, use_nm=True),
              decs.Rosetta_Ref2015_TwoBodyEneriges(individual=True, score_types=[ScoreType.fa_rep,
                                                                                 ScoreType.fa_atr, 
                                                                                 ScoreType.fa_sol, 
                                                                                 ScoreType.lk_ball_wtd, 
                                                                                 ScoreType.fa_elec, 
                                                                                 ScoreType.hbond_sr_bb, 
                                                                                 ScoreType.hbond_lr_bb, 
                                                                                 ScoreType.hbond_bb_sc, 
                                                                                 ScoreType.hbond_sc])]

data_generator = GraphDataGenerator(decorators=decorators,
                                    edge_distance_cutoff_A=config.EDGE_DISTANCE_CUTOFF_A,
                                    max_residues=config.MAX_RESIDUES,
                                    nbr_distance_cutoff_A=config.NBR_DISTANCE_CUTOFF_A)

def build_graphs(args):
    wrapped_pose, pair = args
    cache = data_generator.data_maker.make_data_cache(wrapped_pose)
    X, A, E, resids = data_generator.data_maker.generate_input( wrapped_pose, pair, data_cache=cache )
    return X, A, E

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input', type=str, required=True, help='input file (PDB format)')
    parser.add_argument('-o', '--output', type=str, default='out.csv', help='output file name. default=out.csv')
    parser.add_argument('-m', '--model', type=str, default=config.MODEL_SAVE_FILENAME, help=f'model name. default={config.MODEL_SAVE_FILENAME}')
    parser.add_argument('--fastrelax', action='store_true', help='Flag to perform a fast relax on the structure before analysis')
    args = parser.parse_args()

    pdb = args.input
    pose = pyrosetta.pose_from_pdb(pdb)

    # Perform FastRelax if flag is set
    if args.fastrelax:
        scfxn = pyrosetta.get_fa_scorefxn()
        fast_relax = FastRelax(scorefxn_in = scfxn, standard_repeats=1)
        fast_relax.set_scorefxn(scfxn)
        print('Executing fast relax...')
        fast_relax.apply(pose)
        print('fast relax finished')

    wrapped_pose = mg.RosettaPoseWrapper(pose)
    
    custom_objects = {'ECCConv': ECCConv, 'GATConv': GATConv, 'GlobalSumPool': GlobalSumPool, 'GlobalMaxPool': GlobalMaxPool}
    model = load_model(args.model, custom_objects)

    pairs = []

    for i in range(1, pose.size() + 1):
        for j in range(1, pose.size() +1):
            if j != i and j > i:
                distance = calculate_distance(pose, i, j)
                if distance <= 10:
                    pairs.append([i, j])

    Xs_ = []
    As_ = []
    Es_ = []
    arguments = [(wrapped_pose, pair) for pair in pairs]

    print('Building Graphs...')
    with mp.Pool(processes=mp.cpu_count()) as p:
        results = p.map(build_graphs, arguments)

    for result in results:
        Xs_.append(result[0])
        As_.append(result[1])
        Es_.append(result[2])
    print('Building finished!')

    #prediction
    Xs_ = np.asarray(Xs_)
    As_ = np.asarray(As_)
    Es_ = np.asarray(Es_)
    y_pred = model.predict([Xs_, As_, Es_])

    #get chain & position information from pairs
    pdb_numbering = []
    chains = []
    residues = []

    for i in pairs:
        _1 = pose.pdb_info().number(int(i[0]))
        _2 = pose.pdb_info().number(int(i[1]))
        pdb_numbering.append([_1, _2])
        _1 = pose.pdb_info().chain(int(i[0]))
        _2 = pose.pdb_info().chain(int(i[1]))
        chains.append([_1, _2])
        _1 = pose.residue(int(i[0])).name()
        _2 = pose.residue(int(i[1])).name()
        residues.append([_1, _2])

    #output
    df = pd.DataFrame({'chains': chains,
                       'pdb numbering': pdb_numbering,
                       'rosetta numbering': pairs,
                       'residue': residues,
                       'probability': y_pred.flatten()
                       })
    df.to_csv(args.output, index=False)

if __name__ == "__main__":
    main()
