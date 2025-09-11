import numpy as np
import menten_gcn as mg
import menten_gcn.decorators as decs
from pyrosetta.rosetta.core.scoring import ScoreType

from spektral.layers import *
from tensorflow import keras
from keras.regularizers import l2
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

import matplotlib.pyplot as plt
import seaborn as sns
import math
import random
import glob
from multiprocessing import Pool, cpu_count

import config


class GraphDataGenerator:
    def __init__(self, decorators, edge_distance_cutoff_A, max_residues, nbr_distance_cutoff_A):
        self.data_maker = mg.DataMaker(decorators=decorators,
                                       edge_distance_cutoff_A=edge_distance_cutoff_A,
                                       max_residues=max_residues,
                                       nbr_distance_cutoff_A=nbr_distance_cutoff_A)

    def summary(self):
        self.data_maker.summary()

    def generate_XAE_input_layers(self):
        return self.data_maker.generate_XAE_input_layers()

def calculate_distance(pose, res1, res2):
    coord1 = pose.residue(res1).xyz("CA")
    coord2 = pose.residue(res2).xyz("CA")
    
    # Calculate the Euclidean distance between the two coordinates
    distance = math.sqrt((coord1[0] - coord2[0])**2 + 
                         (coord1[1] - coord2[1])**2 + 
                         (coord1[2] - coord2[2])**2)
    return distance

def get_cys(pose):
    disulfides = []
    for i in range(1, pose.size()+1):
        if pose.residue(i).name() == "CYS":
            res = pose.residue(i)
            disulfide_partner = res.residue_connection_partner(res.n_current_residue_connections())
            if disulfide_partner > i:
                disulfides.append([i, disulfide_partner])
    return disulfides

def get_negative_examples(pose, max_samples=20):
    negative_examples = []
    pairs = []
    disulfides = get_cys(pose)
    disulfide_res = {i for bond in disulfides for i in bond}

    for i in range(1, pose.size() + 1):
        if i not in disulfide_res:
            negative_examples.append(i)

    for i in negative_examples:
        for j in negative_examples:
            if i != j and j > i:
                distance = calculate_distance(pose, i, j)
                if distance <= 10:
                    pairs.append([i, j])
    return random.sample(pairs, min(max_samples, len(pairs)))

def process_and_save_chunk(chunk_index, pdb_chunk, path, edge_distance_cutoff_A, max_residues, nbr_distance_cutoff_A):
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
    data_maker = mg.DataMaker(decorators=decorators,
                               edge_distance_cutoff_A=edge_distance_cutoff_A,
                               max_residues=max_residues,
                               nbr_distance_cutoff_A=nbr_distance_cutoff_A)

    Xs_chunk = []
    As_chunk = []
    Es_chunk = []
    outs_chunk = []

    for pdb_file in pdb_chunk:
        pose = pyrosetta.pose_from_pdb(pdb_file)
        cys = get_cys(pose)
        no_cys = get_negative_examples(pose)
        wrapped_pose = mg.RosettaPoseWrapper(pose)
        cache = data_maker.make_data_cache(wrapped_pose)

        for i in cys:
            X, A, E, resids = data_maker.generate_input( wrapped_pose, i, data_cache=cache )
            Xs_chunk.append(X)
            As_chunk.append(A)
            Es_chunk.append(E)
            outs_chunk.append([1.0,])

        for i in no_cys:
            X, A, E, resids = data_maker.generate_input( wrapped_pose, i, data_cache=cache )
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

def load_npz_files(path):
    data = np.load(f'{path}/merged_file.npz')
    Xs = data['Xs']
    As = data['As']
    Es = data['Es']
    outs = data['outs']
    return Xs, As, Es, outs

class GCNModel:
    def __init__(self, data_maker):
        self.X_in, self.A_in, self.E_in = data_maker.generate_XAE_input_layers()
        self.model = self._build_model()

    def _build_model(self):
        X_noisy = GaussianNoise(0.1)(self.X_in)
        E_noisy = GaussianNoise(0.1)(self.E_in)

        L1 = ECCConv(64, activation=None)([X_noisy, self.A_in, E_noisy])
        L1_bn = BatchNormalization()(L1)
        L1_act = Activation('relu')(L1_bn)
        L1_drop = Dropout(0.2)(L1_act)

        L2 = ECCConv(32, activation=None)([L1_drop, self.A_in, E_noisy])
        L2_bn = BatchNormalization()(L2)
        L2_act = Activation('relu')(L2_bn)
        L2_drop = Dropout(0.2)(L2_act)

        L3 = ECCConv(16, activation=None)([L2_drop, self.A_in, E_noisy])
        L3_bn = BatchNormalization()(L3)
        L3_act = Activation('relu')(L3_bn)
        L3_drop = Dropout(0.2)(L3_act)

        L4 = GATConv(8, attn_heads=2, concat_heads=True, activation=None)([L3_drop, self.A_in])
        L4_bn = BatchNormalization()(L4)
        L4_act = Activation('relu')(L4_bn)
        L4_drop = Dropout(0.2)(L4_act)

        L5 = GlobalMaxPool()(L4_drop)
        L6 = Flatten()(L5)
        output = Dense(1, name="out", activation="sigmoid", kernel_regularizer=l2(0.01))(L6)

        model = Model(inputs=[self.X_in, self.A_in, self.E_in], outputs=output)
        return model

    def compile_model(self, learning_rate=config.LEARNING_RATE):
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=opt, loss='binary_crossentropy')

    def summary(self):
        self.model.summary()

    def fit(self, X_train, A_train, E_train, out_train, X_test, A_test, E_test, out_test, batch_size=config.BATCH_SIZE, epochs=config.EPOCHS, callbacks=None):
        history = self.model.fit(x=[X_train, A_train, E_train], y=out_train, batch_size=batch_size, epochs=epochs, validation_data=([X_test, A_test, E_test], out_test), callbacks=callbacks)
        return history

    def predict(self, X_test, A_test, E_test):
        return self.model.predict([X_test, A_test, E_test])

    def save(self, filename):
        self.model.save(filename)

def apply_sampler(sampler, X, y):
    if sampler is None:
        return X, y
    Xs_reshaped = X.reshape(X.shape[0], -1)
    X_res, y_res = sampler.fit_resample(Xs_reshaped, y)
    return X_res.reshape(-1, *X.shape[1:]), y_res

def get_sampler(X_train, out_train, desired_pos_ratio=config.DESIRED_POS_RATIO):
    n_samples = X_train.shape[0]
    n_pos = int(n_samples * desired_pos_ratio)
    n_neg = n_samples - n_pos

    counter = Counter(out_train.flatten())
    current_pos = counter[1]
    current_neg = counter[0]

    if current_pos < n_pos:
        sampling_strategy = {0: current_neg, 1: n_pos}
        sampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
    elif current_pos > n_pos:
        sampling_strategy = {0: n_neg, 1: current_pos}
        sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
    else:
        sampler = None
    return sampler
