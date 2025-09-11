import numpy as np
import menten_gcn as mg
import menten_gcn.decorators as decs
from pyrosetta.rosetta.core.scoring import ScoreType

from spektral.layers import *
from tensorflow import keras
from keras.regularizers import l2
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
# Removed: from tensorflow.keras import regularizers

from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

import matplotlib.pyplot as plt
import seaborn as sns

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
        X_noisy = GaussianNoise(0.2)(self.X_in)

        L1 = ECCConv(32, activation=None)([X_noisy, self.A_in, self.E_in])
        L1_bn = BatchNormalization()(L1)
        L1_act = Activation('relu')(L1_bn)

        L1_2 = ECCConv(16, activation=None)([L1_act, self.A_in, self.E_in])
        L1_bn2 = BatchNormalization()(L1_2)
        L1_act2 = Activation('relu')(L1_bn2)

        L2 = GATConv(8, attn_heads=2, concat_heads=True, activation=None)([L1_act2, self.A_in])
        L2_bn = BatchNormalization()(L2)
        L2_act = Activation('relu')(L2_bn)
        L2_drop = Dropout(0.2)(L2_act)

        L3 = GlobalMaxPool()(L2_drop)
        L4 = Flatten()(L3)
        output = Dense(1, name="out", activation="sigmoid", kernel_regularizer=l2(0.01))(L4)

        model = Model(inputs=[self.X_in, self.A_in, self.E_in], outputs=output)
        return model

    def compile_model(self, learning_rate=config.LEARNING_RATE):
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=opt, loss='binary_crossentropy')

    def summary(self):
        self.model.summary()

    def fit(self, X_train, A_train, E_train, out_train, X_test, A_test, E_test, out_test, batch_size=50, epochs=500, callbacks=None):
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

def get_sampler(X_train, out_train, desired_pos_ratio=0.07):
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

def get_proline_positions(pose):
    proline_positions = []
    for i in range(1, pose.size() + 1):
        if pose.residue(i).name() == "PRO":
            proline_positions.append(i)
    return proline_positions

def get_non_proline_positions(pose, proline_positions):
    non_proline_positions = [i for i in range(1, pose.size() + 1) if i not in proline_positions]
    return non_proline_positions