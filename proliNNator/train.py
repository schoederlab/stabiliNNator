import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pyrosetta
pyrosetta.init('-mute all')
from pyrosetta.rosetta.core.scoring import ScoreType
import menten_gcn as mg
import menten_gcn.decorators as decs

import utils
from utils import GraphDataGenerator, GCNModel, load_npz_files, get_sampler, apply_sampler
import config

from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_curve, roc_auc_score

plt.rcParams.update({'font.size': 15})
plt.rcParams['axes.linewidth'] = 2


# Extract arrays from DataFrame
X_train, A_train, E_train, out_train = load_npz_files(config.TRAIN_DATA_PATH)
X_test, A_test, E_test, out_test = load_npz_files(config.TEST_DATA_PATH)
X_val, A_val, E_val, out_val = load_npz_files(config.VAL_DATA_PATH)

# Convert lists to numpy arrays
X_train = np.asarray(X_train)
A_train = np.asarray(A_train)
E_train = np.asarray(E_train)
out_train = np.asarray(out_train)

X_test = np.asarray(X_test)
A_test = np.asarray(A_test)
E_test = np.asarray(E_test)
out_test = np.asarray(out_test)

# Pick decorators
decorators = [decs.SequenceSeparation(ln = True),
              decs.SimpleBBGeometry(use_nm = False),
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

# Print summary
data_generator.summary()

# Weighted Random Over Sampling to balance classes according to natural prevalence (5-7% positive)
desired_pos_ratio = config.DESIRED_POS_RATIO
sampler = get_sampler(X_train, out_train, desired_pos_ratio)

X_ros, y_ros = apply_sampler(sampler, X_train, out_train)
A_ros, _     = apply_sampler(sampler, A_train, out_train)
E_ros, _     = apply_sampler(sampler, E_train, out_train)

E_masked = utils.mask_edge_features_by_target_node(X_ros, E_ros)

# Define GCN model
gcn_model = GCNModel(data_generator)
gcn_model.compile_model()
gcn_model.summary()

# Early stopping callback
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    verbose=2,
    patience=config.EARLY_STOPPING_PATIENCE,
    mode='min',
    restore_best_weights=True
)

# Train the model
history = gcn_model.fit(X_ros, A_ros, E_masked, y_ros, X_test, A_test, E_test, out_test, batch_size=config.BATCH_SIZE, epochs=config.EPOCHS, callbacks=[early_stopping])
gcn_model.save(config.MODEL_SAVE_FILENAME)

#further validation
y_pred_prob = gcn_model.predict(X_val, A_val, E_val)
y_pred_prob = np.round(y_pred_prob, 5)
y_pred = (y_pred_prob > 0.5).astype(int)

mcc = matthews_corrcoef(out_test, y_pred)
print('Matthews Coefficient:', mcc)
fpr, tpr, thresholds = roc_curve(out_test, y_pred_prob)
auc = roc_auc_score(out_test, y_pred_prob)
print('auc:', auc)
precision, recall, thresholds = precision_recall_curve(out_test, y_pred_prob)
cm = confusion_matrix(out_test, y_pred)
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

#Plottings
plt.figure(figsize=(6, 6))
plt.plot(history.history['loss'], label='Training Loss', color='#004B6F')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('history.svg')
plt.clf()

plt.figure(figsize=(6, 6))
plt.plot(recall, precision, color='#004B6F')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.savefig('precision-recall.svg')
plt.clf()

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label='ROC curve', color='#004B6F')
plt.xscale('log')  # Set x-axis to logarithmic scale
plt.xlabel('Log False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.xlim([0.00001, 1])  # Set limits for the logarithmic scale
plt.ylim([0.0, 1.05])
plt.title('Log ROC Curve')
plt.savefig('log-roc.svg')
plt.clf()

plt.figure(figsize=(6, 6))
sns.heatmap(cmn, annot=True, fmt='g', cmap='Blues', xticklabels=['Non-Proline', 'Proline'], yticklabels=['Non-Proline', 'Proline'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('conf-m.svg')
plt.clf()