# Loading Packages

from __future__ import division, print_function
import _ctypes
import numpy as np
import pandas as pd
import os
import time
import math
import statistics
from collections import Counter
from Bio import SeqIO
import tensorflow as tf
from keras import backend as K
from keras.utils import np_utils
# from keras.utils.generic_utils import get_custom_objects
from keras.preprocessing import sequence
from keras import initializers
from keras.models import Sequential, load_model, Model
from keras.layers import Embedding, ZeroPadding1D, TimeDistributed, Conv1D, MaxPooling1D, LSTM, Bidirectional, \
    Activation, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.regularizers import l1, l2, l1_l2
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard, ReduceLROnPlateau
from sklearn.utils import shuffle, resample
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, \
    f1_score, matthews_corrcoef, cohen_kappa_score, auc, roc_auc_score, roc_curve
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import h5py

Result_dir = 'Species_Classification_Results'


def species_model(data):
    Check_dir_Result = os.path.isdir(Result_dir)
    if not Check_dir_Result:
        os.makedirs(Result_dir)
        print("created folder : ", Result_dir)

    else:
        print(Result_dir, "folder already exists.")

    filepath = os.path.join('Results', 'classification_feature','6mer_features.csv')
    data = pd.read_csv(filepath)
    data1 = data.iloc[:, ]
    X = data1.iloc[:, 1:]

    scaler = StandardScaler()
    Z = scaler.fit(X)
    # print(Z)
    # print(scaler.mean_)
    X = scaler.transform(X)
    print(X)

    X = X.reshape(X.shape + (1,))
    X = np.array(X, dtype=float)

    species_path = os.path.join(os.getcwd(), 'static', 'Species_names.csv')
    species_names = pd.read_csv(species_path)
    species_names = species_names.iloc[:, ]

    # Species Level Prediction
    model_path = os.path.join(os.getcwd(), 'static',
                              'models', 'Species_Model.h5')
    #species_model = load_model(model_path)
    input_length = 4096
    nb_classes = 429
    
    species_model = Sequential()
    species_model.add(Conv1D(32, 3, input_shape=(input_length, 1), padding="valid")) #input_dim
    species_model.add(Activation('relu'))
    species_model.add(MaxPooling1D(pool_size=2, padding="valid"))
    species_model.add(Conv1D(64, 3, padding="valid"))
    species_model.add(Activation('relu'))
    species_model.add(MaxPooling1D(pool_size=2, padding="valid"))
    species_model.add(Flatten())
    ##
    ##MLP
    #model.add(Dropout(0.2))
    species_model.add(Dense(500))
    species_model.add(Activation('relu'))
    species_model.add(Dropout(0.5))
    species_model.add(Dense(nb_classes))
    species_model.add(Activation('softmax'))
    species_model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
              
    species_model.load_weights(model_path)
    
    
    species_model.summary()

    species_test_prediction = species_model.predict(X, verbose=1)

    predict_probability = pd.DataFrame(species_test_prediction)
    predict_probability.to_csv(
        "Species_Classification_Results/PredictionProbabilities_species.csv")

    # Setting up Appropriate thresholds for designating outlier points as "unknown species"

    predicted_species = []
    unknown = 'unknown'
    for i in species_test_prediction:
        if i.max() < 0.90:
            # print (unknown)
            predicted_species.append(unknown)
        else:
            # print (np.argmax(i))
            j = np.argmax(i)
            species = species_names["Species"][j]
            predicted_species.append(species)

    final_predictions = pd.DataFrame()
    final_predictions['Species'] = predicted_species
    final_predictions.to_csv(
        "Species_Classification_Results/Predicted_Species.csv")
