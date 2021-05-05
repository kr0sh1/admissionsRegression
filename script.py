import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score


def build_model(features, rate):
    model = Sequential(name="outputModel")
    input = tf.keras.Input(shape=(features.shape[1],))
    model.add(input)
    model.add(Dense(128, activation='relu'))
    model.add(layers.Dropout(0.1))  # dropout point added
    model.add(Dense(1))
    opt = tf.keras.optimizers.SGD(learning_rate=rate)
    model.compile(loss='mse', metrics=['mae'], optimizer=opt)
    return model


def fit_model(model, f_train, l_train, learn, n_epochs):
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    history = model.fit(features_train, labels_train, epochs=n_epochs, batch_size=16, verbose=0, validation_split=0.2,
                        callbacks=[es])
    return history


admissionData = pd.read_csv("admissions_data.csv")

admissionData = admissionData.drop(["Serial No."], axis=1)
labels = admissionData.iloc[:, -1]

# remove uni rating and TOEFL score - unethical?
# remove serial no. and research - irrelevant info
features = admissionData.iloc[:, [0, 3, 4, 5, 6]]

# split dataset into train and test
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3,
                                                                            random_state=42)

# scale/normalise dataset features
ct = ColumnTransformer([("normalize", Normalizer(), [0, 1, 2, 3])], remainder='passthrough')
features_train = ct.fit_transform(features_train)
features_test = ct.transform(features_test)

learning_rate = 0.001
num_epochs = 20

# create neural network

#  admissionsModel = build_model(features_train, learning_rate)  # rewrite this function
#  admissionsModel.fit(features_train, labels_train, epochs=20, batch_size=1, verbose=1)
history1 = fit_model(build_model(features_train, learning_rate), features_train, labels_train, learning_rate,
                     num_epochs)

#  need to return the fitted model into a graph somehow here


plt.savefig('perf_graph.png')

# val_mse, val_mae = None, None
# val_mse, val_mae = admissionsModel.evaluate(features_test, labels_test, verbose=0)
#
# print("MAE: ", val_mae)
