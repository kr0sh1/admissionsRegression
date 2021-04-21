import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import kerastuner as kt
from tensorflow	import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score

def build_model(X): # THIS NEEDS REWRITING
    model = Sequential(name="outputModel")
    input = tf.keras.Input(shape=(X.shape[1],))
    model.add(input)
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1))
    opt = tf.keras.optimizers.SGD(learning_rate=0.1)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    return model

admissionData = pd.read_csv("admissions_data.csv")

admissionData = admissionData.drop(["Serial No."], axis=1)
labels = admissionData.iloc[:, -1]


# remove uni rating and TOEFL score - unethical?
# remove serial no. and research - irrelevant info
features = admissionData.iloc[:, [0, 3, 4, 5, 6]]


# split dataset into train and test
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# scale/normalise dataset features
ct = ColumnTransformer([("normalize", Normalizer(), [0, 1, 2, 3])], remainder='passthrough')
features_train = ct.fit_transform(features_train)
features_test = ct.transform(features_test)

learning_rate= 0.1
num_epochs = 200

# create tuner
tuner = kt.Hyperband(build_model, objective='val_accuracy', max_epochs=10, factor=3, directory='my_dir', project_name='intro_to_kt')
stop_early = EarlyStopping(monitor='val_loss', verbose=1, patience=5)
tuner.search(features_train, labels_train, epochs=50, validation_split=0.2, callbacks=[stop_early])

# pass admissionsModel/history
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_hps)
history = model.fit(features_train, labels_train, epochs=50, validation_split=0.2)

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

hypermodel = tuner.hypermodel.build(best_hps)
hypermodel.fit(features_train, labels_train, epochs=best_epoch, validation_split=0.2)

# evaluate
eval_result = hypermodel.evaluate(features_test, labels_test)
print("[test loss, test accuracy]:", eval_result)
