import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow	import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score

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

# create neural network
