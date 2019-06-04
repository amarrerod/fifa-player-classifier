#!/usr/bin/python3


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import re
import random
from datetime import datetime
import tensorflow as tf
from tensorflow import keras

# Constantes
SEED = 13
SCALER = 500000000
NO_VALUE = -100
UNKNOWN = "Unknown"

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


FILENAME = "data.csv"
USELESS = ['Unnamed: 0', 'ID', 'Photo', 'Flag',
           'Club Logo', 'Real Face', 'Jersey Number',
           'Loaned From', 'Contract Valid Until', 'Release Clause']


def get_players_properties():
    data = pd.read_csv(FILENAME)
    print(data.head())
    print(f"Columns: \n{data.columns}")
    print(f"Info about columns: \n{data.info()}")
    # Eliminamos las columnas que no son necesarias
    dataframe = data.drop(columns=USELESS)
    properties = dataframe[['Overall', 'Crossing', 'Finishing', 'HeadingAccuracy',
                            'ShortPassing', 'Volleys', 'Dribbling',
                            'Curve', 'FKAccuracy', 'LongPassing', 'BallControl',
                            'Acceleration', 'SprintSpeed', 'Agility', 'Reactions',
                            'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength',
                            'LongShots', 'Aggression', 'Interceptions', 'Positioning',
                            'Vision', 'Penalties', 'Composure', 'Marking',
                            'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',
                            'GKKicking', 'GKPositioning', 'GKReflexes']]
    properties.dropna(inplace=True)  # Nos quitamos los NaN
    print(f"Properties describe: \n{properties.describe()}")
    return properties


def generate_subsets(X, y, val_size=0.15, test_size=0.15):
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=SEED)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full, test_size=0.15, random_state=SEED)

    return X_train, X_valid, X_test, y_train, y_valid, y_test


if __name__ == "__main__":
    properties = get_players_properties()
    print(properties.corr()["Overall"].sort_values(ascending=False))
    # Eliminamos las variables colineares
    properties = properties[['Overall', 'Strength', 'Stamina',
                             'Jumping', 'Composure', 'Reactions',
                             'ShortPassing', 'GKKicking']]
    print(properties.corr()["Overall"].sort_values(ascending=False).head(12))
    X = properties[['Strength', 'Stamina',
                    'Jumping', 'Composure', 'Reactions',
                    'ShortPassing', 'GKKicking']]
    y = properties["Overall"]
    X_train, X_valid, X_test, y_train, y_valid, y_test = generate_subsets(X, y)

    # StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    model = keras.models.Sequential([
        keras.layers.Dense(7, activation="relu", input_shape=[7]),
        keras.layers.Dense(7, activation="relu"),
        keras.layers.Dense(7, activation="relu"),
        keras.layers.Dense(1)
    ])

    #optimizer = tf.keras.optimizers.RMSprop(0.001)
    lr0 = 0.001
    s = 20 * len(X_train) // 32
    decay = 0.1
    lr_scheduler = keras.optimizers.schedules.ExponentialDecay(lr0, s, decay)
    optimizer = tf.keras.optimizers.SGD(lr_scheduler)
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_squared_error'])

    history = model.fit(X_train, y_train, epochs=100,
                        validation_data=(X_valid, y_valid),
                        callbacks=[keras.callbacks.EarlyStopping(patience=10)])
    pd.DataFrame(history.history).plot()
    plt.show()
