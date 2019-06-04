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
GK = "GK"


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
    properties_w_pos = dataframe[['Overall', 'Crossing', 'Finishing', 'HeadingAccuracy',
                                  'ShortPassing', 'Volleys', 'Dribbling',
                                  'Curve', 'FKAccuracy', 'LongPassing', 'BallControl',
                                  'Acceleration', 'SprintSpeed', 'Agility', 'Reactions',
                                  'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength',
                                  'LongShots', 'Aggression', 'Interceptions', 'Positioning',
                                  'Vision', 'Penalties', 'Composure', 'Marking',
                                  'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',
                                  'GKKicking', 'GKPositioning', 'GKReflexes', 'Position']]
    properties = properties_w_pos.loc[:,
                                      properties_w_pos.columns != "Position"]
    properties.dropna(inplace=True)  # Nos quitamos los NaN
    properties_w_pos.dropna(inplace=True)
    print(f"Properties describe: \n{properties.describe()}")
    return properties, properties_w_pos


def generate_subsets(X, y, val_size=0.15, test_size=0.15):
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=SEED)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full, test_size=0.15, random_state=SEED)

    # StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def build_model(params):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(
        params["n_input"], input_shape=[params["n_input"]]))
    for i in range(params["n_hidden"]):
        model.add(keras.layers.Dense(params["n_neurons"], activation="relu"))

    # Incluimos la ultima capa
    model.add(params["output_layer"])
    lr0 = params["lr0"]
    s = params["s"]
    decay = params["decay"]
    lr_scheduler = keras.optimizers.schedules.ExponentialDecay(lr0, s, decay)
    optimizer = params["optimizer"](lr_scheduler)
    model.compile(loss=params["loss"],
                  optimizer=optimizer, metrics=params["metrics"])
    return model


def train_model(model, X_train, X_valid, X_test, y_train, y_valid, y_test, epochs=100):
    history = model.fit(X_train, y_train, epochs=100,
                        validation_data=(X_valid, y_valid),
                        callbacks=[keras.callbacks.EarlyStopping(patience=10)])
    pd.DataFrame(history.history).plot()
    plt.show()


def predict_overall(X_train, X_valid, X_test, y_train, y_valid, y_test):
    params = {
        "n_input": 7,
        "n_hidden": 2,
        "n_neurons": 7,
        "output_layer": keras.layers.Dense(1),
        "lr0": 0.001,
        "s": 20 * len(X_train) // 32,
        "decay": 0.1,
        "optimizer": keras.optimizers.SGD,
        "loss": "mean_squared_error",
        "metrics": ['mean_squared_error']
    }
    model = build_model(params)
    train_model(model, X_train, X_valid, X_test,
                y_train, y_valid, y_test, epochs=100)


def predict_position(X_train, X_valid, X_test, y_train, y_valid, y_test,
                     n_inputs, n_targets):
    params = {
        "n_input": n_inputs,
        "n_hidden": 4,
        "n_neurons": 90,
        "output_layer": keras.layers.Dense(n_targets, activation="softmax"),
        "lr0": 0.05,
        "s": 20 * len(X_train) // 32,
        "decay": 0.1,
        "optimizer": keras.optimizers.SGD,
        "loss": "sparse_categorical_crossentropy",
        "metrics": ['accuracy']
    }

    model = build_model(params)
    train_model(model, X_train, X_valid, X_test,
                y_train, y_valid, y_test, epochs=200)


def simplify_position(entry):
    entry = str(entry)
    if entry == GK:  # Portero
        return 0
    elif entry.endswith("B"):  # Defensa
        return 1
    elif entry.endswith("M") or entry.endswith("W"):
        return 2
    elif entry.endswith("S") or entry.endswith("T") or entry.endswith("F"):
        return 3
    else:
        return -1


if __name__ == "__main__":
    properties, properties_w_pos = get_players_properties()
    print(properties.corr()["Overall"].sort_values(ascending=False))
    # Eliminamos las variables colineares
    properties = properties[['Overall', 'Strength', 'Stamina',
                             'Jumping', 'Composure', 'Reactions',
                             'ShortPassing', 'GKKicking']]
    print(properties.corr()["Overall"].sort_values(ascending=False).head(12))

    # Estimacion de valoraciones
    X = properties[['Strength', 'Stamina',
                    'Jumping', 'Composure', 'Reactions',
                    'ShortPassing', 'GKKicking']]
    y = properties["Overall"]

    X_train, X_valid, X_test, y_train, y_valid, y_test = generate_subsets(X, y)
    predict_overall(X_train, X_valid, X_test, y_train, y_valid, y_test)

    # Clasificacion de jugadores
    properties_w_pos["Position"] = properties_w_pos.Position.apply(
        simplify_position)
    print(f"Unique positions: \n{properties_w_pos['Position'].unique()}")

    y = properties_w_pos["Position"]
    X = properties_w_pos.drop(columns=["Position"])
    X_train, X_valid, X_test, y_train, y_valid, y_test = generate_subsets(X, y)
    n_targets = len(y.unique())
    n_inputs = len(X.columns)
    predict_position(X_train, X_valid, X_test, y_train,
                     y_valid, y_test, n_inputs, n_targets)
