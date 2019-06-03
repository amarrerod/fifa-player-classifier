#!/usr/bin/python3


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import re
import random
from datetime import datetime
import tensorflow as tf
from tensorflow import keras


SEED = 13
SCALER = 500000000
NO_VALUE = -100
UNKNOWN = "Unknown"

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


filename = "data.csv"
col_to_drop = ["Unnamed: 0", "ID", "Name", "Flag", "Club Logo", "Photo",
               "Real Face", "Body Type", "LS", "ST", "RS", "LW", "LF", "CF",
               "RF", "RW", "LAM", "CAM", "RAM", "LM", "LCM", "CM", "RCM", "RM",
               "LWB", "LDM", "CDM", "RDM", "RWB", "LB", "LCB", "CB", "RCB", "RB",
               "Work Rate", "Loaned From"]


def cast_money_to_float(entry):
    entry = re.sub(r'[€(M|K)$]',  '', str(entry))
    return float(entry) / SCALER


def reformat_date(entry):
    if entry == NO_VALUE:
        return entry
    else:
        date = datetime.strptime(entry, '%b %d, %Y')
        today = datetime.today()
        ldays = today - date
        return int(ldays.days)


def contract_format(entry):
    try:
        datetime.strptime(str(entry), "%Y")
        return int(entry)
    except ValueError:
        if entry != NO_VALUE:
            date = datetime.strptime(entry, '%b %d, %Y')
            return int(date.year)


def generate_dataset():
    dataset = pd.read_csv(filename).drop(columns=col_to_drop)
    dataset = dataset.fillna(NO_VALUE)

    # Extraemos las columnas de texto y numeros
    cat_attribs = dataset.select_dtypes(include="object").columns
    cat_attribs = cat_attribs.drop("Position")
    num_attribs = dataset.select_dtypes(include=["float64", "int"]).columns

    # Cast str yo float
    dataset["Weight"] = dataset["Weight"].apply(
        lambda x: float(re.sub(r'[lbs$]', '', str(x))))
    dataset["Height"] = dataset["Height"].apply(
        lambda x: float(re.sub(r'[\']', '.', str(x))))
    dataset["Release Clause"] = dataset["Release Clause"].apply(
        cast_money_to_float)
    dataset["Value"] = dataset["Value"].apply(cast_money_to_float)
    dataset["Wage"] = dataset["Wage"].apply(cast_money_to_float)

    dataset["Days From Join"] = dataset["Joined"].apply(reformat_date)
    dataset["Contract Valid Until"] = dataset["Contract Valid Until"].apply(
        contract_format)
    dataset.drop(columns="Joined")

    y = dataset["Position"]
    X = dataset.drop(columns=["Position"])
    X["Club"].replace(NO_VALUE, UNKNOWN, inplace=True)
    X["Preferred Foot"].replace(NO_VALUE, UNKNOWN, inplace=True)
    y.replace(NO_VALUE, UNKNOWN, inplace=True)
    cat_attribs = ["Nationality", "Club", "Preferred Foot"]

    c_transformer = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown='ignore'), cat_attribs)
    ])

    X_prepared = c_transformer.fit_transform(X)
    y_prepared = LabelEncoder().fit_transform(y)
    print("Dataset generated")

    return X_prepared, y_prepared


def create_subset(X, y, val_size=0.1, test_size=0.1):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=True, stratify=y)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=val_size, shuffle=True, stratify=y_train)

    # scale dataset
    scaler = StandardScaler(with_mean=False)
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    return X_train, X_valid, X_test, y_train, y_valid, y_test


if __name__ == "__main__":
    X, y = generate_dataset()
    X_train, X_valid, X_test, y_train, y_valid, y_test = create_subset(X, y)
    print(X_train.shape)
    print(len(np.unique(y)))
    model = keras.models.Sequential([
        keras.layers.Dense(1000, input_shape=(819, ), activation="relu"),
        keras.layers.Dense(1000, activation="relu"),
        keras.layers.Dense(1000, activation="relu"),
        keras.layers.Dense(1000, activation="relu"),
        keras.layers.Dense(28, activation="softmax")
    ])
    print(f"Model summary: \n{model.summary()}")
    model.compile(optimizer="SGD", metrics=[
                  "accuracy"], loss="sparse_categorical_crossentropy")

    model.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid),
              callbacks=[keras.callbacks.EarlyStopping(patience=10)])
