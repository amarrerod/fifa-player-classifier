#!/usr/bin/python3


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from money_transformer import MoneyTransformer
import re
import random
from datetime import datetime

SEED = 13
SCALER = 500000000
NO_VALUE = -100
np.random.seed(SEED)
random.seed(SEED)

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
    X["Club"].replace(-100, "Unknown", inplace=True)
    X["Preferred Foot"].replace(-100, "Unknown", inplace=True)

    cat_attribs = ["Nationality", "Club", "Preferred Foot"]

    c_transformer = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown='ignore'), cat_attribs)
    ])

    X_prepared = c_transformer.fit_transform(X)
    print("Dataset generated")
    return X_prepared, y.values


def create_subset(X, y, val_size=0.15, test_size=0.15):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=True)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=val_size, shuffle=True)

    return X_train, X_valid, X_test, y_train, y_valid, y_test


if __name__ == "__main__":
    X, y = generate_dataset()
    X_train, X_valid, X_test, y_train, y_valid, y_test = create_subset(X, y)
