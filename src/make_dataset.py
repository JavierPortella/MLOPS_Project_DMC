import pandas as pd
import numpy as np
import os
from imblearn.over_sampling import RandomOverSampler

FEATURES=[
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]
TARGET=["isFraud"]

def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../data/raw/', filename))
    print(filename, ' cargado correctamente')
    return df

def balance_data(df):
    unds = RandomOverSampler(sampling_strategy=0.6)
    x_us, y_us = unds.fit_resample(df[FEATURES].values, df[TARGET].values)
    df = pd.concat([pd.DataFrame(x_us,columns=FEATURES), pd.DataFrame(y_us,columns=TARGET)], axis = 1)
    print('Balanceo de datos completa')
    return df

def data_exporting(df, features, filename):
    dfp = df[features]
    dfp.to_csv(os.path.join('../data/processed/', filename), index=False)
    print(filename, 'exportado correctamente en la carpeta processed')
    
def main():
    columns_to_work=FEATURES+TARGET
    # Matriz de Entrenamiento
    df = read_file_csv('default_bank_train.csv')
    tdf1 = balance_data(df)
    data_exporting(tdf1, columns_to_work,'bank_train.csv')
    # Matriz de Validación
    df = read_file_csv('default_bank_test.csv')
    data_exporting(df, columns_to_work,'bank_test.csv')
    # Matriz de Scoring
    df = read_file_csv('default_bank_score.csv')
    data_exporting(df, FEATURES,'bank_score.csv')
    
if __name__ == "__main__":
    main()