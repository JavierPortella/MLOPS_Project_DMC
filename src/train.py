import pandas as pd
from sklearn.neural_network import MLPClassifier
import pickle
import os

TARGET=["isFraud"]

def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../data/processed', filename))
    X_train = df.drop(TARGET,axis=1)
    y_train = df[TARGET]
    print(filename, ' cargado correctamente')
    # Entrenamos el modelo con toda la muestra
    mlp=MLPClassifier(random_state=2020, activation='relu', max_iter=200, hidden_layer_sizes=(100,))
    mlp.fit(X_train,y_train)
    print('Modelo entrenado')
    # Guardamos el modelo entrenado para usarlo en produccion
    package = '../models/best_model.pkl'
    pickle.dump(mlp, open(package, 'wb'))
    print('Modelo exportado correctamente en la carpeta models')
    
def main():
    read_file_csv('bank_train.csv')
    print('Finalizó el entrenamiento del Modelo')


if __name__ == "__main__":
    main()