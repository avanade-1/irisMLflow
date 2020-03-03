import os
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import *
from sklearn.model_selection import train_test_split

def ava_innova(data_path, n_estimators, max_depth, max_features):

  with mlflow.start_run() as run:
    # Crear modelo, entrenarlo y crear predicciones
    rf = RandomForestClassifier(**params) #Llamar al modelo
    rf.fit(X_train, y_train) #Enternarlo
    predictions = rf.predict(X_test) #Predicciones

    # Log del modelo
    mlflow.sklearn.log_model(rf, "random-forest-model")

    # Log de los parametros
    [mlflow.log_param(param, value) for param, value in params.items()]

      #Metricas de evaluacion
      #rmse = distancia media cuadrática mínima
      #mae = error medio absoluto
      #r2 = coeficiente de determinacion
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print("  mse: {}".format(mse))
    print("  mae: {}".format(mae))
    print("  R2: {}".format(r2))

    # Log de las métricas
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)  
    mlflow.log_metric("r2", r2)
