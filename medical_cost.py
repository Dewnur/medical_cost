import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from settings import SAVE_MODELS, LOAD_MODELS, best_dt_filename, best_rf_filename, param_grid
from utils import graph
from utils.optimization import get_best_params

if __name__ == '__main__':
    df = pd.read_csv('data/insurance.csv')

    print(df.head())

    df.sex.replace({'female': 1, 'male': 0}, inplace=True)
    df.smoker.replace({'yes': 1, 'no': 0}, inplace=True)
    df.region.replace({'southeast': 0, 'southwest': 1, 'northwest': 2, 'northeast': 3}, inplace=True)
    df.drop(['sex', 'region', 'children'], axis=1, inplace=True)

    print(df.corr())

    X = df.drop('charges', axis=1)
    y = df['charges']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    features = X_train.columns

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    dt = DecisionTreeRegressor(random_state=42)
    rf = RandomForestRegressor(random_state=42)

    if LOAD_MODELS:
        with open(best_dt_filename, 'rb') as file:
            best_dt = pickle.load(file)
        with open(best_rf_filename, 'rb') as file:
            best_rf = pickle.load(file)
    else:
        dt_best_params = get_best_params(dt, X_train, y_train, param_grid)
        rf_best_params = get_best_params(rf, X_train, y_train, param_grid)
        best_dt = DecisionTreeRegressor(**dt_best_params)
        best_rf = RandomForestRegressor(**rf_best_params)
        best_dt.fit(X_train, y_train)
        best_rf.fit(X_train, y_train)

    if SAVE_MODELS:
        with open(best_dt_filename, 'wb') as file:
            pickle.dump(best_dt, file)
        with open(best_rf_filename, 'wb') as file:
            pickle.dump(best_rf, file)

    dt_predict = best_dt.predict(X_val)
    rf_predict = best_rf.predict(X_val)

    print(
        f'Дерево решений\nr2_score: {r2_score(dt_predict, y_val)} | mean_squared_error: {mean_squared_error(dt_predict, y_val)}')
    print(
        f'Случайный лес\nr2_score: {r2_score(rf_predict, y_val)} | mean_squared_error: {mean_squared_error(rf_predict, y_val)}')

    dt_importances = best_dt.feature_importances_
    rf_importances = best_rf.feature_importances_

    dt_indices = np.argsort(dt_importances)
    rf_indices = np.argsort(rf_importances)

    graph.feature_importances(dt_indices, dt_importances, features, 'Дерево решений | Feature Importances')
    graph.feature_importances(rf_indices, rf_importances, features, 'Случайный лес | Feature Importances')
