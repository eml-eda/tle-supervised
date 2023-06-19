from Datasets.Vehicles_Roccaprebalza.get_dataset import get_data

import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline

def compute_accuracy(y_test, y_predicted):
    mse = mean_squared_error(y_test, y_predicted)
    print("MSE:", mse)
    mae = mean_absolute_error(y_test, y_predicted)
    print("MAE:", mae)
    r2 = r2_score(y_test, y_predicted)
    print("R2:", r2)
    mspe = (mse/np.mean(y_test))*100
    print("MSE%:", mspe)
    mape = (mae/np.mean(y_test))*100
    print("MAE%:", mape)

def algorithm(data, labels, number_of_features):
    X_car = SelectKBest(f_regression, k = number_of_features).fit_transform(data, labels["y_car"])
    X_camion = SelectKBest(f_regression, k = number_of_features).fit_transform(data, labels["y_camion"])
    X_train_car, X_test_car, y_train_car, y_test_car = train_test_split(X_car, labels["y_car"])
    X_train_camion, X_test_camion, y_train_camion, y_test_camion = train_test_split(X_camion, labels["y_camion"])
    steps_svr = [( 'scaler', StandardScaler() ), ('svr', SVR(kernel = 'rbf',epsilon=0.1, C=10))]

    pipeline_svr_car = Pipeline(steps_svr)
    pipeline_svr_car.fit(X_train_car,y_train_car)
    y_predicted_car = pipeline_svr_car.predict(X_test_car)
    ### METRICHE MISURA ACCURATEZZA PER CAR
    print("CARS Prediction")
    compute_accuracy(y_test_car, y_predicted_car)
    
    pipeline_svr_camion = Pipeline(steps_svr)
    pipeline_svr_camion.fit(X_train_camion,y_train_camion)
    y_predicted_camion = pipeline_svr_camion.predict(X_test_camion)
    ### METRICHE MISURA ACCURATEZZA PER CAMION
    print("CAMIONS Prediction")
    compute_accuracy(y_test_camion, y_predicted_camion)

def main(directory):
    data, labels = get_data(directory, window_sec_size = 60, shift_sec_size = 2)
    algorithm(data, labels, number_of_features = 50)
    


if __name__ == "__main__":
    dir = "./Datasets/Vehicles_Roccaprebalza/"
    main(dir)