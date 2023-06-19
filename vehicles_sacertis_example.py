from Datasets.Vehicles_Sacertis.get_dataset import get_dataset, get_data

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
    X = SelectKBest(f_regression, k = number_of_features).fit_transform(data, labels)
    X_train, X_test, y_train, y_test = train_test_split(X, labels)
    steps_svr = [( 'scaler', StandardScaler() ), ('svr', SVR(kernel = 'rbf',epsilon=0.1, C=10))]
    pipeline_svr = Pipeline(steps_svr)
    pipeline_svr.fit(X_train,y_train)
    y_predicted = pipeline_svr.predict(X_test)
    print("Prediction")
    compute_accuracy(y_test, y_predicted)


def main(directory):
    data, labels = get_data(directory, True, False, False, sensor = "C1.1.1", time_frequency = "time", features = 'Yes')
    algorithm(data, labels, number_of_features = 12)
    
if __name__ == "__main__":
    dir = "./Datasets/Vehicles_Sacertis/"
    main(dir)