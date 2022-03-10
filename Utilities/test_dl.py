import pickle
import numpy as np
import time
import tensorflow as tf
from tensorflow import keras

model_names = ['model_gru_car_50.keras', 'model_lstm_car_50.keras', 'model_srnn_car_2_100.keras']#, 'model_TCN_car_02.keras']
"""
X_test_list = [np.random.rand(1, 6000, 21), np.random.rand(1, 6000, 21), np.random.rand(1, 6000, 21), np.random.rand(1, 6000, 21),
               np.random.rand(1, 6000, 21), np.random.rand(1, 6000, 21), np.random.rand(1, 6000, 21), np.random.rand(1, 6000, 21),
               np.random.rand(1, 6000, 21), np.random.rand(1, 6000, 21)]
"""
X = np.random.rand(1, 6000, 21)
X_test = np.asarray(X).astype('float32')
for j, model in enumerate(model_names):
    loaded_model = keras.models.load_model(model)
    #X_test = X_test_list[j]
    start_time = time.time()
    for i in np.arange(0, 10):
        result = loaded_model.predict(X_test)
    print("--- %s seconds ---" % (time.time() - start_time))
    print(model)
