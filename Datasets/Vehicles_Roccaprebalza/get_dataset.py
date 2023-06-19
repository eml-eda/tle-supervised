import pandas as pd
import numpy as np
from scipy import stats
import pickle
import itertools

fs = 100
minutes = 50
car = 1
camion = 2

def label_creation(row_label, vehicle_type):
    return sum(row_label==vehicle_type)/10

def featureExtraction(serie, sensNumber):
    axStatistics={
        sensNumber+"mean": [np.mean(serie)],
        sensNumber+"std":  [np.std(serie)],
        sensNumber+"min":  [np.min(serie)],
        sensNumber+"max":  [np.max(serie)],
        sensNumber+"med":  [np.median(serie)],
        sensNumber+"kurt": [stats.kurtosis(serie)],
        sensNumber+"skew": [stats.skew(serie)],
        sensNumber+"rms":  [np.sqrt(np.mean(serie**2))],
        sensNumber+"sabs": [np.sum(np.abs(serie))],
        sensNumber+"eom":  [serie[serie>np.mean(serie)].sum()],
        sensNumber+"ener": [np.sqrt(np.mean(np.array(serie)**2))**2],
        sensNumber+"mad":  [np.median(np.absolute(serie - np.median(serie)))]
    }
    return pd.DataFrame(axStatistics)

def get_data(
        directory,
        window_sec_size = 60,
        shift_sec_size = 2):
    d = pd.read_pickle(directory + './Data/DataFrame__sensori_rilevazioni_granulare_acc_normalizzate.pkl') 
    data = pd.DataFrame(d)

    dataset = pd.DataFrame()
    label_car = []
    label_camion = []

    for index in np.arange(0, minutes * 60 * fs, shift_sec_size * fs):
        if sum(data["rilevazione"][index : (index + window_sec_size * fs)] == "NaN")<1:
            dataset_row = pd.DataFrame()
            for sens in data.keys():
                if "10" in sens and "xyz" not in sens:
                    features = featureExtraction(data[sens][index:(index+window_sec_size*fs)], sens)  
                    dataset_row = pd.concat([dataset_row, features], axis=1)
            y_car = label_creation(data["rilevazione"][index:(index+window_sec_size*fs)], car)
            y_camion = label_creation(data["rilevazione"][index:(index+window_sec_size*fs)], camion)
            label_car.append(y_car)
            label_camion.append(y_camion)
            dataset = pd.concat([dataset, dataset_row],axis=0)
    labels = pd.DataFrame({"y_car": label_car, "y_camion": label_camion})
    return dataset, labels