import torch
import numpy as np
import math
import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import Dataset

import data.Vehicles_Sacertis.get_dataset as Vehicles_Sacertis
import data.AnomalyDetection_SS335.get_dataset as AnomalyDetection_SS335
import data.Vehicles_Roccaprebalza.get_dataset as Vehicles_Roccaprebalza

dim_filtering = 120

def get_all_datasets(dir1: str = None, dir2: str = None, dir3: str = None, window_size: int = 490):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "3"
    print("Creating Training Dataset")
    starting_date = datetime.date(2019,5,22) 
    num_days = 7
    # uc1 data
    directory = dir1
    data_anomaly = AnomalyDetection_SS335.get_data(directory, starting_date, num_days, sensor = 'S6.1.3', time_frequency = "frequency", windowLength = window_size)
    # uc2 data
    directory = dir2
    data_train, _, _, _ = Vehicles_Roccaprebalza.get_data(directory, window_sec_size = 60, shift_sec_size = 2, time_frequency = "frequency", car = "y_camion")
    data_train_2, _, _, _ = Vehicles_Roccaprebalza.get_data(directory, window_sec_size = 60, shift_sec_size = 2, time_frequency = "frequency", car = "y_car")
    # uc3 data
    directory = dir3
    data_sacertis, _ = Vehicles_Sacertis.get_data(directory, True, False, False, time_frequency = "frequency")
    
    data_all = []
    for data in data_sacertis: data_all.append(data[0])
    for i in np.arange(data_anomaly.shape[0]): data_all.append(torch.from_numpy(data_anomaly[i]))
    for data in data_train: data_all.append(data)
    for data in data_train_2: data_all.append(data)

    class Dataset_All(Dataset):
        def __init__(self, data):
            self.data = data
            self.len = len(data)
        def __len__(self):
            return self.len
        def __getitem__(self, index):
            slice = self.data[index]
            return slice, 0
    dataset = Dataset_All(data_all)
    return dataset

def get_model_info(model):
    total_params = sum(param.numel() for param in model.parameters())
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return total_params, size_all_mb

def compute_accuracy(y_test, y_predicted):
    mse = mean_squared_error(y_test, y_predicted)
    # print("MSE:", mse)
    mae = mean_absolute_error(y_test, y_predicted)
    # print("MAE:", mae)
    r2 = r2_score(y_test, y_predicted)
    # print("R2:", r2)
    mspe = (mse/np.mean(y_test))*100
    # print("MSE%:", mspe)
    mape = (mae/np.mean(y_test))*100
    # print("MAE%:", mape)
    return mse, mae, r2, mspe, mape


def millify(n):
    n = float(n)
    millnames = ['',' Thousand',' Million',' Billion',' Trillion']
    millidx = max(0,min(len(millnames)-1,
                        int(math.floor(0 if n == 0 else math.log10(abs(n))/3))))

    return '{:.0f}{}'.format(n / 10**(3 * millidx), millnames[millidx])

def compute_threshold_accuracy(anomalies, normal, ax, min, max, only_acc = 0, dim_filtering = dim_filtering):
    new_normal = []
    for i in np.arange(0, len(normal-dim_filtering)):
        new_normal.append(np.median(normal[i:(i+dim_filtering)]))
    new_anomalies = []
    for i in np.arange(0, len(anomalies-dim_filtering)):
        new_anomalies.append(np.median(anomalies[i:(i+dim_filtering)]))
    th_step = np.median(new_normal[:int(len(new_normal)/4)])
    #increase threshold if necessary
    th = th_step
    while sum(new_normal[:int(len(new_normal)/4)]+np.std(new_normal[:int(len(new_normal)/4)]) < th) != len(new_normal[:int(len(new_normal)/4)]):
        th += th_step/1000
    spec = sum(new_normal < th) / len(new_normal)
    sens = sum(new_anomalies > th) / len(new_anomalies)
    acc = (sum(new_normal < th) + sum(new_anomalies > th)) / (len(new_normal) + len(new_anomalies))
    print(f"Sensitivity: {sens} Spcificity: {spec} Accuracy: {acc}")
    if only_acc == 0:
        ax.axhline(y=(th-min)/(max-min), color='r', linestyle='--', label = "Threshold")
    return spec, sens, acc