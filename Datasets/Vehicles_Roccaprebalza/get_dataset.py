import pandas as pd
import numpy as np
from scipy import stats
import pickle
import itertools
from torch.utils.data import Dataset
import torch 
from scipy import signal
import pickle as pkl
import os 

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

def get_data(directory,
        window_sec_size = 60,
        shift_sec_size = 2,
        time_frequency = "time",
        car = 'none'):
    dataset_train = SHMDataset_Roccaprebalza(directory, window_sec_size, shift_sec_size, time_frequency, car, train = True, test = False)
    dataset_test = SHMDataset_Roccaprebalza(directory, window_sec_size, shift_sec_size, time_frequency, car, train = False, test = True)
    return dataset_train.data, dataset_train.labels[car], dataset_test.data, dataset_test.labels[car]

def get_dataset(
        directory,
        window_sec_size = 60,
        shift_sec_size = 2,
        time_frequency = "time",
        car = 'none'):
    dataset_train = SHMDataset_Roccaprebalza(directory, window_sec_size, shift_sec_size, time_frequency, car, train = True, test = False)
    dataset_test = SHMDataset_Roccaprebalza(directory, window_sec_size, shift_sec_size, time_frequency, car, train = False, test = True)
    return dataset_train, dataset_test


class SHMDataset_Roccaprebalza(Dataset):

    def __init__(self, directory, window_sec_size, shift_sec_size, time_frequency, car, train, test):
        self.directory = directory
        self.window_sec_size = window_sec_size
        self.shift_sec_size = shift_sec_size
        self.time_frequency = time_frequency
        self.fs = 100
        self.minutes = 50
        self.car = 1
        self.camion = 2
        self.sampleRate = 100
        self.frameLength = 198
        self.stepLength = 58
        if f'vehicles_roccaprebalza_{self.time_frequency}.pkl' in os.listdir(self.directory):
            print("Loading Roccaprebalza dataset")
            #to load it
            with open(self.directory+f'vehicles_roccaprebalza_{self.time_frequency}.pkl', "rb") as f:
                self.data, self.labels = pkl.load(f)
        else:
            print("Creating Roccaprebalza dataset")
            self.data, self.labels = self._read_data()
            if isinstance(self.data, pd.DataFrame):
                self.data = self.data.values
            #to save it
            with open(self.directory+f"vehicles_roccaprebalza_{self.time_frequency}.pkl", "wb") as f:
                pkl.dump([self.data, self.labels], f)
            

        import random 
        random.seed(1)
        percentage = 75
        k = len(self.data) * percentage // 100
        indicies = random.sample(range(len(self.data)), k)
        new_data = []
        new_labels = pd.DataFrame(columns=['y_car', 'y_camion'])



        if train == True:
            for i, data in enumerate(self.data):
                if i in indicies:
                    new_data.append(data)
                    dict = {'y_car': [self.labels.values[i][0]], 'y_camion': [self.labels.values[i][1]]}
                    new_labels = pd.concat([new_labels, pd.DataFrame.from_dict(dict)])
        elif test == True:
            for i, data in enumerate(self.data):
                if i not in indicies:
                    new_data.append(data)
                    dict = {'y_car': [self.labels.values[i][0]], 'y_camion': [self.labels.values[i][1]]}
                    new_labels = pd.concat([new_labels, pd.DataFrame.from_dict(dict)])
        self.data = new_data 
        self.labels = new_labels
        self.labels.reset_index(inplace = True)
        self.labels = self.labels.drop('index', axis=1)
        self.car_camion = car

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.car_camion == 'y_car':
            return self.data[index], self.labels['y_car'][index]
        elif self.car_camion == 'y_camion':
            return self.data[index], self.labels['y_camion'][index]
        else:
            sys.exit(0)
        
    def _read_data(self):
        d = pd.read_pickle(self.directory + './Data/DataFrame__sensori_rilevazioni_granulare_acc_normalizzate.pkl') 
        data = pd.DataFrame(d)
        if self.time_frequency == "time":
            dataset = pd.DataFrame()
        else:
            dataset = []
        label_car = []
        label_camion = []
        mins = []
        maxs = []
        for index in np.arange(0, self.minutes * 60 * self.fs, self.shift_sec_size * self.fs):
            if sum(data["rilevazione"][index : (index + self.window_sec_size * self.fs - 10)] == "NaN")<1:
                dataset_row = pd.DataFrame()
                for sens in data.keys():
                    if "10" in sens and "xyz" not in sens:
                        if self.time_frequency == "time":
                            if "z10D41" in sens:
                                features = featureExtraction((data[sens][index:(index + self.window_sec_size * self.fs - 10)].values), sens)
                                dataset_row = pd.concat([dataset_row, features], axis=1)
                        elif self.time_frequency == 'frequency':
                            if "z10D41" in sens:
                                features = torch.tensor(data[sens][index:(index + self.window_sec_size * self.fs - 10)].values)
                                frequencies, times, spectrogram = self._transformation(features)
                                spectrogram = torch.unsqueeze(torch.tensor(spectrogram, dtype=torch.float64), 0)
                                mins.append(np.min(np.array(spectrogram)))
                                maxs.append(np.max(np.array(spectrogram)))
                                dataset.append(torch.transpose(spectrogram, 1, 2))
                y_car = label_creation(data["rilevazione"][index:(index + self.window_sec_size * self.fs - 10)], self.car)
                y_camion = label_creation(data["rilevazione"][index:(index + self.window_sec_size * self.fs - 10)], self.camion)
                label_car.append(y_car)
                label_camion.append(y_camion)
                if self.time_frequency == "time":
                    dataset = pd.concat([dataset, dataset_row],axis=0)
        labels = pd.DataFrame({"y_car": label_car, "y_camion": label_camion})
        if self.time_frequency == 'frequency':
            self.min = np.min(np.array(mins))
            self.max = np.max(np.array(maxs))
            for i in np.arange(len(dataset)):
                dataset[i] = self._normalizer(dataset[i]).type(torch.float16)
        return dataset, labels

    def _transformation(self, slice):
        sliceN = slice-torch.mean(slice)
        frequencies, times, spectrogram = signal.spectrogram(sliceN,self.sampleRate,nfft=self.frameLength,noverlap=(self.frameLength - self.stepLength), nperseg=self.frameLength,mode='psd')
        return frequencies, times, np.log10(spectrogram)
    
    def _normalizer(self, spectrogram):
        spectrogramNorm = (spectrogram - self.min) / (self.max - self.min)
        return spectrogramNorm
