from torch.utils.data import Dataset
import torch
import glob
import pandas as pd
import datetime
import os
import math
from tqdm import tqdm
import numpy as np
import random
from scipy import signal
from pathlib import Path
import sys
'''
There are multiple functions in this file:
- get_dataset, give you back the whole dataset with all the variables
- get_data only give you back the row z axis of acceleration
- SHMDataset class: in this class we have many methods, and need a lot of time to run:
    - readCSV functions:
        - first we read the CSV files and we append all togheter the data;
        - after we create the dataset (around 7 minutes for 1 day of dataset) importing ts and z axis;
    - partitioner function:
        - we generate 500000 training windows, the time and power limits, and the min/max to be used for normalization
    - __getitem__ applies the spectrogram, using the data extracted during init
'''

def get_dataset(directory, 
             starting_date = datetime.date(2019,5,24),
             num_days = 4,
             sensor = 'D6.1.1',
             time_frequency = "time"):
    dataset = SHMDataset(directory + "./Data/", starting_date, num_days, sensor, time_frequency)
    return dataset

def get_data(directory, 
             starting_date = datetime.date(2019,5,24),
             num_days = 4,
             sensor = 'D6.1.1',
             time_frequency = "time"):
    dataset = get_dataset(directory, starting_date, num_days, sensor, time_frequency)
    data_loader_train = torch.utils.data.DataLoader(
        dataset,
        shuffle = False,
        batch_size=1,
        num_workers=1,
        pin_memory='store_true',
        drop_last=True,
    )
    labels = np.asarray([])
    dataset_final = np.asarray([])
    for i, (data, label) in enumerate(data_loader_train): 
        if i == 0:
            dataset_final = np.asarray(data)
        else:
            dataset_final = np.concatenate((dataset_final,np.asarray(data)),axis=0)
    return dataset_final

class SHMDataset(Dataset):
    def __init__(self, data_path, date, num_days, sensor, time_frequency):
        self.day_start = date
        self.num_days = num_days
        self.path = data_path
        self.sensor = sensor
        self.time_frequency = time_frequency
        self.data = self._readCSV()
        self.sampleRate = 100
        self.frameLength = 198
        self.stepLength = 10
        self.windowLength = 1190 #500 ## FORMULA TO COMPUTE THE TIME SAMPLES: 1 + (self.windowLength - self.frameLength) / self.stepLength
        self.windowStep = 100 #500
        self.data, self.limits, self.totalWindows, self.min, self.max = self._partitioner()

    def __len__(self):
        return self.totalWindows

    def __getitem__(self, index):
        start, end, power = self.limits[index]
        slice = self.data[start:end]
        if self.time_frequency == 'time':
            slice = self._normalizer(slice).type(torch.float16)
            return slice, 0
        elif self.time_frequency == 'frequency':
            frequencies, times, spectrogram = self._transformation(slice)
            spectrogram = torch.unsqueeze(torch.tensor(spectrogram, dtype=torch.float64), 0)
            NormSpect = self._normalizer(spectrogram).type(torch.float16)
            return torch.transpose(NormSpect, 1, 2), 0
        else:
            sys.exit(0)

    def _readCSV(self):
        print(f'reading CSV files')
        
        ldf = []
        for x in tqdm(range(self.num_days)): # from 40 to 100 seconds per file
            yy, mm, dd = (self.day_start + datetime.timedelta(days=x)).strftime('%Y,%m,%d').split(",")
            date = f"{int(yy)}{int(mm)}{int(dd)}"
            df = pd.read_csv(self.path + f"ss335-acc-{date}.csv")
            print(f'Read file {x}')
            ldf.append(df.drop(['x','y', "year", "month", "day", "Unnamed: 0"], axis=1))
            print(f'Dropped axes from file {x}')
        df = pd.concat(ldf).sort_values(by=['sens_pos', 'ts'])
        df = df.reset_index(drop=True)
        new_dict = {
            "ts": [],
            "sens_pos": [],
            "z": [],
        }
        conv = (1*2.5)*2**-15
        df = df[df["sens_pos"]==self.sensor]
        # for i in tqdm(range(10000)): to reduce runtime
        for i in tqdm(range(len(df))):
            row = df["z"].iloc[i]
            data_splited = row.replace("\n", "").replace("[", "").replace("]", "").split(" ")
            #data_splited = df["z"][i].split(" ")
            ts = datetime.datetime.utcfromtimestamp(df["ts"].iloc[i]/1000)
            sens = df["sens_pos"].iloc[i]
            
            for idx, data in enumerate(data_splited):
                if data == "":
                    continue
                z = int(data)  
                new_dict["ts"].append(ts + idx*datetime.timedelta(milliseconds=10))
                new_dict["z"].append(z * conv)
                new_dict["sens_pos"].append(sens)

        df_new = pd.DataFrame(new_dict)
        print(f'Finish data reading')
        return df_new
            
    def _partitioner(self):
        sensors = self.data['sens_pos'].unique().tolist()
        print(f'start partitioner')
        partitions = {}
        cumulatedWindows = 0
        limits = dict()
        print(f'Generating windows')
        for sensor in tqdm(sensors):
            sensorData = self.data[self.data['sens_pos']==sensor]
            totalFrames = sensorData.shape[0]
            totalWindows = math.ceil((totalFrames-self.windowLength)/self.windowStep)
            start = cumulatedWindows
            cumulatedWindows += totalWindows
            end = cumulatedWindows
            indexStart = sensorData.index[0]
            partitions[sensor]= (start, end, indexStart)

        timeData = torch.tensor(self.data["z"].values, dtype=torch.float64)
        cummulator = -1

        mins = list()
        maxs = list()
        print(f'Defining useful windows limits')
        noiseFreeSpaces = 1
        indexes = list(range(0, cumulatedWindows))
        random.shuffle(indexes)
        
        for index in tqdm(indexes):
            if cummulator >= 500000: #Number of used examples during training. I do this to avoid large training times.
                break
            for k,v in partitions.items():
                if index in range(v[0], v[1]):
                    start = v[2]+(index-v[0])*self.windowStep
                    filteredSlice = timeData[start: start+self.windowLength]
                    filteredSlice = filteredSlice - (filteredSlice).mean()
                    signalPower = self.power(filteredSlice)
                    if signalPower>(3.125*(10**-6)):
                        cummulator += 1
                        limits[cummulator] = (start, start+self.windowLength, signalPower)
                        slice = timeData[start:start+self.windowLength]
                        
                        if self.time_frequency == 'time':
                            mins.append(np.min(np.array(slice)))
                            maxs.append(np.max(np.array(slice)))
                        elif self.time_frequency == 'frequency':
                            frequencies, times, spectrogram = self._transformation(torch.tensor(slice, dtype=torch.float64))
                            mins.append(np.min(np.array(spectrogram)))
                            maxs.append(np.max(np.array(spectrogram)))
                        else:
                            sys.exit(0)
                    break
        print(f'Total windows in dataset: {cummulator}')
        min = np.min(np.array(mins))
        max = np.max(np.array(maxs))     
        print(f'General min: {min}')
        print(f'General max: {max}')
        return timeData, limits, cummulator, min, max

    def _transformation(self, slice):
        sliceN = slice-torch.mean(slice)
        frequencies, times, spectrogram = signal.spectrogram(sliceN,self.sampleRate,nfft=self.frameLength,noverlap=(self.frameLength - self.stepLength), nperseg=self.frameLength,mode='psd')
        return frequencies, times, np.log10(spectrogram)
    
    def _normalizer(self, spectrogram):
        spectrogramNorm = (spectrogram - self.min) / (self.max - self.min)
        return spectrogramNorm

    def power(self, slice):
        signalPower = np.sum(np.array(slice)**2)/self.windowLength
        return signalPower
