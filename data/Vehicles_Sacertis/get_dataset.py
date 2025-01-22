from torch.utils.data import Dataset
import torch
import glob
import pandas as pd
pd.options.mode.chained_assignment = None
from datetime import datetime
import os
import math
from tqdm import tqdm
import numpy as np
import random
from scipy import stats
from scipy import signal
import json
import pickle as pkl

def get_dataset(directory, 
                isPreTrain, 
                isFineTuning, 
                isEvaluation,
                sensor = 'None',
                time_frequency = "time"):
    dataset = SHMDataset(directory, isPreTrain, isFineTuning, isEvaluation, sensor, time_frequency)
    return dataset

def get_data(directory, 
                isPreTrain, 
                isFineTuning, 
                isEvaluation,
                sensor = 'None',
                time_frequency = "time",
                features = 'No'):
    dataset = get_dataset(directory, 
                isPreTrain, 
                isFineTuning, 
                isEvaluation,
                sensor,
                time_frequency)
    data_loader_train = torch.utils.data.DataLoader(
        dataset, 
        shuffle = False,
        batch_size=1,
        num_workers=1,
        pin_memory='store_true',
        drop_last=True,
    )
    labels = np.asarray([])
    if features == 'Yes':
        dataset_final = pd.DataFrame()
    else:
        dataset_final = []
    for data, label in data_loader_train:
        if features == 'Yes':
            dataset_final = pd.concat([dataset_final, featureExtraction(np.asarray(data))],axis=0)
        else:
            dataset_final.append(data)
        labels = np.concatenate((labels,np.asarray(label)))

    return dataset_final, labels

def featureExtraction(serie):
    serie = serie.reshape(-1)
    axStatistics={
        "mean": [np.mean(serie)],
        "std":  [np.std(serie)],
        "min":  [np.min(serie)],
        "max":  [np.max(serie)],
        "med":  [np.median(serie)],
        "kurt": [stats.kurtosis(serie)],
        "skew": [stats.skew(serie)],
        "rms":  [np.sqrt(np.mean(serie**2))],
        "sabs": [np.sum(np.abs(serie))],
        "eom":  [serie[serie>np.mean(serie)].sum()],
        "ener": [np.sqrt(np.mean(np.array(serie)**2))**2],
        "mad":  [np.median(np.absolute(serie - np.median(serie)))]
    }
    return pd.DataFrame(axStatistics)

class SHMDataset(Dataset):

    def __init__(self, data_path, isPreTrain, isFineTuning, isEvaluation, sensor, time_frequency):
        if isPreTrain:
            self.start_time, self.end_time = "05/12/2021 00:00", "05/12/2021 23:59" #"05/12/2021 00:00", "05/12/2021 23:59"
            # self.start_time, self.end_time = "05/12/2021 04:00", "05/12/2021 04:10" # to test
            self.datasetSize = 500000
        elif isFineTuning:
            self.start_time, self.end_time = "06/12/2021 00:00", "06/12/2021 11:59" #"06/12/2021 00:00", "06/12/2021 11:59"
            # self.start_time, self.end_time = "06/12/2021 00:00", "06/12/2021 00:10" # to test
            self.datasetSize = 200000
        elif isEvaluation:
            self.start_time, self.end_time = "06/12/2021 12:00", "06/12/2021 17:59" #"06/12/2021 12:00", "06/12/2021 17:59"
            # self.start_time, self.end_time = "05/12/2021 12:00", "05/12/2021 12:10" # to test
            self.datasetSize = 50000
        else:
            self.start_time, self.end_time = "06/12/2021 17:59", "06/12/2021 23:59" #"06/12/2021 17:59", "06/12/2021 23:59"
            self.datasetSize = 50000
        self.directory = data_path
        self.noisySensors = ["C12.1.4", "C17.1.2"]
        self.sensor = sensor
        self.time_frequency = time_frequency
        self.minDuration = 0.25
        self.sampleRate = 100
        self.frameLength = 198
        self.stepLength = 58
        self.windowLength= 5990
        self.windowStep = 1500
        if isPreTrain:
            if f'vehicles_sacertis_allVariables_PreTrain.pkl' in os.listdir(self.directory):
                print("Loading Sacertis dataset", end=' ', flush=True)
                with open(self.directory+f'vehicles_sacertis_allVariables_PreTrain.pkl', "rb") as f:
                    self.labelsDf, self.groupsDf, self.data, self.limits, self.totalWindows, min, max = pkl.load(f)
            else:
                print("Creating Sacertis dataset", end=' ', flush=True)
                self.data = self._readCSV()
                self.distanceToSensor = self._readDistanceToSensor()
                self.sensorVarDict = self._calculateThresholds(isPreTrain=isPreTrain)
                self.pesaDataDf = self._readLabels()
                self.labelsDf, self.groupsDf = self._labelAssignment()
                self.data, self.limits, self.totalWindows, min, max = self._partitioner()
                with open(self.directory+f'vehicles_sacertis_allVariables_PreTrain.pkl', "wb") as f:
                    pkl.dump([self.labelsDf, self.groupsDf, self.data, self.limits, self.totalWindows, min, max], f)
        elif isFineTuning:
            if f'vehicles_sacertis_allVariables_Finetuning.pkl' in os.listdir(self.directory):
                print("Loading Sacertis dataset", end=' ', flush=True)
                with open(self.directory+f'vehicles_sacertis_allVariables_Finetuning.pkl', "rb") as f:
                    self.labelsDf, self.groupsDf, self.data, self.limits, self.totalWindows, min, max = pkl.load(f)
            else:
                print("Creating Sacertis dataset", end=' ', flush=True)
                self.data = self._readCSV()
                self.distanceToSensor = self._readDistanceToSensor()
                self.sensorVarDict = self._calculateThresholds(isPreTrain=isPreTrain)
                self.pesaDataDf = self._readLabels()
                self.labelsDf, self.groupsDf = self._labelAssignment()
                self.data, self.limits, self.totalWindows, min, max = self._partitioner()
                with open(self.directory+f'vehicles_sacertis_allVariables_Finetuning.pkl', "wb") as f:
                    pkl.dump([self.labelsDf, self.groupsDf, self.data, self.limits, self.totalWindows, min, max], f)
        elif isEvaluation:
            if f'vehicles_sacertis_allVariables_Evaluation.pkl' in os.listdir(self.directory):
                print("Loading Sacertis dataset", end=' ', flush=True)
                with open(self.directory+f'vehicles_sacertis_allVariables_Evaluation.pkl', "rb") as f:
                    self.labelsDf, self.groupsDf, self.data, self.limits, self.totalWindows, min, max = pkl.load(f)
            else:
                print("Creating Sacertis dataset", end=' ', flush=True)
                self.data = self._readCSV()
                self.distanceToSensor = self._readDistanceToSensor()
                self.sensorVarDict = self._calculateThresholds(isPreTrain=isPreTrain)
                self.pesaDataDf = self._readLabels()
                self.labelsDf, self.groupsDf = self._labelAssignment()
                self.data, self.limits, self.totalWindows, min, max = self._partitioner()
                with open(self.directory+f'vehicles_sacertis_allVariables_Evaluation.pkl', "wb") as f:
                    pkl.dump([self.labelsDf, self.groupsDf, self.data, self.limits, self.totalWindows, min, max], f)
            
        if isPreTrain: #The statistics are calculated during pre-train, during evaluation and fine-tuning use those from train
            self.min = min
            self.max = max
        else:
            self.min = -24.496400092774163
            self.max = -3.3793421008937514

    def __len__(self):
        return self.totalWindows

    def __getitem__(self, index):
        start, end, label, timeSlice, sensor = self.limits[index]
        slice = self.data[start:end]
        if self.time_frequency == 'time':
            return slice, label
        elif self.time_frequency == 'frequency':
            frequencies, times, spectrogram = self._transformation(slice)
            spectrogram = torch.unsqueeze(torch.tensor(spectrogram, dtype=torch.float64), 0)
            NormSpect = self._normalizer(spectrogram).type(torch.float16)
            return torch.transpose(NormSpect, 1, 2), label
        else:
            sys.exit(0)

    def _readCSV(self):
        print(f'reading CSV files')
        start = datetime.strptime(self.start_time, '%d/%m/%Y %H:%M')
        end = datetime.strptime(self.end_time, '%d/%m/%Y %H:%M')

        ldf = list()
        for p in tqdm(glob.glob(self.directory + "./Data/" + "*.csv")):
            name = os.path.split(p)[-1]
            nstr = datetime.strptime(name, 'traffic_%Y%m%dH%H%M%S.csv')
            if start <= nstr < end:
                df_tmp = pd.read_csv(p)
                c_drop = set(df_tmp.columns) - set(["sens_pos", "z", "ts"])
                if len(c_drop) > 0:
                    df_tmp.drop(columns=list(c_drop), inplace=True)
                ldf.append(df_tmp)
        df = pd.concat(ldf).sort_values(by=['sens_pos', 'ts'])
        df.reset_index(inplace=True, drop=True)
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        df['Time'] = df['ts'].dt.strftime('%Y-%m-%d %H:%M:00')
        df["zN"] = df["z"]-np.mean(df["z"]) #Eliminate the constant component of the readings
        df["vars"] = df["zN"].rolling(window=100).var().fillna(0) #Calculate moving variance accross windows
        df["vars"] = df["vars"].rolling(window=100).mean().fillna(0) #Mean of the variances
        if self.sensor != 'None':
            df = df[df["sens_pos"]==self.sensor]
        print(f'finish reading process')
        return df

    def _readDistanceToSensor(self):
        distanceToSensor = {}
        with open(self.directory + './distanceToSensor.csv') as f: #File containing distance from each sensor to the scale
            for line in f.readlines():
                sensor, distance = line.replace("'", "").replace("\n","").split(",")
                distanceToSensor[sensor] = float(distance)
        return distanceToSensor

    def _readLabels(self):        
        print(f"Entering in function _readLabels", end=' ', flush=True)
        start_time = datetime.strptime(self.start_time, '%d/%m/%Y %H:%M')
        end_time = datetime.strptime(self.windowStep.end_time, '%d/%m/%Y %H:%M')
        pesaDataDf = pd.read_csv(self.directory + "./dati_pese_dinamiche/dati 2021-12-04_2021-12-12 pesa km 104,450.csv", sep=";", index_col=0)
        pesaDataDf = pesaDataDf[["Id", "StartTimeStr", "ClassId", "GrossWeight", "Velocity", "VelocityUnit"]]
        pesaDataDf["Time"] = pd.to_datetime(pesaDataDf["StartTimeStr"])
        pesaDataDf["Time"] = pesaDataDf["Time"].dt.strftime('%Y-%d-%m %H:%M:00')
        pesaDataDf["Time"] = pd.to_datetime(pesaDataDf["Time"]) + pd.to_timedelta(-1,'H')
        pesaDataDf.sort_values(by="Id", inplace=True)
        pesaDataDf = pesaDataDf[(pesaDataDf["Time"]>=start_time) & (pesaDataDf["Time"]<=end_time)]
        pesaDataDf.reset_index(drop=True, inplace=True)
        
        return pesaDataDf

    def groupsGenerator(self, sensorData, minTime, maxTime, threshold):
        slice = sensorData[(sensorData["ts"]>= minTime) & (sensorData["ts"]<= maxTime)]
        
        slice["outlier"] = slice["vars"].apply(lambda x: x>=threshold)
        outliers = slice[slice["outlier"] == True].reset_index().to_dict("records")

        if len(outliers) == 0:
            return pd.DataFrame()

        last = minTime
        timeStart = outliers[0]["ts"]
        flag = True
        groups = []
        groupTimes = []
        groupIndexes = []
        groupVars = []
        label = np.nan
        groupId = 0
        for outlier in outliers:
            if ((outlier["ts"] - last).total_seconds() < 2) or flag:
                groupTimes.append(outlier["ts"])
                groupVars.append(outlier["vars"])
                flag = False
                timeEnd = outlier["ts"]
            else:
                start, end = min(groupTimes), max(groupTimes)
                groupSignal = sensorData[(sensorData["ts"]>= start) & (sensorData["ts"]<= end)]["zN"]
                signalPower = np.sqrt(np.mean(np.array(groupSignal)**2))**2 
                pointMaxVar = groupTimes[np.argmax(groupVars)]
                if ((end - start).total_seconds() > self.minDuration):
                    label = {"groupId": groupId,"start": start, "end": end, "signalPower": signalPower, 
                    "pointMaxVar": pointMaxVar}
                    groups.append(label)
                groupId += 1
                groupTimes = [outlier["ts"],]
                groupVars = [outlier["vars"],]
            last = outlier["ts"]

        start, end = min(groupTimes), max(groupTimes)
        groupSignal = sensorData[(sensorData["ts"]>= start) & (sensorData["ts"]<= end)]["zN"]
        signalPower = np.sqrt(np.mean(np.array(groupSignal)**2))**2 
        pointMaxVar = groupTimes[np.argmax(groupVars)]
        if ((end-start).total_seconds() > self.minDuration):
            label = {"groupId": groupId,"start": start, "end": end, "signalPower": signalPower, 
            "pointMaxVar": pointMaxVar}
            groups.append(label)

        if len(groups)>0:
            groupsDf = pd.DataFrame(groups).sort_values("signalPower", ascending=False)
        else:
            groupsDf = pd.DataFrame()

        return groupsDf

    def _labelAssignment(self,):
        print(f"Entering in function _labelAssignment", end=' ', flush=True)
        sensorLabelsDfList = []
        groupsDfList = []
        sensorsList = self.data["sens_pos"].unique()
        for sensor in tqdm(sensorsList):
            if (sensor in self.noisySensors) or (sensor not in self.distanceToSensor.keys()) or (sensor not in self.sensorVarDict.keys()):
                continue
            assignedLabels = {}
            assignedLabels2 = {}
            sensorLabelsDf = self.pesaDataDf.copy(deep=True)
            sensorLabelsDf["EstimatedTime"] = sensorLabelsDf["Time"] + pd.to_timedelta((float(self.distanceToSensor[sensor])/(sensorLabelsDf["Velocity"]/3.6))-20,'S')
            sensorLabelsDf["MaxTime"] = sensorLabelsDf["EstimatedTime"] + pd.to_timedelta(120,'S')
            minTime = sensorLabelsDf["EstimatedTime"].min()
            maxTime = sensorLabelsDf["MaxTime"].max()
            sensorLabelsDf.sort_values("GrossWeight", inplace=True, ascending=False)

            sensorData = self.data[self.data["sens_pos"]==sensor]
            threshold = self.sensorVarDict[sensor]["threshold"]
            groupsDf = self.groupsGenerator(sensorData, minTime, maxTime, threshold)
            print(f"Total groups found for sensor {sensor}: {groupsDf.shape[0]}", end=' ', flush=True)
            if groupsDf.empty:
                continue

            availableGroupsDf = groupsDf.copy(deep=True)
            for index, row in sensorLabelsDf.iterrows():
                if row["Id"] in assignedLabels:
                    continue
                
                if availableGroupsDf.empty:
                    break

                candidatesDf = availableGroupsDf[(row["EstimatedTime"] <= availableGroupsDf["pointMaxVar"]) & (availableGroupsDf["pointMaxVar"] <= row["MaxTime"])]
                if not candidatesDf.empty:
                    assignedLabels[row["Id"]] = candidatesDf.iloc[0].to_dict()
                    assignedLabels2[candidatesDf.iloc[0]["groupId"]] = row["Id"]
                    availableGroupsDf.drop(candidatesDf.index[0], inplace=True)
            
            sensorLabelsDf["sens_pos"] = sensor
            sensorLabelsDf["labels"] = sensorLabelsDf.apply(lambda row: assignedLabels[row["Id"]] if row["Id"] in assignedLabels else np.nan, axis=1)
            sensorLabelsDf.sort_values("Id", inplace=True)
            groupsDf["sens_pos"] = sensor
            groupsDf["labels"] = groupsDf.apply(lambda row: assignedLabels2[row["groupId"]] if row["groupId"] in assignedLabels2 else np.nan, axis=1)
            groupsDf.sort_values("groupId", inplace=True)
            groupsDf.dropna(inplace=True)

            sensorLabelsDfList.append(sensorLabelsDf)
            groupsDfList.append(groupsDf)

        labelsDf = pd.concat(sensorLabelsDfList)
        groupsDf =  pd.concat(groupsDfList)

        print(f"Total labels: {len(labelsDf)}")
        totnan = labelsDf["labels"].isna().sum()
        print(f"Total nan labels: {totnan}")
        print(f"Proportion of match labels: {1-(totnan/len(labelsDf))}")

        return labelsDf, groupsDf

    def _labelAssigner(self, timeSlice, sensor):
        start, end = timeSlice.min(), timeSlice.max()
        vehiclesInSliceDf = self.groupsDf[(self.groupsDf["pointMaxVar"]>=start) &
        (self.groupsDf["pointMaxVar"]<=end) &
        (self.groupsDf["sens_pos"]==sensor)]
        return vehiclesInSliceDf.shape[0]

    def _partitioner(self):
        sensors = self.data['sens_pos'].unique().tolist()
        print(f'start partitioner', end=' ', flush=True)
        partitions = {}
        cumulatedWindows = 0
        limits = dict()
        print(f'Generating windows', end=' ', flush=True)
        for sensor in tqdm(sensors):
            if (sensor in self.noisySensors) or (sensor not in self.distanceToSensor.keys()):
                continue
            sensorData = self.data[self.data['sens_pos']==sensor]
            totalFrames = sensorData.shape[0]
            totalWindows = math.ceil((totalFrames-self.windowLength)/self.windowStep)
            start = cumulatedWindows
            cumulatedWindows += totalWindows
            end = cumulatedWindows
            indexStart = sensorData.index[0]
            partitions[sensor]= (start, end, indexStart)

        timeData = torch.tensor(self.data["z"].values, dtype=torch.float64)
        timestamps = self.data["ts"]
        cummulator = -1

        mins = list()
        maxs = list()
        print(f'Defining useful windows limits', end=' ', flush=True)
        indexes = list(range(0, cumulatedWindows))
        random.shuffle(indexes)

        for index in tqdm(indexes):
            if cummulator >= self.datasetSize:
                break
            for sensor,v in partitions.items():
                if index in range(v[0], v[1]):
                    start = v[2]+(index-v[0])*self.windowStep
                    timeSlice = timestamps[start: start+self.windowLength]
                    label = self._labelAssigner(timeSlice, sensor)
                    filteredSlice = timeData[start: start+self.windowLength]
                    signalPower = self.power(filteredSlice)

                    if (signalPower>1.25*10**-6) or (label>0):
                        cummulator += 1
                        limits[cummulator] = (start, start+self.windowLength, label, timeSlice, sensor)
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
        print(f'Total windows in dataset: {cummulator}', end=' ', flush=True)
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
        return np.sqrt(np.mean(np.array(slice)**2))**2

    def interquartileRule(self, data):
        # Calculate the first quartile (Q1)
        Q1 = np.percentile(data, 25)

        # Calculate the third quartile (Q3)
        Q3 = np.percentile(data, 75)

        # Calculate the interquartile range (IQR)
        IQR = Q3 - Q1

        # Define the upper and lower bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        return lower_bound, upper_bound

    def _calculateThresholds(self, isPreTrain):
        """
        During pre-training the variance of the sensors are calculated to define the thresholds to identify the vehicles, the statistics are save
        for each sensor in the file sensorVarDict.json
        During fine-tuning or evaluation, the statistics are readed from the file
        """
        if isPreTrain and ("sensorVarDict.json" not in os.listdir(self.directory)):
            print(f'Start creating thresholds', end=' ', flush=True)
            varDf = self.data[["sens_pos", "vars"]]
            sensorsList = self.data["sens_pos"].unique()
            sensorVarDict = {}
            for sensor in tqdm(sensorsList):
                if (sensor in self.noisySensors) or (sensor not in self.distanceToSensor.keys()):
                    continue
                sensorVarDf = varDf[varDf["sens_pos"]==sensor]
                lower_bound, upper_bound = self.interquartileRule(sensorVarDf["vars"])
                sensorVarDf = sensorVarDf[(sensorVarDf["vars"]>lower_bound) & (sensorVarDf["vars"]<upper_bound)]
                mean = sensorVarDf["vars"].mean()
                std = sensorVarDf["vars"].std()
                threshold = mean + 3.5 * std
                sensorVarDict[sensor] = {"mean": mean, "std": std, "threshold": threshold}
                with open(self.directory + "./sensorVarDict.json", "w") as f:
                    # Write the dict to the file
                    json.dump(sensorVarDict, f)
            print(f'Finish thresholds creation', end=' ', flush=True)
        else:
            print(f'Start reading thresholds', end=' ', flush=True)
            with open(self.directory + "./sensorVarDict.json", "r") as f:
                # Load the dict from the file
                sensorVarDict = json.load(f)

            print(f'Finish thresholds reading', end=' ', flush=True)

        return sensorVarDict
