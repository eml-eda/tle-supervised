import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

'''
Maintenance intervention day: 9th
Train: 24/05, 25/05, 26/05, 27/05
Validation: 28/05, 29/05, 30/05
Test post intervention: 01/06, 02/06, 03/06, 04/06 Test pre intervention: 01/05, 02/05, 03/05, 04/05

Data used from Amir:
# Week of Anomaly for test - 17 to 23 of April 2019
# Week of Normal for test - 10 to 14 of May 2019 
# Week of Normal for training - 20 or 22 to 29 of May 2019
'''

def plot_results_PCA(data, data2 = None, name = "Base.png", limits = [1,2]):
    plt.figure(figsize=(16,4))
    plt.plot(np.arange(0,len(data)), data, color = 'g', label = 'Pre Intervention', linewidth = 1.5)
    plt.plot(np.arange(len(data),len(data) + len(data2)), data2, color = 'k', label = 'Post Intervention', linewidth = 1.5)

    dim_filtering = 15
    new_data = []
    for i in np.arange(0, len(data-dim_filtering)):
        new_data.append(np.mean(data[i:(i+dim_filtering)]))
    new_data2 = []
    for i in np.arange(0, len(data2-dim_filtering)):
        new_data2.append(np.mean(data2[i:(i+dim_filtering)]))
    plt.plot(np.arange(0,len(new_data)), new_data, color = 'r', label = 'Pre Intervention', linewidth = 1.5)
    plt.plot(np.arange(len(new_data),len(new_data) + len(new_data2)), new_data2, color = 'b', label = 'Post Intervention', linewidth = 1.5)
    plt.grid(axis = 'both')
    plt.legend()
    plt.title('PCA Predicted Values')
    plt.xlabel('Time[days]')
    plt.ylabel('MSE')
    plt.ylim(limits)
    plt.savefig(name)

def plot_results_autoencoder(data, data2 = None, name = "Base.png", limits = [1,2]):
    plt.figure(figsize=(16,4))
    plt.plot(np.arange(0,len(data)), data, color = 'g', label = 'Pre Intervention', linewidth = 1.5)
    plt.plot(np.arange(len(data),len(data) + len(data2)), data2, color = 'k', label = 'Post Intervention', linewidth = 1.5)

    dim_filtering = 100
    new_data = []
    for i in np.arange(0, len(data-dim_filtering)):
        new_data.append(np.mean(data[i:(i+dim_filtering)]))
    new_data2 = []
    for i in np.arange(0, len(data2-dim_filtering)):
        new_data2.append(np.mean(data2[i:(i+dim_filtering)]))
    plt.plot(np.arange(0,len(new_data)), new_data, color = 'r', label = 'Pre Intervention', linewidth = 1.5)
    plt.plot(np.arange(len(new_data),len(new_data) + len(new_data2)), new_data2, color = 'b', label = 'Post Intervention', linewidth = 1.5)
    plt.grid(axis = 'both')
    plt.legend()
    plt.title('Autoencoder Predicted Values')
    plt.xlabel('Time[days]')
    plt.ylabel('MSE')
    plt.ylim(limits)
    plt.savefig(name)

def main(directory):
    data_normal = pd.read_csv(directory + "PCA_All_AllDayTrain_normal.csv")
    data_anomaly = pd.read_csv(directory + "PCA_All_AllDayTrain_anomaly.csv")
    limits = [0,0.00005]
    plot_results_PCA(data_anomaly.values, data_normal.values, f"Results/images/prova_PCA.png", limits)
    data_normal = pd.read_csv(directory + "masked_test_normal.csv")
    data_anomaly = pd.read_csv(directory + "masked_test_anomaly.csv")
    limits = [0.65,0.75]
    plot_results_autoencoder(data_anomaly.values, data_normal.values, f"Results/images/prova_autoencoder.png", limits)
    

if __name__ == "__main__":
    dir = "./Results/"
    main(dir)