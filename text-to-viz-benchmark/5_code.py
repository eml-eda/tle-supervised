import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_threshold_accuracy(anomalies, normal, ax, min, max, only_acc = 0, dim_filtering = 120):
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
    # print(f"Sensitivity: {sens} Spcificity: {spec} Accuracy: {acc}")
    if only_acc == 0:
        ax.axhline(y=(th-min)/(max-min), color='r', linestyle='--', label = "Threshold")
    return spec, sens, acc

def plot_results_PCA(data, data2, fig):
    th = np.median(data) + 1 * np.std(data)
    data_new = data[data<th]
    th = np.median(data) - 1 * np.std(data)
    data_new = data_new[data_new>th]
    th = np.median(data2) - 1 * np.std(data2)
    data2_new = data2[data2>th]
    th = np.median(data) + 1 * np.std(data)
    data2_new = data2_new[data2_new<th]
    norm = np.concatenate([data_new,data2_new])
    max = np.max(norm)
    min = np.min(norm)
    data = (data-min)/(max-min)
    data2 = (data2-min)/(max-min)
    fig.plot(np.arange(len(data2),len(data) + len(data2)), data, color = colors[0], label = 'Pre-Intervention', linewidth = 1.5, alpha = 0.5)
    fig.plot(np.arange(0,len(data2)), data2, color = colors[1], label = 'Post-Intervention', linewidth = 1.5, alpha = 0.5)
    new_data = []
    for i in np.arange(0, len(data-dim_filtering)):
        new_data.append(np.median(data[i:(i+dim_filtering)]))
    new_data2 = []
    for i in np.arange(0, len(data2-dim_filtering)):
        new_data2.append(np.median(data2[i:(i+dim_filtering)]))
    fig.plot(np.arange(len(new_data2),len(new_data) + len(new_data2)), new_data, label = 'Pre-Intervention Filtered', linewidth = 1.5, color = 'green')
    fig.plot(np.arange(0,len(new_data2)), new_data2, label = 'Post-Intervention Filtered', linewidth = 1.5, color = 'royalblue')
    fig.grid(axis = 'both')
    fig.set_title('PCA [36]')
    # fig.set_xlabel('Time [Input windows]', fontsize=12)
    # fig.set_ylabel('Normalized MSE [#]', fontsize=12)
    fig.set_ylabel('MSE', fontsize=12)
    fig.set_ylim([0,1])
    fig.set_xlim([0,len(new_data) + len(new_data2)])
    return min, max

def plot_results_autoencoder(data, data2 , fig):
    th = np.median(data) + 1 * np.std(data)
    data_new = data[data<th]
    th = np.median(data2) - 1 * np.std(data2)
    data2_new = data2[data2>th]
    norm = np.concatenate([data_new,data2_new])
    # norm = np.concatenate([data,data2])
    max = np.max(norm)
    min = np.min(norm)
    data = (data-min)/(max-min)
    data2 = (data2-min)/(max-min)
    fig.plot(np.arange(len(data2),len(data) + len(data2)), data, color = colors[0], label = 'Pre Intervention', linewidth = 1.5, alpha = 0.5)
    fig.plot(np.arange(0,len(data2)), data2, color = colors[1], label = 'Post Intervention', linewidth = 1.5, alpha = 0.5)

    new_data = []
    for i in np.arange(0, len(data-dim_filtering)):
        new_data.append(np.median(data[i:(i+dim_filtering)]))
    new_data2 = []
    for i in np.arange(0, len(data2-dim_filtering)):
        new_data2.append(np.median(data2[i:(i+dim_filtering)]))
    fig.plot(np.arange(len(new_data2),len(new_data) + len(new_data2)), new_data, color = 'green', label = 'Pre Intervention', linewidth = 1.5)
    fig.plot(np.arange(0,len(new_data2)), new_data2, color = 'royalblue', label = 'Post Intervention', linewidth = 1.5)
    fig.grid(axis = 'both')
    # fig.legend()
    fig.set_title('Ours')
    fig.set_xlabel('Time [Input windows]', fontsize=12)
    fig.set_ylabel('MSE', fontsize=12)
    fig.set_ylim([0,1])
    fig.set_xlim([0,len(new_data) + len(new_data2)])
    return min, max    

if __name__ == "__main__":
    directory = "4_data/"
    dim_filtering = 120
    colors = ['#73AD4C', '#6BA7DE']
    data_normal = pd.read_csv(directory + "PCA_1190samples_normal.csv")
    data_anomaly = pd.read_csv(directory + "PCA_1190samples_anomaly.csv")
    # fig, axs = plt.subplots(nrows=2, ncols=1, constrained_layout=True, figsize = (7,7), sharex=True)
    fig, axs = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize = (10,4), sharey=True)
    axs[0].tick_params(labelsize=12)
    min, max = plot_results_PCA(data_anomaly.values, data_normal.values, axs[0])
    compute_threshold_accuracy(data_anomaly.values, data_normal.values, axs[0], min, max, 0, dim_filtering)
    axs[0].legend(loc='upper left', fontsize=12)
    data_normal = pd.read_csv(directory + "ours_1190samples_normal.csv")
    data_anomaly = pd.read_csv(directory + "ours_1190samples_anomaly.csv")
    min, max = plot_results_autoencoder(data_anomaly.values, data_normal.values, axs[1])
    compute_threshold_accuracy(data_anomaly.values, data_normal.values, axs[1], min, max, 0, dim_filtering)
    axs[1].tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig(f"ad_reconstruction_MSE.png", dpi = 600)