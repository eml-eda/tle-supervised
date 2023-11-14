import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 


def main_plot():
    # Create the figure and subplots
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex = True)
    x = np.arange(3)
    bar_width = 0.20
    # Plot acc_PCA and acc_enc in the first subplot
    axs[0].bar(x - bar_width, MSE_pretrain_all, width=bar_width, color='#A9D18E', edgecolor = 'k', label='Pretrain All')
    axs[0].bar(x , MSE_pretrain, width=bar_width, color='lightcyan', edgecolor = 'k', label='Normal')
    axs[0].bar(x + bar_width, MSE_no_pretrain, width=bar_width, color='#9DC3E6', edgecolor = 'k', label='No-Pretrain')
    axs[0].set_ylabel('MSE[%]', fontsize=12)
    axs[0].legend(fontsize=12)
    for ax in axs:
        ax.set_xticks([])
        ax.set_xlabel('')
        ax.yaxis.grid(True)
        ax.tick_params(labelsize=12)
    axs[1].bar(x - bar_width, MAE_pretrain_all, width=bar_width, color='#A9D18E', edgecolor = 'k', label='Pretrain All')
    axs[1].bar(x , MAE_pretrain, width=bar_width, color='lightcyan', edgecolor = 'k', label='Normal')
    axs[1].bar(x + bar_width, MAE_no_pretrain, width=bar_width, color='#9DC3E6', edgecolor = 'k', label='No-Pretrain')
    axs[1].set_ylabel('MAE[%]', fontsize=12)
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(["UC2 - Car", "UC2 - Camion", "UC3"])
    plt.subplots_adjust(hspace=0.1)
    plt.savefig("Results/images/Pretraining.png", dpi=600)

MSE_pretrain_all = [5.82, 18.48, 31.60]
MSE_pretrain = [5.39, 17.97, 39.96 ]
MSE_no_pretrain = [19.11, 151.24, 44.46]

MAE_pretrain_all = [9.98,  13.10, 29.29]
MAE_pretrain = [9.54, 13.09, 33.89 ]
MAE_no_pretrain = [19.39, 39.32, 35.92]


# MSE_original      = [5.39, ]
# MSE_tail          = [19.21, ]
# MSE_tail_modified = [11.47, ]

# MAE_original      = [9.54, ]
# MAE_tail          = [18.73, ]
# MAE_tail_modified = [14.50, ]

if __name__ == "__main__":
    main_plot()