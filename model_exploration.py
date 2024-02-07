import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import datetime
from util.engine_pretrain import train_one_epoch
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from Algorithms.models_audio_mae import audioMae_vit_base
import timm.optim.optim_factory as optim_factory
import util.misc as misc
from util.engine_pretrain import evaluate
import argparse
import os
import csv

import Datasets.Vehicles_Sacertis.get_dataset as Vehicles_Sacertis
import Datasets.AnomalyDetection_SS335.get_dataset as AnomalyDetection_SS335
import Datasets.Vehicles_Roccaprebalza.get_dataset as Vehicles_Roccaprebalza

def get_all_datasets():
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1, 2, 3"
    print("Creating Training Dataset")
    starting_date = datetime.date(2019,5,22) 
    num_days = 7
    window_size = 1190
    directory = "/home/benfenati/code/Datasets/SHM/AnomalyDetection_SS335/"
    data_anomaly = AnomalyDetection_SS335.get_data(directory, starting_date, num_days, sensor = 'S6.1.3', time_frequency = "frequency", windowLength = window_size)
    directory = "/home/benfenati/code/Datasets/SHM/Vehicles_Roccaprebalza/"
    data_train, _, _, _ = Vehicles_Roccaprebalza.get_data(directory, window_sec_size = 60, shift_sec_size = 2, time_frequency = "frequency", car = "y_camion")
    directory = "/home/benfenati/code/Datasets/SHM/Vehicles_Sacertis/"
    data_sacertis, _ = Vehicles_Sacertis.get_data(directory, True, False, False, time_frequency = "frequency")
    data_all = []
    for data in data_sacertis: data_all.append(data[0])
    for i in np.arange(data_anomaly.shape[0]): data_all.append(torch.from_numpy(data_anomaly[i]))
    for data in data_train: data_all.append(data)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Base parameters')
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    print(args)

    # create results file
    filename = '/home/benfenati/code/tle-supervised/Results/model_exploration_results_device{}.csv'.format(args.device) # tag:change name
    header = ["encoder_dim", "decoder_embed_dim", "mar"]
    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)

    device = torch.device('cuda:{}'.format(args.device))
    restart_pretraining = False
    lr = 0.25e-3
    total_epochs = 201
    warmup_epochs = 100
    save_interval_epochs = 50

    # train and val dataset
    all_datasets = get_all_datasets()

    train_size = int(0.8 * len(all_datasets))
    val_size = len(all_datasets) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(all_datasets, [train_size, val_size])

    sampler_train = torch.utils.data.RandomSampler(train_dataset)
    data_loader_train = torch.utils.data.DataLoader(
        train_dataset, sampler=sampler_train,
        batch_size=128,
        num_workers=1,
        pin_memory='store_true',
        drop_last=True)

    data_loader_val = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=128,
        num_workers=1,
        pin_memory='store_true',
        drop_last=True)

    torch.manual_seed(42)
    np.random.seed(42)
    
    combinations = [(768, 512), (384, 256), (192, 128), (96, 64), (48, 32), (24, 16)]
    # combinations = [(768, 512), (24, 16)] # device 0
    # combinations = [(384, 256),(48, 32)] # device 1
    # combinations = [(192, 128), (96, 64)] # device 2

    for comb in combinations:
        print("Testing combination: ", comb)
        embed_dim = comb[0]
        decoder_embed_dim = comb[1]

        model = audioMae_vit_base(embed_dim=embed_dim, decoder_embed_dim=decoder_embed_dim, norm_pix_loss=True)
        model.to(device)

        start_epoch = 0

        param_groups = optim_factory.param_groups_weight_decay(model, 0.05)
        optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
        loss_scaler = NativeScaler()
        print(f"Start pre-training for {total_epochs} epochs")
        for epoch in range(start_epoch, total_epochs):
            train_stats = train_one_epoch(model, data_loader_train, optimizer, device, epoch, loss_scaler, lr, total_epochs, warmup_epochs)
            if epoch % save_interval_epochs == 0:
                misc.save_model(output_dir="/home/benfenati/code/tle-supervised/Results/checkpoints/", model=model, model_without_ddp=model, optimizer=optimizer, 
                                loss_scaler=loss_scaler, epoch=epoch, 
                                name = "{}-{}-pretrain_all".format(embed_dim, decoder_embed_dim))
                
        _, mae = evaluate(data_loader_val, model, device)

        last_row = [embed_dim, decoder_embed_dim, mae]
        with open(filename, 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(last_row)