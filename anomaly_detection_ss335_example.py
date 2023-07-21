from Datasets.AnomalyDetection_SS335.get_dataset import get_dataset, get_data
from Algorithms.pca import pca_class
import datetime
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import time 
import datetime as dt
import sklearn.preprocessing

import torch
from Algorithms.models_audio_mae import audioMae_vit_base
from Algorithms.models_audio_mae_evaluate import audioMae_vit_base_evaluate
import timm
import timm.optim.optim_factory as optim_factory
assert timm.__version__ == "0.3.2"  # version check

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.engine_pretrain import train_one_epoch, evaluate, reconstruct
import argparse
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

def plot_results(data, data2 = None, name = "Base.png"):
    plt.figure(figsize=(16,4))
    plt.plot(np.arange(0,len(data['mse'])), data['mse'], color = 'g', label = 'Pre Intervention', linewidth = 1.5)
    if data2 != None:
        plt.plot(np.arange(len(data['mse']),len(data['mse']) + len(data2['mse'])), data2['mse'], color = 'k', label = 'Post Intervention', linewidth = 1.5)
    plt.grid(axis = 'both')
    plt.legend()
    plt.title('PCA Predicted Values')
    plt.xlabel('Time[days]')
    plt.ylabel('MSE')
    plt.savefig(name)

def main_PCA(args):
### Creating Training Dataset
    starting_date = datetime.date(2019,5,22) 
    num_days = 7
    print("Creating Training Dataset")
    dataset_train = get_data(args.dir, starting_date, num_days, sensor = 'S6.1.3', time_frequency = "time", windowLength = args.window_size)
    pca = pca_class(input_dim= 490, CF = 32)
    print("Fitting PCA")
    Ex, Vx, k = pca.fit(dataset_train)
### Creating Testing Dataset for Normal Data
    starting_date = datetime.date(2019,5,10) 
    num_days = 4
    print("Creating Testing Dataset -- Normal")
    dataset_test_normal = get_data(args.dir, starting_date, num_days, sensor = 'S6.1.3', time_frequency = "time", windowLength = args.window_size)
    pca_result_normal  = pca.predict(dataset_test_normal, Vx)
### Creating Testing Dataset for Anomaly Data
    starting_date = datetime.date(2019,4,17) 
    num_days = 4
    print("Creating Testing Dataset -- Anomaly")
    dataset_test_anomaly = get_data(args.dir, starting_date, num_days, sensor = 'S6.1.3', time_frequency = "time", windowLength = args.window_size)
    pca_result_anomaly  = pca.predict(dataset_test_anomaly, Vx)

    name = f"Results/PCA_{args.window_size}samples"
    # plot_results(pca_result_anomaly, pca_result_normal, f"{name}.png")
    df = pd.DataFrame.from_dict(pca_result_normal)
    df.to_csv(f'{name}_normal.csv', index = False, header = True)
    df = pd.DataFrame.from_dict(pca_result_anomaly)
    df.to_csv(f'{name}_anomaly.csv', index = False, header = True)

def main_masked_autoencoder(args):
### Creating Training 
    lr = 0.25e-2
    total_epochs = 401
    warmup_epochs = 50
    save_interval_epochs = 100
    starting_date = datetime.date(2019,5,22) 
    num_days = 7
    print("Creating Training Dataset")
    dataset = get_dataset(args.dir, starting_date, num_days, sensor = 'S6.1.3', time_frequency = "frequency", windowLength = args.window_size)
    sampler_train = torch.utils.data.RandomSampler(dataset)
    data_loader_train = torch.utils.data.DataLoader(
        dataset, sampler=sampler_train,
        batch_size=64,
        num_workers=1,
        pin_memory='store_true',
        drop_last=True,
    )
    device = torch.device('cuda')
    torch.manual_seed(0)
    np.random.seed(0)
    model = audioMae_vit_base(norm_pix_loss=False)
    model.to(device)
    param_groups = optim_factory.add_weight_decay(model, 0.05)
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()
    print(f"Start training for {total_epochs} epochs")
    for epoch in range(0, total_epochs):
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler, lr, total_epochs, warmup_epochs)
        if epoch % save_interval_epochs == 0:
            misc.save_model(output_dir="Results/checkpoints/", model=model, model_without_ddp=model, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

### Creating Testing Dataset for Normal Data
    starting_date = datetime.date(2019,5,10) 
    num_days = 4
    print("Creating Testing Dataset -- Normal")
    dataset = get_dataset(args.dir, starting_date, num_days, sensor = 'S6.1.3', time_frequency = "frequency", windowLength = args.window_size)
    data_loader_test_normal = torch.utils.data.DataLoader(
        dataset, shuffle=False,
        batch_size=1,
        num_workers=1,
        pin_memory='store_true',
        drop_last=True,
    )
    losses_normal = evaluate(data_loader_test_normal, model, device)
    df = pd.DataFrame.from_dict(losses_normal)
    df.to_csv(f'Results/masked_{args.window_size}samples_normal.csv', index = False, header = True)
        
### Creating Testing Dataset for Anomaly Data
    starting_date = datetime.date(2019,4,17) 
    num_days = 4
    print("Creating Testing Dataset -- Anomaly")
    dataset = get_dataset(args.dir, starting_date, num_days, sensor = 'S6.1.3', time_frequency = "frequency", windowLength = args.window_size)
    data_loader_test_anomaly = torch.utils.data.DataLoader(
        dataset, shuffle=False,
        batch_size=1,
        num_workers=1,
        pin_memory='store_true',
        drop_last=True,
    )
    losses_anomaly = evaluate(data_loader_test_anomaly, model, device)
    df = pd.DataFrame.from_dict(losses_anomaly)
    df.to_csv(f'Results/masked_{args.window_size}samples_anomaly.csv', index = False, header = True)

def evaluate_autoencoder(args):
    print("Loading the checkpoint from memory")
    device = torch.device('cuda')
    model = audioMae_vit_base_evaluate(norm_pix_loss=False)
    model.to(device)
    checkpoint = torch.load(f"Results/checkpoints/checkpoint--200.pth", map_location='cpu')
    checkpoint_model = checkpoint['model']
    msg = model.load_state_dict(checkpoint_model, strict=False)
### Creating Testing Dataset for Normal Data
    starting_date = datetime.date(2019,5,10) 
    num_days = 4
    print("Creating Testing Dataset -- Normal")
    dataset = get_dataset(args.dir, starting_date, num_days, sensor = 'S6.1.3', time_frequency = "frequency", windowLength = args.window_size)
    data_loader_test_normal = torch.utils.data.DataLoader(
        dataset, shuffle=False,
        batch_size=1,
        num_workers=1,
        pin_memory='store_true',
        drop_last=True,
    )
    losses_normal = evaluate(data_loader_test_normal, model, device)
    df = pd.DataFrame.from_dict(losses_normal)
    df.to_csv(f'Results/masked_{args.window_size}samples_normal.csv', index = False, header = True)
        
### Creating Testing Dataset for Anomaly Data
    starting_date = datetime.date(2019,4,17) 
    num_days = 4
    print("Creating Testing Dataset -- Anomaly")
    dataset = get_dataset(args.dir, starting_date, num_days, sensor = 'S6.1.3', time_frequency = "frequency", windowLength = args.window_size)
    data_loader_test_anomaly = torch.utils.data.DataLoader(
        dataset, shuffle=False,
        batch_size=1,
        num_workers=1,
        pin_memory='store_true',
        drop_last=True,
    )
    losses_anomaly = evaluate(data_loader_test_anomaly, model, device)
    df = pd.DataFrame.from_dict(losses_anomaly)
    df.to_csv(f'Results/masked_{args.window_size}samples_anomaly.csv', index = False, header = True)

def reconstruct_autoencoder(args):
    print("Loading the checkpoint from memory")
    device = torch.device('cuda')
    model = audioMae_vit_base_evaluate(norm_pix_loss=False)
    model.to(device)
    checkpoint = torch.load(f"Results/checkpoints/checkpoint--400.pth", map_location='cpu')
    checkpoint_model = checkpoint['model']
    msg = model.load_state_dict(checkpoint_model, strict=False)
    starting_date = datetime.date(2019,5,22) 
    num_days = 7
    print("Creating Training Dataset")
    dataset = get_dataset(args.dir, starting_date, num_days, sensor = 'S6.1.3', time_frequency = "frequency", windowLength = args.window_size)
    sampler_train = torch.utils.data.RandomSampler(dataset)
    data_loader_train = torch.utils.data.DataLoader(
        dataset, sampler=sampler_train,
        batch_size=1,
        num_workers=1,
        pin_memory='store_true',
        drop_last=True,
    )
    img, pred = reconstruct(data_loader_train, model, device, index = 30)
    fig, axs = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize = (8,8))
    axs[0].imshow(img.cpu()[0,0,:,:])
    axs[1].imshow(pred.cpu()[0,0,:,:])
    plt.savefig("Results/prova.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Base parameters')
    parser.add_argument('--dir', type=str, default="/baltic/users/shm_mon/SHM_Datasets_2023/Datasets/AnomalyDetection_SS335/",
                        help='directory')
    parser.add_argument('--model', type=str, default="SOA",
                        help='SOA, autoencoder, autoencoder_evaluate, autoencoder_reconstruct')
    parser.add_argument('--window_size', type=int, default=1190, help='fs = 100')
    args = parser.parse_args()
    model = args.model 
    print(args)
    if model == "SOA":
        main_PCA(args)
    elif model == "autoencoder":
        main_masked_autoencoder(args)
    elif model == "autoencoder_evaluate":
        evaluate_autoencoder(args)
    elif model == "autoencoder_reconstruct":
        reconstruct_autoencoder(args)