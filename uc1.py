from data.AnomalyDetection_SS335.get_dataset import get_dataset, get_data
from models.pca import pca_class
import datetime
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import time 
import datetime as dt
import sklearn.preprocessing

import torch
from models.models_audio_mae import audioMae_vit_base
from models.models_audio_mae_evaluate import audioMae_vit_base_evaluate
import timm
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.engine_pretrain import train_one_epoch, evaluate, reconstruct
import argparse
from utils import *

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
def main_PCA(args):
    ### Creating Training Dataset
    starting_date = datetime.date(2019,5,22) 
    num_days = 7
    print("Creating Training Dataset")
    dataset_train = get_data(args.dir, starting_date, num_days, sensor = 'S6.1.3', time_frequency = "time", windowLength = args.window_size)
    pca = pca_class(input_dim=1190, CF = 32)
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
    # import pdb;pdb.set_trace()
    
    name = f"results/PCA_{args.window_size}samples"
    df = pd.DataFrame.from_dict(pca_result_normal)
    df.to_csv(f'{name}_normal.csv', index = False, header = True)
    df = pd.DataFrame.from_dict(pca_result_anomaly)
    df.to_csv(f'{name}_anomaly.csv', index = False, header = True)

def main_autoencoder(args):
    embed_dim = args.encoder_dim
    decoder_embed_dim = args.decoder_dim
    device = args.device

    print("Testing combination: {}-{}".format(embed_dim, decoder_embed_dim))

    lr = 0.25e-2
    total_epochs = args.epochs
    warmup_epochs = 50
    save_interval_epochs = 100
    starting_date = datetime.date(2019,5,22) 
    num_days = 7
    print("Creating Training Dataset")
    dataset = get_dataset(args.dir, starting_date, num_days, sensor = 'S6.1.3', time_frequency = "frequency", windowLength = args.window_size)
    sampler_train = torch.utils.data.RandomSampler(dataset)
    data_loader_train = torch.utils.data.DataLoader(
        dataset, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=1,
        pin_memory='store_true',
        drop_last=True,
    )
    device = torch.device('cuda:{}'.format(device))
    torch.manual_seed(0)
    np.random.seed(0)
    if args.pretrain_all == True:
        checkpoint = torch.load(f"/home/benfenati/code/tle-supervised/results/checkpoints/checkpoint-{embed_dim}-{decoder_embed_dim}-pretrain_all-200.pth", map_location='cpu')
        checkpoint_model = checkpoint['model']
        msg = model.load_state_dict(checkpoint_model, strict=False)
    else: 
        model = audioMae_vit_base(embed_dim=embed_dim, decoder_embed_dim=decoder_embed_dim, norm_pix_loss=False)
    model.to(device)

    param_groups = optim_factory.param_groups_weight_decay(model, 0.05)
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()
    print(f"Start training for {total_epochs} epochs")
    for epoch in range(0, total_epochs):
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler, lr, total_epochs, warmup_epochs)
        if epoch % save_interval_epochs == 0:
            misc.save_model(output_dir="results/checkpoints/", model=model, model_without_ddp=model, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, name="{}-{}-{}".format(args.window_size, embed_dim, decoder_embed_dim))

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
    losses_normal, _ = evaluate(data_loader_test_normal, model, device)
    df = pd.DataFrame.from_dict(losses_normal)
    df.to_csv(f'results/masked_{args.window_size}samples_normal_{embed_dim}-{decoder_embed_dim}.csv', index = False, header = True)
        
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
    losses_anomaly, _ = evaluate(data_loader_test_anomaly, model, device)
    df = pd.DataFrame.from_dict(losses_anomaly)
    df.to_csv(f'results/masked_{args.window_size}samples_anomaly_{embed_dim}-{decoder_embed_dim}.csv', index = False, header = True)

def evaluate_autoencoder(args):
    # test autoencoder
    directory = "/home/benfenati/code/tle-supervised/results/"
    acc_enc = []
    sens_enc = []
    spec_enc = []

    for dim_filtering in [15,30,60,120, 240]:
        print(f"Dim {dim_filtering}")
        print(f"PCA")
        data_normal = pd.read_csv(directory + f"masked_{args.window_size}_samples_normal.csv")
        data_anomaly = pd.read_csv(directory + f"masked_{args.window_size}samples_anomaly.csv")
        spec, sens, acc = compute_threshold_accuracy(data_anomaly.values, data_normal.values, None, min, max, only_acc = 1, dim_filtering = dim_filtering)
        acc_enc.append(acc*100)
        sens_enc.append(sens*100)
        spec_enc.append(spec*100)

def evaluate_soa(args):
    # test PCA
    directory = "/home/benfenati/code/tle-supervised/results/"
    acc_enc = []
    sens_enc = []
    spec_enc = []

    for dim_filtering in [15,30,60,120, 240]:
        print(f"Dim {dim_filtering}")
        print(f"PCA")
        data_normal = pd.read_csv(directory + f"PCA_{args.window_size}_samples_normal.csv")
        data_anomaly = pd.read_csv(directory + f"PCA_{args.window_size}samples_anomaly.csv")
        spec, sens, acc = compute_threshold_accuracy(data_anomaly.values, data_normal.values, None, min, max, only_acc = 1, dim_filtering = dim_filtering)
        acc_enc.append(acc*100)
        sens_enc.append(sens*100)
        spec_enc.append(spec*100)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Base parameters')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dir', type=str, default="/home/benfenati/data_folder/SHM/AnomalyDetection_SS335/", help='directory')
    parser.add_argument('--model', type=str, default="soa", help="Model to use [soa, autoencoder]")
    parser.add_argument('--window_size', type=int, default=490, help='fs = 100')
    parser.add_argument('--epochs', type=int, default=401)
    parser.add_argument('--pretrain_all', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--encoder_dim', type=int, default=768)
    parser.add_argument('--decoder_dim', type=int, default=512)
    args = parser.parse_args()
    model = args.model 
    print(args)
    
    if args.model == "autoencoder": 
        main_autoencoder(args)
    elif args.model == "evaluate_autoencoder":
        evaluate_autoencoder(args)
    elif args.model == "soa": 
        main_PCA(args)
    elif args.model == "evaluate_soa":
        evaluate_soa(args)
    else: 
        print("Model not found")
        exit(1)