import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import datetime
from util.engine_pretrain import train_one_epoch
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from models.models_audio_mae import audioMae_vit_base
import timm.optim.optim_factory as optim_factory
import util.misc as misc
from util.engine_pretrain import evaluate
import argparse
import os
import csv
from utils import get_all_datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Base parameters')
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    print(args)

    # create results file
    filename = '/home/benfenati/code/tle-supervised/results/model_exploration_results.csv' # tag:change name
    header = ["encoder_dim", "decoder_embed_dim", "mae"]
    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)

    device = torch.device('cuda:{}'.format(args.device))

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

    for comb in combinations:
        print("Testing combination: ", comb)
        embed_dim = comb[0]
        decoder_embed_dim = comb[1]

        model = audioMae_vit_base(embed_dim=embed_dim, decoder_embed_dim=decoder_embed_dim, norm_pix_loss=True)
        model.to(device)

        # setup
        lr = 0.25e-3
        total_epochs = 201
        warmup_epochs = 100
        save_interval_epochs = 50
        param_groups = optim_factory.param_groups_weight_decay(model, 0.05)
        optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
        loss_scaler = NativeScaler()

        print(f"Start pre-training for {total_epochs} epochs")
        for epoch in range(0, total_epochs):
            train_stats = train_one_epoch(model, data_loader_train, optimizer, device, epoch, loss_scaler, lr, total_epochs, warmup_epochs)
            if epoch % save_interval_epochs == 0:
                misc.save_model(output_dir="/home/benfenati/code/tle-supervised/results/checkpoints/model_exploration", 
                                model=model, model_without_ddp=model, optimizer=optimizer, 
                                loss_scaler=loss_scaler, epoch=epoch, 
                                name = "{}-{}-pretrain_all".format(embed_dim, decoder_embed_dim))
                
        _, mae = evaluate(data_loader_val, model, device)

        last_row = [embed_dim, decoder_embed_dim, mae]
        with open(filename, 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(last_row)