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

import optuna

# This is added to test only configuration that have not been tried yet
def get_tried_configurations(filename):
    tried_configs = set()
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                config = (
                    int(row['encoder_depth']),
                    int(row['encoder_heads']),
                    int(row['decoder_depth']),
                    int(row['decoder_heads']),
                    (int(row['encoder_embedding_dim']), int(row['decoder_embedding_dim']))
                )
                tried_configs.add(config)
    return tried_configs

class CustomSampler(optuna.samplers.BaseSampler):
    def __init__(self, tried_configs):
        self.tried_configs = tried_configs
        self.base_sampler = optuna.samplers.RandomSampler()

    def sample_independent(self, study, trial, param_name, param_distribution):
        return self.base_sampler.sample_independent(study, trial, param_name, param_distribution)

    def sample_relative(self, study, trial, search_space):
        while True:
            params = {}
            for name, distribution in search_space.items():
                params[name] = self.base_sampler.sample_independent(study, trial, name, distribution)
            
            config = (
                params['encoder_depth'],
                params['encoder_heads'],
                params['decoder_depth'],
                params['decoder_heads'],
                params['embed_dim']
            )
            
            if config not in self.tried_configs:
                return params

def objective(trial):
    encoder_depth = trial.suggest_categorical("encoder_depth", [1, 3, 5, 7])
    encoder_heads = trial.suggest_categorical("encoder_heads", [8, 12, 16])
    decoder_depth = trial.suggest_categorical("decoder_depth", [2, 4, 6, 8])
    decoder_heads = trial.suggest_categorical("decoder_heads", [8, 16, 24])
    embedding_dim = trial.suggest_categorical("embed_dim", [(1536, 1024), (768, 512), (384, 256), (192, 128)])

    # Train your model with these hyperparameters
    # score = train_and_evaluate(encoder_depth, encoder_heads, decoder_depth, decoder_heads, embedding_dim)

    # params setup
    lr = 0.25e-3
    total_epochs = 20
    warmup_epochs = 10
    save_interval_epochs = 100
    model = audioMae_vit_base(embed_dim=embedding_dim[0], 
                              encoder_depth=encoder_depth,
                              num_heads=encoder_heads,
                              decoder_embed_dim=embedding_dim[1], 
                              decoder_depth=decoder_depth,
                              decoder_num_heads=decoder_heads,
                              norm_pix_loss=True)
    model.to(device)

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
                            name = "{}-{}-{}-{}-{}-{}-pretrain_all".format(encoder_depth, encoder_heads, decoder_depth, decoder_heads, embedding_dim[0], embedding_dim[1]))
            
    _, mae = evaluate(data_loader_val, model, device)

    last_row = [encoder_depth, encoder_heads, decoder_depth, decoder_heads, embedding_dim[0], embedding_dim[1], mae]
    with open(filename, 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(last_row)

    return mae  # E.g., validation loss or another metric

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Base parameters')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dir1', type=str, default='/space/benfenati/data_folder/SHM/AnomalyDetection_SS335/')
    parser.add_argument('--dir2', type=str, default='/space/benfenati/data_folder/SHM/Vehicles_Roccaprebalza/')
    parser.add_argument('--dir3', type=str, default='/space/benfenati/data_folder/SHM/Vehicles_Sacertis/')
    args = parser.parse_args()
    print(args)

    # create results file
    filename = '/home/benfenati/code/tle-supervised/results/model_exploration_results.csv' # tag:change name
    header = ["encoder_depth", "encoder_heads", "decoder_depth", "decoder_heads", "encoder_embedding_dim", "decoder_embedding_dim", "mae"]
    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)

    device = torch.device('cuda:{}'.format(args.device))

    # train and val dataset
    all_datasets = get_all_datasets(args.dir1, args.dir2, args.dir3, window_size=1190)

    # reduce amount of data used
    total_reduced_size = int(0.2 * len(all_datasets))
    reduced_dataset, _ = torch.utils.data.random_split(
        all_datasets, 
        [total_reduced_size, len(all_datasets) - total_reduced_size]
    )
    
    # split reduced dataset into train/val
    train_size = int(0.8 * len(reduced_dataset))
    val_size = len(reduced_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        reduced_dataset, 
        [train_size, val_size]
    )

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

    ## this is the standard setup
    # study = optuna.create_study(direction="minimize")  # or "maximize" depending on your metric
    # study.optimize(objective, n_trials=100)

    ## setup to test only non tested configurations
    # Get tried configurations
    tried_configs = get_tried_configurations(filename)
    
    # Create study with custom sampler
    study = optuna.create_study(
        direction="minimize",
        sampler=CustomSampler(tried_configs)
    )
    
    # Calculate remaining trials
    total_combinations = len([1,3,5,7]) * len([8,12,16]) * len([2,4,6,8]) * len([8,16,24]) * len([(1536,1024), (768,512), (384,256), (192,128)])
    remaining_trials = total_combinations - len(tried_configs)
    
    print(f"Total possible combinations: {total_combinations}")
    print(f"Already tried combinations: {len(tried_configs)}")
    print(f"Remaining combinations to try: {remaining_trials}")
    
    study.optimize(objective, n_trials=remaining_trials)