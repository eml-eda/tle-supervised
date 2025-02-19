from data.Vehicles_Sacertis.get_dataset import get_dataset, get_data

import os
import csv
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

import torch
from models.models_audio_mae import audioMae_vit_base
from models.models_audio_mae_regression import audioMae_vit_base_R
from models.models_tcn import tcn_regression as tcn_regression_simple
from models.models_tcn_regression import tcn_regression as tcn_regression_mae
from models.models_lstm import lstm_regression as lstm_regression_simple
from models.models_lstm_regression import lstm_regression as lstm_regression_mae



import timm.optim.optim_factory as optim_factory

from util.misc import interpolate_pos_embed
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.engine_pretrain import train_one_epoch, train_one_epoch_finetune, evaluate_finetune
import argparse
import pandas as pd

def compute_accuracy(y_test, y_predicted):
    mse = mean_squared_error(y_test, y_predicted)
    print("MSE:", mse)
    mae = mean_absolute_error(y_test, y_predicted)
    print("MAE:", mae)
    r2 = r2_score(y_test, y_predicted)
    print("R2:", r2)
    mspe = (mse/np.mean(y_test))*100
    print("MSE%:", mspe)
    mape = (mae/np.mean(y_test))*100
    print("MAE%:", mape)
    return mse, mae, r2, mspe, mape

def main_soa(args):
    data_train, labels_train = get_data(args.dir, True, False, False, sensor = "None", time_frequency = "time", features = 'Yes')
    data_test, labels_test = get_data(args.dir, False, False, True, sensor = "None", time_frequency = "time", features = 'Yes')

    steps_svr = [( 'scaler', StandardScaler() ), ('svr', SVR(kernel = 'rbf',epsilon=0.1, C=10))]
    steps_DecisionTreeRegressor = [( 'scaler', StandardScaler()), ('model', DecisionTreeRegressor(max_depth=200))]
    steps_MLPRegressor = [('scaler', QuantileTransformer()), ('model', MLPRegressor(hidden_layer_sizes=(100,100,100)))]
    steps_KNeighborsRegressor = [( 'scaler', StandardScaler() ), ('model', KNeighborsRegressor(n_neighbors=7))]
    steps_BayesianRidge = [( 'scaler', StandardScaler() ), ('model', LinearRegression())]
    steps = [steps_svr, steps_DecisionTreeRegressor, steps_MLPRegressor, steps_KNeighborsRegressor, steps_BayesianRidge]
    names = ["SVR", "DT", "MLP", "KNN", "LR"]
    for i, step in enumerate(steps):
        pipeline_svr_car = Pipeline(step)
        pipeline_svr_car.fit(data_train, labels_train)
        y_predicted = pipeline_svr_car.predict(data_test)
        ### METRICHE MISURA ACCURATEZZA PER CAR
        print(f"{names[i]} Prediction")
        compute_accuracy(labels_test, y_predicted)
        df = pd.DataFrame({"Y_true": labels_test, "Y_predicted": y_predicted})
        df.to_csv(f'results/Sacertis_{names[i]}.csv', index = False, header = True)
        
def main_autoencoder(args):
    # create results file
    filename = f'/home/benfenati/code/tle-supervised/results/uc3_results_{args.no_pretrain}-{args.pretrain_uc}-{args.pretrain_all}.csv' # tag:change name
    header = ["embed_dim", "decoder_dim", "mse", "mae", "r2", "mspe", "mape"]
    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)

    embed_dim = args.encoder_dim
    decoder_embed_dim = args.decoder_dim
    device = args.device
    print("Testing combination: {}-{}".format(embed_dim, decoder_embed_dim))

    device = torch.device(f'cuda:{args.device}')
    embed_dim = args.encoder_dim
    decoder_embed_dim = args.decoder_dim
    
    torch.manual_seed(0)
    np.random.seed(0)

    # No Pretrain Setup
    if args.no_pretrain == True:
        model = audioMae_vit_base_R(embed_dim=embed_dim, 
                                    decoder_embed_dim=decoder_embed_dim, 
                                    norm_pix_loss=True, mask_ratio = 0.2)
        model.to(device)

    # Pretrain UC Setup
    elif args.pretrain_uc == True:
        # pre-training data
        dataset = get_dataset(args.dir, True, False, False, sensor = "None", time_frequency = "frequency")
        sampler_train = torch.utils.data.RandomSampler(dataset)
        data_loader_train = torch.utils.data.DataLoader(
            dataset, sampler=sampler_train,
            batch_size=128,
            num_workers=1,
            pin_memory='store_true',
            drop_last=True)
        torch.manual_seed(0)
        np.random.seed(0)

        # training setup
        lr = 0.25e-3
        total_epochs = 201
        warmup_epochs = 50
        save_interval_epochs = 100
        param_groups = optim_factory.param_groups_weight_decay(model, 0.05)
        optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
        loss_scaler = NativeScaler()
        
        model = audioMae_vit_base(embed_dim=embed_dim, 
                                  decoder_embed_dim=decoder_embed_dim,
                                  norm_pix_loss=True)
        model.to(device)
        print(f"Start pre-training for {total_epochs} epochs")
        for epoch in range(0, total_epochs):
            train_stats = train_one_epoch(model, data_loader_train, optimizer, device, epoch, loss_scaler, lr, total_epochs, warmup_epochs)
            if epoch % save_interval_epochs == 0:
                misc.save_model(output_dir="/home/benfenati/code/tle-supervised/results/checkpoints/", model=model, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, name = f"_sacertis")
    
    # Pretrain All Setup
    elif args.pretrain_all == True:
        model = audioMae_vit_base_R(embed_dim=embed_dim, 
                                    decoder_embed_dim=decoder_embed_dim, 
                                    norm_pix_loss=True, mask_ratio = 0.2)
        model.to(device)
        checkpoint = torch.load(f"/home/benfenati/code/tle-supervised/results/checkpoints/checkpoint-{embed_dim}-{decoder_embed_dim}-pretrain_all-200.pth", map_location='cpu') # tag:change name
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        msg = model.load_state_dict(checkpoint_model, strict=False)
        interpolate_pos_embed(model, checkpoint_model)
    
    ##### Fine-tuning (this is valid for both setup)
    # finetuning data
    dataset = get_dataset(args.dir, False, True, False,  sensor = "None", time_frequency = "frequency")
    sampler_train = torch.utils.data.RandomSampler(dataset)
    data_loader_finetune = torch.utils.data.DataLoader(
        dataset, sampler=sampler_train,
        batch_size=128,
        num_workers=1,
        pin_memory='store_true',
        drop_last=True)
    
    # finetuning setup
    lr = 0.25e-5
    total_epochs = 201
    warmup_epochs = 50
    save_interval_epochs = 100
    param_groups = optim_factory.param_groups_weight_decay(model, 0.05)
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()
    criterion = torch.nn.MSELoss()
    # fine-tuning training
    print(f"Start finetuning for {total_epochs} epochs")
    for epoch in range(0, total_epochs):
        train_stats = train_one_epoch_finetune(model, criterion, data_loader_finetune, optimizer, device, epoch, loss_scaler, lr, total_epochs, warmup_epochs)
        if epoch % save_interval_epochs == 0:
            misc.save_model(output_dir="/home/benfenati/code/tle-supervised/results/checkpoints/", model=model, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, 
                            name = f"{embed_dim}-{decoder_embed_dim}_sacertis_finetune_{args.no_pretrain}-{args.pretrain_uc}-{args.pretrain_all}")


    ##### Testing
    dataset = get_dataset(args.dir, False, False, True,  sensor = "None", time_frequency = "frequency")
    data_loader_test = torch.utils.data.DataLoader(
        dataset, shuffle=False,
        batch_size=1,
        num_workers=1,
        pin_memory='store_true',
        drop_last=True,
    )
    y_predicted, y_test = evaluate_finetune(data_loader_test, model, device)
    mse, mae, r2, mspe, mape = compute_accuracy(y_test, y_predicted)

    last_row = [embed_dim, decoder_embed_dim, mse, mae, r2, mspe, mape]
    with open(filename, 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(last_row)

def main_tcn(args):
    # create results file
    filename = f'/home/benfenati/code/tle-supervised/results/uc3_results_tcn.csv' # tag:change name
    header = ["mse", "mae", "r2", "mspe", "mape"]
    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)

    device = args.device
    device = torch.device(f'cuda:{args.device}')
    embed_dim = args.encoder_dim
    decoder_embed_dim = args.decoder_dim
    
    torch.manual_seed(0)
    np.random.seed(0)

    if args.no_pretrain == True:
        model = tcn_regression_simple()
        model.to(device)

    # Pretrain All Setup
    elif args.pretrain_all == True:
        model = tcn_regression_mae(embed_dim=embed_dim, 
                                   decoder_embed_dim=decoder_embed_dim, 
                                   mask_ratio = 0.2)
        model.to(device)
        checkpoint = torch.load(f"/home/benfenati/code/tle-supervised/results/checkpoints/checkpoint-tcn-pretrain_all-200.pth", map_location='cpu') # tag:change name
        checkpoint_model = checkpoint['model']
        msg = model.load_state_dict(checkpoint_model, strict=False)
    
    ##### Fine-tuning (this is valid for both setup)
    # finetuning data
    dataset = get_dataset(args.dir, False, True, False,  sensor = "None", time_frequency = "frequency")
    sampler_train = torch.utils.data.RandomSampler(dataset)
    data_loader_finetune = torch.utils.data.DataLoader(
        dataset, sampler=sampler_train,
        batch_size=128,
        num_workers=1,
        pin_memory='store_true',
        drop_last=True)
    
    # finetuning setup
    lr = 0.25e-5
    total_epochs = 201
    warmup_epochs = 50
    save_interval_epochs = 100
    param_groups = optim_factory.param_groups_weight_decay(model, 0.05)
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()
    criterion = torch.nn.MSELoss()
    # fine-tuning training
    print(f"Start finetuning for {total_epochs} epochs")
    for epoch in range(0, total_epochs):
        train_stats = train_one_epoch_finetune(model, criterion, data_loader_finetune, optimizer, device, epoch, loss_scaler, lr, total_epochs, warmup_epochs)
        if epoch % save_interval_epochs == 0:
            misc.save_model(output_dir="/home/benfenati/code/tle-supervised/results/checkpoints/", model=model, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, 
                            name = f"tcn_sacertis_finetune_{args.no_pretrain}-{args.pretrain_all}")

    ##### Testing
    dataset = get_dataset(args.dir, False, False, True,  sensor = "None", time_frequency = "frequency")
    data_loader_test = torch.utils.data.DataLoader(
        dataset, shuffle=False,
        batch_size=1,
        num_workers=1,
        pin_memory='store_true',
        drop_last=True,
    )
    y_predicted, y_test = evaluate_finetune(data_loader_test, model, device)
    mse, mae, r2, mspe, mape = compute_accuracy(y_test, y_predicted)

    last_row = [mse, mae, r2, mspe, mape]
    with open(filename, 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(last_row)

def main_lstm(args):
    # create results file
    filename = f'/home/benfenati/code/tle-supervised/results/uc3_results_lstm.csv' # tag:change name
    header = ["mse", "mae", "r2", "mspe", "mape"]
    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)

    device = args.device
    device = torch.device(f'cuda:{args.device}')
    embed_dim = args.encoder_dim
    decoder_embed_dim = args.decoder_dim
    
    torch.manual_seed(0)
    np.random.seed(0)

    if args.no_pretrain == True:
        model = lstm_regression_simple()
        model.to(device)

    # Pretrain All Setup
    elif args.pretrain_all == True:
        model = lstm_regression_mae(embed_dim=embed_dim, 
                                   decoder_embed_dim=decoder_embed_dim, 
                                   mask_ratio = 0.2)
        model.to(device)
        checkpoint = torch.load(f"/home/benfenati/code/tle-supervised/results/checkpoints/checkpoint-lstm-pretrain_all-200.pth", map_location='cpu') # tag:change name
        checkpoint_model = checkpoint['model']
        msg = model.load_state_dict(checkpoint_model, strict=False)
    
    ##### Fine-tuning (this is valid for both setup)
    # finetuning data
    dataset = get_dataset(args.dir, False, True, False,  sensor = "None", time_frequency = "frequency")
    sampler_train = torch.utils.data.RandomSampler(dataset)
    data_loader_finetune = torch.utils.data.DataLoader(
        dataset, sampler=sampler_train,
        batch_size=128,
        num_workers=1,
        pin_memory='store_true',
        drop_last=True)
    
    # finetuning setup
    lr = 0.25e-5
    total_epochs = 201
    warmup_epochs = 50
    save_interval_epochs = 100
    param_groups = optim_factory.param_groups_weight_decay(model, 0.05)
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()
    criterion = torch.nn.MSELoss()
    # fine-tuning training
    print(f"Start finetuning for {total_epochs} epochs")
    for epoch in range(0, total_epochs):
        train_stats = train_one_epoch_finetune(model, criterion, data_loader_finetune, optimizer, device, epoch, loss_scaler, lr, total_epochs, warmup_epochs)
        if epoch % save_interval_epochs == 0:
            misc.save_model(output_dir="/home/benfenati/code/tle-supervised/results/checkpoints/", model=model, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, 
                            name = f"lstm_sacertis_finetune_{args.no_pretrain}-{args.pretrain_all}")

    ##### Testing
    dataset = get_dataset(args.dir, False, False, True,  sensor = "None", time_frequency = "frequency")
    data_loader_test = torch.utils.data.DataLoader(
        dataset, shuffle=False,
        batch_size=1,
        num_workers=1,
        pin_memory='store_true',
        drop_last=True,
    )
    y_predicted, y_test = evaluate_finetune(data_loader_test, model, device)
    mse, mae, r2, mspe, mape = compute_accuracy(y_test, y_predicted)

    last_row = [mse, mae, r2, mspe, mape]
    with open(filename, 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(last_row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Base parameters')
    parser.add_argument('--device', type=int, default=0, help='gpu device')
    parser.add_argument('--dir', type=str, default="/home/benfenati/data_folder/SHM/Vehicles_Sacertis/", help='directory')
    parser.add_argument('--model', type=str, default="soa", help='soa, autoencoder')
    parser.add_argument('--no_pretrain', type=bool, default=False)
    parser.add_argument('--pretrain_uc', type=bool, default=False)
    parser.add_argument('--pretrain_all', type=bool, default=True)
    parser.add_argument('--encoder_dim', type=int, default=768)
    parser.add_argument('--decoder_dim', type=int, default=512)
    args = parser.parse_args()
    print(args)

    if args.model == "soa": 
        main_soa(args)
    elif args.model == "autoencoder": 
        main_autoencoder(args)
    elif args.model == "tcn":
        main_tcn(args)
    elif args.model == "lstm":
        main_lstm(args)
    else: 
        print("Model not found")
        exit(1)
