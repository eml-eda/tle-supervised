# Roccaprebalza
import os
import csv
from data.Vehicles_Roccaprebalza.get_dataset import get_data, get_dataset

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split
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
from models.models_tcn_regression import tcn_regression
from models.models_lstm_regression import lstm_regression
import timm
import timm.optim.optim_factory as optim_factory
from util.misc import interpolate_pos_embed
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.engine_pretrain import train_one_epoch, evaluate, train_one_epoch_finetune, evaluate_finetune
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
    data_train, labels_train, data_test, labels_test = get_data(args.dir, window_sec_size = 60, shift_sec_size = 2, time_frequency = "time", car = args.car)
    number_of_features = 12
    model = SelectKBest(f_regression, k = number_of_features).fit(data_train, labels_train)
    X_train = model.transform(data_train)
    X_test = model.transform(data_test)
    steps_svr = [( 'scaler', StandardScaler() ), ('svr', SVR(kernel = 'rbf',epsilon=0.1, C=10))]
    steps_DecisionTreeRegressor = [( 'scaler', StandardScaler()), ('model', DecisionTreeRegressor(max_depth=200))]
    steps_MLPRegressor = [('scaler', QuantileTransformer()), ('model', MLPRegressor(hidden_layer_sizes=(100,100,100)))]
    steps_KNeighborsRegressor = [( 'scaler', StandardScaler() ), ('model', KNeighborsRegressor(n_neighbors=7))]
    steps_BayesianRidge = [( 'scaler', StandardScaler() ), ('model', LinearRegression())]
    steps = [steps_svr, steps_DecisionTreeRegressor, steps_MLPRegressor, steps_KNeighborsRegressor, steps_BayesianRidge]
    names = ["SVR", "DT", "MLP", "KNN", "LR"]
    for i, step in enumerate(steps):
        pipeline_svr_car = Pipeline(step)
        pipeline_svr_car.fit(X_train,labels_train)
        y_predicted_car = pipeline_svr_car.predict(X_test)
        ### METRICHE MISURA ACCURATEZZA PER CAR
        print(f"{args.car} Prediction")
        compute_accuracy(labels_test, y_predicted_car)
        df = pd.DataFrame({"Y_true": labels_test, "Y_predicted": y_predicted_car})
        df.to_csv(f'results/Roccaprebalza_{names[i]}_{args.car}.csv', index = False, header = True)
        
def main_autoencoder(args):
    # create results file
    filename = f'/home/benfenati/code/tle-supervised/results/uc2_results_{args.no_pretrain}-{args.pretrain_uc}-{args.pretrain_all}.csv' # tag:change name
    header = ["embed_dim", "decoder_dim", "car", "mse", "mae", "r2", "mspe", "mape"]
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
    
    # training data (for both pre-training and fine-tuning)
    dataset_train, dataset_test = get_dataset(args.dir, window_sec_size = 60, shift_sec_size = 2, time_frequency = "frequency", car = args.car)
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=8,
        num_workers=1,
        pin_memory='store_true',
        drop_last=True)
    # testing data
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, shuffle=False,
        batch_size=1,
        num_workers=1,
        pin_memory='store_true',
        drop_last=True,
    )

    # No Pretrain Setup
    if args.no_pretrain == True:
        model = audioMae_vit_base_R(embed_dim=embed_dim, 
                                    decoder_embed_dim=decoder_embed_dim, 
                                    norm_pix_loss=True, mask_ratio = 0.2)
        model.to(device)

    # Pretrain UC Setup
    elif args.pretrain_uc == True:
        # training setup
        lr = 0.25e-3
        total_epochs = 201
        warmup_epochs = 100
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
                misc.save_model(output_dir="/home/benfenati/code/tle-supervised/results/checkpoints/", model=model, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, name = f"{args.car}_roccaprebalza")
    
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
    # fine-tuning setup
    lr = 0.25e-5
    total_epochs = 501
    warmup_epochs = 250
    save_interval_epochs = 100
    param_groups = optim_factory.param_groups_weight_decay(model, 0.05)
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()
    criterion = torch.nn.MSELoss()
    # fine-tuning training
    print(f"Start finetuning for {total_epochs} epochs")
    for epoch in range(0, total_epochs):
        train_stats = train_one_epoch_finetune(model, criterion, data_loader_train, optimizer, device, epoch, loss_scaler, lr, total_epochs, warmup_epochs)
        if epoch % save_interval_epochs == 0:
            misc.save_model(output_dir="/home/benfenati/code/tle-supervised/results/checkpoints/", model=model, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, 
                            name = f"{embed_dim}-{decoder_embed_dim}-{args.car}_roccaprebalza_finetune_{args.no_pretrain}-{args.pretrain_uc}-{args.pretrain_all}")

    y_predicted, y_test = evaluate_finetune(data_loader_test, model, device)
    mse, mae, r2, mspe, mape = compute_accuracy(y_test, y_predicted)

    last_row = [embed_dim, decoder_embed_dim, args.car, mse, mae, r2, mspe, mape]
    with open(filename, 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(last_row)

def main_tcn(args):
    # create results file
    filename = f'/home/benfenati/code/tle-supervised/results/uc2_results_tcn.csv' # tag:change name
    header = ["car", "mse", "mae", "r2", "mspe", "mape"]
    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)

    device = args.device
    device = torch.device(f'cuda:{args.device}')
    
    torch.manual_seed(0)
    np.random.seed(0)
    
    # training data (for both pre-training and fine-tuning)
    dataset_train, dataset_test = get_dataset(args.dir, window_sec_size = 60, shift_sec_size = 2, time_frequency = "frequency", car = args.car)
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=8,
        num_workers=1,
        pin_memory='store_true',
        drop_last=True)
    
    # testing data
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, shuffle=False,
        batch_size=1,
        num_workers=1,
        pin_memory='store_true',
        drop_last=True,
    )

    # # no pre-train setup
    # model = tcn_regression(
    #     num_mels=100,
    #     mel_len=100,
    #     num_channels=[100, 100, 100, 100],
    #     kernel_size=2,
    #     dropout=0.2
    # )

    model = tcn_regression()
    model.to(device)
    
    ##### Fine-tuning (this is valid for both setup)
    # fine-tuning setup
    lr = 0.25e-5
    total_epochs = 501
    warmup_epochs = 250
    save_interval_epochs = 100
    param_groups = optim_factory.param_groups_weight_decay(model, 0.05)
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()
    criterion = torch.nn.MSELoss()
    # fine-tuning training
    print(f"Start finetuning for {total_epochs} epochs")
    for epoch in range(0, total_epochs):
        train_stats = train_one_epoch_finetune(model, criterion, data_loader_train, optimizer, device, epoch, loss_scaler, lr, total_epochs, warmup_epochs)
        if epoch % save_interval_epochs == 0:
            misc.save_model(output_dir="/home/benfenati/code/tle-supervised/results/checkpoints/", model=model, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, 
                            name = f"tcn-{args.car}_roccaprebalza_finetune")

    y_predicted, y_test = evaluate_finetune(data_loader_test, model, device)
    mse, mae, r2, mspe, mape = compute_accuracy(y_test, y_predicted)

    last_row = [args.car, mse, mae, r2, mspe, mape]
    with open(filename, 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(last_row)

def main_lstm(args):
    # create results file
    filename = f'/home/benfenati/code/tle-supervised/results/uc2_results_lstm.csv' # tag:change name
    header = ["car", "mse", "mae", "r2", "mspe", "mape"]
    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)

    device = args.device
    device = torch.device(f'cuda:{args.device}')
    
    torch.manual_seed(0)
    np.random.seed(0)
    
    # training data (for both pre-training and fine-tuning)
    dataset_train, dataset_test = get_dataset(args.dir, window_sec_size = 60, shift_sec_size = 2, time_frequency = "frequency", car = args.car)
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=8,
        num_workers=1,
        pin_memory='store_true',
        drop_last=True)
    
    # testing data
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, shuffle=False,
        batch_size=1,
        num_workers=1,
        pin_memory='store_true',
        drop_last=True,
    )

    model = lstm_regression()
    model.to(device)
    
    ##### Fine-tuning (this is valid for both setup)
    # fine-tuning setup
    lr = 0.25e-5
    total_epochs = 501
    warmup_epochs = 250
    save_interval_epochs = 100
    param_groups = optim_factory.param_groups_weight_decay(model, 0.05)
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()
    criterion = torch.nn.MSELoss()
    # fine-tuning training
    print(f"Start finetuning for {total_epochs} epochs")
    for epoch in range(0, total_epochs):
        train_stats = train_one_epoch_finetune(model, criterion, data_loader_train, optimizer, device, epoch, loss_scaler, lr, total_epochs, warmup_epochs)
        if epoch % save_interval_epochs == 0:
            misc.save_model(output_dir="/home/benfenati/code/tle-supervised/results/checkpoints/", model=model, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, 
                            name = f"lstm-{args.car}_roccaprebalza_finetune")

    y_predicted, y_test = evaluate_finetune(data_loader_test, model, device)
    mse, mae, r2, mspe, mape = compute_accuracy(y_test, y_predicted)

    last_row = [args.car, mse, mae, r2, mspe, mape]
    with open(filename, 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(last_row)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Base parameters')
    parser.add_argument('--device', type=int, default=0, help='gpu device')
    parser.add_argument('--dir', type=str, default="/home/benfenati/data_folder/SHM/Vehicles_Roccaprebalza/", help='directory')
    parser.add_argument('--model', type=str, default="soa", help='soa, autoencoder, tcn')
    parser.add_argument('--car', type=str, default="y_car", help='y_camion, y_car')
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
