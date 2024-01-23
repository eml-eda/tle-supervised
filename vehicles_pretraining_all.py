import Datasets.Vehicles_Sacertis.get_dataset as Vehicles_Sacertis
import Datasets.AnomalyDetection_SS335.get_dataset as AnomalyDetection_SS335
import Datasets.Vehicles_Roccaprebalza.get_dataset as Vehicles_Roccaprebalza

import datetime
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

import torch
from Algorithms.models_audio_mae import audioMae_vit_base
from Algorithms.models_audio_mae_regression_modified import audioMae_vit_base_R
import timm
import timm.optim.optim_factory as optim_factory
# assert timm.__version__ == "0.3.2"  # version check # I commented this to solve "torch._six does not exist" error
from util.misc import interpolate_pos_embed
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.engine_pretrain import train_one_epoch, evaluate, train_one_epoch_finetune, evaluate_finetune
import argparse

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

def get_all_datasets():
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1, 2, 3"
    print("Creating Training Dataset")
    starting_date = datetime.date(2019,5,22) 
    num_days = 7
    window_size = 1190
    # directory = "/baltic/users/shm_mon/SHM_Datasets_2023/Datasets/AnomalyDetection_SS335/"
    directory = "/home/benfenati/code/Datasets/SHM/AnomalyDetection_SS335/"
    data_anomaly = AnomalyDetection_SS335.get_data(directory, starting_date, num_days, sensor = 'S6.1.3', time_frequency = "frequency", windowLength = window_size)
    # directory = "/baltic/users/shm_mon/SHM_Datasets_2023/Datasets/Vehicles_Roccaprebalza/"
    directory = "/home/benfenati/code/Datasets/SHM/Vehicles_Roccaprebalza/"
    data_train, _, _, _ = Vehicles_Roccaprebalza.get_data(directory, window_sec_size = 60, shift_sec_size = 2, time_frequency = "frequency", car = "y_camion")
    # directory = "/baltic/users/shm_mon/SHM_Datasets_2023/Datasets/Vehicles_Sacertis/"
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
    
def pretrain(args):
    device = torch.device('cuda:{}'.format(args.device))
    lr = 0.25e-3
    total_epochs = 201
    warmup_epochs = 100
    save_interval_epochs = 100
    dataset_train = get_all_datasets()
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=128,
        num_workers=1,
        pin_memory='store_true',
        drop_last=True)
    torch.manual_seed(0)
    np.random.seed(0)
    embed_dim = 768 # teacher: 768, student 1: 384, student 2: 192, student 3: 96, student 4: 48, student 5: 24
    model = audioMae_vit_base(embed_dim=embed_dim, norm_pix_loss=True)
    model.to(device)

    if args.restart_pretraining:
        print("Loading previous checkpoint model...")
        start_epoch = 101
        checkpoint = torch.load(f"/home/benfenati/code/shm/checkpoints/checkpoint-pretrain_all-100.pth", map_location='cpu')
        # checkpoint = torch.load(f"/home/benfenati/code/shm/checkpoints/checkpoint-student{embed_dim}-pretrain_all-100.pth", map_location='cpu') # student
        checkpoint_model = checkpoint['model']
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print("Done!")
    else: 
        start_epoch = 0

    # param_groups = optim_factory.add_weight_decay(model, 0.05) # modified because 'add_weight_decay' does not exist anymore
    param_groups = optim_factory.param_groups_weight_decay(model, 0.05)
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()
    print(f"Start pre-training for {total_epochs} epochs")
    for epoch in range(start_epoch, total_epochs):
        train_stats = train_one_epoch(model, data_loader_train, optimizer, device, epoch, loss_scaler, lr, total_epochs, warmup_epochs)
        if epoch % save_interval_epochs == 0:
            # misc.save_model(output_dir="/baltic/users/shm_mon/SHM_Datasets_2023/checkpoints/", model=model, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, name = f"pretrain_all")
            misc.save_model(output_dir="/home/benfenati/code/shm/checkpoints/", model=model, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, name = f"pretrain_all")
            # misc.save_model(output_dir="/home/benfenati/code/shm/checkpoints/", model=model, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, name = f"student{embed_dim}-pretrain_all") # student

### Creating Finetuning Roccaprebalza
def finetune_roccaprebalza(args, load_pretrain):
    lr = 0.25e-5
    total_epochs = 501
    save_interval_epochs = 100
    device = torch.device('cuda:{}'.format(args.device))
    warmup_epochs = 100
    dataset_train, dataset_test = Vehicles_Roccaprebalza.get_dataset(args.dir, window_sec_size = 60, shift_sec_size = 2, time_frequency = "frequency", car = args.car)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, shuffle=False,
        batch_size=1,
        num_workers=1,
        pin_memory='store_true',
        drop_last=True,
    )

    torch.manual_seed(0)
    np.random.seed(0)
    model = audioMae_vit_base_R(norm_pix_loss=True, mask_ratio = 0.2)
    model.to(device)
    if load_pretrain == True:
        # checkpoint = torch.load(f"/baltic/users/shm_mon/SHM_Datasets_2023/checkpoints/checkpoint-pretrain_all-200.pth", map_location='cpu')
        checkpoint = torch.load(f"/home/benfenati/code/shm/checkpoints/checkpoint-pretrain_all-200.pth", map_location='cpu')
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        msg = model.load_state_dict(checkpoint_model, strict=False)
        interpolate_pos_embed(model, checkpoint_model)
    # build optimizer with layer-wise lr decay (lrd)
    # param_groups = optim_factory.add_weight_decay(model, 0.05)
    param_groups = optim_factory.param_groups_weight_decay(model, 0.05)
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()
    criterion = torch.nn.MSELoss()
    
    y_predicted, y_test = evaluate_finetune(data_loader_test, model, device)
    compute_accuracy(y_test, y_predicted)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    data_loader_finetune = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=8,
        num_workers=1,
        pin_memory='store_true',
        drop_last=True)
    print(f"Start finetuning for {total_epochs} epochs")
    
    for epoch in range(0, total_epochs):
        train_stats = train_one_epoch_finetune(model, criterion, data_loader_finetune, optimizer, device, epoch, loss_scaler, lr, total_epochs, warmup_epochs)
        if epoch % save_interval_epochs == 0:
            # misc.save_model(output_dir="/baltic/users/shm_mon/SHM_Datasets_2023/checkpoints/", model=model, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, name = f"pretrainig_all_{args.car}_roccaprebalza_finetune")
            misc.save_model(output_dir="/home/benfenati/code/shm/checkpoints/", model=model, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, name = f"pretrainig_all_{args.car}_roccaprebalza_finetune")

    y_predicted, y_test = evaluate_finetune(data_loader_test, model, device)
    compute_accuracy(y_test, y_predicted)

### Creating Finetuning Sacertis
def finetune_sacertis(args, load_pretrain):
    lr = 0.25e-5
    total_epochs = 201
    save_interval_epochs = 50
    device = torch.device('cuda:{}'.format(args.device))
    warmup_epochs = 100

    dataset = Vehicles_Sacertis.get_dataset(args.dir, False, True, False,  sensor = "None", time_frequency = "frequency")
    sampler_train = torch.utils.data.RandomSampler(dataset)
    data_loader_finetune = torch.utils.data.DataLoader(
        dataset, sampler=sampler_train,
        batch_size=128,
        num_workers=1,
        pin_memory='store_true',
        drop_last=True)
    
    torch.manual_seed(0)
    np.random.seed(0)
    model = audioMae_vit_base_R(norm_pix_loss=True, mask_ratio = 0.2)
    model.to(device)
    if load_pretrain == True:
        # checkpoint = torch.load(f"/baltic/users/shm_mon/SHM_Datasets_2023/checkpoints/checkpoint-pretrain_all-200.pth", map_location='cpu')
        checkpoint = torch.load(f"/home/benfenati/code/shm/checkpoints/checkpoint-pretrain_all-200.pth", map_location='cpu')
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        msg = model.load_state_dict(checkpoint_model, strict=False)
        interpolate_pos_embed(model, checkpoint_model)
    # build optimizer with layer-wise lr decay (lrd)
    # param_groups = optim_factory.add_weight_decay(model, 0.05)
    param_groups = optim_factory.param_groups_weight_decay(model, 0.05)
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()
    criterion = torch.nn.MSELoss()

    print(f"Start finetuning for {total_epochs} epochs")
    for epoch in range(0, total_epochs):
        train_stats = train_one_epoch_finetune(model, criterion, data_loader_finetune, optimizer, device, epoch, loss_scaler, lr, total_epochs, warmup_epochs)
        if epoch % save_interval_epochs == 0:
            # misc.save_model(output_dir="/baltic/users/shm_mon/SHM_Datasets_2023/checkpoints/", model=model, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, name = f"pretrainig_all_vehicles_sacertis_finetune")
            misc.save_model(output_dir="/home/benfenati/code/shm/checkpoints/", model=model, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, name = f"pretrainig_all_vehicles_sacertis_finetune")

        dataset = Vehicles_Sacertis.get_dataset(args.dir, False, False, True,  sensor = "None", time_frequency = "frequency")
        data_loader_test = torch.utils.data.DataLoader(
            dataset, shuffle=False,
            batch_size=1,
            num_workers=1,
            pin_memory='store_true',
            drop_last=True,
        )
        y_predicted, y_test = evaluate_finetune(data_loader_test, model, device)
        compute_accuracy(y_test, y_predicted)
        
### Creating Finetuning Anomaly
def finetune_anomaly(args, load_pretrain):
    lr = 0.25e-2
    total_epochs = 401
    warmup_epochs = 50
    save_interval_epochs = 100
    device = torch.device('cuda:{}'.format(args.device))

    starting_date = datetime.date(2019,5,22) 
    num_days = 7
    print("Creating Training Dataset")
    dataset = AnomalyDetection_SS335.get_dataset(args.dir, starting_date, num_days, sensor = 'S6.1.3', time_frequency = "frequency", windowLength = 1190)
    sampler_train = torch.utils.data.RandomSampler(dataset)
    data_loader_train = torch.utils.data.DataLoader(
        dataset, sampler=sampler_train,
        batch_size=64,
        num_workers=1,
        pin_memory='store_true',
        drop_last=True,
    )
    
    torch.manual_seed(0)
    np.random.seed(0)
    model = audioMae_vit_base(norm_pix_loss=False)
    model.to(device)
    if load_pretrain == True:
        # checkpoint = torch.load(f"/baltic/users/shm_mon/SHM_Datasets_2023/checkpoints/checkpoint-pretrain_all-200.pth", map_location='cpu')
        checkpoint = torch.load(f"/home/benfenati/code/shm/checkpoints/checkpoint-pretrain_all-200.pth", map_location='cpu')
        checkpoint_model = checkpoint['model']
        msg = model.load_state_dict(checkpoint_model, strict=False)

    # param_groups = optim_factory.add_weight_decay(model, 0.05)
    param_groups = optim_factory.param_groups_weight_decay(model, 0.05)
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()
    print(f"Start training for {total_epochs} epochs")
    for epoch in range(0, total_epochs):
        train_stats = train_one_epoch(model, data_loader_train, optimizer, device, epoch, loss_scaler, lr, total_epochs, warmup_epochs)
        if epoch % save_interval_epochs == 0:
            # misc.save_model(output_dir="/baltic/users/shm_mon/SHM_Datasets_2023/checkpoints/", model=model, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, name = "pretrainig_all_anomaly")
            misc.save_model(output_dir="/home/benfenati/code/shm/checkpoints/", model=model, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, name = "pretrainig_all_anomaly")

### Creating Testing Dataset for Normal Data
    starting_date = datetime.date(2019,5,10) 
    num_days = 4
    print("Creating Testing Dataset -- Normal")
    dataset = AnomalyDetection_SS335.get_dataset(args.dir, starting_date, num_days, sensor = 'S6.1.3', time_frequency = "frequency", windowLength = 1190)
    data_loader_test_normal = torch.utils.data.DataLoader(
        dataset, shuffle=False,
        batch_size=1,
        num_workers=1,
        pin_memory='store_true',
        drop_last=True,
    )
    losses_normal = evaluate(data_loader_test_normal, model, device)
    df = pd.DataFrame.from_dict(losses_normal)
    df.to_csv(f'Results/Pretrain_all_masked_1190samples_normal.csv', index = False, header = True)
        
### Creating Testing Dataset for Anomaly Data
    starting_date = datetime.date(2019,4,17) 
    num_days = 4
    print("Creating Testing Dataset -- Anomaly")
    dataset = AnomalyDetection_SS335.get_dataset(args.dir, starting_date, num_days, sensor = 'S6.1.3', time_frequency = "frequency", windowLength = 1190)
    data_loader_test_anomaly = torch.utils.data.DataLoader(
        dataset, shuffle=False,
        batch_size=1,
        num_workers=1,
        pin_memory='store_true',
        drop_last=True,
    )
    losses_anomaly = evaluate(data_loader_test_anomaly, model, device)
    df = pd.DataFrame.from_dict(losses_anomaly)
    df.to_csv(f'Results/Pretrain_all_masked_1190samples_anomaly.csv', index = False, header = True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Base parameters')
    # parser.add_argument('--dir', type=str, default="/baltic/users/shm_mon/SHM_Datasets_2023/Datasets/Vehicles_Sacertis/", help='directory')
    parser.add_argument('--device', type=str, default=0)
    parser.add_argument('--dir', type=str, default="/home/benfenati/code/Datasets/SHM/Vehicles_Sacertis/", help='directory')
    parser.add_argument('--car', type=str, default="y_camion", help='y_camion, y_car')
    parser.add_argument('--dataset', type=str, default="Sacertis", help='Roccaprebalza, Anomaly, Sacertis')
    parser.add_argument('--pretrain', type=bool, default=False)
    parser.add_argument('--finetune', type=bool, default=True)
    parser.add_argument('--load_pretrain', type=bool, default=True)
    parser.add_argument('--restart_pretraining', type=bool, default=False)
    args = parser.parse_args()
    print(args)
    if args.pretrain == True:
        pretrain(args)
    if args.finetune:
        if args.dataset == "Anomaly":
            finetune_anomaly(args, load_pretrain=args.load_pretrain)
        elif args.dataset == "Roccaprebalza":
            finetune_roccaprebalza(args, load_pretrain=args.load_pretrain)
        elif args.dataset == "Sacertis":
            finetune_sacertis(args, load_pretrain=args.load_pretrain)
        else:
            print("You can not even read the alternatives...")
            exit(0)