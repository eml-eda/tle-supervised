import Datasets.Vehicles_Sacertis.get_dataset as Vehicles_Sacertis
import Datasets.AnomalyDetection_SS335.get_dataset as AnomalyDetection_SS335
import Datasets.Vehicles_Roccaprebalza.get_dataset as Vehicles_Roccaprebalza

import datetime
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

import torch
from Algorithms.models_audio_mae import audioMae_vit_base
from Algorithms.models_audio_mae_regression import audioMae_vit_base_R
import timm
import timm.optim.optim_factory as optim_factory
assert timm.__version__ == "0.3.2"  # version check
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
    os.environ['CUDA_VISIBLE_DEVICES'] = "2"
    print("Creating Training Dataset")
    starting_date = datetime.date(2019,5,22) 
    num_days = 7
    window_size = 1190
    directory = "/baltic/users/shm_mon/SHM_Datasets_2023/Datasets/AnomalyDetection_SS335/"
    data_anomaly = AnomalyDetection_SS335.get_data(directory, starting_date, num_days, sensor = 'S6.1.3', time_frequency = "frequency", windowLength = window_size)
    directory = "/baltic/users/shm_mon/SHM_Datasets_2023/Datasets/Vehicles_Roccaprebalza/"
    data_train, _, _, _ = Vehicles_Roccaprebalza.get_data(directory, window_sec_size = 60, shift_sec_size = 2, time_frequency = "frequency", car = "y_camion")
    directory = "/baltic/users/shm_mon/SHM_Datasets_2023/Datasets/Vehicles_Sacertis/"
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
    
def main_masked_autoencoder_roccaprebalza(directory, pretrain = True, finetune = True, load_pretrain = True):
    import os 
    os.environ["CUDA_VISIBLE_DEVICES"]="2"
    device = torch.device('cuda')
    lr = 0.25e-3
    total_epochs = 201
    warmup_epochs = 100
    save_interval_epochs = 100

### Creating Training 
    if pretrain == True:
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
        model = audioMae_vit_base(norm_pix_loss=True)
        model.to(device)
        param_groups = optim_factory.add_weight_decay(model, 0.05)
        optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
        loss_scaler = NativeScaler()
        print(f"Start pre-training for {total_epochs} epochs")
        for epoch in range(0, total_epochs):
            train_stats = train_one_epoch(model, data_loader_train, optimizer, device, epoch, loss_scaler, lr, total_epochs, warmup_epochs)
            if epoch % save_interval_epochs == 0:
                misc.save_model(output_dir="Results/checkpoints/", model=model, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, name = f"pretrain_all")
    
        
### Creating Finetuning 
    if finetune == True:
        lr = 0.25e-5
        total_epochs = 501
        dataset_train, dataset_test = get_dataset(args.dir, window_sec_size = 60, shift_sec_size = 2, time_frequency = "frequency", car = args.car)
        sampler_test = torch.utils.data.RandomSampler(dataset_test)
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, shuffle=False,
            batch_size=1,
            num_workers=1,
            pin_memory='store_true',
            drop_last=True,
        )

        torch.manual_seed(0)
        np.random.seed(0)
        model = audioMae_vit_base_R(norm_pix_loss=True, mask_ratio = 0) ## DA CAMBIARE
        model.to(device)
        if load_pretrain == True:
            checkpoint = torch.load(f"Results/checkpoints/checkpoint-pretrain_all-200.pth", map_location='cpu')
            checkpoint_model = checkpoint['model']
            state_dict = model.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
            msg = model.load_state_dict(checkpoint_model, strict=False)
            interpolate_pos_embed(model, checkpoint_model)
        # build optimizer with layer-wise lr decay (lrd)
        param_groups = optim_factory.add_weight_decay(model, 0.05)
        optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
        loss_scaler = NativeScaler()
        criterion = torch.nn.MSELoss()
        
        y_predicted, y_test = evaluate_finetune(data_loader_test, model, device)
        compute_accuracy(y_test, y_predicted)

        dataset_train, dataset_test = get_dataset(args.dir, window_sec_size = 60, shift_sec_size = 2, time_frequency = "frequency", car = args.car)
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
                misc.save_model(output_dir="Results/checkpoints/", model=model, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, name = f"pretrain_all_finetune")
        y_predicted, y_test = evaluate_finetune(data_loader_test, model, device)
        compute_accuracy(y_test, y_predicted)

def main_masked_autoencoder_Sacertis(directory, pretrain = True, finetune = True, load_pretrain = True):
    TODO
    device = torch.device('cuda')
    lr = 0.25e-3
    total_epochs = 201
    warmup_epochs = 50
    save_interval_epochs = 100

### Creating Training 
    if pretrain == True:
        dataset_train = get_all_datasets()
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=16,
            num_workers=1,
            pin_memory='store_true',
            drop_last=True)
        torch.manual_seed(0)
        np.random.seed(0)
        model = audioMae_vit_base(norm_pix_loss=True)
        model.to(device)
        param_groups = optim_factory.add_weight_decay(model, 0.05)
        optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
        loss_scaler = NativeScaler()
        print(f"Start pre-training for {total_epochs} epochs")
        for epoch in range(0, total_epochs):
            train_stats = train_one_epoch(model, data_loader_train, optimizer, device, epoch, loss_scaler, lr, total_epochs, warmup_epochs)
            if epoch % save_interval_epochs == 0:
                misc.save_model(output_dir="Results/checkpoints/", model=model, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, name = f"pretrain_all")
    
        
### Creating Finetuning 
    if finetune == True:
        dataset_train, dataset_test = get_dataset(args.dir, window_sec_size = 60, shift_sec_size = 2, time_frequency = "frequency", car = args.car)
        sampler_test = torch.utils.data.RandomSampler(dataset_test)
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, shuffle=False,
            batch_size=1,
            num_workers=1,
            pin_memory='store_true',
            drop_last=True,
        )

        torch.manual_seed(0)
        np.random.seed(0)
        model = audioMae_vit_base_R(norm_pix_loss=True)
        model.to(device)
        if load_pretrain == True:
            checkpoint = torch.load(f"Results/checkpoints/checkpoint-pretrain_all-200.pth", map_location='cpu')
            checkpoint_model = checkpoint['model']
            state_dict = model.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
            msg = model.load_state_dict(checkpoint_model, strict=False)
            interpolate_pos_embed(model, checkpoint_model)
        # build optimizer with layer-wise lr decay (lrd)
        param_groups = optim_factory.add_weight_decay(model, 0.05)
        optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
        loss_scaler = NativeScaler()
        criterion = torch.nn.MSELoss()
        
        y_predicted, y_test = evaluate_finetune(data_loader_test, model, device)
        compute_accuracy(y_test, y_predicted)

        dataset_train, dataset_test = get_dataset(args.dir, window_sec_size = 60, shift_sec_size = 2, time_frequency = "frequency", car = args.car)
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        data_loader_finetune = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=16,
            num_workers=1,
            pin_memory='store_true',
            drop_last=True)
        print(f"Start finetuning for {total_epochs} epochs")
        for epoch in range(0, total_epochs):
            train_stats = train_one_epoch_finetune(model, criterion, data_loader_finetune, optimizer, device, epoch, loss_scaler, lr, total_epochs, warmup_epochs)
            if epoch % save_interval_epochs == 0:
                misc.save_model(output_dir="Results/checkpoints/", model=model, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, name = f"pretrain_all_finetune")
        y_predicted, y_test = evaluate_finetune(data_loader_test, model, device)
        compute_accuracy(y_test, y_predicted)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Base parameters')
    parser.add_argument('--dir', type=str, default="/baltic/users/shm_mon/SHM_Datasets_2023/Datasets/Vehicles_Roccaprebalza/",
                        help='directory')
    parser.add_argument('--car', type=str, default="y_camion",
                        help='y_camion, y_car')
    args = parser.parse_args()
    main_masked_autoencoder_roccaprebalza(args, pretrain = True, finetune = True, load_pretrain = True)