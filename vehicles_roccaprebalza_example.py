from Datasets.Vehicles_Roccaprebalza.get_dataset import get_data, get_dataset

import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline

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

def algorithm(data, labels, number_of_features):
    X_car = SelectKBest(f_regression, k = number_of_features).fit_transform(data, labels["y_car"])
    X_camion = SelectKBest(f_regression, k = number_of_features).fit_transform(data, labels["y_camion"])
    X_train_car, X_test_car, y_train_car, y_test_car = train_test_split(X_car, labels["y_car"])
    X_train_camion, X_test_camion, y_train_camion, y_test_camion = train_test_split(X_camion, labels["y_camion"])
    steps_svr = [( 'scaler', StandardScaler() ), ('svr', SVR(kernel = 'rbf',epsilon=0.1, C=10))]

    pipeline_svr_car = Pipeline(steps_svr)
    pipeline_svr_car.fit(X_train_car,y_train_car)
    y_predicted_car = pipeline_svr_car.predict(X_test_car)
    ### METRICHE MISURA ACCURATEZZA PER CAR
    print("CARS Prediction")
    compute_accuracy(y_test_car, y_predicted_car)
    
    pipeline_svr_camion = Pipeline(steps_svr)
    pipeline_svr_camion.fit(X_train_camion,y_train_camion)
    y_predicted_camion = pipeline_svr_camion.predict(X_test_camion)
    ### METRICHE MISURA ACCURATEZZA PER CAMION
    print("CAMIONS Prediction")
    compute_accuracy(y_test_camion, y_predicted_camion)

def main_classical(directory):
    data, labels = get_data(directory, window_sec_size = 60, shift_sec_size = 2)
    algorithm(data, labels, number_of_features = 50)

def main_autoencoder(directory, pretrain = True, finetune = True, car = 'y_camion'):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "3"
    device = torch.device('cuda')
    lr = 0.25e-3
    total_epochs = 61
    warmup_epochs = 30

### Creating Training 
    if pretrain == True:
        dataset = get_dataset(directory, window_sec_size = 60, shift_sec_size = 2, time_frequency = "frequency", car = car)
        sampler_train = torch.utils.data.RandomSampler(dataset)
        data_loader_train = torch.utils.data.DataLoader(
            dataset, sampler=sampler_train,
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
        print(f"Start training for {total_epochs} epochs")
        for epoch in range(0, total_epochs):
            train_stats = train_one_epoch(model, data_loader_train, optimizer, device, epoch, loss_scaler, lr, total_epochs, warmup_epochs)
            if epoch % 10 == 0:
                misc.save_model(output_dir="Results/checkpoints/", model=model, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, name = "vehicles_roccaprebalza")
    
### Creating Finetuning 
    if finetune == True:
        dataset = get_dataset(directory, window_sec_size = 60, shift_sec_size = 2, time_frequency = "frequency", car = car)
        sampler_test = torch.utils.data.RandomSampler(dataset)
        data_loader_test = torch.utils.data.DataLoader(
            dataset, shuffle=False,
            batch_size=1,
            num_workers=1,
            pin_memory='store_true',
            drop_last=True,
        )

        torch.manual_seed(0)
        np.random.seed(0)
        model = audioMae_vit_base_R(norm_pix_loss=True)
        model.to(device)
        checkpoint = torch.load("Results/checkpoints/checkpoint-vehicles_roccaprebalza-60.pth", map_location='cpu')
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

        dataset = get_dataset(directory, window_sec_size = 60, shift_sec_size = 2, time_frequency = "frequency", car = car)
        sampler_train = torch.utils.data.RandomSampler(dataset)
        data_loader_finetune = torch.utils.data.DataLoader(
            dataset, sampler=sampler_train,
            batch_size=16,
            num_workers=1,
            pin_memory='store_true',
            drop_last=True)
        print(f"Start finetuning for {total_epochs} epochs")
        for epoch in range(0, total_epochs):
            train_stats = train_one_epoch_finetune(model, criterion, data_loader_finetune, optimizer, device, epoch, loss_scaler, lr, total_epochs, warmup_epochs)
            if epoch % 10 == 0:
                misc.save_model(output_dir="Results/checkpoints/", model=model, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, name = "vehicles_sacertis_finetune")

        y_predicted, y_test = evaluate_finetune(data_loader_test, model, device)
        compute_accuracy(y_test, y_predicted)

if __name__ == "__main__":
    dir = "/baltic/users/shm_mon/SHM_Datasets_2023/Datasets/Vehicles_Roccaprebalza/"
    main_classical(dir)
    main_autoencoder(dir, pretrain = True, finetune = True, car = 'y_camion')