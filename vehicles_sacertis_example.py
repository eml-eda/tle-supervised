from Datasets.Vehicles_Sacertis.get_dataset import get_dataset, get_data

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

def algorithm(X_train, y_train, X_test, y_test, number_of_features):
    steps_svr = [( 'scaler', StandardScaler() ), ('svr', SVR(kernel = 'rbf',epsilon=0.1, C=10))]
    pipeline_svr = Pipeline(steps_svr)
    pipeline_svr.fit(X_train,y_train)
    y_predicted = pipeline_svr.predict(X_test)
    print("Prediction")
    compute_accuracy(y_test, y_predicted)


def main(directory):
    data_train, labels_train = get_data(directory, True, False, False, sensor = "None", time_frequency = "time", features = 'Yes')
    data_test, labels_test = get_data(directory, False, False, True, sensor = "None", time_frequency = "time", features = 'Yes')
    algorithm(data_train, labels_train, data_test, labels_test, number_of_features = 12)
    
def main_masked_autoencoder(directory, pretrain = True, finetune = True):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "3"
    device = torch.device('cuda')
    lr = 0.25e-3
    total_epochs = 201
    warmup_epochs = 50

### Creating Training 
    if pretrain == True:
        dataset = get_dataset(directory, True, False, False, time_frequency = "frequency")
        sampler_train = torch.utils.data.RandomSampler(dataset)
        data_loader_train = torch.utils.data.DataLoader(
            dataset, sampler=sampler_train,
            batch_size=64,
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
                misc.save_model(output_dir="Results/checkpoints/", model=model, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, name = "vehicles_sacertis")
    
### Creating Finetuning 
    if finetune == True:
        dataset = get_dataset(directory, False, False, True, time_frequency = "frequency")
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
        checkpoint = torch.load("Results/checkpoints/checkpoint-vehicles_sacertis-200.pth", map_location='cpu')
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

        dataset = get_dataset(directory, False, True, False, time_frequency = "frequency")
        sampler_train = torch.utils.data.RandomSampler(dataset)
        data_loader_finetune = torch.utils.data.DataLoader(
            dataset, sampler=sampler_train,
            batch_size=64,
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
    parser = argparse.ArgumentParser(description='Base parameters')
    parser.add_argument('--dir', type=str, default="/baltic/users/shm_mon/SHM_Datasets_2023/Datasets/Vehicles_Sacertis/",
                        help='directory')
    parser.add_argument('--model', type=str, default="SOA",
                        help='SOA, autoencoder')
    args = parser.parse_args()
    dir = args.dir 
    model = args.model
    if model == "SOA":
        main(dir)
    elif model == "autoencoder":
        main_masked_autoencoder(dir, pretrain = False)