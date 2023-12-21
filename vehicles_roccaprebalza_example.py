from Datasets.Vehicles_Roccaprebalza.get_dataset import get_data, get_dataset

import numpy as np
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
from Algorithms.models_audio_mae import audioMae_vit_base
from Algorithms.models_audio_mae_regression_modified import audioMae_vit_base_R
import timm
import timm.optim.optim_factory as optim_factory
# assert timm.__version__ == "0.3.2"  # version check
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

def algorithm(data_train, labels_train, data_test, labels_test, car, number_of_features):
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
        print(f"{car} Prediction")
        compute_accuracy(labels_test, y_predicted_car)
        df = pd.DataFrame({"Y_true": labels_test, "Y_predicted": y_predicted_car})
        df.to_csv(f'Results/Roccaprebalza_{names[i]}_{car}.csv', index = False, header = True)
        
        # pipeline_svr_camion = Pipeline(steps_svr)
        # pipeline_svr_camion.fit(X_train_camion,y_train_camion)
        # y_predicted_camion = pipeline_svr_camion.predict(X_test_camion)
        # ### METRICHE MISURA ACCURATEZZA PER CAMION
        # print("CAMIONS Prediction")
        # compute_accuracy(y_test_camion, y_predicted_camion)

def main_classical(args):
    data_train, labels_train, data_test, labels_test = get_data(args.dir, window_sec_size = 60, shift_sec_size = 2, time_frequency = "time", car = args.car)
    algorithm(data_train, labels_train, data_test, labels_test, args.car, number_of_features = 12)

def main_autoencoder(args, pretrain = True, finetune = True, load_pretrain = True):
    import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = "3"
    device = torch.device('cuda')
    lr = 0.25e-3
    total_epochs = 501
    warmup_epochs = 250
    save_interval_epochs = 100

### Creating Training 
    if pretrain == True:
        dataset_train, dataset_test = get_dataset(args.dir, window_sec_size = 60, shift_sec_size = 2, time_frequency = "frequency", car = args.car)
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=8,
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
            if epoch % save_interval_epochs == 0:
                misc.save_model(output_dir="/baltic/users/shm_mon/SHM_Datasets_2023/checkpoints/", model=model, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, name = f"{args.car}_roccaprebalza")
    
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
        model = audioMae_vit_base_R(norm_pix_loss=True, mask_ratio = 0.2)
        model.to(device)
        if load_pretrain == True:
            checkpoint = torch.load(f"/baltic/users/shm_mon/SHM_Datasets_2023/checkpoints/checkpoint-{args.car}_roccaprebalza-500.pth", map_location='cpu')
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
                misc.save_model(output_dir="/baltic/users/shm_mon/SHM_Datasets_2023/checkpoints/", model=model, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, name = f"{args.car}_roccaprebalza_finetune")

        y_predicted, y_test = evaluate_finetune(data_loader_test, model, device)
        compute_accuracy(y_test, y_predicted)

def evaluate_autoencoder(args, pretrain = True, finetune = True, load_pretrain = True):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    device = torch.device('cuda')
    dataset_train, dataset_test = get_dataset(args.dir, window_sec_size = 60, shift_sec_size = 2, time_frequency = "frequency", car = args.car)
    sampler_test = torch.utils.data.RandomSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, shuffle=False,
        batch_size=1,
        num_workers=1,
        pin_memory='store_true',
        drop_last=True,
    )
    # import pdb;pdb.set_trace()

    model = audioMae_vit_base_R(norm_pix_loss=True, mask_ratio = 0.2)
    model.to(device)
    # checkpoint = torch.load(f"/baltic/users/shm_mon/SHM_Datasets_2023/checkpoints/checkpoint-{args.car}_roccaprebalza_finetune-500.pth", map_location='cpu')
    checkpoint = torch.load(f"/home/benfenati/code/shm/checkpoints/checkpoint-pretrainig_all_{args.car}_roccaprebalza_finetune-500.pth", map_location='cpu')
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    # for k in ['head.weight', 'head.bias']:
    #     if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
    #         print(f"Removing key {k} from pretrained checkpoint")
    #         del checkpoint_model[k]
    msg = model.load_state_dict(checkpoint_model, strict=True)
    # interpolate_pos_embed(model, checkpoint_model)
    y_predicted, y_test = evaluate_finetune(data_loader_test, model, device)
    compute_accuracy(y_test, y_predicted)

    df = pd.DataFrame({"Y_true": y_test, "Y_predicted": y_predicted})
    # df.to_csv(f'Results/Roccaprebalza_autoencoder_{args.car}.csv', index = False, header = True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Base parameters')
    parser.add_argument('--dir', type=str, default="/baltic/users/shm_mon/SHM_Datasets_2023/Datasets/Vehicles_Roccaprebalza/",
                        help='directory')
    parser.add_argument('--model', type=str, default="SOA",
                        help='SOA, autoencoder')
    parser.add_argument('--car', type=str, default="y_car",
                        help='y_camion, y_car')
    parser.add_argument('--pretrain', type=bool, default=True, help = 'pass '' for false')
    parser.add_argument('--finetune', type=bool, default=True, help = 'pass '' for false')
    parser.add_argument('--load_pretrain', type=bool, default=True, help = 'pass '' for false')
    parser.add_argument('--train', type=str, default="Yes", help = 'Yes or No')
    args = parser.parse_args()
    model = args.model 
    print(args)
    # if args.train == 'Yes':
    #     if model == "SOA":
    #         main_classical(args)
    #     elif model == "autoencoder":
    #         main_autoencoder(args, pretrain = args.pretrain, finetune = args.finetune, load_pretrain = args.load_pretrain)
    # else:
    #     if model == "SOA":
    #         main_classical(args)
    #     elif model == "autoencoder":
    #         evaluate_autoencoder(args, pretrain = args.pretrain, finetune = args.finetune, load_pretrain = args.load_pretrain)
    evaluate_autoencoder(args, pretrain = args.pretrain, finetune = args.finetune, load_pretrain = args.load_pretrain)