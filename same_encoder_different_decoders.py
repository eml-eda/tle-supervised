import Datasets.Vehicles_Sacertis.get_dataset as Vehicles_Sacertis
import Datasets.AnomalyDetection_SS335.get_dataset as AnomalyDetection_SS335
import Datasets.Vehicles_Roccaprebalza.get_dataset as Vehicles_Roccaprebalza

import datetime
from torch.utils.data import Dataset
import numpy as np

import torch
from Algorithms.models_audio_mae import audioMae_vit_base
from Algorithms.models_audio_mae_regression_modified import audioMae_vit_base_R
import timm
import timm.optim.optim_factory as optim_factory
assert timm.__version__ == "0.3.2"  # version check
from util.misc import interpolate_pos_embed
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.engine_pretrain import train_one_epoch, evaluate, train_one_epoch_finetune, evaluate_finetune
import argparse

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

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
    
def main(args):
    device = torch.device('cuda')
    torch.manual_seed(0)
    np.random.seed(0)
    task = args.task
    if task == "Roccaprebalza":
        model = audioMae_vit_base_R(norm_pix_loss=True, mask_ratio = 0.2)
        total_epochs = 501
        warmup_epochs = 100
    elif task == "Sacertis":
        model = audioMae_vit_base_R(norm_pix_loss=True, mask_ratio = 0.2)
        total_epochs = 201
    elif task == "AnomalyDetection":
        model = audioMae_vit_base(norm_pix_loss=True, mask_ratio = 0.2)
        total_epochs = 401
    else:
        print("Task not provided")
        exit(0)
    model.to(device)
    checkpoint = torch.load(f"Results/checkpoints/checkpoint-y_camion_roccaprebalza-500.pth", map_location='cpu')
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]
    msg = model.load_state_dict(checkpoint_model, strict=False)
    interpolate_pos_embed(model, checkpoint_model)

    save_interval_epochs = 100
    # build optimizer with layer-wise lr decay (lrd)
    param_groups = optim_factory.add_weight_decay(model, 0.05)
    loss_scaler = NativeScaler()
    criterion = torch.nn.MSELoss()
    if args.tail == "Yes":
        for param in model.encoder.parameters(): param.requires_grad = False
    elif args.tail == "No":
        pass
    else:
        exit(0)
    if task == "Roccaprebalza":
        lr = 0.25e-5
        optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
        dataset_train, dataset_test = Vehicles_Roccaprebalza.get_dataset(args.dir, window_sec_size = 60, shift_sec_size = 2, time_frequency = "frequency", car = args.car)
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, shuffle=False,
            batch_size=1,
            num_workers=1,
            pin_memory='store_true',
            drop_last=True,
        )
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        data_loader_finetune = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=8,
            num_workers=1,
            pin_memory='store_true',
            drop_last=True)
    if args.train == "Yes":
        print(f"Start finetuning for {total_epochs} epochs")
        for epoch in range(0, total_epochs):
            train_stats = train_one_epoch_finetune(model, criterion, data_loader_finetune, optimizer, device, epoch, loss_scaler, lr, total_epochs, warmup_epochs)
            # model.encoder.layers[0][0].fn.to_qkv.weight
            # model.fc1.weight
            # model.decoder_blocks.mlp_block.fn.fn.net[0].weight
            if epoch % save_interval_epochs == 0:
                misc.save_model(output_dir="Results/checkpoints/", model=model, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, name = f"freeze_{args.task}")
        y_predicted, y_test = evaluate_finetune(data_loader_test, model, device)
        compute_accuracy(y_test, y_predicted)
    else:
        model = audioMae_vit_base_R(norm_pix_loss=True, mask_ratio = 0.2)
        model.to(device)
        checkpoint = torch.load(f"Results/checkpoints/checkpoint-freeze_-500.pth", map_location='cpu')
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        msg = model.load_state_dict(checkpoint_model, strict=True)
        y_predicted, y_test = evaluate_finetune(data_loader_test, model, device)
        compute_accuracy(y_test, y_predicted)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Base parameters')
    parser.add_argument('--dir', type=str, default="/baltic/users/shm_mon/SHM_Datasets_2023/Datasets/Vehicles_Roccaprebalza/",
                        help='directory')
    parser.add_argument('--car', type=str, default="y_camion",
                        help='y_camion, y_car')
    parser.add_argument('--task', type=str, default="Roccaprebalza",
                        help='Roccaprebalza, Sacertis, AnomalyDetection')
    parser.add_argument('--train', type=str, default="Yes",
                        help='Yes, No')
    parser.add_argument('--tail', type=str, default="No",
                        help='Yes, No')
    args = parser.parse_args()
    print(args)
    main(args)