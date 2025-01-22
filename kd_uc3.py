import os
import torch
import numpy as np
import argparse
import csv

from models.models_audio_mae_regression import audioMae_vit_base_R

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from time import time
from utils import *
import util.misc as misc
from util.misc import interpolate_pos_embed

import datetime
from data.Vehicles_Sacertis.get_dataset import get_dataset as get_dataset_sacertis

from util.engine_pretrain import evaluate_finetune

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Base parameters')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dir', type=str, default="/home/benfenati/data_folder/SHM/Vehicles_Sacertis/")
    args = parser.parse_args()
    print(args)

    device = torch.device("cuda:{}".format(args.device))

    # create results file
    filename = '/home/benfenati/code/tle-supervised/Results/uc3_results_kd_finetuning_pt2.csv' # tag:change name
    header = ["embed_dim", "decoder_dim", "mse", "mae", "r2", "mspe", "mape"]
    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)

    lr = 0.25e-5
    total_epochs = 201
    warmup_epochs = 50
    save_interval_epochs = 100
    directory = args.dir

    # combinations_kd = [(384, 512), (192, 512), (96, 512), (48, 512), (24, 512)] #tag:change name
    combinations_kd = [(384, 256), (192, 128), (96, 64), (48, 32), (24, 16)] 

    # teacher model
    teacher = audioMae_vit_base_R(norm_pix_loss=True, mask_ratio = 0.2)
    teacher.to(device)
    checkpoint = torch.load(f"/home/benfenati/code/tle-supervised/Results/checkpoints/uc3/checkpoint-768-512_vehicles_sacertis_finetune-200.pth", map_location='cpu')
    checkpoint_model = checkpoint['model']
    state_dict = teacher.state_dict()
    msg = teacher.load_state_dict(checkpoint_model, strict=True)
    params, size = get_model_info(teacher)
    print("N. params = {}; Size = {:.3f}".format(params, size))


    for comb in combinations_kd:
        embed_dim = comb[0]
        decoder_embed_dim = comb[1]

        # student model
        student = audioMae_vit_base_R(embed_dim=embed_dim, decoder_embed_dim=decoder_embed_dim, 
                                    norm_pix_loss=True, mask_ratio = 0.2)
        student.to(device)

        # checkpoint = torch.load(f"/home/benfenati/code/tle-supervised/Results/checkpoints/uc23/checkpoint-{embed_dim}-{decoder_embed_dim}-pretrain_all-200.pth", map_location='cpu') #tag:change name
        checkpoint = torch.load(f"/home/benfenati/code/tle-supervised/Results/checkpoints/uc1/checkpoint-{embed_dim}-{decoder_embed_dim}-pretrain_all-200.pth", map_location='cpu')
        checkpoint_model = checkpoint['model']
        state_dict = student.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        msg = student.load_state_dict(checkpoint_model, strict=False)
        interpolate_pos_embed(student, checkpoint_model)

        params, size = get_model_info(student)
        print("N. params = {}; Size = {:.3f}".format(params, size))

        # training
        dataset_train = get_dataset_sacertis(args.dir, False, True, False,  sensor = "None", time_frequency = "frequency")
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=128,
            num_workers=1,
            pin_memory='store_true',
            drop_last=True)

        torch.manual_seed(0)
        np.random.seed(0)

        optimizer = optim.Adam(student.parameters(), lr=lr, weight_decay=1e-6)
        loss_fn_1 = nn.L1Loss()
        loss_fn_2 = nn.L1Loss()
        loss_fn_3 = nn.MSELoss()

        teacher.eval()

        b = 0.5
        g = 0.6667

        best_loss = 100000000
        best_epoch = 0

        for epoch in range(total_epochs):
            student.train()
            train_loss = 0
            counter = 0
            for samples, targets in data_loader_train:
                samples = samples.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
            
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    middle_student, final_student = student(samples)

                with torch.no_grad() and torch.cuda.amp.autocast():
                    teacher.eval()
                    middle_teacher, final_teacher = teacher(samples)
                
                final_student = final_student.squeeze()
                final_teacher = final_teacher.squeeze()
                loss_1 = loss_fn_1(final_student, targets.float())
                loss_2 = loss_fn_2(final_student, final_teacher)

                # loss_3 = loss_fn_3(middle_student, middle_teacher) #tag:change kd
                
                # loss = g*(b*loss_1 + (1-b)*loss_2) + (1-g)*loss_3 #tag:change kd
                loss = b*loss_1 + (1-b)*loss_2 #tag:change kd

                loss.backward()
                optimizer.step()

                train_loss += loss_fn_1(final_student, targets).item()
                # train_loss += loss_1.item()
                counter +=1
            
            print(f"Epoch: {epoch}, Loss: {train_loss}")

        torch.save(student,  f"/home/benfenati/code/tle-supervised/Results/checkpoints/uc3_checkpoint-{embed_dim}-{decoder_embed_dim}-vehicles_sacertis_finetune_KD.pth")
        
        # testing
        dataset = get_dataset_sacertis(args.dir, False, False, True,  sensor = "None", time_frequency = "frequency")
        data_loader_test = torch.utils.data.DataLoader(
            dataset, shuffle=False,
            batch_size=1,
            num_workers=1,
            pin_memory='store_true',
            drop_last=True,
        )        

        y_predicted, y_test = evaluate_finetune(data_loader_test, student, device)
        mse, mae, r2, mspe, mape = compute_accuracy(y_test, y_predicted)

        last_row = [embed_dim, decoder_embed_dim, mse, mae, r2, mspe, mape]
        with open(filename, 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(last_row)