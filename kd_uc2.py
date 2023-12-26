import os
import torch
import numpy as np
import argparse

# from Algorithms.models_audio_mae_regression import audioMae_vit_base_R
from Algorithms.models_audio_mae_regression_modified import audioMae_vit_base_R

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from time import time
from utils import *
import util.misc as misc
from util.misc import interpolate_pos_embed

import datetime
from Datasets.Vehicles_Roccaprebalza.get_dataset import get_dataset as get_dataset_roccaprebalza

from util.engine_pretrain import evaluate_finetune
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Base parameters')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--car', type=str, default="y_car", help='y_camion, y_car')
    parser.add_argument('--dir', type=str, default="/home/benfenati/code/Datasets/Vehicles_Roccaprebalza/")
    parser.add_argument('--lr', type=float, default=0.25e-5)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    print(args)

    device = torch.device("cuda:{}".format(args.device))

    # teacher model
    teacher = audioMae_vit_base_R(norm_pix_loss=True, mask_ratio = 0.2)
    teacher.to(device)
    checkpoint = torch.load(f"/home/benfenati/code/shm/checkpoints/checkpoint-pretrainig_all_{args.car}_roccaprebalza_finetune-500.pth", map_location='cpu')
    checkpoint_model = checkpoint['model']
    state_dict = teacher.state_dict()
    msg = teacher.load_state_dict(checkpoint_model, strict=True)
    params, size = get_model_info(teacher)
    print("N. params = {}; Size = {:.3f}".format(params, size))

    # student model
    embed_dim = 384 # 384, 768 (original)
    decoder_embed_dim = 512 # 256, 512 (original)
    student = audioMae_vit_base_R(embed_dim=embed_dim, decoder_embed_dim=decoder_embed_dim, 
                                norm_pix_loss=True, mask_ratio = 0.2)
    student.to(device)

    checkpoint = torch.load(f"/home/benfenati/code/shm/checkpoints/checkpoint-student-pretrain_all-200.pth", map_location='cpu')
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
    dataset_train, dataset_test = get_dataset_roccaprebalza(args.dir, window_sec_size = 60, shift_sec_size = 2, time_frequency = "frequency", car = args.car)
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=8,
        num_workers=1,
        pin_memory='store_true',
        drop_last=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, shuffle=False,
        batch_size=1,
        num_workers=1,
        pin_memory='store_true',
        drop_last=True,
    )

    torch.manual_seed(0)
    np.random.seed(0)

    optimizer = optim.Adam(student.parameters(), lr=args.lr, weight_decay=1e-6)
    loss_fn_1 = nn.L1Loss()
    loss_fn_2 = nn.L1Loss()
    loss_fn_3 = nn.MSELoss()

    teacher.eval()

    b = 0.5
    g = 0.6667

    best_loss = 100000000
    best_epoch = 0

    for epoch in range(args.epochs):
        if (epoch+1) % 10 == 0: 
            print("Epoch ", epoch+1)

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


            loss_3 = loss_fn_3(middle_student, middle_teacher)
            
            loss = g*(b*loss_1 + (1-b)*loss_2) + (1-g)*loss_3
            # loss = b*loss_1 + (1-b)*loss_2

            loss.backward()
            optimizer.step()

            train_loss += loss_fn_1(final_student, targets).item()
            # train_loss += loss_1.item()
            counter +=1

    # testing
    y_predicted, y_test = evaluate_finetune(data_loader_test, teacher, device)
    compute_accuracy(y_test, y_predicted)