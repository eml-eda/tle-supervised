import torch
import pandas as pd
import numpy as np
import argparse
import util.misc as misc

from Algorithms.models_audio_mae import audioMae_vit_base
from Algorithms.models_audio_mae_evaluate import audioMae_vit_base_evaluate

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import *
from util.engine_pretrain import evaluate
from utils import compute_threshold_accuracy

import datetime

from Datasets.AnomalyDetection_SS335.get_dataset import get_dataset as get_dataset_ss335

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Base parameters')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dir', type=str, default="/home/benfenati/code/Datasets/SHM/AnomalyDetection_SS335/")
    parser.add_argument('--window_size', type=int, default=1190)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    print(args)

    device = torch.device("cuda:{}".format(args.device))

    # teacher model
    teacher = audioMae_vit_base_evaluate(norm_pix_loss=False)
    teacher.to(device)
    checkpoint = torch.load(f"/home/benfenati/code/tle-supervised/Results/checkpoints/uc1/checkpoint--400.pth", map_location='cpu')
    checkpoint_model = checkpoint['model']
    msg = teacher.load_state_dict(checkpoint_model, strict=False)
    params, size = get_model_info(teacher)
    print("N. params = {}; Size = {:.3f}".format(params, size))


    combinations_uc1 = [(384, 256), (192, 128), (96, 64), (48, 32), (24, 16)]
    for comb in combinations_uc1:
        ### Creating Training 
        print("Testing combination: ", comb)
        embed_dim = comb[0]
        decoder_embed_dim = comb[1]
        
        # student model
        student = audioMae_vit_base_evaluate(embed_dim=embed_dim, decoder_embed_dim=decoder_embed_dim)
        student.to(device)
        params, size = get_model_info(student)
        print("N. params = {}; Size = {:.3f}".format(params, size))

        # training
        starting_date = datetime.date(2019,5,22) 
        num_days = 7
        print("Creating Training Dataset")
        dataset = get_dataset_ss335(args.dir, starting_date, num_days, sensor = 'S6.1.3', time_frequency = "frequency", windowLength = args.window_size)
        sampler_train = torch.utils.data.RandomSampler(dataset)
        data_loader_train = torch.utils.data.DataLoader(
            dataset, sampler=sampler_train,
            batch_size=64,
            num_workers=1,
            pin_memory='store_true',
            drop_last=True,
        )
        device = torch.device(device)
        torch.manual_seed(0)
        np.random.seed(0)

        optimizer = optim.Adam(student.parameters(), lr=0.001, weight_decay=1e-6)
        loss_fn_1 = nn.L1Loss()
        loss_fn_2 = nn.L1Loss()
        loss_fn_3 = nn.MSELoss()

        teacher.eval()

        b = 0.5

        best_loss = 100000000
        best_epoch = 0

        for epoch in range(args.epochs):
            student.train()
            train_loss = 0
            counter = 0
            for samples, targets in data_loader_train:
                samples = samples.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
            
                
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    loss_student, pred_student, _ = student(samples, mask_ratio=0.8)

                with torch.no_grad() and torch.cuda.amp.autocast():
                    teacher.eval()
                    loss_teacher, pred_teacher, _ = teacher(samples, mask_ratio=0.8)
                
                
                loss_1 = loss_student
                loss_2 = loss_fn_3(pred_student, pred_teacher)
                
                loss = b*loss_1 + (1-b)*loss_2

                loss.backward()
                optimizer.step()

                train_loss += loss_student.item()
                # train_loss += loss.item()
                counter +=1
            
            print(f"Epoch {epoch} -- Loss: {train_loss/counter}")

        torch.save(student.state_dict(), f"Results/checkpoints/uc1_checkpoint-{embed_dim}_{decoder_embed_dim}-KD.pth")
        
        # testing
        # 1) Produce predictions
        ### Creating Testing Dataset for Normal Data
        starting_date = datetime.date(2019,5,10)
        num_days = 4
        print("Creating Testing Dataset -- Normal")
        dataset = get_dataset_ss335(args.dir, starting_date, num_days, sensor = 'S6.1.3', time_frequency = "frequency", windowLength = args.window_size)
        data_loader_test_normal = torch.utils.data.DataLoader(
            dataset, shuffle=False,
            batch_size=1,
            num_workers=1,
            pin_memory='store_true',
            drop_last=True,
        )
        losses_normal, _ = evaluate(data_loader_test_normal, student, device)
        df = pd.DataFrame.from_dict(losses_normal)
        df.to_csv(f'Results/masked_{args.window_size}samples_normal_{embed_dim}-{decoder_embed_dim}-KD.csv', index = False, header = True)
            
        ### Creating Testing Dataset for Anomaly Data
        starting_date = datetime.date(2019,4,17) 
        num_days = 4
        print("Creating Testing Dataset -- Anomaly")
        dataset = get_dataset_ss335(args.dir, starting_date, num_days, sensor = 'S6.1.3', time_frequency = "frequency", windowLength = args.window_size)
        data_loader_test_anomaly = torch.utils.data.DataLoader(
            dataset, shuffle=False,
            batch_size=1,
            num_workers=1,
            pin_memory='store_true',
            drop_last=True,
        )
        losses_anomaly, _ = evaluate(data_loader_test_anomaly, student, device)
        df = pd.DataFrame.from_dict(losses_anomaly)
        df.to_csv(f'Results/masked_{args.window_size}samples_anomaly_{embed_dim}-{decoder_embed_dim}-KD.csv', index = False, header = True)