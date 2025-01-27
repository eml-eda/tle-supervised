import torch
import numpy as np
from util.engine_pretrain import train_one_epoch
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from models.models_audio_mae import audioMae_vit_base
from models.models_tcn_mae import tcn_mae
from models.models_lstm_mae import lstm_mae
import timm.optim.optim_factory as optim_factory
import util.misc as misc
import argparse
from utils import get_all_datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Base parameters')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--model', type=str, default='mae', help='mae, tcn')
    parser.add_argument('--encoder_dim', type=int, default=768)
    parser.add_argument('--decoder_dim', type=int, default=512)
    parser.add_argument('--window_size', type=int, default=1190)
    parser.add_argument('--dir1', type=str, default='/space/benfenati/data_folder/SHM/AnomalyDetection_SS335/')
    parser.add_argument('--dir2', type=str, default='/space/benfenati/data_folder/SHM/Vehicles_Roccaprebalza/')
    parser.add_argument('--dir3', type=str, default='/space/benfenati/data_folder/SHM/Vehicles_Sacertis/')
    args = parser.parse_args()
    print(args)

    device = torch.device('cuda:{}'.format(args.device))
    restart_pretraining = False
    lr = 0.25e-3
    total_epochs = 201
    warmup_epochs = 100
    save_interval_epochs = 50

    # train and val dataset
    dataset_train = get_all_datasets(args.dir1, args.dir2, args.dir3, args.window_size)
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=128,
        num_workers=1,
        pin_memory='store_true',
        drop_last=True)

    torch.manual_seed(42)
    np.random.seed(42)
    
    embed_dim = args.encoder_dim
    decoder_embed_dim = args.decoder_dim

    print("Testing combination: {}-{}".format(embed_dim, decoder_embed_dim))

    if args.model == "mae":
        model = audioMae_vit_base(embed_dim=embed_dim, decoder_embed_dim=decoder_embed_dim, norm_pix_loss=True)
        model.to(device)
    elif args.model == "tcn": 
        model = tcn_mae(embed_dim=embed_dim, decoder_embed_dim=decoder_embed_dim)
        model.to(device)
    elif args.model == "lstm":
        model = lstm_mae(embed_dim=embed_dim, decoder_embed_dim=decoder_embed_dim)
        model.to(device)

    param_groups = optim_factory.param_groups_weight_decay(model, 0.05)
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()
    print(f"Start pre-training for {total_epochs} epochs")
    for epoch in range(0, total_epochs):
        train_stats = train_one_epoch(model, data_loader_train, optimizer, device, epoch, loss_scaler, lr, total_epochs, warmup_epochs)
        if epoch % save_interval_epochs == 0:
            if args.model == "mae":
                misc.save_model(output_dir="/home/benfenati/code/tle-supervised/results/checkpoints/", model=model, model_without_ddp=model, optimizer=optimizer, 
                                loss_scaler=loss_scaler, epoch=epoch, 
                                name = "{}-{}-pretrain_all".format(embed_dim, decoder_embed_dim))
            elif args.model == "tcn":
                misc.save_model(output_dir="/home/benfenati/code/tle-supervised/results/checkpoints/", model=model, model_without_ddp=model, optimizer=optimizer, 
                loss_scaler=loss_scaler, epoch=epoch, 
                name = "tcn-pretrain_all")
            elif args.model == "lstm":
                misc.save_model(output_dir="/home/benfenati/code/tle-supervised/results/checkpoints/", model=model, model_without_ddp=model, optimizer=optimizer, 
                loss_scaler=loss_scaler, epoch=epoch, 
                name = "lstm-pretrain_all")