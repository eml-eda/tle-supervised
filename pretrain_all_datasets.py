import torch
import numpy as np
from util.engine_pretrain import train_one_epoch
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from Algorithms.models_audio_mae import audioMae_vit_base
import timm.optim.optim_factory as optim_factory
import util.misc as misc
import argparse
from utils import get_all_datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Base parameters')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--encoder_dim', type=int, default=768)
    parser.add_argument('--decoder_dim', type=int, default=512)
    args = parser.parse_args()
    print(args)

    device = torch.device('cuda:{}'.format(args.device))
    restart_pretraining = False
    lr = 0.25e-3
    total_epochs = 201
    warmup_epochs = 100
    save_interval_epochs = 50

    # train and val dataset
    dataset_train = get_all_datasets()
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

    model = audioMae_vit_base(embed_dim=embed_dim, decoder_embed_dim=decoder_embed_dim, norm_pix_loss=True)
    model.to(device)

    param_groups = optim_factory.param_groups_weight_decay(model, 0.05)
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()
    print(f"Start pre-training for {total_epochs} epochs")
    for epoch in range(0, total_epochs):
        train_stats = train_one_epoch(model, data_loader_train, optimizer, device, epoch, loss_scaler, lr, total_epochs, warmup_epochs)
        if epoch % save_interval_epochs == 0:
            misc.save_model(output_dir="/home/benfenati/code/tle-supervised/Results/checkpoints/", model=model, model_without_ddp=model, optimizer=optimizer, 
                            loss_scaler=loss_scaler, epoch=epoch, 
                            name = "{}-{}-pretrain_all".format(embed_dim, decoder_embed_dim))