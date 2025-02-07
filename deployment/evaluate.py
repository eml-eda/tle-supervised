import torch
import time

from models.models_audio_mae import audioMae_vit_base
from models.models_audio_mae_regression import audioMae_vit_base_R

embed_dim = 768
decoder_embed_dim = 512
ntests = 100

## UC1 eval
data_uc1 = torch.load('/home/benfenati/code/tle-supervised/deployment/data/uc1_data.pth', map_location='cpu', weights_only=True)
model_uc1 = audioMae_vit_base(embed_dim=embed_dim, decoder_embed_dim=decoder_embed_dim, norm_pix_loss=False)
checkpoint_uc1 = torch.load('/home/benfenati/code/tle-supervised/deployment/checkpoints/checkpoint-UC1.pth', map_location='cpu', weights_only=True)
checkpoint_model = checkpoint_uc1['model']
msg = model_uc1.load_state_dict(checkpoint_model, strict=False)

model_uc1.half().eval()
latencies = []
losses = []
for _ in range(ntests):
    samples = data_uc1[0]
    targets = data_uc1[-1]
    start = time.time()
    # compute output
    loss, _, _ = model_uc1  (samples)
    end = time.time()
    latencies.append(end - start)
print(f'average latency on UC1: {sum(latencies)/len(latencies):.3f} seconds')

## UC2 and UC3 eval
data_uc2_car = torch.load('/home/benfenati/code/tle-supervised/deployment/data/uc2_data_car.pth', map_location='cpu', weights_only=True)
data_uc2_camion = torch.load('/home/benfenati/code/tle-supervised/deployment/data/uc2_data_camion.pth', map_location='cpu', weights_only=True)
data_uc3 = torch.load('/home/benfenati/code/tle-supervised/deployment/data/uc3_data.pth', map_location='cpu', weights_only=True)
data = [data_uc1, data_uc2_car, data_uc2_camion, data_uc3]

checkpoint_uc2_car = torch.load('/home/benfenati/code/tle-supervised/deployment/checkpoints/checkpoint-UC2-car.pth', map_location='cpu', weights_only=True)
checkpoint_uc2_camion = torch.load('/home/benfenati/code/tle-supervised/deployment/checkpoints/checkpoint-UC2-camion.pth', map_location='cpu', weights_only=True)
checkpoint_uc3 = torch.load('/home/benfenati/code/tle-supervised/deployment/checkpoints/checkpoint-UC3.pth', map_location='cpu', weights_only=True)
checkpoints = [checkpoint_uc2_car, checkpoint_uc2_camion, checkpoint_uc3]

model_uc2_car = audioMae_vit_base_R(embed_dim=embed_dim, 
                                    decoder_embed_dim=decoder_embed_dim, 
                                    norm_pix_loss=True, mask_ratio = 0.2)
model_uc2_camion = audioMae_vit_base_R(embed_dim=embed_dim, 
                                       decoder_embed_dim=decoder_embed_dim, 
                                       norm_pix_loss=True, mask_ratio = 0.2)
model_uc3 = audioMae_vit_base_R(embed_dim=embed_dim, 
                                decoder_embed_dim=decoder_embed_dim, 
                                norm_pix_loss=True, mask_ratio = 0.2)

models = [model_uc2_car, model_uc2_camion, model_uc3]
names = ["UC2-car", "UC2-camion", "UC3"]
for batch, model, checkpoint, name in zip(data, models, checkpoints, names):
    checkpoint_model = checkpoint['model']
    msg = model.load_state_dict(checkpoint_model, strict=False)
    model.half().eval()
    latencies = []
    losses = []
    for _ in range(ntests):
        samples = data_uc1[0]
        targets = data_uc1[-1]
        start = time.time()
        # compute output
        loss, _ = model(samples)
        end = time.time()
        latencies.append(end - start)
    print(f'average latency on {name}: {sum(latencies)/len(latencies):.3f} seconds')