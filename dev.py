#%% params
import torch
import time

from models.models_audio_mae import audioMae_vit_base
from models.models_audio_mae_regression import audioMae_vit_base_R
from models.models_lstm_regression import lstm_regression
from models.models_tcn_regression import tcn_regression

embed_dim = 768
decoder_embed_dim = 512
ntests = 100

model_uc1 = audioMae_vit_base(embed_dim=embed_dim, decoder_embed_dim=decoder_embed_dim, norm_pix_loss=False)
checkpoint_uc1 = torch.load('/home/benfenati/code/tle-supervised/deployment/checkpoints/checkpoint-UC1.pth', map_location='cpu')
checkpoint_model = checkpoint_uc1['model']
msg = model_uc1.load_state_dict(checkpoint_model, strict=False)

checkpoint_uc2_car = torch.load('/home/benfenati/code/tle-supervised/deployment/checkpoints/checkpoint-UC2-car.pth', map_location='cpu')
model_uc2_car = audioMae_vit_base_R(embed_dim=embed_dim, decoder_embed_dim=decoder_embed_dim, norm_pix_loss=True, mask_ratio = 0.2)
checkpoint_model = checkpoint_uc2_car['model']
msg = model_uc2_car.load_state_dict(checkpoint_model, strict=False)

checkpoint_lstm = torch.load("/home/benfenati/code/tle-supervised/results/checkpoints/checkpoint-lstm-y_car_roccaprebalza_finetune-500.pth", map_location='cpu')
model_lstm = lstm_regression()
checkpoint_model = checkpoint_lstm['model']
msg = model_lstm.load_state_dict(checkpoint_model, strict=False)

checkpoint_tcn = torch.load("/home/benfenati/code/tle-supervised/results/checkpoints/checkpoint-tcn-y_car_roccaprebalza_finetune-500.pth", map_location='cpu')
model_tcn = tcn_regression()
checkpoint_model = checkpoint_tcn['model']
msg = model_tcn.load_state_dict(checkpoint_model, strict=False)

# count model parameters
num_params = sum(p.numel() for p in model_uc1.parameters())
print(f'Number of parameters UC1: {num_params:,}')

# count model parameters
num_params = sum(p.numel() for p in model_uc2_car.parameters())
print(f'Number of parameters model UC2/UC3: {num_params:,}')

# count model parameters
num_params = sum(p.numel() for p in model_lstm.parameters())
print(f'Number of parameters model LSTM: {num_params:,}')

# count model parameters
num_params = sum(p.numel() for p in model_tcn.parameters())
print(f'Number of parameters model TCN: {num_params:,}')

# %% lenet
import torch
from models.models_lenet import lenet_regression

model = lenet_regression()
x = torch.randn(1, 1, 100, 100)
with torch.no_grad():
    output = model(x)
print(output.shape)  # Should print: torch.Size([32, 8, 1])

# %%
