#%%
import torch
from models.models_tcn import tcn_regression

#%%
# Example usage
model = tcn_regression(
    num_mels=100,
    mel_len=100,
    num_channels=[100, 100, 100, 100],
    kernel_size=2,
    dropout=0.2
)

# Test with dummy input
x = torch.randn(8, 1, 100, 100)  # [batch, channel, mel_len, num_mels]
output = model(x)  # Returns [batch, 1]

# %%
output