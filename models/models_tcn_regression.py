'''
Implementation of TCN as a Masked Autoencoder-like architecture (this is used during fine-tuning)
'''
import torch
import torch.nn as nn
from .modules import PatchEmbed, TemporalConvNet, SwinBlock
from .utils import get_2d_sincos_pos_embed

class TCNRegression(nn.Module):
    def __init__(self, num_mels=100, mel_len=100, patch_size=5, in_chans=1,
                 embed_dim=768, encoder_channels=[768, 768, 768],
                 decoder_embed_dim=512, decoder_num_heads=16,
                 kernel_size=2, dropout=0.2, mask_ratio=0.8, mlp_ratio=4.):
        super().__init__()
        
        # Encoder specifics
        self.mask_ratio = mask_ratio
        self.patch_embed = PatchEmbed((mel_len, num_mels), (patch_size, patch_size), 
                                    in_chans, embed_dim)
        self.grid_h = mel_len // patch_size
        self.grid_w = num_mels // patch_size
        num_patches = self.patch_embed.num_patches
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim),
                                    requires_grad=False)
        
        # Encoder TCN
        self.encoder_tcn = TemporalConvNet(
            num_inputs=embed_dim,
            num_channels=encoder_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        # Regression components
        self.decoder_embed = nn.Linear(encoder_channels[-1], decoder_embed_dim)
        
        self.decoder_blocks = SwinBlock(
            decoder_embed_dim, decoder_num_heads, 
            decoder_embed_dim // decoder_num_heads,
            int(mlp_ratio * decoder_embed_dim),
            shifted=True, window_size=4, 
            relative_pos_embedding=True
        )
        
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        
        # Fix regression input shape calculation - remove mask_ratio
        self.regressionInputShape = decoder_embed_dim * self.grid_h * self.grid_w
        self.fc1 = nn.Linear(self.regressionInputShape, 1)
        
        self.initialize_weights()
        
    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], 
                                          (self.grid_h, self.grid_w), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # Initialize patch_embed like MAE
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
    def forward_encoder(self, x):
        # Patch embedding
        x = self.patch_embed(x)
        x = x + self.pos_embed
        
        # TCN encoding
        x = x.transpose(1, 2)
        x = self.encoder_tcn(x)
        x = x.transpose(1, 2)
        
        return x
        
    def forward_regression(self, x):
        x = self.decoder_embed(x)  # [B, L, decoder_embed_dim]
        b, l, c = x.shape
        
        x = x.view(b, self.grid_h, self.grid_w, c)  # [B, 20, 20, 512]
        x = self.decoder_blocks(x)
        x = self.decoder_norm(x)
        out1 = x
        
        # Flatten for regression
        x = x.reshape(b, -1)  # [B, 20*20*512]
        out2 = self.fc1(x)
        
        return out1, out2

    def forward(self, imgs):
        latent = self.forward_encoder(imgs)
        pred = self.forward_regression(latent)
        return pred

def tcn_regression(**kwargs):
    model = TCNRegression(**kwargs)
    return model