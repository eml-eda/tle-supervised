import torch
import torch.nn as nn
from .modules import PatchEmbed, SwinBlock
from .utils import get_2d_sincos_pos_embed

class LSTMRegression(nn.Module):
    def __init__(self, num_mels=100, mel_len=100, patch_size=5, in_chans=1,
                 embed_dim=768, encoder_hidden=768, encoder_layers=3,
                 decoder_embed_dim=512, decoder_num_heads=16,
                 dropout=0.2, mask_ratio=0.8, mlp_ratio=4., bidirectional=True):
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
        
        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=encoder_hidden,
            num_layers=encoder_layers,
            dropout=dropout if encoder_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Regression components
        encoder_output_dim = encoder_hidden * 2 if bidirectional else encoder_hidden
        self.decoder_embed = nn.Linear(encoder_output_dim, decoder_embed_dim)
        
        self.decoder_blocks = SwinBlock(
            decoder_embed_dim, decoder_num_heads,
            decoder_embed_dim // decoder_num_heads,
            int(mlp_ratio * decoder_embed_dim),
            shifted=True, window_size=4,
            relative_pos_embedding=True
        )
        
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        
        # Fix regression input shape calculation
        self.regressionInputShape = decoder_embed_dim * self.grid_h * self.grid_w
        self.fc1 = nn.Linear(self.regressionInputShape, 1)
        
        self.initialize_weights()
        
    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], 
                                          (self.grid_h, self.grid_w), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
    def forward_encoder(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        
        x, _ = self.encoder_lstm(x)
        
        return x
        
    def forward_regression(self, x):
        x = self.decoder_embed(x)
        b, l, c = x.shape
        
        x = x.view(b, self.grid_h, self.grid_w, c)
        x = self.decoder_blocks(x)
        x = self.decoder_norm(x)
        out1 = x
        
        x = x.reshape(b, -1)
        out2 = self.fc1(x)
        
        return out1, out2

    def forward(self, imgs):
        latent = self.forward_encoder(imgs)
        pred = self.forward_regression(latent)
        return pred

def lstm_regression(**kwargs):
    model = LSTMRegression(**kwargs)
    return model