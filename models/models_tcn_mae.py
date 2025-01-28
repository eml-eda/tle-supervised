'''
Implementation of TCN as a Masked Autoencoder-like architecture (this is used during pre-training)
'''

import torch
import torch.nn as nn
from .modules import PatchEmbed, TemporalConvNet
from .utils import get_2d_sincos_pos_embed

class TCNMaskedAutoencoder(nn.Module):
    def __init__(self, num_mels=100, mel_len=100, patch_size=5, in_chans=1,
                 embed_dim=768, encoder_channels=[768, 768, 768], 
                 decoder_embed_dim=512, decoder_channels=[512, 512, 512],
                 kernel_size=2, dropout=0.2, mask_ratio=0.8):
        super().__init__()
        
        # Encoder specifics
        self.mask_ratio = mask_ratio
        self.patch_embed = PatchEmbed((mel_len, num_mels), (patch_size, patch_size), 
                                    in_chans, embed_dim)
        self.grid_h = mel_len // patch_size
        self.grid_w = num_mels // patch_size
        
        # Positional embedding
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim),
                                    requires_grad=False)
        
        # Encoder TCN
        self.encoder_tcn = TemporalConvNet(
            num_inputs=embed_dim,
            num_channels=encoder_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        # Decoder specifics
        self.decoder_embed = nn.Linear(encoder_channels[-1], decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, 
                                            decoder_embed_dim), requires_grad=False)
        
        # Decoder TCN
        self.decoder_tcn = TemporalConvNet(
            num_inputs=decoder_embed_dim,
            num_channels=decoder_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        # Final projection
        self.decoder_pred = nn.Linear(decoder_channels[-1], patch_size**2 * in_chans)
        
        self.initialize_weights()
        
    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], 
                                          (self.grid_h, self.grid_w), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                  (self.grid_h, self.grid_w), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        
        torch.nn.init.normal_(self.mask_token, std=.02)

    def patchify(self, imgs):
        """
        imgs: (N, 1, H, W)
        x: (N, L, patch_size**2)
        """
        p = self.patch_embed.patch_size[0]
        h = self.grid_h
        w = self.grid_w
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2)
        imgs: (N, 1, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = self.grid_h
        w = self.grid_w
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, w * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        x: [N, L, D], sequence
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 1, H, W]
        pred: [N, L, p*p*1]
        mask: [N, L], 0 is keep, 1 is remove
        """
        target = self.patchify(imgs)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = loss.sum()  # mean loss on all patches
        return loss
        
    def forward_encoder(self, x):
        # Patch embedding
        x = self.patch_embed(x)
        x = x + self.pos_embed
        
        # Masking
        x, mask, ids_restore = self.random_masking(x, self.mask_ratio)
        
        # TCN encoding
        x = x.transpose(1, 2)  # TCN expects [B, C, L]
        x = self.encoder_tcn(x)
        x = x.transpose(1, 2)  # Back to [B, L, C]
        
        return x, mask, ids_restore
        
    def forward_decoder(self, x, ids_restore):
        # Embed tokens
        x = self.decoder_embed(x)
        
        # Append mask tokens
        mask_tokens = self.mask_token.repeat(x.shape[0], 
                                           ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x_, dim=1, 
                        index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        
        # Add positional embedding
        x = x + self.decoder_pos_embed
        
        # TCN decoding
        x = x.transpose(1, 2)
        x = self.decoder_tcn(x)
        x = x.transpose(1, 2)
        
        # Predict patches
        x = self.decoder_pred(x)
        
        return x

    def forward(self, imgs,  mask_ratio=0.8):
        latent, mask, ids_restore = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        pred = self.unpatchify(pred)
        return loss, pred, mask

def tcn_mae(**kwargs):
    model = TCNMaskedAutoencoder(**kwargs)
    return model