import torch
from torch import nn
import numpy as np
import pandas as pd
from models.models_audio_mae import audioMae_vit_base
from models.models_audio_mae_regression import audioMae_vit_base_R
from models.modules import Attention, WindowAttention, SwinBlock

def count_macs_audio_mae(model, input_shape=(1, 1, 100, 100), mask_ratio=0.8):
    """
    Count MAC operations for AudioMaskedAutoencoderViT using hooks
    """
    # Initialize dictionaries to store counts
    macs_dict = {}
    total_macs = 0
    
    # Register hooks for different module types
    hooks = []
    
    # Conv2D MAC counter
    def count_conv2d(module, input, output, name):
        input_shape = input[0].shape
        output_shape = output.shape
        batch_size = input_shape[0]
        
        in_channels = module.in_channels
        out_channels = module.out_channels 
        kernel_h, kernel_w = module.kernel_size
        
        # MACs per position
        macs_per_position = kernel_h * kernel_w * in_channels * out_channels
        
        # Total positions
        output_h, output_w = output_shape[2], output_shape[3]
        total_positions = batch_size * output_h * output_w
        
        # Total MACs
        total_macs = macs_per_position * total_positions
        macs_dict[name] = total_macs
    
    # Linear MAC counter
    def count_linear(module, input, output, name):
        input_shape = input[0].shape
        batch_size = input_shape[0]
        
        if len(input_shape) > 2:  # Handle sequence inputs [batch, seq, dim]
            seq_len = input_shape[1]
            macs = batch_size * seq_len * module.in_features * module.out_features
        else:  # Standard linear [batch, dim]
            macs = batch_size * module.in_features * module.out_features
            
        macs_dict[name] = macs
    
    # Attention MAC counter
    def count_attention(module, input, output, name):
        input_shape = input[0].shape
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        dim = input_shape[2]
        heads = module.heads
        head_dim = dim // heads
        inner_dim = head_dim * heads
        
        # QKV projections
        qkv_macs = batch_size * seq_len * dim * inner_dim * 3
        
        # Attention computation
        attn_macs = batch_size * heads * seq_len * seq_len * head_dim
        
        # Output projection
        out_macs = batch_size * seq_len * inner_dim * dim
        
        macs_dict[name] = qkv_macs + attn_macs + out_macs
    
    # Window Attention MAC counter
    def count_window_attention(module, input, output, name):
        input_shape = input[0].shape
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        dim = input_shape[3]
        heads = module.heads
        head_dim = dim // heads
        inner_dim = head_dim * heads
        window_size = module.window_size
        
        # Number of windows
        n_windows = (height // window_size) * (width // window_size)
        
        # QKV projections
        qkv_macs = batch_size * n_windows * window_size**2 * dim * inner_dim * 3
        
        # Attention computation
        attn_macs = batch_size * heads * n_windows * window_size**4 * head_dim
        
        # Output projection
        out_macs = batch_size * n_windows * window_size**2 * inner_dim * dim
        
        macs_dict[name] = qkv_macs + attn_macs + out_macs
    
    # Register hooks for all relevant modules
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(
                lambda mod, inp, out, name=name: count_conv2d(mod, inp, out, name)))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(
                lambda mod, inp, out, name=name: count_linear(mod, inp, out, name)))
        elif isinstance(module, Attention):
            hooks.append(module.register_forward_hook(
                lambda mod, inp, out, name=name: count_attention(mod, inp, out, name)))
        elif isinstance(module, WindowAttention):
            hooks.append(module.register_forward_hook(
                lambda mod, inp, out, name=name: count_window_attention(mod, inp, out, name)))
    
    # Create dummy input and perform forward pass
    dummy_input = torch.randn(input_shape)
    with torch.no_grad():
        model(dummy_input, mask_ratio=mask_ratio)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Calculate total MACs
    total_macs = sum(macs_dict.values())
    
    # Create DataFrame for better visualization
    df = pd.DataFrame(list(macs_dict.items()), columns=['Layer', 'MACs'])
    df['MACs'] = df['MACs'].astype(float)
    df['Percentage'] = df['MACs'] / total_macs * 100
    df = df.sort_values('MACs', ascending=False)
    
    return df, total_macs

def main():
    # Create model with best configuration from your exploration
    # Based on your CSV data, the best configuration appears to be:
    # embed_dim=1536,
    # encoder_depth=7,
    # num_heads=8,
    # decoder_embed_dim=1024,
    # decoder_depth=2,
    # decoder_num_heads=24,
    # norm_pix_loss=True
    model = audioMae_vit_base()
    
    # Profile model
    df, total_macs = count_macs_audio_mae(model)
    
    # Print results
    print(f"Total MACs: {total_macs/1e9:.2f} GMACs")
    print("\nTop 10 layers by MAC count:")
    print(df.head(10).to_string(index=False))
    
    # Summarize by module type
    print("\nMACs by module type:")
    module_types = {}
    for name in df['Layer']:
        if 'conv' in name.lower():
            module_type = 'Conv2d'
        elif 'linear' in name.lower() or 'fc' in name.lower() or 'pred' in name.lower():
            module_type = 'Linear'
        elif 'attn' in name.lower():
            module_type = 'Attention'
        else:
            module_type = 'Other'
        
        if module_type not in module_types:
            module_types[module_type] = 0
        
        module_types[module_type] += df[df['Layer'] == name]['MACs'].values[0]
    
    for module_type, macs in module_types.items():
        print(f"{module_type}: {macs/1e9:.2f} GMACs ({macs/total_macs*100:.2f}%)")

if __name__ == "__main__":
    main()