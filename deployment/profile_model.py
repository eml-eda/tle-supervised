import torch
import numpy as np
import pandas as pd
import argparse
from pathlib import Path

from models.pca import pca_class
from models.models_audio_mae import audioMae_vit_base
from models.models_audio_mae_regression import audioMae_vit_base_R
from models.models_tcn_regression import tcn_regression as tcn_regression_mae
from models.models_lstm_regression import lstm_regression as lstm_regression_mae
from models.soa import TorchStandardScaler, TorchSVR, TorchDecisionTree, TorchMLP, TorchKNN, TorchLinearRegression, TorchPipeline
from models.modules import Attention, WindowAttention, SwinBlock, PatchEmbed, TemporalConvNet

def count_macs(model, input_shape, mask_ratio=0.8):
    """
    Count MAC operations for any model using hooks
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
        kernel_h, kernel_w = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
        
        # MACs per position
        macs_per_position = kernel_h * kernel_w * in_channels * out_channels
        
        # Total positions
        if len(output_shape) == 4:  # 2D convolution
            output_h, output_w = output_shape[2], output_shape[3]
            total_positions = batch_size * output_h * output_w
        else:  # 1D convolution
            output_len = output_shape[2]
            total_positions = batch_size * output_len
        
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
        
        if len(input_shape) == 3:  # Standard attention [batch, seq, dim]
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
        else:
            # Fallback for other shapes
            dim = input_shape[-1]
            heads = module.heads
            head_dim = dim // heads
            inner_dim = head_dim * heads
            
            # Estimate based on last dimension
            qkv_macs = batch_size * np.prod(input_shape[1:-1]) * dim * inner_dim * 3
            attn_macs = batch_size * heads * np.prod(input_shape[1:-1]) * np.prod(input_shape[1:-1]) * head_dim
            out_macs = batch_size * np.prod(input_shape[1:-1]) * inner_dim * dim
        
        macs_dict[name] = qkv_macs + attn_macs + out_macs
    
    # Window Attention MAC counter
    def count_window_attention(module, input, output, name):
        input_shape = input[0].shape
        batch_size = input_shape[0]
        
        if len(input_shape) == 4:  # [batch, height, width, dim]
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
        else:
            # Fallback estimation
            dim = input_shape[-1]
            heads = module.heads
            head_dim = dim // heads
            inner_dim = head_dim * heads
            window_size = module.window_size
            
            # Rough estimate based on dimensions
            total_elements = np.prod(input_shape[1:-1])
            n_windows = total_elements // (window_size**2)
            
            qkv_macs = batch_size * total_elements * dim * inner_dim * 3
            attn_macs = batch_size * heads * n_windows * window_size**4 * head_dim
            out_macs = batch_size * total_elements * inner_dim * dim
            
        macs_dict[name] = qkv_macs + attn_macs + out_macs
    
    # LSTM MAC counter
    def count_lstm(module, input, output, name):
        input_shape = input[0].shape if isinstance(input, tuple) else input.shape
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        input_size = input_shape[2]
        
        hidden_size = module.hidden_size
        num_layers = module.num_layers
        bidirectional = 2 if module.bidirectional else 1
        
        # For each gate in LSTM: input, forget, cell, output
        gates_per_cell = 4
        
        # Calculate MACs for each timestep and each layer
        layer_input_size = input_size
        total_layer_macs = 0
        
        for layer in range(num_layers):
            # Input projection
            total_layer_macs += batch_size * seq_len * layer_input_size * hidden_size * gates_per_cell * bidirectional
            
            # Hidden state projection
            total_layer_macs += batch_size * seq_len * hidden_size * hidden_size * gates_per_cell * bidirectional
            
            # Update layer input size for next layer
            layer_input_size = hidden_size * bidirectional
            
        macs_dict[name] = total_layer_macs
    
    # TCN (Temporal Conv Net) MAC counter
    def count_temporal_conv(module, input, output, name):
        if not hasattr(module, 'network'):
            return
            
        input_shape = input[0].shape
        batch_size = input_shape[0]
        channels = input_shape[1]
        seq_len = input_shape[2]
        
        # Process all blocks in the network
        total_macs = 0
        
        # Simplified calculation assuming main computation is in the convolutions
        for i, block in enumerate(module.network):
            if hasattr(block, 'conv1') and hasattr(block, 'conv2'):
                # First conv
                in_ch = block.conv1.in_channels
                out_ch = block.conv1.out_channels
                kernel_size = block.conv1.kernel_size[0]
                dilation = block.conv1.dilation[0]
                
                # MACs for first conv
                macs = batch_size * seq_len * in_ch * out_ch * kernel_size
                total_macs += macs
                
                # Second conv
                in_ch = block.conv2.in_channels
                out_ch = block.conv2.out_channels
                kernel_size = block.conv2.kernel_size[0]
                
                # MACs for second conv
                macs = batch_size * seq_len * in_ch * out_ch * kernel_size
                total_macs += macs
                
                # Downsample if present
                if block.downsample is not None:
                    # MACs for 1x1 conv
                    in_ch = block.downsample.in_channels
                    out_ch = block.downsample.out_channels
                    macs = batch_size * seq_len * in_ch * out_ch
                    total_macs += macs
                
        macs_dict[name] = total_macs
    
    # Register hooks for all relevant modules
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            hooks.append(module.register_forward_hook(
                lambda mod, inp, out, name=name: count_conv2d(mod, inp, out, name)))
        elif isinstance(module, torch.nn.Conv1d):
            hooks.append(module.register_forward_hook(
                lambda mod, inp, out, name=name: count_conv2d(mod, inp, out, name)))
        elif isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_hook(
                lambda mod, inp, out, name=name: count_linear(mod, inp, out, name)))
        elif isinstance(module, Attention):
            hooks.append(module.register_forward_hook(
                lambda mod, inp, out, name=name: count_attention(mod, inp, out, name)))
        elif isinstance(module, WindowAttention):
            hooks.append(module.register_forward_hook(
                lambda mod, inp, out, name=name: count_window_attention(mod, inp, out, name)))
        elif isinstance(module, torch.nn.LSTM):
            hooks.append(module.register_forward_hook(
                lambda mod, inp, out, name=name: count_lstm(mod, inp, out, name)))
        elif isinstance(module, TemporalConvNet):
            hooks.append(module.register_forward_hook(
                lambda mod, inp, out, name=name: count_temporal_conv(mod, inp, out, name)))
    
    # Create dummy input and perform forward pass
    try:
        if hasattr(model, "half"):
            model = model.half().eval()
            dummy_input = torch.randn(input_shape, dtype=torch.float16)
        else:
            model = model.eval()
            dummy_input = torch.randn(input_shape, dtype=torch.float32)
        
        # Move to the same device as the model
        device = next(model.parameters()).device
        dummy_input = dummy_input.to(device)
        
        # Forward pass with mask_ratio if model supports it
        with torch.no_grad():
            if hasattr(model, "mask_ratio"):
                if 'audio_mae' in model.__class__.__name__.lower():
                    model(dummy_input, mask_ratio=mask_ratio)
                else:
                    model(dummy_input)
            else:
                model(dummy_input)
    except Exception as e:
        print(f"Error during forward pass: {e}")
        # Try a different approach for models that don't follow standard input patterns
        try:
            # Special handling for different model types
            if hasattr(model, "predict"):
                # For models with predict method (like PCA or scikit-learn style models)
                dummy_data = np.random.rand(1, input_shape[1])
                dummy_Vx = np.random.rand(input_shape[1], input_shape[1])
                model.predict(dummy_data, dummy_Vx)
            else:
                print(f"Could not profile model of type {type(model)}")
        except Exception as e:
            print(f"Failed alternative profiling approach: {e}")
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Calculate total MACs
    total_macs = sum(macs_dict.values())
    
    # Create DataFrame for better visualization
    df = pd.DataFrame(list(macs_dict.items()), columns=['Layer', 'MACs'])
    df['MACs'] = df['MACs'].astype(float)
    if len(df) > 0:  # Check if dataframe is not empty
        df['Percentage'] = df['MACs'] / total_macs * 100
        df = df.sort_values('MACs', ascending=False)
    
    return df, total_macs

def profile_models():
    """Profile all available models"""
    
    models_to_profile = {
        # UC1 Models
        "AudioMAE": (audioMae_vit_base(), (1, 1, 100, 100), 0.8),
        
        # UC2/UC3 Models
        "AudioMAE-R": (audioMae_vit_base_R(embed_dim=768, decoder_embed_dim=512, norm_pix_loss=True, mask_ratio=0.2), 
                       (1, 1, 100, 100), 0.2),
        "TCN": (tcn_regression_mae(embed_dim=768, decoder_embed_dim=512, mask_ratio=0.2), 
                (1, 1, 100, 100), 0.2),
        "LSTM": (lstm_regression_mae(embed_dim=768, decoder_embed_dim=512, mask_ratio=0.2), 
                 (1, 1, 100, 100), 0.2),
    }
    
    results = {}
    
    # Profile each model
    for model_name, (model, input_shape, mask_ratio) in models_to_profile.items():
        print(f"Profiling {model_name}...")
        try:
            if hasattr(model, "cuda"):
                model = model.cuda()
            
            df, total_macs = count_macs(model, input_shape, mask_ratio)
            results[model_name] = {
                "table": df,
                "total_macs": total_macs
            }
            print(f"  Total MACs: {total_macs/1e9:.2f} GMACs")
        except Exception as e:
            print(f"  Error profiling {model_name}: {e}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Profile models and count MACs')
    parser.add_argument('--model', choices=['all', 'audioMAE', 'audioMAE-R', 'tcn', 'lstm'], 
                        default='all', help='Model to profile (default: all)')
    parser.add_argument('--output', type=str, default='profile_results',
                        help='Output directory for profile results')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if args.model == 'all':
        results = profile_models()
        
        # Print summary
        print("\n===== Model Profiling Summary =====")
        for model_name, result in results.items():
            print(f"{model_name}: {result['total_macs']/1e9:.2f} GMACs")
            
            # Save detailed results to files
            result['table'].to_csv(output_dir / f"{model_name}_layers.csv", index=False)
            
            # Show top 10 layers
            print(f"\nTop 10 layers in {model_name}:")
            print(result['table'].head(10).to_string(index=False))
            print("\n")
            
        # Create summary CSV
        summary_df = pd.DataFrame([
            {"Model": name, "GMACs": result['total_macs']/1e9} 
            for name, result in results.items()
        ])
        summary_df.to_csv(output_dir / "model_summary.csv", index=False)
    else:
        # Profile specific model
        if args.model == 'audioMAE':
            model = audioMae_vit_base()
            input_shape = (1, 1, 100, 100)
            mask_ratio = 0.8
        elif args.model == 'audioMAE-R':
            model = audioMae_vit_base_R(embed_dim=768, decoder_embed_dim=512, norm_pix_loss=True, mask_ratio=0.2)
            input_shape = (1, 1, 100, 100)
            mask_ratio = 0.2
        elif args.model == 'tcn':
            model = tcn_regression_mae(embed_dim=768, decoder_embed_dim=512, mask_ratio=0.2)
            input_shape = (1, 1, 100, 100)
            mask_ratio = 0.2
        elif args.model == 'lstm':
            model = lstm_regression_mae(embed_dim=768, decoder_embed_dim=512, mask_ratio=0.2)
            input_shape = (1, 1, 100, 100)
            mask_ratio = 0.2
            
        # Move to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
            
        # Profile
        df, total_macs = count_macs(model, input_shape, mask_ratio)
        print(f"\n===== {args.model.upper()} Model Profiling =====")
        print(f"Total MACs: {total_macs/1e9:.2f} GMACs")
        print("\nTop 10 layers by MAC count:")
        print(df.head(10).to_string(index=False))
        
        # Summarize by module type
        print("\nMACs by module type:")
        module_types = {}
        for name in df['Layer']:
            if 'conv' in name.lower():
                module_type = 'Conv'
            elif 'linear' in name.lower() or 'fc' in name.lower() or 'pred' in name.lower():
                module_type = 'Linear'
            elif 'attn' in name.lower():
                module_type = 'Attention'
            elif 'lstm' in name.lower():
                module_type = 'LSTM'
            elif 'tcn' in name.lower() or 'temporal' in name.lower():
                module_type = 'TCN'
            else:
                module_type = 'Other'
            
            if module_type not in module_types:
                module_types[module_type] = 0
            
            module_types[module_type] += df[df['Layer'] == name]['MACs'].values[0]
        
        for module_type, macs in module_types.items():
            print(f"{module_type}: {macs/1e9:.2f} GMACs ({macs/total_macs*100:.2f}%)")
            
        # Save results
        df.to_csv(output_dir / f"{args.model}_layers.csv", index=False)
        
        # Save module type summary
        module_df = pd.DataFrame([
            {"Module Type": module_type, "GMACs": macs/1e9, "Percentage": macs/total_macs*100} 
            for module_type, macs in module_types.items()
        ])
        module_df.to_csv(output_dir / f"{args.model}_modules.csv", index=False)

if __name__ == "__main__":
    main()