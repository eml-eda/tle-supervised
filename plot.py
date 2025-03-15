import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('results/model_exploration_results.csv')

# Clean MAE values by taking only the first number before parentheses
df['mae'] = df['mae'].apply(lambda x: float(x.split('(')[0].strip()))

# Find the row with minimum MAE
best_config = df.loc[df['mae'].idxmin()]

print("\n=== Best Model Configuration ===")
print(f"MAE: {best_config['mae']:.2f}")
print("\nArchitecture:")
print(f"• Encoder: {best_config['encoder_depth']} layers, {best_config['encoder_heads']} heads")
print(f"• Decoder: {best_config['decoder_depth']} layers, {best_config['decoder_heads']} heads")
print(f"• Embedding dimensions: {best_config['encoder_embedding_dim']}/{best_config['decoder_embedding_dim']}")