'''
Plain implementation of TCN
'''
# Original model from here (https://github.com/locuslab/TCN/blob/master/TCN/tcn.py)
import torch.nn as nn
from .modules import TemporalConvNet
    

# TCN for regression
class TemporalConvNetRegression(nn.Module):
    def __init__(self, num_mels=100, mel_len=100, num_channels=[100, 100, 100, 100], kernel_size=2, dropout=0.2):
        super(TemporalConvNetRegression, self).__init__()
        
        self.mel_len = mel_len
        self.num_mels = num_mels
        
        self.tcn = TemporalConvNet(
            num_inputs=num_mels,  # each time step has num_mels features
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        # regression head
        final_channels = num_channels[-1]
        self.regression_head = nn.Sequential(
            nn.Linear(final_channels * mel_len, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        batch_size = x.size(0)

        x = x.squeeze(1)  # [batch, mel_len, num_mels]

        tcn_out = self.tcn(x)  # [batch, final_channels, mel_len]
        
        tcn_out = tcn_out.reshape(batch_size, -1)
        
        out = self.regression_head(tcn_out)
        
        return None, out # put None because engine_pretrain loop expects 2 outputs

def tcn_regression(**kwargs):
    model = TemporalConvNetRegression(**kwargs)
    return model