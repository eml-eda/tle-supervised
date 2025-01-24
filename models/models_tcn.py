# Original model from here (https://github.com/locuslab/TCN/blob/master/TCN/tcn.py)
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from .modules import PatchEmbed


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    

# TCN for regression
class TemporalConvNetRegression(nn.Module):
    def __init__(self, num_mels=100, mel_len=100, num_channels=[100, 100, 100, 100, 100, 100, 100, 100], kernel_size=2, dropout=0.2):
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