import torch.nn as nn

class LSTMRegression(nn.Module):
    def __init__(self, num_mels=100, mel_len=100, 
                 hidden_size=100, num_layers=8, 
                 dropout=0.2, bidirectional=True):
        super(LSTMRegression, self).__init__()
        
        self.mel_len = mel_len
        self.num_mels = num_mels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=num_mels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate final features size
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Regression head (similar to TCN)
        self.regression_head = nn.Sequential(
            nn.Linear(lstm_output_size * mel_len, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        
        # Remove channel dimension
        x = x.squeeze(1)  # [batch, mel_len, num_mels]
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # [batch, mel_len, hidden_size*2]
        
        # Flatten for regression
        lstm_out = lstm_out.reshape(batch_size, -1)
        
        # Final regression
        out = self.regression_head(lstm_out)
        
        return None, out  # Match TCN interface

def lstm_regression(**kwargs):
    model = LSTMRegression(**kwargs)
    return model