import torch
import torch.nn as nn

class LeNetRegression(nn.Module):
    def __init__(self, in_channels=1):
        """
        LeNet-5 architecture adapted for regression with 100x100 input
        Args:
            in_channels (int): Number of input channels (default: 1 for spectrograms)
        """
        super(LeNetRegression, self).__init__()
        
        # First convolutional block
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Second convolutional block
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Calculate the size of flattened features
        # After layer1: 100x100 -> 48x48
        # After layer2: 48x48 -> 22x22
        self.flat_features = 16 * 22 * 22
        
        # Fully connected layers
        self.fc = nn.Linear(self.flat_features, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, 8)  # Output size 8 for regression
        
        self.initialize_weights()
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # First conv block
        x = self.layer1(x)
        
        # Second conv block
        x = self.layer2(x)
        
        # Flatten
        x = x.reshape(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        
        # Reshape to [batch, 8, 1]
        x = x.view(-1, 8, 1)
        
        return x

def lenet_regression(**kwargs):
    model = LeNetRegression(**kwargs)
    return model