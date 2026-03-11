import torch
import torch.nn as nn
import math

class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding='same', dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding='same', dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return self.relu(out + residual)

class HybridNuMTModel(nn.Module):
    def __init__(self, 
                 in_channels: int = 4, 
                 conv_hidden: int = 64, 
                 dropout: float = 0.2):
        super().__init__()
        
        # 1. Stem (Initial projection)
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, conv_hidden, kernel_size=8, padding='same'),
            nn.BatchNorm1d(conv_hidden),
            nn.ReLU()
        )
        
        # 2. Body (Dilated Residual Blocks - TCN style)
        self.blocks = nn.Sequential(
            ResidualBlock(conv_hidden, dilation=1),
            ResidualBlock(conv_hidden, dilation=2),
            ResidualBlock(conv_hidden, dilation=4),
            ResidualBlock(conv_hidden, dilation=8)
        )
        
        # 3. Head (Classification)
        self.head = nn.Sequential(
            nn.Linear(conv_hidden, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        """
        x: (Batch, 4, SeqLen)
        """
        x = self.stem(x) # (Batch, conv_hidden, SeqLen)
        
        x = self.blocks(x) # (Batch, conv_hidden, SeqLen)
        
        # Global Average Pooling over the sequence dimension
        x = x.mean(dim=2) # (Batch, conv_hidden)
        
        out = self.head(x) # (Batch, 1)
        return out.squeeze(-1)

if __name__ == "__main__":
    # Test the model with dummy data
    model = HybridNuMTModel()
    print("Model Architecture:")
    print(model)
    
    # Dummy batch of 32 sequences, 4 channels (A,C,G,T), 200 length
    dummy_input = torch.randn(32, 4, 200)
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("Forward pass successful!")
