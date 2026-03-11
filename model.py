import torch
import torch.nn as nn
import math

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False)
        )
        
    def forward(self, x):
        # x: (Batch, SeqLen, HiddenDim)
        attn_weights = self.attention(x) # (Batch, SeqLen, 1)
        attn_weights = torch.softmax(attn_weights, dim=1) # Normalize over SeqLen
        
        # Weighted sum of BiLSTM outputs
        context = torch.sum(attn_weights * x, dim=1) # (Batch, HiddenDim)
        return context

class HybridNuMTModel(nn.Module):
    def __init__(self, 
                 in_channels: int = 4, 
                 conv_hidden: int = 64, 
                 kernel_size: int = 8, 
                 lstm_hidden: int = 64, 
                 num_layers: int = 2, 
                 dropout: float = 0.2):
        super().__init__()
        
        # 1. Stem (Local Motifs + Downsampling)
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=conv_hidden, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm1d(conv_hidden),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=conv_hidden, out_channels=conv_hidden*2, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm1d(conv_hidden*2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout)
        )
        
        # 2. Body (BiLSTM)
        self.lstm = nn.LSTM(
            input_size=conv_hidden*2,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 3. Attention Layer
        self.attention = Attention(lstm_hidden * 2)
        
        # 4. Head (Classification)
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        """
        x: (Batch, 4, SeqLen)
        """
        x = self.stem(x) # (Batch, conv_hidden*2, SeqLen/4)
        
        # Prepare for LSTM
        x = x.transpose(1, 2) # (Batch, SeqLen/4, conv_hidden*2)
        
        lstm_out, _ = self.lstm(x) # (Batch, SeqLen/4, lstm_hidden*2)
        
        # Apply Attention to gather focused context
        context = self.attention(lstm_out) # (Batch, lstm_hidden*2)
        
        out = self.head(context) # (Batch, 1)
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
