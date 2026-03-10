import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return x

class HybridNuMTModel(nn.Module):
    def __init__(self, 
                 in_channels: int = 4, 
                 conv_hidden: int = 64, 
                 kernel_size: int = 8, 
                 num_heads: int = 4, 
                 num_layers: int = 2, 
                 dropout: float = 0.1):
        super().__init__()
        
        # 1. Stem (Local Motifs)
        # Input: (Batch, Channels, SeqLen) -> (B, 4, L)
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=conv_hidden, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm1d(conv_hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # We need a fixed positional embedding or sinusoidal. Since sequence length might vary slightly
        # or we just use sinusoidal. PyTorch Transformer uses [SeqLen, Batch, Dim] if batch_first=False
        self.pos_encoder = PositionalEncoding(d_model=conv_hidden, max_len=500)
        
        # 2. Body (Global Context via Transformer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=conv_hidden, 
            nhead=num_heads, 
            dim_feedforward=conv_hidden * 4, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Head (Classification)
        self.head = nn.Sequential(
            nn.Linear(conv_hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        """
        x: (Batch, 4, SeqLen)
        """
        # Stem
        x = self.stem(x) # (Batch, conv_hidden, SeqLen)
        
        # Prepare for Transformer (BatchFirst=True -> Batch, SeqLen, Dim)
        x = x.transpose(1, 2) # (Batch, SeqLen, conv_hidden)
        
        # Add Positional Encoding (PositionalEncoding expects SeqLen, Batch, Dim, so we transpose back and forth)
        # Wait, our PositionalEncoding expects (SeqLen, Batch, Dim). Let's reshape momentarily.
        x = x.transpose(0, 1) # (SeqLen, Batch, Dim)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1) # (Batch, SeqLen, Dim)
        
        # Transformer
        x = self.transformer(x) # (Batch, SeqLen, Dim)
        
        # Global Average Pooling over the sequence dimension
        x = x.mean(dim=1) # (Batch, Dim)
        
        # Classification Head
        out = self.head(x) # (Batch, 1)
        
        # Use BCEWithLogitsLoss during training, so NO sigmoid here!
        return out.squeeze(-1) # (Batch,)

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
