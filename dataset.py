import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from Bio import SeqIO
from sklearn.model_selection import train_test_split

# One-hot encoding dictionary
# Maps a nucleotide to a (4,) array: [A, C, G, T]
NUC_MAP = {
    'A': [1.0, 0.0, 0.0, 0.0],
    'C': [0.0, 1.0, 0.0, 0.0],
    'G': [0.0, 0.0, 1.0, 0.0],
    'T': [0.0, 0.0, 0.0, 1.0],
    'N': [0.25, 0.25, 0.25, 0.25] # Fallback, though we filtered them out
}

def sequence_to_tensor(sequence: str) -> torch.Tensor:
    """
    Converts a string of nucleotides into a PyTorch tensor.
    Expects shape to be (4, sequence_length) for Conv1D input.
    """
    encoded = []
    for nuc in sequence:
        encoded.append(NUC_MAP.get(nuc.upper(), NUC_MAP['N']))
    
    # encoded is shape (length, 4)
    tensor = torch.tensor(encoded, dtype=torch.float32)
    # Transpose to (4, length) to act as sequence channels
    return tensor.transpose(0, 1)

class NuMTDataset(Dataset):
    def __init__(self, sequences, labels):
        """
        sequences: List of tensor sequences of shape (4, L)
        labels: List or array of labels (1 for mtDNA, 0 for NuMT)
        """
        self.sequences = sequences
        self.labels = torch.tensor(labels, dtype=torch.float32)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def load_fasta_to_tensors(fasta_path: str, label: int):
    """
    Reads a FASTA file and returns a list of tensors and a list of labels.
    """
    sequences = []
    labels = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        seq_str = str(record.seq).upper()
        tensor = sequence_to_tensor(seq_str)
        sequences.append(tensor)
        labels.append(label)
    return sequences, labels

def get_dataloaders(
    positive_fasta="data/processed/mtDNA_200bp.fasta",
    negative_fasta="data/processed/NUMT_200bp.fasta",
    batch_size=32,
    num_workers=0,
    seed=42
):
    """
    Loads data, splits into 80/10/10 (Train/Val/Test), and returns DataLoaders.
    """
    # 1. Load data
    print("Loading positive sequences (mtDNA)...")
    pos_seqs, pos_labels = load_fasta_to_tensors(positive_fasta, label=1)
    
    print("Loading negative sequences (NuMT)...")
    neg_seqs, neg_labels = load_fasta_to_tensors(negative_fasta, label=0)
    
    # Combine
    all_seqs = pos_seqs + neg_seqs
    all_labels = pos_labels + neg_labels
    
    print(f"Total sequences: {len(all_labels)} (Pos: {len(pos_labels)}, Neg: {len(neg_labels)})")
    
    # 2. Split (80/10/10 -> Train/Val/Test)
    # First split off 20% for val/test combined
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        all_seqs, all_labels, 
        test_size=0.20, 
        random_state=seed, 
        stratify=all_labels
    )
    
    # Then split the 20% in half to get 10% val and 10% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, 
        test_size=0.50, 
        random_state=seed, 
        stratify=y_tmp
    )
    
    print(f"Train split: {len(y_train)} sequences")
    print(f"Val split:   {len(y_val)} sequences")
    print(f"Test split:  {len(y_test)} sequences")
    
    # 3. Create Datasets
    train_dataset = NuMTDataset(X_train, y_train)
    val_dataset = NuMTDataset(X_val, y_val)
    test_dataset = NuMTDataset(X_test, y_test)
    
    # 4. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()
    for batch_x, batch_y in train_loader:
        print(f"Batch X shape (Batch, Channels, Length): {batch_x.shape}")
        print(f"Batch Y shape: {batch_y.shape}")
        break
