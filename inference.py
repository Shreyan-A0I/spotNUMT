import torch
from dataset import sequence_to_tensor
from model import HybridNuMTModel

def load_inference_model(weights_path: str = 'best_model.pt', device: str = 'cpu') -> HybridNuMTModel:
    model = HybridNuMTModel().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    return model

def predict_sequence(model: HybridNuMTModel, sequence: str, device: str = 'cpu') -> float:
    # 1. Provide some basic validation
    sequence = sequence.strip().upper()
    valid_chars = set('ACGTN')
    if not all(c in valid_chars for c in sequence):
        raise ValueError("Sequence contains invalid characters. Only A, C, G, T, N are allowed.")
    
    # 2. Convert to tensor
    tensor_seq = sequence_to_tensor(sequence).unsqueeze(0).to(device) # Shape: (1, 4, L)
    
    # 3. Inference
    with torch.no_grad():
        logits = model(tensor_seq)
        prob = torch.sigmoid(logits).item()
        
    return prob

if __name__ == "__main__":
    # Test with a dummy sequence
    device_str = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    try:
        model = load_inference_model(device=device_str)
        dummy_seq = "GATCACAGGTCTATCACCCTATTAACCACTCACGGGAGCTCTCCATGCATTTGGTATTTTCGTCTGGGGG" * 3
        prob = predict_sequence(model, dummy_seq[:200], device=device_str)
        print(f"Test Probability (mtDNA): {prob:.4f}")
    except FileNotFoundError:
        print("Model weights 'best_model.pt' not found. Train the model first.")
