import gradio as gr
import torch
from inference import load_inference_model, predict_sequence

# Load model once globally
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
try:
    model = load_inference_model('best_model.pt', device=device)
    MODEL_LOADED = True
except Exception as e:
    print(f"Warning: Model could not be loaded. Error: {e}")
    model = None
    MODEL_LOADED = False

def predict(sequence):
    if not MODEL_LOADED:
        return {"Error": 0.0}, "Model weights not found. Please train the model first."
        
    sequence = sequence.strip().replace(" ", "").replace("\n", "").upper()
    
    if len(sequence) < 50:
        return None, "Error: Sequence too short. Provide at least 50bp."
        
    try:
        prob = predict_sequence(model, sequence, device=device)
        # return a dictionary for Gradio Label component
        result = {
            "True mtDNA": prob,
            "NuMT (Pseudogene)": 1.0 - prob
        }
        
        msg = f"Analysis complete. Sequence length: {len(sequence)}bp"
        return result, msg
        
    except ValueError as e:
        return None, f"Error: {str(e)}"
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"

# Define Gradio Interface
with gr.Blocks(title="spotNUMT - NuMT Detector", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🧬 spotNUMT: Mitochondrial Pseudogene Detector")
    gr.Markdown("Identify NuMTs (Nuclear Mitochondrial DNA sequences) using a Hybrid Conv1D-Transformer deep learning model.")
    
    with gr.Row():
        with gr.Column(scale=2):
            seq_input = gr.Textbox(
                lines=5, 
                placeholder="Paste DNA sequence here (A, C, G, T)...",
                label="Input Sequence (FASTA string)"
            )
            submit_btn = gr.Button("Analyze Sequence", variant="primary")
            
            gr.Examples(
                examples=[
                    ["GATCACAGGTCTATCACCCTATTAACCACTCACGGGAGCTCTCCATGCATTTGGTATTTTCGTCTGGGGGGTATGCACGCGATAGCATTGCGAGACGCTGGAGCCGGAGCACCCTATGTCGCAGTATCTGTCTTTGATTCCTGCCTCATCCTATTATTTATCGCACCTACGTTCAATATTACAGGCGAACATACTTACTAAAGTGTGTTA"],
                    ["CAAAGGGAGTCCGAACTAGTCTCAGGCTTCAACATCGAATACGCCGCAGGCCCCTTCGCCCTATTCTTCATAGCCGAATACACAAACATTATTATAATAAACACCCTCACCACTACAATCTTCCTAGGAACAACATATAACGCACTCTCCCCTGAACTCTACACAACATATTTTGTCACCAAGACCCTACTTCTGACCTC"]
                ],
                inputs=seq_input,
                label="Examples (mtDNA vs NuMT)"
            )
            
        with gr.Column(scale=1):
            label_output = gr.Label(label="Prediction Confidence")
            msg_output = gr.Textbox(label="Status message", interactive=False)

    submit_btn.click(
        fn=predict,
        inputs=seq_input,
        outputs=[label_output, msg_output]
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
