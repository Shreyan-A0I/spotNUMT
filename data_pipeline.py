import os
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

def process_fasta(input_file: str, output_file: str, window_size: int = 200):
    """
    Reads a FASTA file, slices the sequences into uniform windows,
    filters out sequences with 'N's, and discards leftover fragments.
    """
    print(f"Processing {input_file} -> {output_file} (Window Size: {window_size}bp)")
    
    valid_records = []
    total_windows = 0
    dropped_n = 0
    
    # Read all sequences from the input FASTA
    for record in SeqIO.parse(input_file, "fasta"):
        sequence_str = str(record.seq).upper()
        
        # Slice into chunks of `window_size`
        for i in range(0, len(sequence_str), window_size):
            chunk = sequence_str[i:i + window_size]
            
            # Discard fragments smaller than the window size
            if len(chunk) < window_size:
                continue
                
            total_windows += 1
            
            # Quality Control: Filter out chunks with unknown bases 'N'
            if 'N' in chunk:
                dropped_n += 1
                continue
                
            # Create a new SeqRecord for the valid chunk
            chunk_id = f"{record.id}_window_{i}_{i+window_size}"
            valid_record = SeqRecord(
                Seq(chunk),
                id=chunk_id,
                description=f"length={window_size}"
            )
            valid_records.append(valid_record)
            
    # Write valid standardized chunks to the output FASTA
    SeqIO.write(valid_records, output_file, "fasta")
    
    print(f"Generated {len(valid_records)} valid {window_size}bp windows.")
    if dropped_n > 0:
        print(f"Dropped {dropped_n} windows containing 'N'.")
        
    return valid_records

if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    
    # Process positive class (true mtDNA)
    process_fasta(
        input_file="data/sequence.fasta",
        output_file="data/processed/mtDNA_200bp.fasta",
        window_size=200
    )
    
    # Process negative class (NuMTs)
    process_fasta(
        input_file="data/NUMT.fasta",
        output_file="data/processed/NUMT_200bp.fasta",
        window_size=200
    )
    
    print("Data processing complete. Standardized files saved to data/processed/")
