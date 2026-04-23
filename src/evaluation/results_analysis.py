import os
import re
import string
import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import REFERENCE_FILES, TRANSCRIPTIONS_OUTPUT, ANALYSIS_OUTPUT, ensure_output_dirs

def normalize_text(text):
    """Lowercases, removes punctuation, and strips extra whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return " ".join(text.split())

def calculate_wer(reference, hypothesis):
    """A basic implementation of Word Error Rate (Levenshtein distance at word level)."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    # Initialize matrix
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1): d[i][0] = i
    for j in range(len(hyp_words) + 1): d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion = d[i][j-1] + 1
                deletion = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)
    
    return d[len(ref_words)][len(hyp_words)] / len(ref_words) if len(ref_words) > 0 else 0

def compare_transcriptions(ground_truth_path, folder_path, output_file="analysis_results.txt"):
    ground_truth = {}
    try:
        with open(ground_truth_path, 'r') as f:
            content = f.read()
            matches = re.findall(r'(sp\d{2})\s+(.*?)(?=\n|\[source|$)', content, re.DOTALL)
            for sp_id, text in matches:
                ground_truth[sp_id] = text.strip()
    except FileNotFoundError:
        print("Error: Ground truth file not found.")
        return
    
    # Write header for the output files
    with open(ANALYSIS_OUTPUT / output_file, 'w') as f:
        f.write(f"{'ID':<6} | {'Norm Match':<12} | {'WER (%)':<8} | {'Details'}\n")
        f.write("-" * 75 + "\n")
    

    print(f"{'ID':<6} | {'Norm Match':<12} | {'WER (%)':<8} | {'Details'}")
    print("-" * 75)

    total_wer = 0
    match_count = 0

    for sp_id, true_text in ground_truth.items():
        file_path = os.path.join(folder_path, f"{sp_id}_train_sn10.txt")
        
        if not os.path.exists(file_path):
            continue
            
        with open(file_path, 'r') as f:
            model_output = f.read().split('\n')[0].strip()

        # Take out the header if it exists
        if model_output.startswith("TEXT:"):
            model_output = model_output[5:].strip()
        
        # 1. Normalize both for the "True" Match
        norm_ref = normalize_text(true_text)
        norm_hyp = normalize_text(model_output)
        
        # 2. Metrics
        is_match = (norm_ref == norm_hyp)
        wer_score = calculate_wer(norm_ref, norm_hyp)
        
        total_wer += wer_score
        if is_match: match_count += 1

        with open(ANALYSIS_OUTPUT / output_file, 'a') as f:
            f.write(f"{sp_id:<6} | {'YES' if is_match else 'NO':<12} | {wer_score*100:>7.1f}% | {model_output[:30]}...\n")

        status = "YES" if is_match else "NO"
        print(f"{sp_id:<6} | {status:<12} | {wer_score*100:>7.1f}% | {model_output[:30]}...")

    # Write summary to file
    with open(ANALYSIS_OUTPUT / output_file, 'a') as f:
        f.write("-" * 75 + "\n")
        f.write(f"OVERALL ACCURACY (Normalized): {(match_count / len(ground_truth) * 100):.2f}%\n")
        f.write(f"AVERAGE WORD ERROR RATE: {(total_wer / len(ground_truth) * 100):.2f}%\n")

    # Summary Analysis
    avg_wer = (total_wer / len(ground_truth)) * 100
    accuracy = (match_count / len(ground_truth)) * 100
    
    print("-" * 75)
    print(f"OVERALL ACCURACY (Normalized): {accuracy:.2f}%")
    print(f"AVERAGE WORD ERROR RATE: {avg_wer:.2f}%")

if __name__ == "__main__":
    ensure_output_dirs()
    print("-" * 40)
    print("Analyzing transcriptions for test_tiny...")
    print("-" * 40)
    compare_transcriptions(str(REFERENCE_FILES["ground_truth"]), str(TRANSCRIPTIONS_OUTPUT / "test_tiny",), output_file="analysis_results_tiny.txt")
    print("-" * 40)
    print("Analyzing transcriptions for test_base...")
    print("-" * 40)
    compare_transcriptions(str(REFERENCE_FILES["ground_truth"]), str(TRANSCRIPTIONS_OUTPUT / "test_base"), output_file="analysis_results_base.txt")
    print("-" * 40)
    print("Analyzing transcriptions for test_small...")
    print("-" * 40)
    compare_transcriptions(str(REFERENCE_FILES["ground_truth"]), str(TRANSCRIPTIONS_OUTPUT / "test_small"), output_file="analysis_results_small.txt")
    print("-" * 40)
    print("Analyzing transcriptions for test_medium...")
    print("-" * 40)
    compare_transcriptions(str(REFERENCE_FILES["ground_truth"]), str(TRANSCRIPTIONS_OUTPUT / "test_medium"), output_file="analysis_results_medium.txt")