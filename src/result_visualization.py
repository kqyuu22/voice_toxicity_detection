import os
import re
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PERF_SUMMARY_OUTPUT, FILE_BREAKDOWN_OUTPUT, TRANSCRIPTIONS_OUTPUT, ensure_output_dirs

def visualize_results(folder_path):
    ensure_output_dirs()
    data = []

    # 1. Parse the files
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".txt") or filename.endswith(".text"):
            with open(os.path.join(folder_path, filename), "r") as f:
                content = f.read()
                
                # Extract numbers using regex
                conf_match = re.search(r"CONFIDENCE:\s*([\d\.]+)", content)
                lat_match = re.search(r"LATENCY:\s*([\d\.]+)s", content)
                
                if conf_match and lat_match:
                    data.append({
                        "File": filename.replace(".txt", "").replace(".text", ""),
                        "Confidence": float(conf_match.group(1)),
                        "Latency": float(lat_match.group(1))
                    })

    df = pd.DataFrame(data)
    if df.empty:
        print("No valid data found in folder.")
        return

    # 2. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Chart A: Confidence Distribution
    axes[0].hist(df['Confidence'], bins=10, color='skyblue', edgecolor='black')
    axes[0].set_title('Model Confidence Distribution')
    axes[0].set_xlabel('Confidence Score')
    axes[0].set_ylabel('Number of Files')

    # Chart B: Confidence vs Latency
    axes[1].scatter(df['Latency'], df['Confidence'], color='coral', alpha=0.7)
    axes[1].set_title('Confidence vs. Latency')
    axes[1].set_xlabel('Latency (seconds)')
    axes[1].set_ylabel('Confidence')

    plt.tight_layout()
    plt.savefig(str(PERF_SUMMARY_OUTPUT))
    
    # Chart C: Per-file breakdown
    plt.figure(figsize=(12, 5))
    df_sorted = df.sort_values('Confidence')
    plt.bar(df_sorted['File'], df_sorted['Confidence'], color='teal')
    plt.xticks(rotation=90)
    plt.title('Individual File Confidence (Sorted)')
    plt.ylabel('Confidence')
    plt.tight_layout()
    plt.savefig(str(FILE_BREAKDOWN_OUTPUT))
    
    print(f"Visualizations saved to:")
    print(f"  - {PERF_SUMMARY_OUTPUT}")
    print(f"  - {FILE_BREAKDOWN_OUTPUT}")

if __name__ == "__main__":
    ensure_output_dirs()
    visualize_results(str(TRANSCRIPTIONS_OUTPUT))