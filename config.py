"""
Configuration and path management for the audio processing pipeline.
This module centralizes all path definitions to keep the codebase organized and maintainable.
"""

import os
import sys
from pathlib import Path

# Project root directory (same directory as config.py)
PROJECT_ROOT = Path(__file__).parent

# Data directories
DATA_ROOT = PROJECT_ROOT / "data"
DATA_DIRS = {
    "raw": DATA_ROOT / "raw",
    "training": DATA_ROOT / "training",
    "reference": DATA_ROOT / "reference",
}

# Input directories
INPUT_DIRS = {
    "raw_audio": DATA_ROOT / "raw" / "10dB",
    "samples": DATA_ROOT / "raw" / "samples",
}

# Output directories
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
OUTPUT_DIRS = {
    "audio": OUTPUT_ROOT / "audio",
    "visualizations": OUTPUT_ROOT / "visualizations",
    "transcriptions": OUTPUT_ROOT / "transcriptions",
    "analysis": OUTPUT_ROOT / "analysis",
}

# Legacy/reference directories (for backward compatibility)
LEGACY_ROOT = PROJECT_ROOT / "legacy"
LEGACY_DIRS = {
    "enhanced_results": LEGACY_ROOT / "enhanced_results",
    "transcriptions_10dB": LEGACY_ROOT / "transcriptions_10dB",
    "transcriptions_cleaned": LEGACY_ROOT / "transcriptions_10dB_cleaned",
}

# Reference files
REFERENCE_FILES = {
    "ground_truth": DATA_ROOT / "reference" / "transcriptions_true.txt",
}

def ensure_output_dirs():
    """Create all output directories if they don't exist."""
    for dir_name, dir_path in OUTPUT_DIRS.items():
        dir_path.mkdir(parents=True, exist_ok=True)

# Output file paths
AUDIO_OUTPUT = OUTPUT_DIRS["audio"] / "final_output.wav"
TRANSCRIPTION_OUTPUT = OUTPUT_DIRS["audio"] / "final_output_transcription.txt"
PERF_SUMMARY_OUTPUT = OUTPUT_DIRS["visualizations"] / "performance_summary.png"
FILE_BREAKDOWN_OUTPUT = OUTPUT_DIRS["visualizations"] / "file_breakdown.png"
TRANSCRIPTIONS_OUTPUT = OUTPUT_DIRS["transcriptions"]
ANALYSIS_OUTPUT = OUTPUT_DIRS["analysis"]

if __name__ == "__main__":
    print("Output directory structure:")
    for dir_name, dir_path in OUTPUT_DIRS.items():
        print(f"  {dir_name}: {dir_path}")
