import subprocess
import sys
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parents[2]
    classifier_script = project_root / "models" / "text_classification.py"

    command = [
        sys.executable,
        str(classifier_script),
        "predict",
        *sys.argv[1:],
    ]

    completed = subprocess.run(command, cwd=str(project_root))
    sys.exit(completed.returncode)


if __name__ == "__main__":
    main()
