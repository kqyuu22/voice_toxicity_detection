import argparse
import subprocess
import sys
from pathlib import Path

from config import VAD_THRESHOLD_FILE


def run_step(name, command, cwd, timeout=None, continue_on_timeout=False):
    print("\n" + "=" * 60)
    print(f"STEP: {name}")
    print("=" * 60)
    print("Command:", " ".join(command))

    try:
        completed = subprocess.run(command, cwd=str(cwd), timeout=timeout)
    except subprocess.TimeoutExpired:
        if continue_on_timeout:
            print(f"\n{name} finished after {timeout} seconds. Continuing...")
            return
        raise

    if completed.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {completed.returncode}.")


def ensure_venv_python(project_root):
    venv_python = project_root / "venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        return

    try:
        using_venv_python = Path(sys.executable).resolve() == venv_python.resolve()
    except OSError:
        using_venv_python = False

    if using_venv_python:
        return

    print(f"Using virtual environment interpreter: {venv_python}")
    command = [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]]
    completed = subprocess.run(command, cwd=str(project_root))
    sys.exit(completed.returncode)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the audio recording, transcription, and toxicity prediction pipeline."
    )
    parser.add_argument(
        "--skip-threshold",
        action="store_true",
        help="Skip the microphone threshold monitoring step.",
    )
    parser.add_argument(
        "--threshold-seconds",
        type=int,
        default=5,
        help="Seconds to run threshold_testing.py before continuing.",
    )
    parser.add_argument(
        "--skip-predict",
        action="store_true",
        help="Only record and transcribe; do not run toxicity prediction.",
    )
    parser.add_argument(
        "--record-timeout-seconds",
        type=int,
        default=120,
        help="Maximum seconds to wait for recorder.py before failing.",
    )
    parser.add_argument(
        "--predict-threshold",
        type=float,
        default=None,
        help="Forward a custom toxicity threshold to predict_toxicity.py.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Forward --json to predict_toxicity.py.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    ensure_venv_python(project_root)
    pipeline_dir = project_root / "src" / "pipeline"
    evaluation_dir = project_root / "src" / "evaluation"
    fallback_flag = project_root / "outputs" / "pipeline_results" / "used_dataset_fallback.flag"

    try:
        if not args.skip_threshold and args.threshold_seconds > 0:
            run_step(
                name="Threshold test",
                command=[
                    sys.executable,
                    str(evaluation_dir / "threshold_testing.py"),
                    "--duration",
                    str(args.threshold_seconds),
                    "--output",
                    str(VAD_THRESHOLD_FILE),
                ],
                cwd=project_root,
                timeout=args.threshold_seconds + 2,
                continue_on_timeout=True,
            )

        run_step(
            name="Record audio",
            command=[sys.executable, str(pipeline_dir / "record.py")],
            cwd=project_root,
            timeout=args.record_timeout_seconds,
        )

        if fallback_flag.exists():
            source = fallback_flag.read_text(encoding="utf-8", errors="ignore").strip()
            print("\nRecorder timeout fallback was used.")
            if source:
                print(f"Fallback source: {source}")
            print("Skipping transcription step and using prepared transcript output.")
            fallback_flag.unlink()
        else:
            run_step(
                name="Transcribe audio",
                command=[sys.executable, str(pipeline_dir / "transcribe.py")],
                cwd=project_root,
            )

        if not args.skip_predict:
            predict_command = [sys.executable, str(pipeline_dir / "predict_toxicity.py")]
            if args.predict_threshold is not None:
                predict_command.extend(["--threshold", str(args.predict_threshold)])
            if args.json:
                predict_command.append("--json")

            run_step(
                name="Predict toxicity",
                command=predict_command,
                cwd=project_root,
            )

    except KeyboardInterrupt:
        print("\nPipeline stopped by user.")
        sys.exit(130)
    except Exception as exc:
        print(f"\nPipeline failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
