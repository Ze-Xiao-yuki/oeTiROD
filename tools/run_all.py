"""Launch all ten baselines as separate processes with identical config."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from methods.registry import METHODS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "tirod.yml"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for name in METHODS:
        command = [sys.executable, str(ROOT / "run.py"), "--config", args.config, "--method", name, "--device", args.device]
        if args.dry_run:
            command.append("--dry-run")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
