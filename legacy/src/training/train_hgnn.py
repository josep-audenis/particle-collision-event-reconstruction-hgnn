from __future__ import annotations

import argparse

from legacy.src.training.config import load_config
from legacy.src.training.trainer import train_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the particle-collision HGNN.")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--dry-run", action="store_true", help="Validate the config and print the planned experiment.")
    args = parser.parse_args()

    results = train_experiment(load_config(args.config), dry_run=args.dry_run)
    print(results["summary"] if "summary" in results else results)


if __name__ == "__main__":
    main()

