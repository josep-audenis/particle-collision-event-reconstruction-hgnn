from __future__ import annotations

import argparse

from legacy.src.data.dataset import summarize_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a serialized HGNN dataset split.")
    parser.add_argument("--split-dir", required=True, help="Directory containing input_*.npy and target_*.npy files")
    parser.add_argument("--max-events", type=int, default=32, help="Maximum number of events to inspect")
    args = parser.parse_args()

    summary = summarize_split(args.split_dir, max_events=args.max_events)
    for key, value in summary.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
