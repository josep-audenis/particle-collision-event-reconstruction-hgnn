from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def summarize_version(root: Path, version: int):
    path = root / "IFT" / f"version_{version}" / "metrics.csv"
    print(f"=== IFT version_{version} ===")
    if not path.exists():
        print("missing metrics.csv")
        return

    df = pd.read_csv(path)
    for col in [
        "val_ft_loss",
        "val_ft_auc",
        "val_ft_tagging_power_proxy",
        "val_ft_wrong_fraction_proxy",
    ]:
        if col not in df:
            continue
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if values.empty:
            print(f"{col}: empty")
            continue
        print(
            f"{col}: min={values.min():.6f} max={values.max():.6f} "
            f"last={values.iloc[-1]:.6f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="LHCb_logs")
    parser.add_argument("versions", nargs="*", type=int)
    args = parser.parse_args()

    root = Path(args.log_dir)
    versions = args.versions
    if not versions:
        versions = sorted(
            int(path.name.removeprefix("version_"))
            for path in (root / "IFT").glob("version_*")
            if path.name.removeprefix("version_").isdigit()
        )

    for version in versions:
        summarize_version(root, version)


if __name__ == "__main__":
    main()
