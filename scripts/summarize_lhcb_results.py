from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def summarize_csv(path: Path) -> None:
    df = pd.read_csv(path)
    print(f"- {path}: rows={len(df)} cols={len(df.columns)}")

    for col in [
        "NumBackgroundParticlesInEvent",
        "NumBkgParticles_noniso",
        "SigMatch",
        "PerfectReco",
        "AllParticles",
        "NoneIso",
        "PartReco",
        "NotFound",
    ]:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if col in {"NumBackgroundParticlesInEvent", "NumBkgParticles_noniso"}:
            neg = int(((values < 0) & (values != -999)).sum())
            print(f"  {col}: min={values.min()} max={values.max()} neg_ex_sentinel={neg}")
        else:
            print(f"  {col}: sum={values.sum()} mean={values.mean():.6g}")


def summarize_metrics(path: Path) -> None:
    if not path.exists():
        return
    df = pd.read_csv(path)
    print(f"- {path}: rows={len(df)}")
    if df.empty:
        return
    cols = [
        col
        for col in df.columns
        if any(key in col.lower() for key in ["loss", "acc", "auc", "eff", "power"])
    ]
    if cols:
        print(df[cols].tail(3).to_string(index=False))


def summarize_ft(path: Path) -> None:
    text = path.read_text(errors="ignore")
    print(f"- {path}: nan={text.lower().count('nan')} chars={len(text)}")
    for line in text.splitlines()[:18]:
        print(f"  {line}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="LHCb_logs")
    args = parser.parse_args()

    root = Path(args.log_dir)
    for model in ["DFEI", "IFT"]:
        version_dir = root / model / "version_0"
        print(f"\n## {model}")
        if not version_dir.exists():
            print(f"- missing {version_dir}")
            continue
        for path in sorted(version_dir.glob("*.csv")):
            summarize_csv(path)
        summarize_metrics(version_dir / "metrics.csv")

    print("\n## FT reports")
    for path in sorted(root.glob("IFT/version_0/info_*_FT.txt")):
        summarize_ft(path)


if __name__ == "__main__":
    main()
