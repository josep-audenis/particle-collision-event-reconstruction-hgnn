from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


CASES = [
    ("v3_old_best", Path("LHCb_logs/IFT/version_3"), "__cache100000_dfeichain_e04_n05"),
    ("v4_trained_best", Path("LHCb_logs/IFT/version_4"), "__cache100000_dfeichain_trained_e04_n06"),
]


def numeric_summary(df: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        q = s.quantile([0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0])
        print(
            f"  {col}: n={len(s)} mean={s.mean():.6g} std={s.std():.6g} "
            f"min={q.loc[0.0]:.6g} p01={q.loc[0.01]:.6g} p10={q.loc[0.1]:.6g} "
            f"p50={q.loc[0.5]:.6g} p90={q.loc[0.9]:.6g} p99={q.loc[0.99]:.6g} max={q.loc[1.0]:.6g}"
        )


def binary_rates(df: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            print(f"  {col}: mean={s.mean():.6g} sum={s.sum():.0f} n={s.notna().sum()}")


def find_cols(df: pd.DataFrame, needles: tuple[str, ...]) -> list[str]:
    found = []
    for col in df.columns:
        low = col.lower()
        if any(n in low for n in needles):
            found.append(col)
    return found


def ft_text_summary(path: Path) -> None:
    text = path.read_text(errors="ignore")
    print(f"  ft_nan_count={text.lower().count('nan')}")
    for line in text.splitlines():
        low = line.lower()
        if any(key in low for key in ("tagging power", "wrong fraction", "epsilon", "dsquared")):
            print(f"  {line.strip()}")


for name, root, suffix in CASES:
    print(f"\n## {name}")
    event_path = root / f"event_reco_df_00353851_inclusive{suffix}.csv"
    signal_path = root / f"signal_reco_df_00353851_inclusive{suffix}.csv"
    ft_path = root / f"info_00353851_inclusive{suffix}_FT.txt"

    event = pd.read_csv(event_path)
    signal = pd.read_csv(signal_path)
    print(f"event_rows={len(event)} signal_rows={len(signal)}")
    print(f"event_cols={list(event.columns)}")
    print(f"signal_cols={list(signal.columns)}")

    print("event binary rates:")
    binary_rates(event, ["PerfectReco", "AllParticles", "NoneIso", "PartReco", "NotFound", "SigMatch"])
    print("signal binary rates:")
    binary_rates(signal, ["PerfectReco", "AllParticles", "NoneIso", "PartReco", "NotFound", "SigMatch"])

    print("event score-like cols:")
    score_cols = find_cols(event, ("prob", "score", "pred", "tag", "eta", "omega", "mistag", "ft", "bq", "q"))
    numeric_summary(event, score_cols)

    print("signal score-like cols:")
    score_cols = find_cols(signal, ("prob", "score", "pred", "tag", "eta", "omega", "mistag", "ft", "bq", "q"))
    numeric_summary(signal, score_cols)

    print("background/count cols:")
    numeric_summary(signal, find_cols(signal, ("bkg", "num", "track", "particle")))

    print("ft text:")
    ft_text_summary(ft_path)
