from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


IFT_CASES = [
    ("50k truth/PV replacement", Path("LHCb_logs/IFT/version_0"), ""),
    ("50k DFEI-chain 0.5/0.5", Path("LHCb_logs/IFT/version_1"), "__dfeichain_thr05"),
    ("100k truth/PV replacement", Path("LHCb_logs/IFT/version_2"), "__cache100000_truth"),
    ("100k DFEI-chain 0.5/0.5", Path("LHCb_logs/IFT/version_3"), "__cache100000_dfeichain_thr05"),
    ("100k DFEI-chain edge=0.5 node=0.4", Path("LHCb_logs/IFT/version_3"), "__cache100000_dfeichain_e05_n04"),
    ("100k DFEI-chain edge=0.4 node=0.5", Path("LHCb_logs/IFT/version_3"), "__cache100000_dfeichain_e04_n05"),
    ("100k DFEI-chain edge=0.6 node=0.5", Path("LHCb_logs/IFT/version_3"), "__cache100000_dfeichain_e06_n05"),
    ("100k DFEI-chain edge=0.5 node=0.6", Path("LHCb_logs/IFT/version_3"), "__cache100000_dfeichain_e05_n06"),
    ("100k DFEI-chain trained edge=0.4 node=0.5", Path("LHCb_logs/IFT/version_4"), "__cache100000_dfeichain_trained_e04_n05"),
    ("100k DFEI-chain trained edge=0.4 node=0.4", Path("LHCb_logs/IFT/version_4"), "__cache100000_dfeichain_trained_e04_n04"),
    ("100k DFEI-chain trained edge=0.3 node=0.5", Path("LHCb_logs/IFT/version_4"), "__cache100000_dfeichain_trained_e03_n05"),
    ("100k DFEI-chain trained edge=0.4 node=0.5 repeat", Path("LHCb_logs/IFT/version_4"), "__cache100000_dfeichain_trained_e04_n05_repeat"),
    ("100k DFEI-chain trained edge=0.5 node=0.5", Path("LHCb_logs/IFT/version_4"), "__cache100000_dfeichain_trained_e05_n05"),
    ("100k DFEI-chain trained edge=0.4 node=0.6", Path("LHCb_logs/IFT/version_4"), "__cache100000_dfeichain_trained_e04_n06"),
]

DFEI_CASES = [
    ("50k DFEI", Path("LHCb_logs/DFEI/version_0"), ""),
    ("100k DFEI", Path("LHCb_logs/DFEI/version_1"), ""),
]


def parse_ft(path: Path) -> dict[str, str]:
    text = path.read_text(errors="ignore")
    out = {}
    for key, pattern in {
        "candidates": r"Tagging performance for .*\((\d+)\):",
        "tagging_power": r"Tagging power\s+:\s+\(([^)]+)\)%",
        "wrong_fraction": r"Wrong fraction\s+:\s+\(([^)]+)\)%",
        "epsilon": r"Epsilon\s+:\s+\(([^)]+)\)%",
        "dsquared": r"Dsquared\s+:\s+\(([^)]+)\)%",
        "auc": r"AUC\s+:\s+([0-9.]+|nan)",
        "loss": r"Loss\s+:\s+([0-9.]+|nan)",
    }.items():
        match = re.search(pattern, text)
        out[key] = match.group(1) if match else ""
    out["nan_count"] = str(text.lower().count("nan"))
    return out


def summarize_signal(path: Path) -> dict[str, float | int]:
    signal = pd.read_csv(path)
    return {
        "rows": len(signal),
        "sigmatch": int(signal["SigMatch"].sum()) if "SigMatch" in signal else 0,
        "perfect": float(signal["PerfectReco"].mean()) if "PerfectReco" in signal else 0.0,
        "all": float(signal["AllParticles"].mean()) if "AllParticles" in signal else 0.0,
        "noneiso": float(signal["NoneIso"].mean()) if "NoneIso" in signal else 0.0,
        "part": float(signal["PartReco"].mean()) if "PartReco" in signal else 0.0,
        "notfound": float(signal["NotFound"].mean()) if "NotFound" in signal else 0.0,
        "neg_bkg": int(
            ((signal["NumBkgParticles_noniso"] < 0) & (signal["NumBkgParticles_noniso"] != -999)).sum()
        )
        if "NumBkgParticles_noniso" in signal
        else 0,
    }


def best_metric(root: Path, col: str) -> str:
    metrics = root / "metrics.csv"
    if not metrics.exists():
        return ""
    df = pd.read_csv(metrics)
    if col not in df:
        return ""
    values = pd.to_numeric(df[col], errors="coerce").dropna()
    return f"{values.min():.6f}" if len(values) else ""


print("## DFEI")
for name, root, suffix in DFEI_CASES:
    signal_path = root / f"signal_reco_df_00353851_inclusive{suffix}.csv"
    if not signal_path.exists():
        continue
    s = summarize_signal(signal_path)
    print(
        f"{name}: rows={s['rows']} best_val_combined_loss={best_metric(root, 'val_combined_loss')} "
        f"perfect={s['perfect']:.4f} all={s['all']:.4f} noneiso={s['noneiso']:.4f} "
        f"part={s['part']:.4f} notfound={s['notfound']:.4f} neg_bkg={s['neg_bkg']}"
    )

print("\n## IFT")
for name, root, suffix in IFT_CASES:
    signal_path = root / f"signal_reco_df_00353851_inclusive{suffix}.csv"
    ft_path = root / f"info_00353851_inclusive{suffix}_FT.txt"
    if not signal_path.exists() or not ft_path.exists():
        continue
    s = summarize_signal(signal_path)
    ft = parse_ft(ft_path)
    print(
        f"{name}: rows={s['rows']} ft_candidates={ft['candidates']} "
        f"best_val_ft_loss={best_metric(root, 'val_ft_loss')} "
        f"perfect={s['perfect']:.4f} all={s['all']:.4f} noneiso={s['noneiso']:.4f} "
        f"part={s['part']:.4f} notfound={s['notfound']:.4f} "
        f"tagging_power={ft['tagging_power']} wrong_fraction={ft['wrong_fraction']} "
        f"epsilon={ft['epsilon']} auc={ft['auc']} loss={ft['loss']} "
        f"ft_nan={ft['nan_count']} neg_bkg={s['neg_bkg']}"
    )
