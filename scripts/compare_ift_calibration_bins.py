from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


CASES = [
    ("v3_old_best", Path("LHCb_logs/IFT/version_3"), "__cache100000_dfeichain_e04_n05"),
    ("v4_trained_best", Path("LHCb_logs/IFT/version_4"), "__cache100000_dfeichain_trained_e04_n06"),
]

ETA_BINS = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0])


def process(root: Path, suffix: str) -> pd.DataFrame:
    df = pd.read_csv(root / f"signal_reco_df_00353851_inclusive{suffix}.csv")
    sig = df[df["SigMatch"] == 1].copy()
    reco_events = sig[sig["AllParticles"] == 1]["EventNumber"].unique()
    sig = sig[sig["EventNumber"].isin(reco_events)].copy()
    sig["eta"] = 1 - sig[["ft_b_score", "ft_bbar_score"]].max(axis=1)
    sig["pred"] = np.argmax(sig[["ft_b_score", "ft_bbar_score"]].to_numpy(), axis=1) * 2 - 1
    sig["truth"] = np.sign(sig["B_id"])
    sig["right"] = sig["pred"] == sig["truth"]
    sig["margin"] = (sig["ft_b_score"] - sig["ft_bbar_score"]).abs()
    sig["is_pos"] = sig["truth"] == 1
    return sig


for name, root, suffix in CASES:
    df = process(root, suffix)
    print(f"\n## {name}")
    print(f"processed_rows={len(df)} right={df['right'].sum()} wrong={(~df['right']).sum()} wrong_frac={(~df['right']).mean()*100:.4f}%")
    print(f"truth_pos={df['is_pos'].mean()*100:.4f}% pred_pos={(df['pred'].eq(1)).mean()*100:.4f}%")
    print(f"eta_mean={df['eta'].mean():.6f} eta_p50={df['eta'].median():.6f} margin_mean={df['margin'].mean():.6f}")
    print("eta_bin,n,right,wrong,wrong_frac,tag_power,truth_pos,pred_pos,eta_mean,margin_mean")
    for lo, hi in zip(ETA_BINS[:-1], ETA_BINS[1:]):
        b = df[(df["eta"] >= lo) & (df["eta"] < hi)]
        n = len(b)
        if n == 0:
            print(f"{lo:.1f}-{hi:.1f},0,0,0,0,0,0,0,0,0")
            continue
        wrong = int((~b["right"]).sum())
        right = int(b["right"].sum())
        wf = wrong / n
        power = (1 - 2 * wf) ** 2 * 100
        print(
            f"{lo:.1f}-{hi:.1f},{n},{right},{wrong},{wf*100:.4f},{power:.4f},"
            f"{b['is_pos'].mean()*100:.4f},{b['pred'].eq(1).mean()*100:.4f},"
            f"{b['eta'].mean():.6f},{b['margin'].mean():.6f}"
        )
