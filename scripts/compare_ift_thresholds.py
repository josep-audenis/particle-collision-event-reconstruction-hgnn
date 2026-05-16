from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


CASES = [
    ("truth_replacement_v0_thr09", Path("LHCb_logs/IFT/version_0"), ""),
    ("dfei_chain_v1_thr09", Path("LHCb_logs/IFT/version_1"), ""),
    ("dfei_chain_v1_thr07", Path("LHCb_logs/IFT/version_1"), "__dfeichain_thr07"),
    ("dfei_chain_v1_thr05", Path("LHCb_logs/IFT/version_1"), "__dfeichain_thr05"),
    ("dfei_chain_v1_e05_n04", Path("LHCb_logs/IFT/version_1"), "__dfeichain_e05_n04"),
    ("dfei_chain_v1_e04_n05", Path("LHCb_logs/IFT/version_1"), "__dfeichain_e04_n05"),
    ("dfei_chain_v1_e06_n05", Path("LHCb_logs/IFT/version_1"), "__dfeichain_e06_n05"),
    ("dfei_chain_v1_e05_n06", Path("LHCb_logs/IFT/version_1"), "__dfeichain_e05_n06"),
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
    }.items():
        match = re.search(pattern, text)
        out[key] = match.group(1) if match else ""
    out["nan_count"] = str(text.lower().count("nan"))
    return out


for name, root, suffix in CASES:
    signal_path = root / f"signal_reco_df_00353851_inclusive{suffix}.csv"
    event_path = root / f"event_reco_df_00353851_inclusive{suffix}.csv"
    ft_path = root / f"info_00353851_inclusive{suffix}_FT.txt"
    if not signal_path.exists():
        continue

    signal = pd.read_csv(signal_path)
    event = pd.read_csv(event_path)
    ft = parse_ft(ft_path)
    neg_signal = int(((signal["NumBkgParticles_noniso"] < 0) & (signal["NumBkgParticles_noniso"] != -999)).sum())
    neg_event = int(
        ((event["NumBackgroundParticlesInEvent"] < 0) & (event["NumBackgroundParticlesInEvent"] != -999)).sum()
    )
    print(f"\n## {name}")
    print(f"signal_rows={len(signal)} event_rows={len(event)} ft_candidates={ft['candidates']}")
    print(
        "perfect={:.4f} all={:.4f} noneiso={:.4f} part={:.4f} notfound={:.4f}".format(
            signal["PerfectReco"].mean(),
            signal["AllParticles"].mean(),
            signal["NoneIso"].mean(),
            signal["PartReco"].mean(),
            signal["NotFound"].mean(),
        )
    )
    print(f"neg_signal_bkg={neg_signal} neg_event_bkg={neg_event} ft_nan={ft['nan_count']}")
    print(f"tagging_power={ft['tagging_power']}")
    print(f"wrong_fraction={ft['wrong_fraction']}")
    print(f"epsilon={ft['epsilon']}")
    print(f"dsquared={ft['dsquared']}")
