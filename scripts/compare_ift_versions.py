from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def first_line(text: str, prefix: str) -> str:
    match = re.search(rf"^{re.escape(prefix)}.*$", text, flags=re.MULTILINE)
    return match.group(0) if match else ""


for version in [0, 1]:
    root = Path(f"LHCb_logs/IFT/version_{version}")
    print(f"\n## IFT version_{version}")
    metrics = pd.read_csv(root / "metrics.csv")
    val = pd.to_numeric(metrics["val_ft_loss"], errors="coerce").dropna()
    train = pd.to_numeric(metrics["train_ft_loss"], errors="coerce").dropna()
    print(f"metrics_rows={len(metrics)} best_val_ft_loss={val.min() if len(val) else None} last_train_ft_loss={train.iloc[-1] if len(train) else None}")

    signal = pd.read_csv(root / "signal_reco_df_00353851_inclusive.csv")
    neg_bkg = int(((signal["NumBkgParticles_noniso"] < 0) & (signal["NumBkgParticles_noniso"] != -999)).sum())
    print(
        "signal_rows={rows} sigmatch={sigmatch} perfect={perfect:.6f} all={allp:.6f} "
        "noneiso={noneiso:.6f} part={part:.6f} notfound={notfound:.6f} neg_bkg={neg_bkg}".format(
            rows=len(signal),
            sigmatch=int(signal["SigMatch"].sum()),
            perfect=signal["PerfectReco"].mean(),
            allp=signal["AllParticles"].mean(),
            noneiso=signal["NoneIso"].mean(),
            part=signal["PartReco"].mean(),
            notfound=signal["NotFound"].mean(),
            neg_bkg=neg_bkg,
        )
    )

    event = pd.read_csv(root / "event_reco_df_00353851_inclusive.csv")
    neg_event = int(
        ((event["NumBackgroundParticlesInEvent"] < 0) & (event["NumBackgroundParticlesInEvent"] != -999)).sum()
    )
    print(f"event_rows={len(event)} neg_event_bkg={neg_event}")

    ft = (root / "info_00353851_inclusive_FT.txt").read_text(errors="ignore")
    print(first_line(ft, "Tagging performance for"))
    print(first_line(ft, "Tagging power"))
    print(first_line(ft, "Wrong fraction"))
    print(first_line(ft, "Epsilon"))
    print(first_line(ft, "Dsquared"))
