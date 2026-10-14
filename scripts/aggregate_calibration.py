import pandas as pd, numpy as np, sys

d = pd.read_csv(sys.argv[1], sep="\t")
# dedup (exp,channel,seed) keeping last occurrence (latest re-run)
d = d.drop_duplicates(subset=["exp", "channel", "seed"], keep="last")

order = ["baseline", "agg_mean", "agg_attention", "nblocks3", "nblocks4",
         "loss_focal", "loss_allclass", "lca_soft", "data_200k", "tracks_all",
         "lca_soft_nb3", "data200k_lca_nb3", "data200k_tracks_lca_nb3"]

def agg(g):
    return pd.Series({
        "nseed": len(g),
        "flat_tp": g["flat_tp"].mean(), "flat_tp_sd": g["flat_tp"].std(ddof=0),
        "cal_tp": g["cal_tp"].mean(), "cal_tp_sd": g["cal_tp"].std(ddof=0),
        "fc_flat": g["fc_flat"].mean(), "fc_flat_sd": g["fc_flat"].std(ddof=0),
        "fc_cal": g["fc_cal"].mean(), "fc_cal_sd": g["fc_cal"].std(ddof=0),
        "auc": g["auc"].mean(),
    })

a = d.groupby(["exp", "channel"]).apply(agg, include_groups=False).reset_index()
a["exp"] = pd.Categorical(a["exp"], categories=order, ordered=True)
a = a.sort_values(["exp", "channel"])

print(f"{'exp':<26}{'ch':<4}{'ns':<3}{'flat_tp':<14}{'cal_tp':<14}{'fc_flat':<14}{'fc_cal':<14}{'auc':<7}")
for _, r in a.iterrows():
    print(f"{r['exp']:<26}{r['channel']:<4}{int(r['nseed']):<3}"
          f"{r['flat_tp']:.2f}±{r['flat_tp_sd']:.2f}    "
          f"{r['cal_tp']:.2f}±{r['cal_tp_sd']:.2f}    "
          f"{r['fc_flat']:.2f}±{r['fc_flat_sd']:.2f}    "
          f"{r['fc_cal']:.2f}±{r['fc_cal_sd']:.2f}    "
          f"{r['auc']:.3f}")
