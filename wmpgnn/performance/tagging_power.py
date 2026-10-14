import os

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import log_loss as sklearn_log_loss
    from sklearn.metrics import roc_auc_score as sklearn_roc_auc_score
except ImportError:
    sklearn_log_loss = None
    sklearn_roc_auc_score = None

try:
    from wmpgnn.performance.tagging_power_classes import *
except ImportError:
    from tagging_power_classes import *


def write_tagging_power(file, metric, label):
    file.write(f"{'=' * 20}\n")
    file.write(f"{label}\n")
    file.write(f"Tagging power  : ({metric.power[0]:.4f} +/- {metric.power[1]:.4f})%\n")
    file.write(f"Wrong fraction : ({metric.wrong_fraction[0]:.4f} +/- {metric.wrong_fraction[1]:.4f})%\n")
    file.write(f"Epsilon        : ({metric.epsilon[0]:.4f} +/- {metric.epsilon[1]:.4f})%\n")
    file.write(f"Dsquared       : ({metric.d_squared[0]:.4f} +/- {metric.d_squared[1]:.4f})%\n")


def write_correctness_ratio(file, ratio, frag_results=None):
    file.write(f"Correctness ratio: ({ratio[0] * 100:.2f} +/- {ratio[1] * 100:.2f})%\n")
    if frag_results is not None:
        file.write(f"With fragmentation: ({frag_results['with_frag'][0] * 100:.2f} +/- "
                   f"{frag_results['with_frag'][1] * 100:.2f})%\n")
        file.write(f"Without fragmentation: ({frag_results['without_frag'][0] * 100:.2f} +/- "
                   f"{frag_results['without_frag'][1] * 100:.2f})%\n")


def compute_benchmark_metrics(df: pd.DataFrame) -> dict[str, float | int]:
    """Compute DeepSets-comparable IFT classification metrics on B-candidate rows."""
    required = {"B_id", "ft_b_score", "ft_bbar_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing benchmark metric columns: {sorted(missing)}")

    metric_df = df.loc[:, ["B_id", "ft_b_score", "ft_bbar_score"]].dropna()
    metric_df = metric_df[np.sign(metric_df["B_id"]) != 0]
    if metric_df.empty:
        return {"rows": 0, "auc": np.nan, "loss": np.nan}

    y_true = (np.sign(metric_df["B_id"]) == 1).astype(int).to_numpy()
    y_score = metric_df["ft_bbar_score"].to_numpy(dtype=float)
    y_prob = metric_df[["ft_b_score", "ft_bbar_score"]].to_numpy(dtype=float)
    y_prob = np.clip(y_prob, 1e-12, 1.0)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)

    auc = np.nan
    if np.unique(y_true).size == 2:
        if sklearn_roc_auc_score is not None:
            auc = sklearn_roc_auc_score(y_true, y_score)
        else:
            auc = binary_auc(y_true, y_score)

    return {
        "rows": int(len(metric_df)),
        "auc": float(auc),
        "loss": float(cross_entropy_loss(y_true, y_prob)),
    }


def binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)

    sorted_scores = y_score[order]
    unique_scores, starts, counts = np.unique(sorted_scores, return_index=True, return_counts=True)
    del unique_scores
    for start, count in zip(starts, counts):
        if count > 1:
            tied = order[start:start + count]
            ranks[tied] = ranks[tied].mean()

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos == 0 or n_neg == 0:
        return np.nan

    rank_sum_pos = np.sum(ranks[y_true == 1])
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def cross_entropy_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if sklearn_log_loss is not None:
        return sklearn_log_loss(y_true, y_prob, labels=[0, 1])
    return -float(np.mean(np.log(y_prob[np.arange(len(y_true)), y_true])))


def write_benchmark_metrics(file, metrics: dict[str, float | int], label: str):
    file.write(f"{'=' * 20}\n")
    file.write(f"{label}\n")
    file.write(f"Rows           : {metrics['rows']}\n")
    file.write(f"AUC            : {metrics['auc']:.5f}\n")
    file.write(f"Loss           : {metrics['loss']:.5f}\n")


def write_selection_metadata(file, raw_df: pd.DataFrame, processed_df: pd.DataFrame, full_df: pd.DataFrame):
    raw_signal = raw_df[raw_df["SigMatch"] == 1] if "SigMatch" in raw_df else raw_df.iloc[0:0]
    raw_reco_signal = raw_signal[raw_signal["AllParticles"] == 1] if "AllParticles" in raw_signal else raw_signal
    file.write(f"{'=' * 20}\n")
    file.write("Selection and denominator metadata\n")
    file.write("Metric scope    : reconstructed B-candidate rows after EventReconstruction output\n")
    file.write("TP denominator  : rows in each reported subset, not all generated events\n")
    file.write("AUC/Loss labels : sign(B_id), with positive class mapped to ft_bbar_score\n")
    file.write("Calibration     : uncalibrated model scores; eta = 1 - max(ft_b_score, ft_bbar_score)\n")
    file.write(f"Input rows      : {len(raw_df)}\n")
    file.write(f"Input events    : {raw_df['EventNumber'].nunique() if 'EventNumber' in raw_df else 0}\n")
    file.write(f"Signal rows     : {len(raw_signal)}\n")
    file.write(f"Signal AllParticles rows: {len(raw_reco_signal)}\n")
    file.write(f"Processed rows  : {len(processed_df)}\n")
    file.write(f"Full metric rows: {len(full_df)}\n")


def analyze_tagging_power(df: pd.DataFrame, version: str, signal: str, log_dir: str = "lightning_logs"):
    """
    Main function for tagging power values and plots

    Args:
        df: DataFrame with B meson tagging data
        version: Analysis version identifier
        signal: Signal type identifier
        log_dir: Logging directory
    """
    analyzer = TaggingPowerAnalyzer(version, signal, log_dir)
    classifier = EventClassifier()
    calc = CorrectnessCalculator()


    raw_df = df.copy()
    df = analyzer.process_df(df)
    event_ids, event_counts = classifier.get_event_counts(df)

    file = f"{log_dir}/IFT/version_{version}/info_{signal}_FT.txt"

    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, "w") as f:
        f.write("=" * 50 + "\n")
        f.write("FLAVOUR TAGGING POWER \n")

        """1. Full tagging power (all signal-matched events)"""
        full_df = df[df["SigMatch"] == 1]
        write_selection_metadata(f, raw_df, df, full_df)
        metrics_full = analyzer.compute_tagging_power_per_eta(full_df)
        analyzer.plot_tagging_power(metrics_full, full_df["eta"], "eta", "tagging_power")
        analyzer.plot_tagging_power(metrics_full, full_df["num_pvs"], "npvs", "npvs_tagging_power")
        label = f"Tagging performance for {signal} ({len(full_df)}):"
        write_tagging_power(f, metrics_full, label)
        benchmark_metrics = compute_benchmark_metrics(full_df)
        write_benchmark_metrics(f, benchmark_metrics, "DeepSets-comparable metrics on full signal-matched subset:")

        """2. Single B events"""
        single_b_df = classifier.filter_n_b_events(df, 1, event_ids, event_counts)
        signal_single_b = single_b_df[single_b_df["SigMatch"] == 1]
        metrics_single = analyzer.compute_tagging_power_per_eta(signal_single_b)
        analyzer.plot_tagging_power(metrics_single, signal_single_b["eta"], "eta", "singleB_tagging_power")
        label = f"1 B candidate present in event ({np.sum(event_counts == 1)}):"
        write_tagging_power(f, metrics_single, label)

        ratio = calc.calculate_prediction_correctness(signal_single_b)
        frag_results = calc.calculate_by_fragmentation(signal_single_b)
        write_correctness_ratio(f, ratio, frag_results)

        """3. Two B events (opposite-side tagging possible)"""
        two_b_df = classifier.filter_n_b_events(df, 2, event_ids, event_counts)
        signal_two_b = two_b_df[two_b_df["SigMatch"] == 1]
        metrics_two = analyzer.compute_tagging_power_per_eta(signal_two_b)
        analyzer.plot_tagging_power(metrics_two, signal_two_b["eta"], "eta", "doubleB_tagging_power")
        label = f"2 B candidate present in event ({np.sum(event_counts == 2)}):"
        write_tagging_power(f, metrics_two, label)
        ratio = calc.calculate_prediction_correctness(signal_two_b)
        frag_results = calc.calculate_by_fragmentation(signal_two_b)
        write_correctness_ratio(f, ratio, frag_results)

        """4. B+/- specific analysis"""
        bpm_df = classifier.filter_bpm_events(two_b_df)


        bpm_signal = bpm_df[bpm_df["SigMatch"] == 1]
        metrics_bpm_sig = analyzer.compute_tagging_power_per_eta(bpm_signal)
        analyzer.plot_tagging_power(metrics_bpm_sig, bpm_signal["eta"], "eta", "doubleB_ospm_tagging_power")
        label = f"Signal has OS B+/- Events: {len(bpm_signal)}"
        write_tagging_power(f, metrics_bpm_sig, label)


        bpm_os = bpm_df[bpm_df["SigMatch"] == 0]
        metrics_bpm_os = analyzer.compute_tagging_power_per_eta(bpm_os)
        analyzer.plot_tagging_power(metrics_bpm_os, bpm_os["eta"], "eta", "osB_pm_tagging_power")
        charge_frac, charge_correct_df = calc.verify_charge_reconstruction(bpm_os)
        label = f"OS B+/- Events: {len(bpm_os)}"
        write_tagging_power(f, metrics_bpm_os, label)
        f.write(f"B+/- charge reconstruction correctness: {charge_frac * 100:.2f}%\n")


        metrics_charge = analyzer.compute_tagging_power_per_eta(charge_correct_df)
        analyzer.plot_tagging_power(metrics_charge, charge_correct_df["eta"], "eta", "osB_pm_true_charge_tagging_power")
        label = "Charged correct OS B+/- Events:"
        write_tagging_power(f, metrics_charge, label)

        """5. Bias study of signal"""

        neg_df = full_df[np.sign(full_df["B_id"]) == -1]
        metrics_full = analyzer.compute_tagging_power_per_eta(neg_df)
        analyzer.plot_tagging_power(metrics_full, neg_df["eta"], "eta", f"neg_tagging_power")
        label = f"Tagging performance for negative {signal} ({len(neg_df)}):"
        write_tagging_power(f, metrics_full, label)
        ratio = calc.calculate_prediction_correctness(neg_df)
        write_correctness_ratio(f, ratio)

        neg_single_bool = (single_b_df["SigMatch"] == 1) & (np.sign(single_b_df["B_id"]) == -1)
        neg_sig_df = single_b_df[neg_single_bool]
        metrics_full = analyzer.compute_tagging_power_per_eta(neg_sig_df)
        analyzer.plot_tagging_power(metrics_full, neg_sig_df["eta"], "eta", f"neg_single_tagging_power")
        label = f"Tagging performance for negative single B {signal} ({len(neg_sig_df)}):"
        write_tagging_power(f, metrics_full, label)
        ratio = calc.calculate_prediction_correctness(neg_sig_df)
        write_correctness_ratio(f, ratio)

        neg_double_bool = (two_b_df["SigMatch"] == 1) & (np.sign(two_b_df["B_id"]) == -1)
        neg_sig_df = two_b_df[neg_double_bool]
        metrics_full = analyzer.compute_tagging_power_per_eta(neg_sig_df)
        analyzer.plot_tagging_power(metrics_full, neg_sig_df["eta"], "eta", f"neg_double_tagging_power")
        label = f"Tagging performance for negative double B {signal} ({len(neg_sig_df)}):"
        write_tagging_power(f, metrics_full, label)
        ratio = calc.calculate_prediction_correctness(neg_sig_df)
        write_correctness_ratio(f, ratio)


        pos_df = full_df[np.sign(full_df["B_id"]) == 1]
        metrics_full = analyzer.compute_tagging_power_per_eta(pos_df)
        analyzer.plot_tagging_power(metrics_full, pos_df["eta"], "eta", f"pos_tagging_power")
        label = f"Tagging performance for positive {signal} ({len(pos_df)}):"
        write_tagging_power(f, metrics_full, label)
        ratio = calc.calculate_prediction_correctness(pos_df)
        write_correctness_ratio(f, ratio)

        pos_single_bool = (single_b_df["SigMatch"] == 1) & (np.sign(single_b_df["B_id"]) == 1)
        pos_sig_df = single_b_df[pos_single_bool]
        metrics_full = analyzer.compute_tagging_power_per_eta(pos_sig_df)
        analyzer.plot_tagging_power(metrics_full, pos_sig_df["eta"], "eta", f"pos_single_tagging_power")
        label = f"Tagging performance for positive single B {signal} ({len(pos_sig_df)}):"
        write_tagging_power(f, metrics_full, label)
        ratio = calc.calculate_prediction_correctness(pos_sig_df)
        write_correctness_ratio(f, ratio)

        pos_double_bool = (two_b_df["SigMatch"] == 1) & (np.sign(two_b_df["B_id"]) == 1)
        pos_sig_df = two_b_df[pos_double_bool]
        metrics_full = analyzer.compute_tagging_power_per_eta(pos_sig_df)
        analyzer.plot_tagging_power(metrics_full, pos_sig_df["eta"], "eta", f"pos_double_tagging_power")
        label = f"Tagging performance for positive double B {signal} ({len(pos_sig_df)}):"
        write_tagging_power(f, metrics_full, label)
        ratio = calc.calculate_prediction_correctness(pos_sig_df)
        write_correctness_ratio(f, ratio)

        """7. Performance based on OS reconstruction type"""
        os_incl_df = two_b_df[two_b_df["SigMatch"] != 1]
        os_incl_df.loc[os_incl_df["PerfectReco"] == 1, "AllParticles"] = 0
        sig_df = two_b_df[two_b_df["SigMatch"] == 1]
        conditions = ["PerfectReco", "AllParticles", "NoneIso", "PartReco", "NotFound"]
        for condition in conditions:
            evts = os_incl_df[os_incl_df[condition] == 1]["EventNumber"]
            usage_df = sig_df[sig_df["EventNumber"].isin(evts)]
            metrics_two = analyzer.compute_tagging_power_per_eta(usage_df)
            analyzer.plot_tagging_power(metrics_two, usage_df["eta"], "eta", f"doubleB_OS{condition}_tagging_power")
            label = f"2 B candidate present in event with OS being {condition} ({len(usage_df)}):"
            write_tagging_power(f, metrics_two, label)
            ratio = calc.calculate_prediction_correctness(usage_df)
            frag_results = calc.calculate_by_fragmentation(usage_df)
            write_correctness_ratio(f, ratio, frag_results)


if __name__ == "__main__":
    df = pd.read_csv(
        "/eos/user/y/yukaiz/DFEI_IFT/IFT_training/wmpgnn/analysis/LHCb_logs/IFT/version_1/signal_df_00299103_Bs_Jpsiphi.csv")
    analyze_tagging_power(df, "-1", "Bs_Jpsiphi")
