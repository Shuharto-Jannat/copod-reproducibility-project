"""Create verified APS report figures from saved scores, summaries and resamples.

Run: python scripts/plot_aps_full_results.py
No model fitting or bootstrap resampling is performed. Existing differing files
are never overwritten; identical figures and curve tables can be verified again.
"""

from pathlib import Path
from importlib.metadata import version
import argparse
import hashlib
import io
import json
import platform
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve

import summarise_aps_full_results as summary
import bootstrap_aps_full_results as bootstrap


ROOT = Path(__file__).resolve().parents[1]
require = summary.require
COLORS = {"pyod_batch": "#0072B2", "fixed_training_reference": "#D55E00", "isolation_forest": "#00866A"}
LABELS = {"pyod_batch": "Batch COPOD", "fixed_training_reference": "Training-only COPOD", "isolation_forest": "Isolation Forest"}
MARKERS = {"pyod_batch": "o", "fixed_training_reference": "s", "isolation_forest": "^"}
CAPTIONS = {
    "aps_performance_comparison": "Full APS supplied split: 60,000 training and 16,000 test rows, "
        "with 375 positive test labels. Primary preprocessing retains 169 predictors; excluding predictors "
        "with more than 50% missing training values retains 161. COPOD symbols are single deterministic results. "
        "Isolation Forest small dots show seeds 42-46, large symbols show their mean, and bars show +/-1 sample SD "
        "of five fits on the same data, not confidence intervals. Axes are zoomed; both metrics have full range 0-1. "
        "Batch COPOD uses training plus test features for scoring; the other methods use a training-only reference. "
        "The secondary condition has one exact retained profile shared by a negative training and negative test row; "
        "it appears at column selection and all supplied rows are retained.",
    "aps_primary_roc_pr": "Primary COPOD ROC and precision-recall curves from the original saved test scores. "
        "Both axes use the full 0-1 range. The ROC reference is random ranking; the PR reference is the observed "
        "test prevalence, 375/16000. Legend AP values use non-interpolated average precision, not trapezoidal PR area. "
        "No Isolation Forest seed was selected for these COPOD protocol curves. The curves nearly overlap; "
        "the paired-difference figure reports the numerical contrast.",
    "aps_copod_paired_differences": "Primary fixed training-reference minus batch COPOD. Dots show observed "
        "metric differences and bars show 95% percentile resampling intervals for the observed score-label pairs, "
        "using 2,000 paired class-stratified draws with seed 20261026 and linear quantiles. Original scores, "
        "fitted preprocessing/models, batch reference and prevalence remain fixed. Rows are treated as independent, "
        "although truck-level independence is not established. These conditional intervals omit retraining, "
        "changed-batch reference, prevalence changes and population shift; no confirmatory significance, "
        "equivalence or deployment claim is made.",
}


def checked_tables(folder, metadata_name, expected_files, hashes):
    path = folder / metadata_name
    require(path.is_file(), f"Missing saved analysis: {path}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    require(set(metadata["output_sha256"]) == set(expected_files), f"Unexpected analysis files: {folder}")
    summary.remember_file(path, hashes)
    tables = {}
    for name, digest in metadata["output_sha256"].items():
        current = folder / name
        require(current.is_file() and summary.sha256(current) == digest, f"Saved analysis changed: {current}")
        summary.remember_file(current, hashes)
        tables[name] = pd.read_csv(current, float_precision="round_trip")
    return metadata, tables


def assert_same_table(saved, recomputed, name):
    # CSV blanks reload as NaN; nullable integer columns use pd.NA in memory.
    # Normalise only missing-value representations before comparing values.
    saved = saved.astype(object).where(saved.notna(), None)
    recomputed = recomputed.astype(object).where(recomputed.notna(), None)
    try:
        pd.testing.assert_frame_equal(saved, recomputed, check_dtype=False, check_exact=False, rtol=0, atol=1e-12)
    except AssertionError as error:
        raise ValueError(f"Saved {name} differs from recomputed values.") from error


def load_plot_evidence(run_name):
    evidence = summary.load_verified_results(run_name)
    hashes = dict(evidence["input_sha256"])
    expected = {"all_metrics.csv": evidence["metrics"], **summary.build_summaries(evidence["metrics"]),
                "preprocessing_summary.csv": evidence["preprocessing"], "duplicate_summary.csv": evidence["duplicates"],
                "numerical_diagnostics.csv": evidence["numerics"], "warning_log.csv": evidence["warnings"]}
    metadata, tables = checked_tables(evidence["folder"] / "summary", "summary_metadata.json", expected, hashes)
    require(metadata["schema"] == "aps_full_summary_v1" and metadata["checks"] == "passed"
            and metadata["verified_model_results"] == 14, "Saved summary checks differ.")
    summary.evaluation.check_input_hashes(metadata["input_sha256"])
    for name, frame in expected.items():
        assert_same_table(tables[name], frame, name)
    boot_metadata, boot_tables = checked_tables(evidence["folder"] / "paired_bootstrap_primary", "bootstrap_metadata.json",
                                               ["replicate_metrics.csv", "percentile_intervals.csv"], hashes)
    config = boot_metadata["configuration"]
    expected_config = bootstrap.configuration(evidence)
    # The plotting environment is recorded separately; it need not equal the historical resampling environment.
    for key in set(expected_config) - {"versions", "metric_source_sha256", "evaluation_versions"}:
        require(config[key] == expected_config[key], f"Bootstrap configuration differs: {key}")
    summary.evaluation.check_input_hashes(config["input_sha256"])
    checks = boot_metadata["checks"]
    require(checks["status"] == "passed" and checks["completed_replicates"] == 2000
            and checks["original_arrays_unchanged"] is True and checks["all_replicates_have_fixed_class_counts"] is True
            and checks["shared_indices_for_both_methods"] is True and checks["paired_difference_identity"] == "passed"
            and checks["nonfinite_metrics"] == 0, "Saved paired resampling checks did not pass.")
    replicates = boot_tables["replicate_metrics.csv"]
    require(len(replicates) == 2000 and np.array_equal(replicates.replicate_id_one_based, np.arange(1, 2001))
            and replicates.draw_indices_sha256.str.fullmatch(r"[0-9a-f]{64}").all(), "Incomplete replicate records.")
    for metric in bootstrap.METRICS:
        for method in bootstrap.METHODS[:2]:
            values = replicates[method + "_" + metric].to_numpy()
            require(np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(), "Invalid bootstrap metrics.")
        np.testing.assert_allclose(replicates["fixed_minus_batch_" + metric],
                                   replicates["fixed_training_reference_" + metric] - replicates["pyod_batch_" + metric],
                                   rtol=0, atol=1e-15)
    scores = evidence["scores"][("primary", "copod")]
    point_estimates = bootstrap.paired_metrics(evidence["labels"], scores.pyod_batch, scores.fixed_training_reference)
    intervals = bootstrap.make_intervals(replicates, point_estimates)
    assert_same_table(boot_tables["percentile_intervals.csv"], intervals, "percentile_intervals.csv")
    require(len(intervals) == 6, "Expected six method-metric intervals.")
    for path in [Path(__file__).resolve(), Path(summary.__file__).resolve(), Path(bootstrap.__file__).resolve()]:
        summary.remember_file(path, hashes)
    evidence.update({"input_sha256": hashes, "performance": tables["performance_summary.csv"],
                     "intervals": intervals, "summary_versions": metadata["summary_versions"],
                     "bootstrap_versions": config["versions"], "bootstrap_warnings": boot_metadata["warnings"]})
    print("Verified saved summaries, 2,000 paired replicates and all six percentile intervals.", flush=True)
    return evidence


def curve_tables(evidence):
    labels = evidence["labels"]
    scores = evidence["scores"][("primary", "copod")]
    primary = evidence["metrics"].query('condition == "primary"').set_index("method")
    roc_tables, pr_tables, validations = [], [], []
    for method in summary.METHODS[:2]:
        values = scores[method].to_numpy()
        fpr, tpr, roc_thresholds = roc_curve(labels, values, pos_label=1, drop_intermediate=False)
        precision, recall, pr_thresholds = precision_recall_curve(labels, values, pos_label=1, drop_intermediate=False)
        require(np.isfinite(np.concatenate([fpr, tpr, precision, recall])).all(), "Non-finite curve coordinates.")
        require((np.diff(fpr) >= 0).all() and (np.diff(tpr) >= 0).all()
                and fpr[0] == tpr[0] == 0 and fpr[-1] == tpr[-1] == 1, "Invalid ROC endpoints/order.")
        require((np.diff(recall) <= 0).all() and recall[0] == 1 and recall[-1] == 0 and precision[-1] == 1,
                "Invalid PR endpoints/order.")
        require(np.isposinf(roc_thresholds[0]) and np.isfinite(roc_thresholds[1:]).all()
                and np.isfinite(pr_thresholds).all(), "Unexpected curve threshold semantics.")
        roc_area = float(auc(fpr, tpr))
        # Non-interpolated AP, using descending recall; do not use trapezoidal PR area.
        noninterpolated_ap = float(-np.sum(np.diff(recall) * precision[:-1]))
        np.testing.assert_allclose([roc_area, noninterpolated_ap], primary.loc[method, ["roc_auc", "average_precision"]].to_numpy(dtype=float),
                                   rtol=0, atol=1e-12)
        roc_tables.append(pd.DataFrame({"method": method, "curve_point": np.arange(len(fpr)),
            "false_positive_rate": fpr, "true_positive_rate": tpr,
            "threshold_score": np.concatenate([[np.nan], roc_thresholds[1:]]),
            "threshold_kind": ["above_maximum_score"] + ["observed_score"] * (len(fpr) - 1)}))
        pr_tables.append(pd.DataFrame({"method": method, "curve_point": np.arange(len(recall)),
            "recall": recall, "precision": precision, "threshold_score": np.concatenate([pr_thresholds, [np.nan]]),
            "threshold_kind": ["observed_score"] * len(pr_thresholds) + ["no_corresponding_threshold"]}))
        validations.append({"method": method, "roc_points": len(fpr), "pr_points": len(recall),
            "roc_curve_area": roc_area, "saved_roc_auc": float(primary.loc[method, "roc_auc"]),
            "noninterpolated_curve_ap": noninterpolated_ap, "saved_average_precision": float(primary.loc[method, "average_precision"]),
            "checks": "passed"})
    return {"primary_roc_curves.csv": pd.concat(roc_tables, ignore_index=True),
            "primary_pr_curves.csv": pd.concat(pr_tables, ignore_index=True), "curve_validation.csv": pd.DataFrame(validations)}


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.titlesize": 11,
        "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.2,
        "axes.linewidth": 0.7, "pdf.fonttype": 42, "figure.facecolor": "white", "savefig.facecolor": "white"})


def clean_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.set_axisbelow(True)


def performance_figure(evidence):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 5.6))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.75, bottom=0.31, wspace=0.34)
    fig.suptitle("APS performance across preprocessing conditions", fontsize=13.5, y=0.97)
    fig.text(0.5, 0.91, "Higher is better | Same supplied training and test rows", ha="center", fontsize=9.5)
    handles = [Line2D([], [], linestyle="none", marker=MARKERS[m], color=COLORS[m], markersize=6, label=LABELS[m]) for m in summary.METHODS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.87), ncol=3, frameon=False, columnspacing=1.1)
    for ax, metric, title, bounds in zip(axes, ["roc_auc", "average_precision"], ["ROC AUC", "Average precision"], [(0.970, 1.000), (0.50, 0.63)]):
        for i, condition in enumerate(summary.evaluation.CONDITIONS):
            for j, method in enumerate(summary.METHODS):
                group = evidence["metrics"].loc[evidence["metrics"].condition.eq(condition) & evidence["metrics"].method.eq(method)].sort_values("model_seed")
                values, x = group[metric].to_numpy(), i + [-0.22, 0, 0.22][j]
                require(((values >= bounds[0]) & (values <= bounds[1])).all(), "A result falls outside the documented plot limits.")
                if method == "isolation_forest":
                    mean, sd = values.mean(), values.std(ddof=1)
                    require(bounds[0] <= mean - sd and mean + sd <= bounds[1], "An SD bar would be clipped.")
                    ax.scatter(x + np.linspace(-0.045, 0.045, len(values)), values, s=16, color=COLORS[method], alpha=0.5, zorder=3)
                    ax.errorbar(x, mean, yerr=sd, fmt=MARKERS[method], color=COLORS[method], markersize=6, capsize=4, linewidth=1.2, zorder=4)
                else:
                    ax.plot(x, values[0], marker=MARKERS[method], color=COLORS[method], markersize=6, linestyle="none", zorder=4)
        ax.set_title(title + " (zoomed)", pad=9)
        ax.set_xticks([0, 1], ["Primary\n169 predictors", "Drop >50% missing\n161 predictors"])
        ax.set_xlim(-0.48, 1.48)
        ax.set_ylim(*bounds)
        ax.set_ylabel(title)
        clean_axis(ax)
    axes[0].set_yticks([0.97, 0.98, 0.99, 1.0])
    axes[0].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axes[1].set_yticks([0.50, 0.55, 0.60])
    fig.text(0.5, 0.205, "IF: small dots = all five seeds; large symbols = means; bars = +/-1 sample SD.", ha="center", fontsize=8.2)
    fig.text(0.5, 0.162, "SD measures fitting randomness, not a confidence interval. COPOD has one deterministic run.", ha="center", fontsize=8.2)
    fig.text(0.5, 0.119, "Axes are zoomed; full metric ranges are 0-1. References outside the axes: ROC 0.50; AP 0.02344.", ha="center", fontsize=8.2)
    fig.text(0.5, 0.076, "60,000 training / 16,000 test rows; 375 APS-positive test cases. All rows retained.", ha="center", fontsize=8.2)
    fig.text(0.5, 0.033, "Batch COPOD uses train + test features; training-only COPOD and IF use training features only.", ha="center", fontsize=8.2)
    return fig


def curves_figure(evidence, curves):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 5.3))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.79, bottom=0.25, wspace=0.31)
    fig.suptitle("APS primary ROC and precision-recall curves", fontsize=13.5, y=0.97)
    fig.text(0.5, 0.91, "169 predictors | 16,000 test cases | APS-positive prevalence: 2.34375%", ha="center", fontsize=9.3)
    metrics = evidence["metrics"].query('condition == "primary"').set_index("method")
    for method in summary.METHODS[:2]:
        roc = curves["primary_roc_curves.csv"].query("method == @method")
        pr = curves["primary_pr_curves.csv"].query("method == @method")
        draw = dict(color=COLORS[method], linewidth=2.3 if method == "pyod_batch" else 1.4,
                    linestyle="-" if method == "pyod_batch" else "--", alpha=0.9)
        axes[0].plot(roc.false_positive_rate, roc.true_positive_rate,
                     label=f"{LABELS[method]}: {metrics.loc[method, 'roc_auc']:.4f}", **draw)
        axes[1].step(pr.recall, pr.precision, where="post",
                     label=f"{LABELS[method]}: {metrics.loc[method, 'average_precision']:.4f}", **draw)
    axes[0].plot([0, 1], [0, 1], color="#777777", linestyle=":", linewidth=1, label="Random ranking: 0.50")
    axes[1].axhline(evidence["labels"].mean(), color="#777777", linestyle=":", linewidth=1, label="Prevalence: 0.02344")
    for ax, title, x, y in zip(axes, ["ROC (legend: AUC)", "Precision-recall (legend: AP)"],
                               ["False positive rate", "Recall"], ["True positive rate", "Precision"]):
        ax.set_title(title, pad=10)
        ax.set(xlabel=x, ylabel=y, xlim=(0, 1), ylim=(0, 1))
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks(np.linspace(0, 1, 6))
        clean_axis(ax)
        ax.legend(loc="lower right" if ax is axes[0] else "upper right", frameon=True, framealpha=0.95, fontsize=7.5)
    fig.text(0.5, 0.145, "AP is non-interpolated average precision; curves use the original saved test scores.", ha="center", fontsize=8.3)
    fig.text(0.5, 0.099, "Batch COPOD uses train + test features; training-only COPOD keeps the training reference fixed.", ha="center", fontsize=8.3)
    fig.text(0.5, 0.053, "The curves nearly overlap; the paired-difference figure shows the small numerical contrast.", ha="center", fontsize=8.3)
    return fig


def differences_figure(evidence):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.0))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.70, bottom=0.36, wspace=0.28)
    fig.suptitle("APS: paired differences between COPOD procedures", fontsize=13, y=0.97)
    fig.text(0.5, 0.88, "Fixed training-reference minus batch COPOD | Primary condition", ha="center", fontsize=9.5)
    differences = evidence["intervals"].query('method_or_difference == "fixed_minus_batch"').set_index("metric")
    for ax, metric, title in zip(axes, bootstrap.METRICS, ["ROC AUC difference", "Average precision difference"]):
        row = differences.loc[metric]
        point, low, high = row.point_estimate, row.lower_2_5_percentile, row.upper_97_5_percentile
        # Draw endpoints directly: percentile intervals need not contain the observed estimate in general.
        ax.hlines(0, low, high, color="#514A86", linewidth=2)
        ax.vlines([low, high], -0.045, 0.045, color="#514A86", linewidth=1.4)
        ax.plot(point, 0, "o", color="#514A86", markersize=6)
        ax.axvline(0, color="#777777", linestyle="--", linewidth=1)
        ax.text(0, -0.21, "No change", ha="center", fontsize=8, color="#555555")
        ax.text(0.5, 0.89, f"Observed: {point:+.6f}\nInterval: [{low:+.6f}, {high:+.6f}]",
                ha="center", va="top", transform=ax.transAxes, fontsize=8.5)
        span = max(high, point, 0) - min(low, point, 0)
        require(np.isfinite([point, low, high]).all() and low <= high and span > 0, "Invalid interval plotting range.")
        ax.set_xlim(min(low, point, 0) - 0.18 * span, max(high, point, 0) + 0.15 * span)
        ax.set_ylim(-0.35, 0.55)
        ax.set_yticks([])
        ax.set_xlabel(title)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.5f" if metric == "roc_auc" else "%.4f"))
        clean_axis(ax)
        ax.spines["left"].set_visible(False)
    fig.text(0.5, 0.215, "Dots: observed differences. Bars: 95% percentile resampling intervals (2,000 paired draws).", ha="center", fontsize=8.2)
    fig.text(0.5, 0.157, "Conditional on observed score-label pairs; original scores, batch reference and prevalence fixed.", ha="center", fontsize=8.2)
    fig.text(0.5, 0.099, "No refitting or new-population uncertainty is represented. Values are in raw metric units.", ha="center", fontsize=8.2)
    fig.text(0.5, 0.041, "Positive differences mean higher fixed-reference metrics; the observed improvements are small.", ha="center", fontsize=8.2)
    return fig


def figure_payloads(name, figure):
    payloads = {}
    for extension in ["png", "pdf"]:
        buffer = io.BytesIO()
        metadata = {"Title": name, "Creator": "COPOD reproducibility project", "CreationDate": None, "ModDate": None} if extension == "pdf" else {"Title": name}
        figure.savefig(buffer, format=extension, dpi=240, metadata=metadata)
        payloads[name + "." + extension] = buffer.getvalue()
    plt.close(figure)
    return payloads


def save_outputs(evidence, curves, figures, warning_records):
    figure_dir = ROOT / "figures/aps_full_evaluation" / evidence["folder"].name
    curve_dir = evidence["folder"] / "curves"
    payloads = {figure_dir / name: content for name, content in figures.items()}
    payloads.update({curve_dir / name: frame.to_csv(index=False).encode("utf-8") for name, frame in curves.items()})
    summary.evaluation.check_input_hashes(evidence["input_sha256"])
    metadata = {"schema": "aps_full_report_figures_v1", "input_sha256": evidence["input_sha256"],
        "versions": {"python": platform.python_version(), **{name: version(name) for name in
                     ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "pillow", "pyod"]}},
        "summary_versions": evidence["summary_versions"], "bootstrap_versions": evidence["bootstrap_versions"],
        "checks": "passed", "captions": CAPTIONS, "plotting_warnings": warning_records,
        "recorded_evaluation_warning_count": len(evidence["warnings"]), "recorded_bootstrap_warnings": evidence["bootstrap_warnings"],
        "curve_threshold_semantics": "Blank ROC threshold in first row denotes above-maximum-score endpoint. "
                                     "Blank final PR threshold has no corresponding score threshold. These are intentional.",
        "verification_scope": "Source/bundle hashes, all 14 metrics, saved summary values, paired differences, "
                              "six quantiles recomputed from saved replicates, ROC AUC and non-interpolated AP from curves. "
                              "No model fitting or resampling draws.",
        "output_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(content).hexdigest() for path, content in payloads.items()}}
    payloads[figure_dir / "figure_metadata.json"] = summary.evaluation.json_bytes(metadata)
    for path, content in payloads.items():
        require(not path.exists() or path.read_bytes() == content,
                f"Existing figure/curve file differs: {path}. Inspect changed inputs, code or rendering environment before replacing evidence.")
    for path, content in payloads.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(content)
        print("Saved or verified:", path.relative_to(ROOT), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="v1")
    args = parser.parse_args()
    evidence = load_plot_evidence(args.run_name)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        style()
        curves = curve_tables(evidence)
        figures = {}
        figures.update(figure_payloads("aps_performance_comparison", performance_figure(evidence)))
        figures.update(figure_payloads("aps_primary_roc_pr", curves_figure(evidence, curves)))
        figures.update(figure_payloads("aps_copod_paired_differences", differences_figure(evidence)))
    warning_records = [{"category": warning.category.__name__, "message": str(warning.message)} for warning in caught]
    for item in warning_records:
        print(f"Warning [{item['category']}]: {item['message']}", flush=True)
    save_outputs(evidence, curves, figures, warning_records)
    print(f"\nThree figures saved in PNG and PDF, plus curve tables and provenance. Plotting warnings: {len(warning_records)}.")
    print("No models or bootstrap resamples were run. Open all three PNGs to review them.")


if __name__ == "__main__":
    main()
