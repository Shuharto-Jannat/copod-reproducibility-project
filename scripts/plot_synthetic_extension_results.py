"""Create three verified synthetic-extension figures from saved results.

Run: python scripts/plot_synthetic_extension_results.py
Requires the completed ten-seed summary and its underlying evidence. No models
are fitted and no datasets or resamples are generated. Saves PNG/PDF figures,
separate captions, exact plotting tables and provenance. Differing existing
outputs are never overwritten.
"""

import argparse
import io
import json
import platform
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

import summarise_synthetic_extension as summary


ROOT = Path(__file__).resolve().parents[1]
generation, evaluation = summary.generation, summary.evaluation
require, digest = generation.require, generation.digest
METHODS, SEEDS, CONDITIONS = summary.METHODS, summary.SEEDS, summary.CONDITIONS
METRICS = ("roc_auc", "average_precision")
FAMILIES = ("marginal_shift", "dependence_violation")
COMPARISONS = ("fixed_minus_batch", "ledoit_wolf_minus_fixed")
COLORS = {"pyod_batch": "#0072B2", "fixed_training_reference": "#D55E00", "ledoit_wolf": "#00866A"}
MARKERS = {"pyod_batch": "o", "fixed_training_reference": "s", "ledoit_wolf": "^"}
LABELS = {"pyod_batch": "Batch COPOD", "fixed_training_reference": "Fixed-reference COPOD",
          "ledoit_wolf": "Ledoit-Wolf distance"}
METRIC_LABELS = {"roc_auc": "ROC AUC", "average_precision": "Average precision"}
FIGURES = ("ten_seed_performance", "ten_seed_method_differences", "ten_seed_strength_differences")
FOOTER = "Small dots: ten seeds | Large markers: means +/- sample SD (not confidence intervals)"
CAPTIONS = {
    "ten_seed_performance":
        "Performance across data seeds 100-109. Each realization contains 10,000 rows and 20 features, "
        "with 500 designated positives. The shared stratified split gives 6,000 training rows "
        "(300 positives) and 4,000 test rows (200 positives); no preprocessing is applied. "
        "The left panels vary the one-feature shift magnitude at normal within-block correlation 0.8. "
        "The right panels vary normal within-block correlation while positive features remain independent. "
        "Delta 0 and rho 0 are separate family-specific null controls without a deliberately introduced signal. "
        "Small dots show every seed, large markers show means, and bars show +/-1 sample SD across ten seeds, "
        "not confidence intervals. Both metrics have range 0-1; plotting margins retain all SD endpoints "
        "even if they slightly exceed that range. Dashed reference lines indicate random-ranking ROC AUC 0.5 "
        "and AP prevalence 0.05; finite-sample null AP is not required to equal 0.05. Horizontal offsets "
        "separate methods and seeds only for visibility; tick labels identify the actual parameter levels. "
        "Lines connect the method means as a guide to the eye, not as estimates at untested strengths. "
        "Coincident values, including perfect rankings, are retained. Batch COPOD uses training plus test "
        "features for its reference; fixed COPOD and Ledoit-Wolf use training features only. "
        "The covariance comparator was chosen for these Gaussian mechanisms, not as a universally superior detector.",
    "ten_seed_method_differences":
        "Paired method comparisons across all seven conditions and data seeds 100-109. "
        "Left panels show fixed-reference COPOD minus batch COPOD; right panels show Ledoit-Wolf "
        "squared Mahalanobis distance minus fixed-reference COPOD. Positive differences favour the first "
        "method in each title. Each small dot is a difference computed on the same seed, condition and "
        "test rows. Large markers and bars show the mean and +/-1 sample SD of those ten differences; "
        "the bars are not confidence intervals. The two comparison columns use different horizontal "
        "scales to keep the smaller scoring-protocol differences visible. All points and SD endpoints "
        "are included. Dashed vertical lines mark zero difference. Vertical jitter only separates seeds. "
        "The family divider separates single-feature shift settings from dependence settings; the controls "
        "are not pooled. Ledoit-Wolf minus fixed-reference COPOD at rho 0.8 is the prespecified principal "
        "relational comparison. These are descriptive comparisons, without significance or equivalence tests.",
    "ten_seed_strength_differences":
        "Paired changes from each family's own null control. For each method and seed, the left panels "
        "subtract the delta-0 metric from the corresponding delta-2, delta-4 or delta-6 metric. "
        "The right panels subtract the rho-0 metric from the corresponding rho-0.4 or rho-0.8 metric. "
        "Conditions within a family share the planned latent draws, row identities and split. The two "
        "families have different draws and their controls remain separate. Every seed difference is shown; "
        "large markers and bars show the mean and +/-1 sample SD of the ten paired differences, not "
        "confidence intervals. Negative values are retained. Dashed lines indicate zero difference. "
        "Axes include every seed and SD endpoint, with the same vertical scale across families for a "
        "given metric. Horizontal offsets only separate methods and seeds. Connecting lines guide the eye "
        "between tested levels. A larger single-feature shift also changes positive-group correlations; "
        "these results should not be interpreted as an intervention on individual distributions alone.",
}


def load_plot_evidence(run_name):
    verified = summary.load_verified_results(run_name)
    hashes = dict(verified["input_sha256"])
    folder = ROOT / "results/synthetic_extension" / run_name / "summary"

    def remember(path, expected=None):
        content = path.read_bytes()
        actual, key = digest(content), path.relative_to(ROOT).as_posix()
        require(expected is None or actual == expected, f"Saved summary checksum differs: {path}")
        require(key not in hashes or hashes[key] == actual, f"Input changed during plotting: {path}")
        hashes[key] = actual
        return content

    metadata = json.loads(remember(folder / "summary_metadata.json"))
    require(metadata["schema"] == summary.SCHEMA and metadata["run_name"] == run_name
            and metadata["checks"] == "passed" and metadata["datasets_verified"] == 70
            and metadata["results_verified"] == 210 and metadata["models_fitted"] == 0
            and metadata["data_generated"] is False and metadata["resamples_run"] == 0
            and metadata["planned_data_seeds"] == list(SEEDS)
            and metadata["conditions"] == list(CONDITIONS) and metadata["methods"] == list(METHODS)
            and metadata["plan_sha256"] == generation.PLAN_SHA256
            and metadata["plan_commit"] == generation.PLAN_COMMIT
            and metadata["input_sha256"] == verified["input_sha256"]
            and metadata["original_evaluation_runtime"] == verified["manifest"]["runtime"],
            "Saved summary does not match the complete verified experiment.")
    remember(Path(summary.__file__).resolve(), metadata["summary_script_sha256"])
    recomputed = summary.build_tables(verified["all_metrics"])
    recomputed.update({"numerical_diagnostics.csv": verified["numerical_diagnostics"],
                       "verified_inventory.csv": verified["inventory"], "warning_log.csv": verified["warnings"]})
    require(set(metadata["output_sha256"]) == set(recomputed)
            and metadata["output_rows"] == {name: len(frame) for name, frame in recomputed.items()},
            "Saved summary inventory differs.")
    tables = {}
    for name, expected in recomputed.items():
        content = remember(folder / name, metadata["output_sha256"][name])
        table = pd.read_csv(io.BytesIO(content), float_precision="round_trip")
        evaluation.assert_table(table, expected, name)
        tables[name] = table
    print("Verified all nine summary tables, 210 results and the planned paired comparisons.", flush=True)
    return {"tables": tables, "input_sha256": hashes, "summary_metadata": metadata,
            "original_evaluation_runtime": verified["manifest"]["runtime"]}


def make_plot_tables(tables):
    """Save the exact positions and values used for every point and SD bar."""
    records = {name: [] for name in FIGURES}
    checks = []
    seed_jitter = np.linspace(-1, 1, 10)

    def add_group(figure, panel, spec, series, metric, values, mean, sd, position, horizontal=False):
        require(np.array_equal(values.data_seed, SEEDS), "Plot group must contain each planned seed once.")
        numbers = values.value.to_numpy(dtype=float)
        np.testing.assert_allclose([mean, sd], [numbers.mean(), numbers.std(ddof=1)], rtol=0, atol=1e-12)
        require(np.isfinite(numbers).all() and np.isfinite([mean, sd]).all() and sd >= 0,
                "Invalid plotting values.")
        family = spec["family"]
        parameter = "delta" if family == "marginal_shift" else "rho"
        jitter = .15 if horizontal else (.065 if family == "marginal_shift" else .013)
        common = {"figure": figure, "panel": panel, "condition": spec["condition"],
                  "family": family, "series": series, "metric": metric,
                  "parameter": parameter, "parameter_value": spec[parameter],
                  "null_control": spec["null_control"], "error_direction": "x" if horizontal else "y"}
        for seed, value, offset in zip(SEEDS, numbers, seed_jitter*jitter):
            records[figure].append({**common, "kind": "seed", "data_seed": seed,
                "value": value, "sample_sd": np.nan, "error_low": np.nan, "error_high": np.nan,
                "x": value if horizontal else position+offset,
                "y": position+offset if horizontal else value})
        records[figure].append({**common, "kind": "mean", "data_seed": np.nan,
            "value": mean, "sample_sd": sd, "error_low": mean-sd, "error_high": mean+sd,
            "x": mean if horizontal else position, "y": position if horizontal else mean})
        checks.append({**common, "n_seed_points": len(numbers), "mean": mean, "sample_sd": sd,
                       "error_low": mean-sd, "error_high": mean+sd, "checks": "passed"})

    performance = tables["performance_summary.csv"].set_index(["condition", "method"])
    all_metrics = tables["all_metrics.csv"]
    method_pairs, method_means = tables["method_paired_differences.csv"], tables["method_paired_summary.csv"]
    strength_pairs, strength_means = tables["strength_paired_differences.csv"], tables["strength_paired_summary.csv"]
    for spec in CONDITIONS:
        name, family = spec["condition"], spec["family"]
        parameter = "delta" if family == "marginal_shift" else "rho"
        separation = .20 if family == "marginal_shift" else .04
        for metric in METRICS:
            panel = metric+"/"+family
            for i, method in enumerate(METHODS):
                current = all_metrics[(all_metrics.condition == name) & (all_metrics.method == method)].sort_values("data_seed")
                values = current[["data_seed", metric]].rename(columns={metric: "value"})
                row = performance.loc[(name, method)]
                add_group(FIGURES[0], panel, spec, method, metric, values,
                          row[metric+"_mean"], row[metric+"_sd"], spec[parameter]+(i-1)*separation)
                if not spec["null_control"]:
                    current = strength_pairs[(strength_pairs.condition == name) & (strength_pairs.method == method)].sort_values("data_seed")
                    values = current[["data_seed", metric+"_difference"]].rename(columns={metric+"_difference": "value"})
                    selected = strength_means[(strength_means.condition == name) & (strength_means.method == method)]
                    require(len(selected) == 1, "Missing or duplicate strength summary.")
                    row = selected.iloc[0]
                    add_group(FIGURES[2], panel, spec, method, metric, values,
                              row[metric+"_difference_mean"], row[metric+"_difference_sd"], spec[parameter]+(i-1)*separation)
            for comparison in COMPARISONS:
                current = method_pairs[(method_pairs.condition == name) & (method_pairs.comparison == comparison)].sort_values("data_seed")
                values = current[["data_seed", metric+"_difference"]].rename(columns={metric+"_difference": "value"})
                selected = method_means[(method_means.condition == name) & (method_means.comparison == comparison)]
                require(len(selected) == 1, "Missing or duplicate method summary.")
                row = selected.iloc[0]
                add_group(FIGURES[1], metric+"/"+comparison, spec, comparison, metric, values,
                          row[metric+"_difference_mean"], row[metric+"_difference_sd"],
                          [s["condition"] for s in CONDITIONS].index(name), horizontal=True)
    frames = {name: pd.DataFrame(rows) for name, rows in records.items()}
    for name, seed_points, mean_points in zip(FIGURES, (420, 280, 300), (42, 28, 30)):
        require(frames[name].kind.eq("seed").sum() == seed_points
                and frames[name].kind.eq("mean").sum() == mean_points, "Plot inventory differs.")
    return frames, pd.DataFrame(checks)


def limits_for(data, performance=False):
    bounds = np.r_[data.value, data.loc[data.kind.eq("mean"), "error_low"],
                   data.loc[data.kind.eq("mean"), "error_high"], 0]
    if performance:
        bounds = np.r_[bounds, 1]
    low, high = float(bounds.min()), float(bounds.max())
    padding = max(high-low, .001)*.07
    return low-padding, high+padding


def style_axis(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E1E5EA", linewidth=.6)
    ax.tick_params(length=3, color="#8B949E")
    for spine in ax.spines.values():
        spine.set_color("#B0B7C0")


def check_artists(ax, group, horizontal, color, marker, lines=False):
    points = group[group.kind.eq("seed")]
    means = group[group.kind.eq("mean")].sort_values("parameter_value")
    collection = ax.scatter(points.x, points.y, s=10, color=color, alpha=.45, linewidths=0, zorder=3)
    errors = ax.errorbar(means.x, means.y, xerr=means.sample_sd if horizontal else None,
                        yerr=None if horizontal else means.sample_sd, fmt=marker,
                        color=color, markersize=5.0, markeredgecolor="white", markeredgewidth=.45,
                        elinewidth=1.3, capsize=3, zorder=4,
                        linestyle="-" if lines else "none", linewidth=1.05)
    np.testing.assert_allclose(np.asarray(collection.get_offsets()), points[["x", "y"]], rtol=0, atol=1e-14)
    np.testing.assert_allclose(np.asarray(errors.lines[0].get_xdata(), dtype=float), means.x, rtol=0, atol=1e-14)
    np.testing.assert_allclose(np.asarray(errors.lines[0].get_ydata(), dtype=float), means.y, rtol=0, atol=1e-14)
    segments = errors.lines[2][0].get_segments()
    require(len(segments) == len(means), "Missing plotted SD bar.")
    for segment, (_, row) in zip(segments, means.iterrows()):
        expected = ([[row.error_low, row.y], [row.error_high, row.y]] if horizontal
                    else [[row.x, row.error_low], [row.x, row.error_high]])
        np.testing.assert_allclose(segment, expected, rtol=0, atol=1e-14)


def check_axis_bounds(ax, data, horizontal):
    xlim, ylim = sorted(ax.get_xlim()), sorted(ax.get_ylim())
    xs, ys = data.x.to_numpy(), data.y.to_numpy()
    means = data[data.kind.eq("mean")]
    if horizontal:
        xs = np.r_[xs, means.error_low, means.error_high]
    else:
        ys = np.r_[ys, means.error_low, means.error_high]
    require(np.isfinite(xs).all() and np.isfinite(ys).all()
            and (xs >= xlim[0]).all() and (xs <= xlim[1]).all()
            and (ys >= ylim[0]).all() and (ys <= ylim[1]).all(), "A plotted point or SD bar would be clipped.")
    return {"xlim": list(ax.get_xlim()), "ylim": list(ax.get_ylim()),
            "seed_points": int(data.kind.eq("seed").sum()), "mean_markers": int(data.kind.eq("mean").sum())}


def add_method_legend(fig):
    handles = [Line2D([], [], color=COLORS[m], marker=MARKERS[m], markersize=5,
                      linewidth=1, label=LABELS[m]) for m in METHODS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.52, .939), ncol=3,
               frameon=False, handlelength=1.7, columnspacing=1.8, fontsize=8.5)


def family_figure(data, strength=False):
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.25))
    fig.subplots_adjust(left=.10, right=.985, top=.81, bottom=.13, wspace=.29, hspace=.46)
    fig.suptitle("Synthetic extension: changes from each null control" if strength else
                 "Synthetic extension: performance", y=.988, fontsize=12.5, fontweight="semibold")
    add_method_legend(fig)
    fig.text(.5, .872, "Dashed lines: no change (0)" if strength else
             "Dashed lines: random-ranking ROC AUC 0.50 | AP prevalence 0.05",
             ha="center", fontsize=8.2, color="#424B57")
    fig.text(.5, .035, FOOTER, ha="center", fontsize=7.9, color="#424B57")
    axis_records = {}
    for i, metric in enumerate(METRICS):
        limits = limits_for(data[data.metric.eq(metric)], performance=not strength)
        for j, family in enumerate(FAMILIES):
            ax = axes[i, j]
            current = data[(data.metric == metric) & (data.family == family)]
            style_axis(ax)
            for method in METHODS:
                check_artists(ax, current[current.series.eq(method)], False, COLORS[method], MARKERS[method], lines=True)
            levels = ([2, 4, 6] if strength else [0, 2, 4, 6]) if j == 0 else ([.4, .8] if strength else [0, .4, .8])
            margin = .52 if j == 0 else .105
            ax.set_xlim(min(levels)-margin, max(levels)+margin)
            ax.set_xticks(levels, [str(value) for value in levels])
            ax.set_ylim(*limits)
            if not strength:
                ax.set_yticks(np.linspace(0, 1, 6))
            else:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.set_xlabel("Shift magnitude, delta" if j == 0 else "Normal within-block correlation, rho", fontsize=8.8)
            ax.set_ylabel(METRIC_LABELS[metric]+(" difference" if strength else ""), fontsize=9)
            if i == 0:
                ax.set_title(("Single-feature shifts" if j == 0 else "Dependence violations")
                             + ((" (vs delta = 0)" if j == 0 else " (vs rho = 0)") if strength else ""),
                             fontsize=9.6, pad=10)
            reference = 0 if strength else (.5 if metric == "roc_auc" else .05)
            ax.axhline(reference, color="#6B7280", linewidth=.9, linestyle="--", zorder=2)
            axis_records[metric+"/"+family] = check_axis_bounds(ax, current, False)
    return fig, axis_records


def method_figure(data):
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.7))
    fig.subplots_adjust(left=.17, right=.98, top=.82, bottom=.12, wspace=.73, hspace=.38)
    fig.suptitle("Synthetic extension: paired method differences", y=.985, fontsize=12.5, fontweight="semibold")
    fig.text(.5, .938, "Positive values favour the first method | Horizontal scales differ between columns",
             ha="center", fontsize=8.2, color="#424B57")
    fig.text(.5, .89, "Dashed lines: no difference (0)", ha="center", fontsize=8.2, color="#424B57")
    fig.text(.5, .035, FOOTER, ha="center", fontsize=7.9, color="#424B57")
    labels = ["Shift 0 (null)", "Shift 2", "Shift 4", "Shift 6", "Rho 0 (null)", "Rho 0.4", "Rho 0.8"]
    axis_records = {}
    for i, metric in enumerate(METRICS):
        for j, comparison in enumerate(COMPARISONS):
            ax = axes[i, j]
            current = data[(data.metric == metric) & (data.series == comparison)]
            style_axis(ax)
            ax.grid(axis="y", visible=False)
            ax.grid(axis="x", color="#E1E5EA", linewidth=.6)
            color = COLORS["fixed_training_reference" if j == 0 else "ledoit_wolf"]
            check_artists(ax, current, True, color, "D")
            ax.set_xlim(*limits_for(current))
            ax.set_ylim(6.5, -.55)
            ax.set_yticks(range(7), labels, fontsize=8.4)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.set_xlabel(METRIC_LABELS[metric]+" difference", fontsize=8.8)
            ax.axvline(0, color="#555D69", linestyle="--", linewidth=.9, zorder=2)
            ax.axhline(3.5, color="#C9CFD7", linewidth=.8)
            if i == 0:
                ax.set_title("Fixed COPOD - batch COPOD" if j == 0 else "Ledoit-Wolf - fixed COPOD",
                             fontsize=9.3, pad=10)
            axis_records[metric+"/"+comparison] = check_axis_bounds(ax, current, True)
    return fig, axis_records


def build_outputs(run_name):
    evidence = load_plot_evidence(run_name)
    plot_tables, checks = make_plot_tables(evidence["tables"])
    figure_dir = ROOT / "figures/synthetic_extension" / run_name / "results"
    table_dir = ROOT / "results/synthetic_extension" / run_name / "plot_data"
    files = {table_dir/(name+".csv"): generation.csv_bytes(frame) for name, frame in plot_tables.items()}
    files[table_dir/"plot_checks.csv"] = generation.csv_bytes(checks)
    axis_records = {}
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#222B36", "text.color": "#222B36", "figure.facecolor": "white"}):
        for name in FIGURES:
            data = plot_tables[name]
            fig, axis_records[name] = (method_figure(data) if name == FIGURES[1]
                                      else family_figure(data, strength=name == FIGURES[2]))
            try:
                for extension in ("png", "pdf"):
                    buffer = io.BytesIO()
                    metadata = ({"Software": "COPOD synthetic extension figures"} if extension == "png" else
                                {"Creator": "COPOD synthetic extension figures", "CreationDate": None, "ModDate": None})
                    fig.savefig(buffer, format=extension, dpi=220, metadata=metadata)
                    files[figure_dir/(name+"."+extension)] = buffer.getvalue()
            finally:
                plt.close(fig)
    captions = "# Synthetic extension figure captions\n\n" + "\n\n".join(
        "## "+name+"\n\n"+CAPTIONS[name] for name in FIGURES) + "\n"
    files[figure_dir/"figure_captions.md"] = captions.encode("utf-8")
    source = Path(__file__).resolve()
    evidence["input_sha256"][source.relative_to(ROOT).as_posix()] = digest(source.read_bytes())
    evaluation.check_hashes(evidence["input_sha256"])
    return files, evidence, axis_records, figure_dir, table_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="v1", help="Existing generation/evaluation/summary namespace.")
    args = parser.parse_args()
    require(re.fullmatch(r"[a-z][a-z0-9_]{0,47}", args.run_name) is not None, "Invalid run name.")
    result, warning_log = evaluation.capture_warnings("plotting", lambda: build_outputs(args.run_name))
    files, evidence, axes, figure_dir, table_dir = result
    metadata = {"schema": "synthetic_extension_result_figures_v1", "run_name": args.run_name,
                "data_seeds": list(SEEDS), "conditions": list(CONDITIONS), "methods": list(METHODS),
                "saved_summary_checks": "passed", "artist_value_checks": "passed", "axis_bounds_checks": "passed",
                "models_fitted": 0, "data_generated": False, "resamples_run": 0,
                "sample_sd_ddof": 1, "error_bars": "mean +/- 1 sample SD; not confidence intervals",
                "jitter": "Deterministic display offsets; values and parameter labels are unchanged.",
                "captions": CAPTIONS, "axes": axes, "input_sha256": evidence["input_sha256"],
                "evaluation_runtime": evidence["original_evaluation_runtime"],
                "summary_runtime": evidence["summary_metadata"]["summary_runtime"],
                "plot_runtime": {"python": platform.python_version(), "numpy": np.__version__,
                                 "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
                "historical_warning_count": evidence["summary_metadata"]["historical_warning_count"],
                "saved_summary_warning_count": evidence["summary_metadata"]["summary_warning_count"],
                "plot_warnings": warning_log,
                "output_sha256": {path.relative_to(ROOT).as_posix(): digest(content) for path, content in files.items()}}
    files[figure_dir/"figure_metadata.json"] = generation.json_bytes(metadata)
    created = generation.save_or_verify(files)
    print(f"\nCreated {created} files; verified {len(files)-created} existing files.")
    print("Three figures in PNG and PDF, separate captions and provenance:", figure_dir.relative_to(ROOT))
    print("Three exact plotting tables and 100 group checks:", table_dir.relative_to(ROOT))
    print("Plotting warnings:", len(warning_log))
    print("No models, data generation or resampling were run. Open the three PNGs to review them.")


if __name__ == "__main__":
    main()
