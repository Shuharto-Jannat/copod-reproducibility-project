"""Plot the seven preselected seed-100 construction checks, without fitting models.

Run: python scripts/plot_synthetic_extension_diagnostics.py
Reads existing datasets, annotations, splits and validation evidence. Saves seven
PNG/PDF figures, numerical plot tables and provenance; never replaces differing
outputs. --run-name selects the same namespace used by the generator (default v1).
"""

import argparse
import io
import json
import platform
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import generate_synthetic_extension as generation


ROOT = Path(__file__).resolve().parents[1]
SEED = 100  # Chosen in the committed plan, before detector outcomes.
GROUPS = {0: "Normal origin", 1: "Positive origin"}
COLORS = {0: "#0072B2", 1: "#D55E00"}
require = generation.require
digest = generation.digest
TOLERANCE = 2e-12


def assert_table(saved, recomputed, name):
    try:
        pd.testing.assert_frame_equal(saved, recomputed, check_dtype=False,
                                      check_exact=False, rtol=0, atol=TOLERANCE)
    except AssertionError as error:
        raise ValueError(f"Saved {name} disagrees with the underlying data.") from error


def load_evidence(run_name):
    """Verify the complete generation record, then read its saved arrays."""
    data_dir = ROOT / "data_new/synthetic_extension" / run_name / f"seed{SEED}"
    result_dir = ROOT / "results/synthetic_extension" / run_name / f"seed{SEED}"
    hashes = {}

    def remember(path, expected=None):
        require(path.is_file(), f"Missing input: {path}")
        content = path.read_bytes()
        actual = digest(content)
        require(expected is None or actual == expected, f"Input hash changed: {path}")
        hashes[path.relative_to(ROOT).as_posix()] = actual
        return content

    def read_json(path, expected=None):
        return json.loads(remember(path, expected))

    def read_csv(path):
        return pd.read_csv(io.BytesIO(remember(path)), float_precision="round_trip")

    manifest_path = result_dir.parent / "generation_manifest.json"
    manifest = read_json(manifest_path)
    manifest_hash = digest(manifest_path.read_bytes())
    require(manifest["schema"] == "synthetic_extension_generation_v1"
            and manifest["run_name"] == run_name
            and manifest["conditions"] == list(generation.CONDITIONS)
            and manifest["planned_seeds"] == list(generation.PLANNED_SEEDS)
            and manifest["feature_columns"] == generation.FEATURES
            and (manifest["n_rows"], manifest["n_features"], manifest["n_positive"],
                 manifest["block_size"]) == (10000, 20, 500, 5)
            and manifest["preprocessing"] == "none"
            and manifest["models_run"] is False, "Unexpected generation configuration.")
    require(manifest["plan_path"] == generation.PLAN_PATH
            and manifest["plan_commit"] == generation.PLAN_COMMIT
            and manifest["plan_sha256"] == generation.PLAN_SHA256,
            "The generation record does not identify the frozen plan.")
    remember(ROOT / generation.PLAN_PATH, generation.PLAN_SHA256)
    remember(Path(generation.__file__).resolve(), manifest["generator_sha256"])
    require(manifest["split"] == {
        "test_size": 0.4, "stratify": "label", "random_state": 42,
        "train_rows": 6000, "test_rows": 4000,
        "train_positives": 300, "test_positives": 200}, "Unexpected split protocol.")

    seed_record = read_json(result_dir / "seed_metadata.json")
    names = [spec["condition"] for spec in generation.CONDITIONS]
    require(seed_record["schema"] == "synthetic_extension_seed_generation_v1"
            and seed_record["seed"] == SEED
            and seed_record["generation_manifest_sha256"] == manifest_hash
            and seed_record["completed_conditions"] == names
            and seed_record["integrity_checks"] == "passed"
            and seed_record["pairing_checks"] == "passed"
            and seed_record["models_run"] is False, "Seed generation is incomplete or mismatched.")
    expected_paths = {result_dir / "validation_summary.csv", result_dir / "pairing_checks.csv"}
    for name in names:
        expected_paths.update({data_dir / f"{name}.csv", data_dir / f"{name}_annotations.csv"})
        expected_paths.update(result_dir / name / item for item in (
            "split.csv", "feature_summary.csv", "correlation_summary.csv", "metadata.json", "validation.json"))
    expected_keys = {path.relative_to(ROOT).as_posix() for path in expected_paths}
    require(set(seed_record["output_sha256"]) == expected_keys,
            "Generation inventory must include all seven datasets and their validation files.")
    for key, expected in seed_record["output_sha256"].items():
        remember(ROOT / key, expected)

    cases, anchors = {}, {}
    for spec in generation.CONDITIONS:
        name = spec["condition"]
        target = result_dir / name
        metadata = read_json(target / "metadata.json")
        validation = read_json(target / "validation.json")
        data_path, annotation_path = data_dir / f"{name}.csv", data_dir / f"{name}_annotations.csv"
        require(metadata["schema"] == "synthetic_extension_dataset_v1"
                and metadata["seed"] == SEED and metadata["specification"] == spec
                and metadata["generation_manifest_sha256"] == manifest_hash
                and metadata["feature_columns"] == generation.FEATURES
                and metadata["data_path"] == data_path.relative_to(ROOT).as_posix()
                and metadata["annotation_path"] == annotation_path.relative_to(ROOT).as_posix(),
                f"Dataset metadata differs: {name}")
        expected_artifacts = {data_path, annotation_path, target / "split.csv",
                              target / "feature_summary.csv", target / "correlation_summary.csv"}
        require(metadata["output_sha256"] == {
            path.relative_to(ROOT).as_posix(): hashes[path.relative_to(ROOT).as_posix()]
            for path in expected_artifacts}, f"Dataset hash chain differs: {name}")
        require(validation["schema"] == "synthetic_extension_validation_v1"
                and validation["condition"] == name and validation["seed"] == SEED
                and validation["metadata_sha256"] == digest((target / "metadata.json").read_bytes())
                and validation["data_sha256"] == digest(data_path.read_bytes())
                and validation["integrity_checks"] == "passed"
                and validation["paired_construction"] == "passed"
                and validation["models_run"] is False
                and validation["null_control"] == spec["null_control"]
                and not any(validation["duplicates"].values()), f"Validation differs: {name}")
        data, annotations, split = read_csv(data_path), read_csv(annotation_path), read_csv(target / "split.csv")
        require(list(data.columns) == generation.FEATURES + ["label"] and data.shape == (10000, 21),
                f"Unexpected data shape or columns: {name}")
        X, y = data[generation.FEATURES].to_numpy(), data.label.to_numpy()
        require(np.isfinite(X).all() and np.array_equal(np.unique(y, return_counts=True)[0], [0, 1])
                and np.array_equal(np.unique(y, return_counts=True)[1], [9500, 500]),
                f"Invalid features or labels: {name}")
        require(len(annotations) == len(split) == len(y)
                and np.array_equal(annotations.row_index, np.arange(10000))
                and np.array_equal(split.row_index, np.arange(10000))
                and np.array_equal(annotations.label, y) and np.array_equal(split.label, y)
                and np.array_equal(annotations.source_row_id, split.source_row_id)
                and np.array_equal(np.sort(annotations.source_row_id), np.arange(10000))
                and np.array_equal(y, (annotations.source_row_id >= 9500).astype(int))
                and (annotations.family == spec["family"]).all()
                and (annotations.condition == name).all(), f"Row alignment differs: {name}")
        ordered = {}
        for partition, count, positives in (("train", 6000, 300), ("test", 4000, 200)):
            part = split[split.split == partition].sort_values("partition_position")
            require(len(part) == count and part.label.sum() == positives
                    and np.array_equal(part.partition_position, np.arange(count)), f"Invalid split: {name}")
            ordered[partition] = part.row_index.to_numpy()
            require(generation.array_digest(ordered[partition])
                    == metadata["stream_provenance"][partition + "_row_order_sha256"],
                    f"Split order changed: {name}")

        features, correlations = generation.distribution_summaries(X, y, spec)
        assert_table(read_csv(target / "feature_summary.csv"), features, name + " feature summary")
        assert_table(read_csv(target / "correlation_summary.csv"), correlations, name + " correlation summary")
        matrices = {label: np.corrcoef(X[y == label], rowvar=False) for label in (0, 1)}
        require(all(np.isfinite(matrix).all() for matrix in matrices.values()), "Invalid correlation matrix.")

        anchor = anchors.setdefault(spec["family"], {"X": X, "y": y, "annotations": annotations, "split": split})
        pd.testing.assert_frame_equal(split, anchor["split"])
        require(np.array_equal(annotations.source_row_id, anchor["annotations"].source_row_id)
                and np.array_equal(y, anchor["y"]), "Paired row identities changed.")
        if spec["family"] == "marginal_shift":
            selected, signs = annotations.selected_feature_index.to_numpy(), annotations.shift_sign.to_numpy()
            require((selected[y == 0] == -1).all() and (signs[y == 0] == 0).all()
                    and np.isin(selected[y == 1], np.arange(20)).all()
                    and np.isin(signs[y == 1], [-1, 1]).all()
                    and np.array_equal(selected, anchor["annotations"].selected_feature_index)
                    and np.array_equal(signs, anchor["annotations"].shift_sign)
                    and np.array_equal(annotations.applied_shift, signs * spec["delta"]),
                    "Marginal construction annotations changed.")
            expected = anchor["X"].copy()
            chosen = np.flatnonzero(y == 1)
            expected[chosen, selected[chosen]] += signs[chosen] * spec["delta"]
            np.testing.assert_allclose(X, expected, rtol=0, atol=1e-12)
        else:
            require((annotations.selected_feature_index == -1).all()
                    and (annotations.shift_sign == 0).all() and (annotations.applied_shift == 0).all()
                    and np.array_equal(X[y == 1], anchor["X"][y == 1]),
                    "Dependence pairing or annotations changed.")
        cases[name] = {"spec": spec, "X": X, "y": y, "matrices": matrices,
                       "correlations": correlations, "maximum": np.abs(X).max(axis=1)}
        print(f"Verified saved inputs, pairing and distribution summaries: {name}", flush=True)
    return {"cases": cases, "input_sha256": hashes, "manifest": manifest,
            "generation_warnings": seed_record["warnings"], "result_dir": result_dir}


def prepare_plot_tables(cases):
    """Use common family scales; include every observation in each histogram."""
    scales, histogram_rows, matrix_rows, check_rows = {}, [], [], []
    for family in generation.FAMILIES:
        members = [case for case in cases.values() if case["spec"]["family"] == family]
        scatter_max = max(np.abs(case["X"][:, :2]).max() for case in members)
        scatter_limit = float(np.ceil(scatter_max * 1.05 * 2) / 2)
        histogram_max = float(np.ceil(max(case["maximum"].max() for case in members) * 2) / 2)
        bins = np.linspace(0, histogram_max, 41)
        ymax = 0.0
        for case in members:
            case["histograms"] = {}
            for label in (0, 1):
                values = case["maximum"][case["y"] == label]
                counts, _ = np.histogram(values, bins=bins)
                density = counts / (len(values) * np.diff(bins))
                area = float(np.sum(density * np.diff(bins)))
                require(counts.sum() == len(values) and abs(area - 1.0) <= 1e-12,
                        "Histogram excludes rows or is not normalized within group.")
                ymax = max(ymax, float(density.max()))
                case["histograms"][label] = density
                for i, count in enumerate(counts):
                    histogram_rows.append({"condition": case["spec"]["condition"], "label": label,
                        "bin_left": bins[i], "bin_right": bins[i + 1], "count": int(count),
                        "density": density[i]})
                matrix = case["matrices"][label]
                for i, first in enumerate(generation.FEATURES):
                    for j, second in enumerate(generation.FEATURES):
                        matrix_rows.append({"condition": case["spec"]["condition"], "label": label,
                            "feature_1": first, "feature_2": second, "pearson_correlation": matrix[i, j]})
                within = case["correlations"].query("label == @label and pair_type == 'within_block'").iloc[0]
                check_rows.append({"condition": case["spec"]["condition"], "label": label,
                    "null_control": case["spec"]["null_control"], "rows_plotted": len(values),
                    "histogram_area": area, "within_block_correlation": within.mean_correlation,
                    "expected_within_block_correlation": within.expected_correlation,
                    "saved_summaries_agree": True, "finite_correlations": True})
        scales[family] = {"scatter_limit": scatter_limit, "histogram_edges": bins.tolist(),
                          "density_upper_limit": ymax * 1.08}
    return scales, {"histogram_bins.csv": pd.DataFrame(histogram_rows),
                    "correlation_matrices.csv": pd.DataFrame(matrix_rows),
                    "plot_checks.csv": pd.DataFrame(check_rows)}


def caption(spec):
    return (f"Seed {SEED}, {spec['condition']}: all 10,000 generated rows, 20 predictors, "
        "9,500 normal-origin and 500 positive-origin rows. Scatter shows the first two features only; "
        "a shift in one of the other 18 features need not be visible here. Histogram shows the largest "
        "absolute feature value in each row, normalized separately within each group; this is a "
        "construction summary, not a fitted detector score. Equal individual feature distributions do "
        "not imply equal distributions of this row maximum. Pearson heatmaps include all 20 features; "
        "boundaries mark four blocks of five. Reported within-block means exclude self-correlations "
        "and average 40 distinct feature pairs. Heatmaps share range [-1,1]; scatter limits, histogram "
        "bins and density scales are fixed within each family. Null labels identify independent groups "
        "with the same population distribution, without a deliberately introduced anomaly signal. "
        "Single-feature shifts also change positive-group correlations, whose target is "
        "0.8/(1+delta^2/20). Sample correlations need not equal their population targets. "
        "Seed 100 was selected in the committed plan; no models were fitted to make these figures.")


def make_figure(case, scale):
    spec, X, y = case["spec"], case["X"], case["y"]
    if spec["family"] == "marginal_shift":
        title = f"Single-feature shift | magnitude {spec['delta']:g}"
    else:
        title = f"Broken relationships | normal correlation {spec['rho']:.1f}"
    if spec["null_control"]:
        title += " | NULL CONTROL"
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 9.2))
    fig.subplots_adjust(left=.085, right=.94, bottom=.09, top=.80, hspace=.49, wspace=.38)
    fig.suptitle(title, x=.5, y=.97, fontsize=16, fontweight="bold")
    fig.text(.5, .928, f"Construction diagnostics | seed {SEED} | all 10,000 rows | 20 features",
             ha="center", fontsize=10, color="#424B54")
    handles = [Line2D([], [], color=COLORS[0], marker="o", linestyle="none", markersize=6,
                      label="Normal origin (9,500 rows)"),
               Line2D([], [], color=COLORS[1], marker="o", markerfacecolor="none", linestyle="none",
                      markersize=6, label="Positive origin (500 rows)")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .907), ncol=2,
               frameon=False, fontsize=10)
    if spec["null_control"]:
        note = "Null control: both groups follow the same distribution; no anomaly pattern was added."
    elif spec["family"] == "marginal_shift":
        note = "Each positive row has one shifted feature; this also weakens its group's correlations."
    else:
        note = "Individual feature distributions match in the population; positive features are independent."
    fig.text(.5, .848, note, ha="center", fontsize=9.3, color="#424B54")

    ax = axes[0, 0]
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=5, alpha=.18, color=COLORS[0],
               edgecolors="none", rasterized=True)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=15, alpha=.75, edgecolors=COLORS[1],
               facecolors="none", linewidths=.6, rasterized=True)
    limit = scale["scatter_limit"]
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), xlabel="x01", ylabel="x02")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("A  |  Two example features", loc="left", fontsize=11, pad=10)

    ax = axes[0, 1]
    for label in (0, 1):
        ax.stairs(case["histograms"][label], scale["histogram_edges"], fill=True,
                  color=COLORS[label], alpha=.30, linewidth=0)
        ax.stairs(case["histograms"][label], scale["histogram_edges"],
                  color=COLORS[label], linewidth=1.4, linestyle="-" if label == 0 else "--")
    ax.set(xlim=(0, scale["histogram_edges"][-1]), ylim=(0, scale["density_upper_limit"]),
           xlabel="Largest absolute feature value per row", ylabel="Density within each group")
    ax.set_title("B  |  Extremeness across all 20 features", loc="left", fontsize=11, pad=10)
    ax.grid(axis="y", color="#E5E8EB", linewidth=.6)
    ax.set_axisbelow(True)

    for label, ax in zip((0, 1), axes[1]):
        im = ax.imshow(case["matrices"][label], cmap="RdBu_r", vmin=-1, vmax=1,
                       interpolation="nearest", origin="upper")
        for boundary in (4.5, 9.5, 14.5):
            ax.axhline(boundary, color="#444444", linewidth=.6, alpha=.5)
            ax.axvline(boundary, color="#444444", linewidth=.6, alpha=.5)
        ticks = [0, 5, 10, 15, 19]
        ax.set_xticks(ticks, [generation.FEATURES[i] for i in ticks], rotation=45, ha="right")
        ax.set_yticks(ticks, [generation.FEATURES[i] for i in ticks])
        within = case["correlations"].query("label == @label and pair_type == 'within_block'").iloc[0]
        panel = "C" if label == 0 else "D"
        ax.set_title(f"{panel}  |  {GROUPS[label]} correlations\n"
                     f"Within-block mean {within.mean_correlation:.3f}; target {within.expected_correlation:.3f}",
                     loc="left", fontsize=10, pad=10)
        bar = fig.colorbar(im, ax=ax, fraction=.046, pad=.035, ticks=[-1, -.5, 0, .5, 1])
        bar.set_label("Pearson correlation", fontsize=9)
        bar.ax.tick_params(labelsize=8)
    for ax in axes.flat:
        ax.tick_params(labelsize=9)
    fig.text(.5, .022, "Shared scales within each family | Histogram is a construction summary | No models fitted",
             ha="center", fontsize=9, color="#424B54")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="v1", help="Same output namespace used for generation.")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,47}", args.run_name):
        parser.error("Run name must start with a lowercase letter and use letters/digits/underscores.")
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        evidence = load_evidence(args.run_name)
        scales, tables = prepare_plot_tables(evidence["cases"])
        figure_dir = ROOT / "figures/synthetic_extension" / args.run_name / f"construction_seed{SEED}"
        table_dir = evidence["result_dir"] / "diagnostics"
        files = {table_dir / name: generation.csv_bytes(table) for name, table in tables.items()}
        with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                             "pdf.fonttype": 42, "axes.spines.top": False,
                             "axes.spines.right": False, "figure.facecolor": "white"}):
            for name, case in evidence["cases"].items():
                figure = make_figure(case, scales[case["spec"]["family"]])
                try:
                    for extension in ("png", "pdf"):
                        buffer = io.BytesIO()
                        metadata = ({"Software": "COPOD synthetic construction diagnostics"} if extension == "png"
                                    else {"Creator": "COPOD synthetic construction diagnostics",
                                          "CreationDate": None, "ModDate": None})
                        figure.savefig(buffer, format=extension, dpi=180, metadata=metadata)
                        files[figure_dir / f"{name}_diagnostics.{extension}"] = buffer.getvalue()
                finally:
                    plt.close(figure)
    warning_log = [{"category": item.category.__name__, "message": str(item.message),
                    "filename": item.filename, "line": item.lineno} for item in recorded]
    source = Path(__file__).resolve()
    evidence["input_sha256"][source.relative_to(ROOT).as_posix()] = digest(source.read_bytes())
    # Check that none of the inputs changed while they were being plotted.
    for key, expected in evidence["input_sha256"].items():
        require(digest((ROOT / key).read_bytes()) == expected, f"Input changed during plotting: {key}")
    metadata = {"schema": "synthetic_extension_construction_figures_v1", "seed": SEED,
        "run_name": args.run_name, "completed_conditions": list(evidence["cases"]),
        "models_run": False, "data_regenerated": False, "all_saved_input_checks": "passed",
        "rows_per_figure": 10000, "row_selection": "all rows; no subsampling",
        "seed_selection": "seed 100 prespecified in the committed plan",
        "scales": scales, "heatmap_range": [-1, 1], "summary_comparison_atol": TOLERANCE,
        "plot_versions": {"python": platform.python_version(), "numpy": np.__version__,
                          "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
        "generation_versions": evidence["manifest"]["versions"],
        "generation_warnings": evidence["generation_warnings"], "plot_warnings": warning_log,
        "captions": {spec["condition"]: caption(spec) for spec in generation.CONDITIONS},
        "input_sha256": evidence["input_sha256"],
        "output_sha256": {path.relative_to(ROOT).as_posix(): digest(content) for path, content in files.items()}}
    files[figure_dir / "figure_metadata.json"] = generation.json_bytes(metadata)
    created = generation.save_or_verify(files)
    print(f"\nCreated {created} files; verified {len(files) - created} existing files.")
    print("Seven construction figures saved in PNG and PDF:", figure_dir.relative_to(ROOT))
    print("Three numerical plot tables:", table_dir.relative_to(ROOT))
    print("Generation warnings:", len(evidence["generation_warnings"]), "| Plotting warnings:", len(warning_log))
    print("No data were regenerated and no models were fitted.")
    print("Open the seven PNGs and review the construction patterns before model evaluation.")


if __name__ == "__main__":
    main()
