"""Generate and validate the frozen synthetic extension; no detectors are fitted.

Start with --seeds 100. Other planned seeds are 101 through 109.
Raw datasets/annotations go under data_new/ (Git ignored); validation evidence
goes under results/. Representative diagnostic plots are a separate next step.
"""

import argparse
import hashlib
import io
import json
import platform
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = "notes/synthetic_extension_plan.md"
PLAN_COMMIT = "39a2cb60318ecd73f9e017f03a94adae25a2678b"
PLAN_SHA256 = "8f91155a425bc6714a4c8e47ada30e5a1a5128bdd7b38b0dea4b7609abebbcb9"
PLANNED_SEEDS = tuple(range(100, 110))
N_ROWS, N_FEATURES, N_POSITIVE, BLOCK_SIZE = 10000, 20, 500, 5
N_NORMAL = N_ROWS - N_POSITIVE
FEATURES = [f"x{j + 1:02d}" for j in range(N_FEATURES)]
FAMILIES = ("marginal_shift", "dependence_violation")
STREAM_NAMES = ("normal_latents", "positive_latents", "shift_feature",
                "shift_sign", "row_permutation")
CONDITIONS = tuple(
    {"condition": f"marginal_delta{delta}", "family": FAMILIES[0],
     "rho": 0.8, "delta": float(delta), "null_control": delta == 0}
    for delta in (0, 2, 4, 6)
) + tuple(
    {"condition": f"dependence_rho{tag}", "family": FAMILIES[1],
     "rho": rho, "delta": 0.0, "null_control": rho == 0}
    for tag, rho in (("0p0", 0.0), ("0p4", 0.4), ("0p8", 0.8))
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(content):
    return hashlib.sha256(content).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
            + "\n").encode("utf-8")


def csv_bytes(frame):
    # Seventeen significant digits preserve float64 values on round-trip reads.
    return frame.to_csv(index=False, float_format="%.17g",
                        lineterminator="\n").encode("utf-8")


def array_digest(values):
    array = np.asarray(values)
    dtype = "<f8" if array.dtype.kind == "f" else "<i8"
    canonical = np.ascontiguousarray(array, dtype=dtype)
    header = json_bytes({"shape": list(array.shape), "dtype": dtype})
    return digest(header + canonical.tobytes())


def block_values(shared, residual, rho):
    # repeat, not tile: x01-x05 share the first factor, x06-x10 the next.
    return (np.sqrt(rho) * np.repeat(shared, BLOCK_SIZE, axis=1)
            + np.sqrt(1.0 - rho) * residual)


def make_families(seed):
    """Create fixed independent family streams and paired within-family draws."""
    root_sequence = np.random.SeedSequence([seed, 20261029])
    families = {}
    for name, sequence in zip(FAMILIES, root_sequence.spawn(2)):
        children = sequence.spawn(len(STREAM_NAMES))
        rngs = [np.random.Generator(np.random.PCG64(child))
                for child in children]
        # Factors must be consumed before residuals, even if factors are unused.
        normal_s = rngs[0].standard_normal((N_NORMAL, N_FEATURES // BLOCK_SIZE))
        normal_e = rngs[0].standard_normal((N_NORMAL, N_FEATURES))
        positive_s = rngs[1].standard_normal((N_POSITIVE, N_FEATURES // BLOCK_SIZE))
        positive_e = rngs[1].standard_normal((N_POSITIVE, N_FEATURES))
        selected = np.full(N_ROWS, -1, dtype=np.int64)
        signs = np.zeros(N_ROWS, dtype=np.int64)
        if name == "marginal_shift":
            selected[N_NORMAL:] = rngs[2].integers(0, N_FEATURES, N_POSITIVE)
            signs[N_NORMAL:] = rngs[3].choice([-1, 1], N_POSITIVE)
        order = rngs[4].permutation(N_ROWS)
        source_labels = np.r_[np.zeros(N_NORMAL, dtype=np.int64),
                              np.ones(N_POSITIVE, dtype=np.int64)]
        labels = source_labels[order]
        train, test = train_test_split(np.arange(N_ROWS), test_size=0.4,
                                      stratify=labels, random_state=42)
        require(len(train) == 6000 and len(test) == 4000, "Wrong split sizes.")
        require(np.array_equal(np.sort(np.r_[train, test]), np.arange(N_ROWS)),
                "Split rows must cover every row exactly once.")
        require(labels[train].sum() == 300 and labels[test].sum() == 200,
                "Wrong split class counts.")
        partition = np.full(N_ROWS, "train", dtype="<U5")
        partition[test] = "test"
        position = np.empty(N_ROWS, dtype=np.int64)
        position[train], position[test] = np.arange(6000), np.arange(4000)
        split = pd.DataFrame({"row_index": np.arange(N_ROWS),
                              "source_row_id": order, "label": labels,
                              "split": partition, "partition_position": position})
        latent = {"normal_factors": normal_s, "normal_residuals": normal_e,
                  "positive_factors": positive_s, "positive_residuals": positive_e}
        provenance = {
            "entropy": [seed, 20261029], "family_spawn_key": list(sequence.spawn_key),
            "stream_spawn_keys": {role: list(child.spawn_key)
                                  for role, child in zip(STREAM_NAMES, children)},
            "latent_array_sha256": {key: array_digest(value)
                                    for key, value in latent.items()},
            "source_order_sha256": array_digest(order),
            "selected_features_sha256": array_digest(selected),
            "shift_signs_sha256": array_digest(signs),
            "train_row_order_sha256": array_digest(train),
            "test_row_order_sha256": array_digest(test),
        }
        families[name] = dict(latent, order=order, labels=labels,
                              selected=selected, signs=signs, train=train, test=test,
                              split=split, provenance=provenance)
    return families


def make_case(spec, family):
    normal = block_values(family["normal_factors"], family["normal_residuals"],
                          spec["rho"])
    if spec["family"] == "marginal_shift":
        positive_base = block_values(family["positive_factors"],
                                     family["positive_residuals"], 0.8)
        positive = positive_base.copy()
        if spec["delta"]:
            positive[np.arange(N_POSITIVE), family["selected"][N_NORMAL:]] += (
                family["signs"][N_NORMAL:] * spec["delta"])
    else:
        positive_base = family["positive_residuals"]
        positive = positive_base.copy()
    order, labels = family["order"], family["labels"]
    X = np.vstack([normal, positive])[order]
    base = np.vstack([normal, positive_base])[order]
    selected, signs = family["selected"][order], family["signs"][order]
    shifts = signs * spec["delta"] if spec["delta"] else np.zeros(N_ROWS)
    annotations = pd.DataFrame({
        "row_index": np.arange(N_ROWS), "source_row_id": order,
        "family": spec["family"], "condition": spec["condition"], "label": labels,
        "selected_feature_index": selected, "shift_sign": signs,
        "applied_shift": shifts,
    })
    require(X.shape == (N_ROWS, N_FEATURES) and np.isfinite(X).all(),
            "Invalid feature array.")
    require(np.array_equal(labels, (order >= N_NORMAL).astype(int)),
            "Labels disagree with source-row identities.")
    expected_change = np.zeros_like(X)
    chosen = np.flatnonzero(selected >= 0)
    expected_change[chosen, selected[chosen]] = shifts[chosen]
    np.testing.assert_allclose(X - base, expected_change, rtol=0, atol=1e-12)
    unselected = np.ones_like(X, dtype=bool)
    unselected[chosen, selected[chosen]] = False
    require(np.array_equal(X[unselected], base[unselected]),
            "An unselected coordinate changed.")
    restored = X - expected_change
    np.testing.assert_allclose(restored, base, rtol=0, atol=1e-12)
    if spec["null_control"]:
        require(np.array_equal(X, base), "Null construction changed base values.")
    if spec["family"] == "marginal_shift":
        require(np.isin(signs[labels == 1], [-1, 1]).all(), "Missing shift signs.")
        require(((selected[labels == 1] >= 0) &
                 (selected[labels == 1] < N_FEATURES)).all(), "Invalid selection.")
    else:
        require((selected == -1).all() and (signs == 0).all(),
                "Dependence rows must not have shift annotations.")
    data = pd.DataFrame(X, columns=FEATURES)
    data["label"] = labels
    return {"data": data, "annotations": annotations, "X": X,
            "restored": restored, "base": base,
            "undo_max_abs_difference": float(np.max(np.abs(restored - base)))}


def duplicate_checks(X, family):
    # np.unique compares complete numeric rows; identifiers/labels are excluded.
    _, group, counts = np.unique(X, axis=0, return_inverse=True, return_counts=True)
    train_counts = np.bincount(group[family["train"]], minlength=len(counts))
    test_counts = np.bincount(group[family["test"]], minlength=len(counts))
    positives = np.bincount(group, weights=family["labels"], minlength=len(counts))
    result = {
        "all_extra_copies": int(np.maximum(counts - 1, 0).sum()),
        "training_extra_copies": int(np.maximum(train_counts - 1, 0).sum()),
        "test_extra_copies": int(np.maximum(test_counts - 1, 0).sum()),
        "shared_training_test_profiles": int(((train_counts > 0) &
                                               (test_counts > 0)).sum()),
        "conflicting_label_profiles": int(((positives > 0) &
                                             (positives < counts)).sum()),
    }
    require(not any(result.values()),
            "Unexpected duplicate profiles; investigate without dropping rows: " + str(result))
    return result


def distribution_summaries(X, labels, spec):
    """Population targets are descriptive references, never acceptance thresholds."""
    feature_rows, correlation_rows = [], []
    i, j = np.triu_indices(N_FEATURES, k=1)
    within = i // BLOCK_SIZE == j // BLOCK_SIZE
    require(within.sum() == 40 and (~within).sum() == 150, "Invalid block pairs.")
    for label, group_name in ((0, "normal_origin"), (1, "positive_origin")):
        values = X[labels == label]
        variance, rho = 1.0, spec["rho"]
        if label == 1:
            if spec["family"] == "marginal_shift":
                variance = 1.0 + spec["delta"] ** 2 / N_FEATURES
                rho = 0.8 / variance
            else:
                rho = 0.0
        mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
        require(np.isfinite(sd).all() and (sd > 0).all(), "Invalid sample SD.")
        quantiles = np.quantile(values, [.01, .25, .50, .75, .99], axis=0,
                                method="linear")
        for col, feature in enumerate(FEATURES):
            feature_rows.append({
                "label": label, "group": group_name, "n_rows": len(values),
                "feature": feature, "expected_mean": 0.0,
                "expected_variance": variance, "expected_sd": np.sqrt(variance),
                "mean": mean[col], "sample_sd_ddof1": sd[col],
                "minimum": values[:, col].min(), "q01": quantiles[0, col],
                "q25": quantiles[1, col], "median": quantiles[2, col],
                "q75": quantiles[3, col], "q99": quantiles[4, col],
                "maximum": values[:, col].max(),
            })
        pairs = np.corrcoef(values, rowvar=False)[i, j]
        require(np.isfinite(pairs).all(), "Non-finite Pearson correlation.")
        for kind, selected, expected in (("within_block", within, rho),
                                          ("between_block", ~within, 0.0)):
            observed = pairs[selected]
            correlation_rows.append({
                "label": label, "group": group_name, "n_rows": len(values),
                "pair_type": kind, "n_pairs": len(observed),
                "expected_correlation": expected,
                "mean_correlation": observed.mean(),
                "minimum_correlation": observed.min(),
                "maximum_correlation": observed.max(),
            })
    return pd.DataFrame(feature_rows), pd.DataFrame(correlation_rows)


def verified_csv(frame):
    content = csv_bytes(frame)
    restored = pd.read_csv(io.BytesIO(content), float_precision="round_trip")
    require(list(restored.columns) == list(frame.columns), "CSV column mismatch.")
    require(restored.shape == frame.shape, "CSV shape mismatch.")
    for column in frame:
        require(np.array_equal(restored[column].to_numpy(), frame[column].to_numpy()),
                "CSV round-trip mismatch: " + column)
    return content


def generation_manifest(root, run_name):
    plan = root / PLAN_PATH
    require(plan.is_file() and digest(plan.read_bytes()) == PLAN_SHA256,
            "The plan does not match committed 39a2cb6. Use the committed plan unchanged.")
    return {
        "schema": "synthetic_extension_generation_v1", "run_name": run_name,
        "plan_path": PLAN_PATH, "plan_commit": PLAN_COMMIT, "plan_sha256": PLAN_SHA256,
        "generator_sha256": digest(Path(__file__).read_bytes()),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit-learn": sklearn.__version__},
        "planned_seeds": list(PLANNED_SEEDS), "conditions": list(CONDITIONS),
        "n_rows": N_ROWS, "n_features": N_FEATURES, "n_positive": N_POSITIVE,
        "block_size": BLOCK_SIZE, "feature_columns": FEATURES,
        "rng": "Generator(PCG64(SeedSequence([data_seed, 20261029]) child))",
        "family_order": list(FAMILIES), "within_family_stream_order": list(STREAM_NAMES),
        "latent_draw_order": "(group_rows,4) factors, then (group_rows,20) residuals",
        "factor_rule": "sqrt(rho)*repeat(S,5,axis=1) + sqrt(1-rho)*E",
        "split": {"test_size": 0.4, "stratify": "label", "random_state": 42,
                  "train_rows": 6000, "test_rows": 4000,
                  "train_positives": 300, "test_positives": 200},
        "preprocessing": "none", "sample_sd_ddof": 1, "quantile_method": "linear",
        "csv_float_format": "%.17g", "csv_reader": "pandas float_precision=round_trip",
        "construction_check_rtol": 0.0, "construction_check_atol": 1e-12,
        "label_meaning": {"0": "normal-origin process", "1": "positive-origin process"},
        "null_label_limit": "Null groups have identical population distributions; labels are not distinguishable anomalies.",
        "population_targets": "Descriptive, not thresholds for rejecting/redrawing seeds.",
        "completion_scope": "This manifest describes planned settings; seed_metadata.json records each completed seed.",
        "models_run": False,
    }


def seed_bundle(root, run_name, seed, manifest_content):
    data_dir = root / "data_new" / "synthetic_extension" / run_name / f"seed{seed}"
    result_dir = root / "results" / "synthetic_extension" / run_name / f"seed{seed}"
    families = make_families(seed)
    files, summary_rows, pairing_rows, anchors = {}, [], [], {}
    for spec in CONDITIONS:
        name, family_name = spec["condition"], spec["family"]
        family = families[family_name]
        case = make_case(spec, family)
        X, labels = case["X"], family["labels"]
        duplicates = duplicate_checks(X, family)
        features, correlations = distribution_summaries(X, labels, spec)
        data_path = data_dir / f"{name}.csv"
        annotation_path = data_dir / f"{name}_annotations.csv"
        target = result_dir / name
        split_content = verified_csv(family["split"])
        artifacts = {
            data_path: verified_csv(case["data"]),
            annotation_path: verified_csv(case["annotations"]),
            target / "split.csv": split_content,
            target / "feature_summary.csv": csv_bytes(features),
            target / "correlation_summary.csv": csv_bytes(correlations),
        }
        anchor = anchors.setdefault(family_name, {
            "condition": name, "X": X.copy(), "order": family["order"].copy(),
            "labels": labels.copy(), "split": split_content,
            "selected": case["annotations"]["selected_feature_index"].to_numpy().copy(),
            "signs": case["annotations"]["shift_sign"].to_numpy().copy(),
        })
        require(np.array_equal(family["order"], anchor["order"])
                and np.array_equal(labels, anchor["labels"])
                and split_content == anchor["split"], "Pair identities or splits changed.")
        if family_name == "marginal_shift":
            require(np.array_equal(X[labels == 0], anchor["X"][labels == 0]),
                    "Marginal normal rows changed across levels.")
            require(np.array_equal(case["annotations"]["selected_feature_index"], anchor["selected"])
                    and np.array_equal(case["annotations"]["shift_sign"], anchor["signs"]),
                    "Marginal feature/sign choices changed across levels.")
            np.testing.assert_allclose(case["restored"], anchor["X"], rtol=0, atol=1e-12)
            pair_difference = float(np.max(np.abs(case["restored"] - anchor["X"])))
        else:
            require(np.array_equal(X[labels == 1], anchor["X"][labels == 1]),
                    "Dependence positive rows changed across levels.")
            pair_difference = 0.0
        pairing_rows.append({
            "family": family_name, "condition": name, "anchor_condition": anchor["condition"],
            "source_rows_labels_and_split": "identical",
            "normal_latent_sha256": family["provenance"]["latent_array_sha256"]["normal_residuals"],
            "required_feature_pairing": "passed", "max_paired_recovery_difference": pair_difference,
        })
        metadata = {
            "schema": "synthetic_extension_dataset_v1", "seed": seed, "specification": spec,
            "generation_manifest_sha256": digest(manifest_content),
            "feature_columns": FEATURES, "stream_provenance": family["provenance"],
            "data_path": data_path.relative_to(root).as_posix(),
            "annotation_path": annotation_path.relative_to(root).as_posix(),
            "annotation_join": "row_index is zero-based CSV row position; source_row_id is stable within family/seed",
            "split_order": "Select split, then sort by partition_position to recover train_test_split order",
            "selected_feature_convention": "0=x01,...,19=x20; -1=none; marginal delta0 keeps selected feature and +/-1 sign",
            "output_sha256": {path.relative_to(root).as_posix(): digest(content)
                              for path, content in artifacts.items()},
        }
        artifacts[target / "metadata.json"] = json_bytes(metadata)
        validation = {
            "schema": "synthetic_extension_validation_v1", "seed": seed, "condition": name,
            "integrity_checks": "passed", "models_run": False,
            "metadata_sha256": digest(artifacts[target / "metadata.json"]),
            "data_sha256": digest(artifacts[data_path]),
            "dimensions": [N_ROWS, N_FEATURES], "normal_rows": 9500, "positive_rows": 500,
            "train_rows": 6000, "test_rows": 4000, "train_positives": 300, "test_positives": 200,
            "missing_or_nonfinite_values": 0, "duplicates": duplicates,
            "csv_round_trip": "exact numerical/string value agreement",
            "annotations_and_allowed_coordinate_changes": "passed",
            "undo_shift_max_abs_difference": case["undo_max_abs_difference"],
            "paired_construction": "passed", "null_control": spec["null_control"],
            "null_check": ("Same distribution for both independent groups by construction"
                           if spec["null_control"] else "not_applicable"),
            "distribution_summary_policy": "Population targets recorded; no empirical threshold or score-based seed rejection",
        }
        artifacts[target / "validation.json"] = json_bytes(validation)
        files.update(artifacts)
        within = correlations[correlations["pair_type"] == "within_block"].set_index("label")
        summary_rows.append({
            "seed": seed, "condition": name, "family": family_name,
            "null_control": spec["null_control"], "rho": spec["rho"], "delta": spec["delta"],
            "normal_rows": 9500, "positive_rows": 500, "train_rows": 6000, "test_rows": 4000,
            "normal_within_correlation": within.loc[0, "mean_correlation"],
            "normal_expected_within_correlation": within.loc[0, "expected_correlation"],
            "positive_within_correlation": within.loc[1, "mean_correlation"],
            "positive_expected_within_correlation": within.loc[1, "expected_correlation"],
            "positive_expected_sd": float(features.loc[features["label"] == 1, "expected_sd"].iloc[0]),
            "duplicate_feature_rows": duplicates["all_extra_copies"], "checks": "passed",
        })
    files[result_dir / "validation_summary.csv"] = csv_bytes(pd.DataFrame(summary_rows))
    files[result_dir / "pairing_checks.csv"] = csv_bytes(pd.DataFrame(pairing_rows))
    return files, pd.DataFrame(summary_rows), result_dir


def save_or_verify(files):
    # Preflight the entire requested seed before creating any of its files.
    for path, content in files.items():
        if path.exists() and path.read_bytes() != content:
            raise FileExistsError("Existing output differs; review without overwriting: " + str(path))
    created = 0
    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            # Exclusive creation also prevents accidentally replacing an existing file.
            with path.open("xb") as handle:
                handle.write(content)
            created += 1
    return created


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", choices=PLANNED_SEEDS, default=[100])
    parser.add_argument("--run-name", default="v1",
                        help="Output namespace; use v1 for the planned experiment.")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Do not repeat a seed in the same command.")
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,47}", args.run_name):
        parser.error("Run name must start with a lowercase letter and use letters/digits/underscores.")
    manifest_content = json_bytes(generation_manifest(ROOT, args.run_name))
    result_root = ROOT / "results" / "synthetic_extension" / args.run_name
    manifest_path = result_root / "generation_manifest.json"
    for seed in args.seeds:
        print(f"Generating and validating all seven settings: seed {seed}...", flush=True)
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            files, summary, result_dir = seed_bundle(ROOT, args.run_name, seed, manifest_content)
        warning_log = [{"category": item.category.__name__, "message": str(item.message),
                        "filename": item.filename, "line": item.lineno} for item in recorded]
        metadata = {
            "schema": "synthetic_extension_seed_generation_v1", "seed": seed,
            "generation_manifest_sha256": digest(manifest_content),
            "completed_conditions": [spec["condition"] for spec in CONDITIONS],
            "integrity_checks": "passed", "pairing_checks": "passed", "warnings": warning_log,
            "models_run": False, "diagnostic_plots": "separate step; seed 100 selected by plan",
            "output_sha256": {path.relative_to(ROOT).as_posix(): digest(content)
                              for path, content in files.items()},
        }
        # The completion record is written last, after all referenced files.
        bundle = {manifest_path: manifest_content, **files,
                  result_dir / "seed_metadata.json": json_bytes(metadata)}
        created = save_or_verify(bundle)
        print(summary[["condition", "null_control", "normal_within_correlation",
                       "positive_within_correlation", "positive_expected_within_correlation",
                       "checks"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"Created {created} files; verified {len(bundle) - created} existing files.")
        print("Data:", (ROOT / "data_new" / "synthetic_extension" / args.run_name / f"seed{seed}").relative_to(ROOT))
        print("Evidence:", result_dir.relative_to(ROOT))
        print("Recorded warnings:", len(warning_log))
        if warning_log:
            print("Review warnings in seed_metadata.json before model evaluation.")
    print("\nRequested seeds generated and validated. No detectors were fitted.")
    print("Population targets are descriptive; sample summaries need not match them exactly.")
    print("Next: review seed-100 construction diagnostics before evaluation.")


if __name__ == "__main__":
    main()
