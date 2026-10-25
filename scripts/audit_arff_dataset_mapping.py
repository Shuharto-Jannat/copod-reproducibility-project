"""Audit ARFF column selection and saved metadata; do not run any models."""

from pathlib import Path
import ast
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess

import arff
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "scripts/reproduce_benchmark_arff.py"
AUTHORS = ROOT / "original_code/COPOD"
ORIGINAL = AUTHORS / "results/benchmark_arff.py"
RESULTS = ROOT / "results/reproduction_arff_16_10_trials"
OUTPUT = ROOT / "results/audits/arff_dataset_mapping"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalise(text):
    return "".join(c for c in text.casefold() if c.isalnum())


def configuration(tree):
    assignments = {
        node.targets[0].id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign) and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    names = ast.literal_eval(assignments["file_names"])
    misplaced = ast.literal_eval(assignments["misplaced_list"])

    # Evaluate only strings, __file__, and the three path functions below.
    def path_value(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name) and node.id == "__file__":
            return str(BENCHMARK)
        functions = {"os.path.join": os.path.join, "os.path.dirname": os.path.dirname,
                     "os.path.abspath": os.path.abspath}
        if isinstance(node, ast.Call) and not node.keywords:
            name = ast.unparse(node.func)
            if name in functions:
                return functions[name](*(path_value(arg) for arg in node.args))
        raise ValueError("Unrecognised dataset path expression; inspect the benchmark.")

    paths = [Path(path_value(node)).resolve() for node in assignments["arff_list"].elts]
    require(len(names) == len(paths) == 16, "Expected 16 ARFF datasets.")
    require(len(set(names)) == len(set(paths)) == 16, "Duplicate names or paths.")
    return names, paths, misplaced


def isolated_reader(tree, filename):
    # Compile ONLY the reviewed read_arff helper, not imports or benchmark loops.
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == "read_arff"]
    require(len(functions) == 1, f"Expected one read_arff helper in {filename}.")
    module = ast.Module(body=functions, type_ignores=[])
    namespace = {"np": np, "arff": arff}
    exec(compile(module, str(filename), "exec"), namespace)
    return namespace["read_arff"]


def main():
    tree = ast.parse(BENCHMARK.read_text(encoding="utf-8"))
    original_tree = ast.parse(ORIGINAL.read_text(encoding="utf-8"))
    names, paths, misplaced = configuration(tree)
    reader = isolated_reader(tree, BENCHMARK)
    original_reader = isolated_reader(original_tree, ORIGINAL)
    original_assignments = {node.targets[0].id: node.value for node in original_tree.body
                            if isinstance(node, ast.Assign) and len(node.targets) == 1
                            and isinstance(node.targets[0], ast.Name)}
    original_misplaced = ast.literal_eval(original_assignments["misplaced_list"])
    result_paths = [RESULTS / name for name in ["roc.csv", "ap.csv", "prc.csv", "time.csv"]]
    protected = [BENCHMARK, ORIGINAL, *result_paths, *paths]
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in protected}
    tables = {}
    for path in result_paths:
        table = pd.read_csv(path)
        require(len(table) == 16 and not table["Data"].duplicated().any()
                and set(table["Data"]) == set(names), f"Dataset names differ in {path.name}.")
        tables[path.name] = table.set_index("Data")

    summaries, attributes, disagreements, details = [], [], [], []
    for name, path in zip(names, paths):
        print(f"Checking {name}...", flush=True)
        with path.open(encoding="utf-8") as handle:
            raw = arff.load(handle)
        attrs = raw["attributes"]
        attr_names = [str(item[0]) for item in attrs]
        values = np.asarray(raw["data"], dtype=object)
        require(values.ndim == 2 and values.shape[1] == len(attrs), f"Invalid shape: {name}.")
        ids = [i for i, item in enumerate(attr_names) if item.casefold() == "id"]
        labels = [i for i, item in enumerate(attr_names) if item.casefold() == "outlier"]
        require(len(ids) == len(labels) == 1, f"Expected one id and outlier column: {name}.")
        id_i, label_i = ids[0], labels[0]
        expected_features = [i for i in range(len(attrs)) if i not in [id_i, label_i]]
        raw_labels = values[:, label_i].astype(str)
        require(set(raw_labels) == {"no", "yes"}, f"Unexpected outlier labels: {name}.")
        named_y = (raw_labels == "yes").astype(int)
        named_X = values[:, expected_features].astype(float)

        # Obtain the actual project loader's output, without running the benchmark.
        X, y, _ = reader(str(path), misplaced)
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        used_features = list(range(len(attrs) - 2))
        used_label = len(attrs) - (2 if any(item in str(path) for item in misplaced) else 1)
        expected_legacy_y = np.array([
            0 if str(v) == "no" else 1 if str(v) == "yes" else int(float(v))
            for v in values[:, used_label]
        ])
        require(np.array_equal(X, values[:, used_features].astype(float), equal_nan=True)
                and np.array_equal(y, expected_legacy_y),
                f"Loader selection differs from the reviewed logic: {name}.")
        require(np.isin(y, [0, 1]).all(), f"Non-binary benchmark labels: {name}.")
        original_X, original_y, _ = original_reader(str(path), original_misplaced)
        original_agrees = bool(np.array_equal(X, np.asarray(original_X, dtype=float), equal_nan=True)
                               and np.array_equal(y, original_y))
        metadata = [len(y), X.shape[1], round(float(np.count_nonzero(y) / len(y) * 100), 4)]
        metadata_checks = {
            filename: bool(np.allclose(table.loc[name, ["# Samples", "# Dimensions", "Outlier Perc"]]
                                      .to_numpy(dtype=float), metadata, rtol=0, atol=1e-8))
            for filename, table in tables.items()
        }
        path_name_ok = (normalise(path.parent.name) == normalise(name)
                        and path.name.casefold().startswith(name.casefold() + "_"))
        relation_ok = normalise(raw["relation"].split(",")[0]) == normalise(name)
        missing_features = [attr_names[i] for i in expected_features if i not in used_features]
        unexpected_features = [attr_names[i] for i in used_features if i not in expected_features]
        label_diff = np.flatnonzero(y != named_y)
        issues = []
        if not (path_name_ok and relation_ok):
            issues.append("dataset name/path/header mismatch")
        if used_label != label_i:
            issues.append("wrong label column")
        if missing_features or unexpected_features:
            issues.append("wrong predictor columns")
        if not (np.isfinite(X).all() and np.isfinite(named_X).all()):
            issues.append("non-finite predictors")
        if not all(metadata_checks.values()):
            issues.append("saved metadata mismatch")
        if not original_agrees:
            issues.append("authors/project loader difference")

        summaries.append({
            "dataset": name, "status": "ISSUE" if issues else "PASS", "issues": "; ".join(issues),
            "n_rows": len(y), "n_predictors_loaded": X.shape[1],
            "n_predictors_by_name": len(expected_features),
            "benchmark_label_attribute": attr_names[used_label],
            "benchmark_positive_rows": int(np.count_nonzero(y)),
            "named_outlier_positive_rows": int(named_y.sum()),
            "label_disagreements": len(label_diff),
            "unexpected_predictors": "; ".join(unexpected_features),
            "omitted_predictors": "; ".join(missing_features),
            "all_saved_metadata_match": all(metadata_checks.values()),
            "authors_project_loader_agreement": original_agrees,
        })
        for i, (attribute, datatype) in enumerate(attrs):
            attributes.append({
                "dataset": name, "column_index_zero_based": i, "attribute": attribute,
                "declared_type": json.dumps(datatype), "used_as_predictor": i in used_features,
                "used_as_label": i == used_label, "expected_predictor": i in expected_features,
                "role_by_name": "id" if i == id_i else "outlier" if i == label_i else "predictor",
            })
        for row_i in label_diff:
            disagreements.append({
                "dataset": name, "row_index_zero_based": int(row_i),
                "record_id": str(values[row_i, id_i]), "benchmark_label": int(y[row_i]),
                "named_outlier_label": int(named_y[row_i]),
                "raw_value_used_as_label": str(values[row_i, used_label]),
            })
        details.append({
            "dataset": name, "path": str(path.relative_to(ROOT)), "relation": raw["relation"],
            "id_index_zero_based": id_i, "outlier_index_zero_based": label_i,
            "benchmark_label_index_zero_based": used_label,
            "raw_label_values_used": sorted(set(str(v) for v in values[:, used_label])),
            "metadata_checks": metadata_checks,
        })

    require(hashes == {str(p.relative_to(ROOT)): sha256(p) for p in protected},
            "A protected input changed during the audit.")
    source_commit = subprocess.run(["git", "-C", str(AUTHORS), "rev-parse", "HEAD"],
                                   capture_output=True, text=True, check=True).stdout.strip()
    metadata = {
        "scope": "Current loader and raw-file consistency audit; no models run.",
        "authors_repository": "https://github.com/winstonll/COPOD",
        "authors_checkout_commit": source_commit,
        "input_sha256": hashes, "audit_script_sha256": sha256(Path(__file__)),
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "liac_arff": importlib.metadata.version("liac-arff"),
        "datasets": details,
        "limitations": [
            "Agreement with saved metadata does not prove which exact bytes were used historically.",
            "Matching dimensions can hide an included ID and an omitted measurement.",
            "This audit does not calculate corrected model scores or confirm a paper-table error.",
            "The authors' helper is tested on the same supplied files, not its original path list.",
        ],
    }
    summary = pd.DataFrame(summaries)
    disagreement_columns = ["dataset", "row_index_zero_based", "record_id", "benchmark_label",
                            "named_outlier_label", "raw_value_used_as_label"]
    payloads = {
        "dataset_mapping_summary.csv": summary.to_csv(index=False),
        "attribute_selection.csv": pd.DataFrame(attributes).to_csv(index=False),
        "label_disagreements.csv": pd.DataFrame(disagreements, columns=disagreement_columns).to_csv(index=False),
        "audit_metadata.json": json.dumps(metadata, indent=2, allow_nan=False) + "\n",
    }
    for filename, content in payloads.items():
        path = OUTPUT / filename
        if path.exists() and path.read_bytes() != content.encode("utf-8"):
            raise FileExistsError(f"Different existing audit output: {path}. Archive it before rerunning.")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for filename, content in payloads.items():
        (OUTPUT / filename).write_bytes(content.encode("utf-8"))

    print("\nARFF dataset mapping audit completed; no models were run.")
    print(summary[["dataset", "status", "benchmark_label_attribute", "unexpected_predictors",
                   "omitted_predictors", "label_disagreements"]].to_string(index=False))
    print(f"\nDatasets with issues: {int((summary['status'] == 'ISSUE').sum())} of {len(summary)}")
    print("All four saved tables agree with current loader metadata:", bool(summary["all_saved_metadata_match"].all()))
    print("Authors' and project loaders agree on these files:", bool(summary["authors_project_loader_agreement"].all()))
    print("Original datasets, scripts and historical results are unchanged.")
    print(f"Saved four audit files: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
