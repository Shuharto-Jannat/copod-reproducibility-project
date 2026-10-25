"""Explore Table II row alignment without changing historical results."""

from pathlib import Path
import hashlib
import json
import platform

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = "https://arxiv.org/pdf/2009.09463v1"
METHODS = ["ABOD", "HBOS", "KNN", "LOF", "OCSVM"]

# Table II, page 6: ARFF rows exactly as printed in arXiv v1.
# Columns: printed dataset name, five baseline APs, COPOD AP.
# Checked against the PDF on 2026-10-25; IEEE version not yet checked.
PUBLISHED = [
    ["Arrhythmia",      .6987, .7504, .7118, .7040, .7164, .7519],
    ["Cardiotocography",.2470, .3664, .3108, .2582, .4185, .3778],
    ["HeartDisease",    .5341, .6247, .5384, .4779, .5128, .6400],
    ["Hepatitis",       .3304, .4732, .4752, .4986, .4274, .5849],
    ["InternetAds",     .2760, .5347, .2805, .2622, .3157, .5102],
    ["Ionosphere",      .5055, .5281, .5250, .4584, .4779, .5302],
    ["KDDCup99",        .3798, .5249, .4219, .3489, .4115, .5667],
    ["Lymphography",    .2465, .3983, .3412, .2291, .3464, .4637],
    ["Pima",            .9153, .3637, .9145, .8110, .8215, .7031],
    ["Shuttle",         .0182, .2781, .0457, .0280, .1249, .2389],
    ["SpamBase",        .8013, .9783, .9417, .9254, .8367, .9817],
    ["Stamps",          .2626, .0804, .3597, .3945, .3042, .0854],
    ["Waveform",        .0656, .0547, .1343, .1080, .0702, .0785],
    ["WBC",             .5422, .6935, .5806, .3303, .7018, .8381],
    ["WDBC",            .4300, .7952, .6535, .7038, .6123, .8407],
    ["WPBC",            .2095, .2303, .2328, .2248, .2226, .2438],
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths = {
        "arff": ROOT / "results/reproduction_arff_16_10_trials/ap.csv",
        "mat": ROOT / "results/reproduction_mat_14_10_trials/ap.csv",
        "comparison": ROOT / "results/comparison/copod_published_vs_reproduced.csv",
    }
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in paths.values()}
    tables = {key: pd.read_csv(path) for key, path in paths.items()}
    paper = pd.DataFrame(PUBLISHED, columns=["Data", *METHODS, "COPOD"])
    paper = paper.set_index("Data")

    for key, expected_count in [("arff", 16), ("mat", 14)]:
        table = tables[key]
        require(len(table) == expected_count, f"Unexpected {key} row count.")
        require(not table["Data"].duplicated().any(), f"Duplicate {key} names.")
        tables[key] = table.set_index("Data")
    arff, mat = tables["arff"], tables["mat"]
    comparison = tables["comparison"].copy()
    require(set(arff.index) == set(paper.index), "ARFF dataset names differ.")
    expected_keys = {("ARFF", n) for n in arff.index} | {("MAT", n) for n in mat.index}
    actual_keys = list(zip(comparison["Format"], comparison["Data"]))
    require(len(actual_keys) == 30 and set(actual_keys) == expected_keys,
            "Comparison must contain exactly the expected 30 datasets.")
    for values in [arff[METHODS + ["COD"]], mat[["COD"]],
                   comparison[["Paper_AP", "Reproduced_AP"]]]:
        array = values.to_numpy(dtype=float)
        require(np.isfinite(array).all() and ((array >= 0) & (array <= 1)).all(),
                "AP values must be finite and between zero and one.")
    for _, row in comparison.iterrows():
        table = arff if row["Format"] == "ARFF" else mat
        require(np.isclose(row["Reproduced_AP"], table.loc[row["Data"], "COD"],
                           rtol=0, atol=1e-12), "Saved AP files disagree.")
        if row["Format"] == "ARFF":
            require(np.isclose(row["Paper_AP"], paper.loc[row["Data"], "COPOD"],
                               rtol=0, atol=1e-12), "Paper AP transcription differs.")

    # Match using the FIVE BASELINES only. COPOD is excluded from matching.
    # These baselines reduce sensitivity to random initialization; versions
    # and numerical differences can still affect their results.
    distances = pd.DataFrame(index=paper.index, columns=arff.index, dtype=float)
    rows = []
    for printed_name in paper.index:
        errors = arff[METHODS].sub(paper.loc[printed_name, METHODS]).abs().mean(axis=1)
        distances.loc[printed_name] = errors
        ranked = errors.sort_values(kind="stable")
        require(not np.isclose(ranked.iloc[0], ranked.iloc[1], rtol=0, atol=1e-12),
                f"Ambiguous nearest baseline match for {printed_name}.")
        rows.append({
            "printed_label": printed_name,
            "candidate_dataset": ranked.index[0],
            "same_name_baseline_mae": errors.loc[printed_name],
            "candidate_baseline_mae": ranked.iloc[0],
            "runner_up_dataset": ranked.index[1],
            "runner_up_baseline_mae": ranked.iloc[1],
        })
    matches = pd.DataFrame(rows)
    require(matches["candidate_dataset"].nunique() == 16,
            "Nearest matches are not one-to-one; manual review required.")
    mapping = dict(zip(matches["printed_label"], matches["candidate_dataset"]))
    inverse_mapping = {dataset: label for label, dataset in mapping.items()}

    # Keep the original columns and add explicitly hypothetical comparisons.
    comparison["Candidate_paper_label"] = [
        inverse_mapping[r.Data] if r.Format == "ARFF" else r.Data
        for r in comparison.itertuples()
    ]
    comparison["Candidate_Paper_AP"] = [
        paper.loc[r.Candidate_paper_label, "COPOD"] if r.Format == "ARFF" else r.Paper_AP
        for r in comparison.itertuples()
    ]
    comparison["As_printed_AP_abs_gap"] = (
        comparison["Reproduced_AP"] - comparison["Paper_AP"]
    ).abs()
    comparison["Candidate_AP_abs_gap"] = (
        comparison["Reproduced_AP"] - comparison["Candidate_Paper_AP"]
    ).abs()
    require(np.isclose(comparison["Paper_AP"].mean(),
                       comparison["Candidate_Paper_AP"].mean(), rtol=0, atol=1e-12),
            "A row permutation should preserve the overall paper mean.")

    scope_rows = []
    changed = comparison["Data"] != comparison["Candidate_paper_label"]
    for scope, subset in [
        ("all_30", comparison),
        ("arff_16", comparison[comparison["Format"] == "ARFF"]),
        ("changed_arff_rows", comparison[changed]),
    ]:
        if len(subset):
            scope_rows.append({
                "scope": scope, "n_datasets": len(subset),
                "as_printed_ap_mae": subset["As_printed_AP_abs_gap"].mean(),
                "candidate_ap_mae": subset["Candidate_AP_abs_gap"].mean(),
            })
    summary = pd.DataFrame(scope_rows)
    metadata = {
        "status": "exploratory alignment hypothesis; NOT a confirmed correction",
        "source": SOURCE, "table": "II", "pdf_page": 6,
        "source_checked_date": "2026-10-25", "ieee_version_checked": False,
        "matching_methods": METHODS,
        "matching_rule": "Nearest mean absolute AP difference across five baselines, among all 16 ARFF datasets.",
        "copod_used_for_matching": False,
        "candidate_mapping_printed_to_reproduced": mapping,
        "input_sha256": hashes, "script_sha256": sha256(Path(__file__)),
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "limitations": [
            "Exploratory analysis of existing results, not independent confirmation.",
            "Numeric nearest matches do not establish the correct dataset identities.",
            "Raw dataset mappings and the final IEEE table still require checking.",
            "Only trial means are available here; no confidence intervals are estimated.",
            "ROC results and original comparison files are unchanged.",
        ],
    }
    output = ROOT / "results/audits/published_ap_alignment"
    payloads = {
        "published_arff_ap_as_printed.csv": paper.reset_index().to_csv(index=False),
        "baseline_candidate_matches.csv": matches.to_csv(index=False),
        "baseline_distance_matrix.csv": distances.to_csv(index_label="printed_label"),
        "copod_ap_alignment_hypothesis.csv": comparison.to_csv(index=False),
        "ap_gap_summary.csv": summary.to_csv(index=False),
        "audit_metadata.json": json.dumps(metadata, indent=2, allow_nan=False) + "\n",
    }
    # Refuse to overwrite any different audit output; historical files are inputs only.
    for name, content in payloads.items():
        path = output / name
        if path.exists() and path.read_bytes() != content.encode("utf-8"):
            raise FileExistsError(f"Different existing audit output: {path}. Archive it before rerunning.")
    output.mkdir(parents=True, exist_ok=True)
    for name, content in payloads.items():
        (output / name).write_bytes(content.encode("utf-8"))
    require(hashes == {str(p.relative_to(ROOT)): sha256(p) for p in paths.values()},
            "Input files changed during the audit.")

    print("AP alignment audit: exploratory hypothesis, not a confirmed correction.")
    print("All 16 ARFF rows checked; COPOD excluded from baseline matching.")
    print("\nRows whose nearest baseline match has a different name:")
    print(matches[matches["printed_label"] != matches["candidate_dataset"]].to_string(
        index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nCOPOD mean absolute AP gaps (smaller means closer):")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nOverall paper and reproduced means are unchanged.")
    print("Historical input files are unchanged. No models were rerun.")
    print(f"Saved six audit files: {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
