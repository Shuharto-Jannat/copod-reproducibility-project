"""Prepare frozen claim-review inputs; no models, judges or human verdicts run.

Run from the project: python scripts/prepare_llm_judge_evidence.py
Uses only Python's standard library and local Git history. Original data/results
are read from a fixed commit, not from potentially changed working-tree files.
"""

from pathlib import Path
import csv
import hashlib
import io
import json
import math
import re
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
BASELINE = "ed494a3b3c78ecd17f5436f6c9bcfae0050603fe"
PLAN_COMMIT = "3721a7dde5543c53666efa3177eaad270312e65e"
PLAN_PATH = "notes/llm_judge_evaluation_plan.md"
PLAN_SHA256 = "dfa2f9b59ba1de23ae62f18d7e184380b4d835ccbd583a6da33b762e66b8b59f"
OUT = ROOT / "results/llm_judge/v1"
VERDICTS = ["SUPPORTED", "PARTIALLY_SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"]
CONDITIONS = ["marginal_delta0", "marginal_delta2", "marginal_delta4", "marginal_delta6",
              "dependence_rho0p0", "dependence_rho0p4", "dependence_rho0p8"]
METHODS = ["pyod_batch", "fixed_training_reference", "ledoit_wolf"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def csv_text(rows, columns):
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def git_bytes(path, commit=BASELINE):
    result = subprocess.run(["git", "--no-pager", "show", f"{commit}:{path}"],
                            cwd=ROOT, capture_output=True, check=False)
    require(result.returncode == 0,
            f"Cannot read {path} at {commit}. Local Git history must contain the "
            "experiment baseline and committed plan. No files were replaced.")
    return result.stdout


class Packet:
    """Explicit source selections with auditable exclusions and byte hashes."""

    def __init__(self):
        self.cache = {}
        self.items = []
        self.sections = []

    def read(self, path):
        if path not in self.cache:
            self.cache[path] = git_bytes(path)
        return self.cache[path]

    def rows(self, path):
        reader = csv.DictReader(io.StringIO(self.read(path).decode("utf-8")))
        columns = reader.fieldnames
        rows = list(reader)
        require(columns and len(columns) == len(set(columns)), f"Invalid CSV header: {path}")
        require(all(None not in row and all(v is not None for v in row.values()) for row in rows),
                f"Malformed CSV rows: {path}")
        return columns, rows

    def add(self, block, path, title, selection, language, body):
        item_id = f"{block}.{sum(x['evidence_id'] == block for x in self.items) + 1:02d}"
        digest = sha(self.read(path))
        self.items.append({"item_id": item_id, "evidence_id": block, "source_path": path,
                           "source_commit": BASELINE, "source_sha256": digest,
                           "selection": selection, "excerpt_sha256": sha(body.encode("utf-8"))})
        self.sections.append(
            f"### {item_id} — {title}\n\nSource: `{path}`\n\n"
            f"Commit: `{BASELINE}`; source SHA-256: `{digest}`.\n\n"
            f"Selection: {json.dumps(selection, ensure_ascii=False)}\n\n"
            f"```{language}\n{body.rstrip()}\n```\n")

    def table(self, block, path, title, columns=None, where=None, expected=None):
        original_columns, all_rows = self.rows(path)
        columns = columns or original_columns
        require(set(columns) <= set(original_columns), f"Unknown selected columns: {path}")
        where = where or {}
        selected = [(i, row) for i, row in enumerate(all_rows, 1)
                    if all(row[k] in values for k, values in where.items())]
        if expected is not None:
            require(len(selected) == expected, f"Unexpected selected row count: {path}")
        selection = {"row_filter": where or "all rows", "source_data_rows_1based": [i for i, _ in selected],
                     "total_source_rows": len(all_rows), "selected_columns": columns,
                     "omitted_columns": [c for c in original_columns if c not in columns],
                     "omitted_row_count": len(all_rows) - len(selected),
                     "numeric_values": "original CSV strings; no rounding or metric recomputation"}
        self.add(block, path, title, selection, "csv", csv_text([r for _, r in selected], columns))

    def fields(self, block, path, title, pointers):
        data = json.loads(self.read(path))
        values = {}
        for pointer in pointers:
            value = data
            for key in pointer.strip("/").split("/"):
                value = value[key.replace("~1", "/").replace("~0", "~")]
            values[pointer] = value
        self.add(block, path, title,
                 {"json_pointers": pointers, "omissions": "All other JSON fields excluded."},
                 "json", json_text(values))

    def lines(self, block, path, title, ranges):
        source = self.read(path).decode("utf-8").splitlines()
        body = []
        for start, end in ranges:
            require(1 <= start <= end <= len(source), f"Invalid source line range: {path}")
            body.append("\n".join(f"{i}: {source[i - 1]}" for i in range(start, end + 1)))
        self.add(block, path, title,
                 {"inclusive_source_line_ranges": ranges, "source_lines": len(source),
                  "omissions": "All source lines outside the listed ranges excluded."},
                 "text", "\n\n".join(body) + "\n")


def build_evidence(p):
    comp = "results/comparison/"
    p.table("E01", comp + "copod_comparison_summary.csv", "Recorded aggregate ROC AUC",
            ["Metric", "Paper", "Reproduced"], {"Metric": ["Mean ROC-AUC"]}, 1)
    p.table("E01", comp + "copod_published_vs_reproduced.csv", "Thirty-dataset ROC comparison",
            ["Format", "Data", "Paper_ROC", "Reproduced_ROC", "ROC_Difference"], expected=30)

    p.table("E02", "results/audits/arff_dataset_mapping/dataset_mapping_summary.csv",
            "ARFF datasets with mapping issues", where={"status": ["ISSUE"]}, expected=4)
    arff = "results/loader_corrections/v1/summary/"
    p.table("E02", arff + "performance_summary.csv", "All ARFF correction conditions", expected=12)
    p.table("E02", arff + "paired_summary.csv", "Column correction with true labels held fixed",
            where={"comparison": ["column_correction_with_true_labels"]}, expected=4)

    for dataset in ["wine", "wbc", "satimage-2"]:
        prefix = f"results/audits/mat_reproduction_gaps/v1/{dataset}/summary/"
        p.table("E03", prefix + "historical_comparison.csv", f"{dataset}: historical and paper comparison", expected=2)
        p.table("E03", prefix + "all_metrics.csv", f"{dataset}: all original diagnostic trials", expected=10)
        p.table("E03", prefix + "score_stability.csv", f"{dataset}: all-row score stability", expected=10)
        config_keys = ["schema", "python", "package_versions", "authors_commit", "test_fraction",
                       "stratified", "subsampled", "scaling", "scoring_reference", "threshold_policy",
                       "dataset", "n_rows", "n_features", "n_positive", "duplicate_feature_rows"]
        p.fields("E03", prefix + "summary_metadata.json", f"{dataset}: recorded provenance and scope",
                 ["/configuration/" + k for k in config_keys] + ["/split_seeds", "/interpretation_limits"])
    p.lines("E03", "scripts/diagnose_mat_reproduction_gaps.py", "MAT split, scoring and summary logic",
            [[137, 154], [240, 266]])

    aps = "results/aps_full_evaluation/v1/"
    p.table("E04", aps + "summary/all_metrics.csv", "Every full APS model-condition result",
            ["condition", "method", "model_seed", "n_train", "n_test", "n_features", "train_positives",
             "test_positives", "test_prevalence", "roc_auc", "average_precision"], expected=14)
    p.table("E04", aps + "summary/performance_summary.csv", "APS performance across model seeds",
            ["condition", "method", "n_runs", "n_features", "test_prevalence", "score_reference",
             "sd_interpretation", "roc_auc_mean", "roc_auc_sd", "average_precision_mean", "average_precision_sd"],
            expected=6)
    p.table("E04", aps + "summary/preprocessing_summary.csv", "Retained and excluded APS features",
            ["condition", "n_train", "n_test", "n_features", "excluded_features",
             "training_missing_cells_in_retained_features", "test_missing_cells_in_retained_features", "checks"],
            expected=2)
    p.table("E04", aps + "summary/missingness_summary.csv", "Secondary minus primary: method means", expected=3)
    p.table("E04", aps + "summary/missingness_paired_differences.csv", "Secondary minus primary: all paired runs", expected=7)
    p.fields("E04", aps + "run_manifest.json", "Recorded APS split, parameters and label use",
             ["/schema", "/plan_commit", "/planned_conditions", "/isolation_forest_seeds",
              "/isolation_forest_parameters", "/copod_parameters", "/label_use", "/row_reference", "/partitions"])
    for condition in ["primary", "drop_high_missingness"]:
        p.fields("E04", aps + condition + "/preprocessing/metadata.json", condition + ": preprocessing configuration",
                 ["/configuration", "/details/all_rows_retained", "/details/checks"])
    p.lines("E04", "notes/aps_full_evaluation_plan.md", "Prespecified APS protocol and scope",
            [[9, 13], [27, 59], [63, 93]])

    ext = "results/synthetic_extension/v1/"
    p.table("E05", ext + "summary/performance_summary.csv", "All seven settings and three methods", expected=21)
    p.table("E05", ext + "summary/method_paired_differences.csv", "All ten principal relational pairs",
            where={"principal_relational_comparison": ["True"]}, expected=10)
    p.table("E05", ext + "summary/method_paired_summary.csv", "Both method contrasts across all settings", expected=14)
    p.table("E05", ext + "summary/verified_inventory.csv", "Completed seed-condition inventory",
            ["data_seed", "condition", "family", "rho", "delta", "null_control", "n_methods", "n_test",
             "test_positives", "checks"], expected=70)

    p.lines("E06", "notes/synthetic_extension_plan.md", "Construction, pairing, evaluation and limits",
            [[13, 13], [53, 71], [79, 95], [99, 115], [138, 150], [168, 174]])
    p.lines("E06", "scripts/generate_synthetic_extension.py", "Implemented random streams and construction",
            [[24, 42], [73, 107], [135, 158]])
    p.lines("E06", "scripts/evaluate_synthetic_extension.py", "Implemented fitting and test metrics",
            [[431, 451], [465, 477]])
    p.fields("E06", ext + "generation_manifest.json", "Generation specification (planned inventory)",
             ["/conditions", "/planned_seeds", "/n_rows", "/n_features", "/n_positive", "/block_size",
              "/factor_rule", "/label_meaning", "/null_label_limit", "/rng", "/family_order",
              "/within_family_stream_order", "/latent_draw_order", "/split", "/preprocessing",
              "/population_targets", "/completion_scope", "/plan_commit", "/plan_sha256"])
    p.fields("E06", ext + "evaluation_manifest.json", "Recorded synthetic scoring specification",
             ["/methods", "/planned_data_seeds", "/scoring_references", "/copod_parameters",
              "/fixed_reference_probability_floor", "/ledoit_wolf_parameters", "/labels_in_fit",
              "/training_rows", "/training_positives_retained", "/split", "/metrics", "/metrics_use",
              "/preprocessing", "/performance_acceptance_thresholds", "/runtime", "/timing_policy"])
    for condition in ["marginal_delta0", "dependence_rho0p0"]:
        p.fields("E06", ext + "seed100/" + condition + "/metadata.json", condition + ": representative construction provenance",
                 ["/schema", "/seed", "/specification", "/annotation_join", "/selected_feature_convention",
                  "/stream_provenance/entropy", "/stream_provenance/family_spawn_key",
                  "/stream_provenance/stream_spawn_keys"])


def close(actual, expected, label):
    require(math.isfinite(actual) and math.isfinite(expected)
            and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12),
            f"Saved arithmetic disagrees: {label}: {actual} versus {expected}")


def unique_rows(rows, columns, label):
    keys = [tuple(row[c] for c in columns) for row in rows]
    require(len(keys) == len(set(keys)), f"Duplicate keys: {label}")


def verify_tables(p):
    """Check inventories and selected saved arithmetic, not claim verdicts."""
    def rows(path):
        return p.rows(path)[1]

    comp = "results/comparison/"
    table = rows(comp + "copod_published_vs_reproduced.csv")
    unique_rows(table, ["Format", "Data"], "benchmark")
    require(len(table) == 30, "Expected 30 benchmark datasets.")
    aggregate = next(r for r in rows(comp + "copod_comparison_summary.csv") if r["Metric"] == "Mean ROC-AUC")
    for output, source in [("Paper", "Paper_ROC"), ("Reproduced", "Reproduced_ROC")]:
        close(float(aggregate[output]), statistics.mean(float(r[source]) for r in table), "benchmark " + output)

    arff = rows("results/loader_corrections/v1/summary/performance_summary.csv")
    unique_rows(arff, ["dataset", "condition"], "ARFF")
    datasets = {"HeartDisease", "KDDCup99", "Lymphography", "Pima"}
    arms = {"legacy_columns_legacy_labels", "legacy_columns_true_labels", "corrected_columns_true_labels"}
    require({(r["dataset"], r["condition"]) for r in arff} == {(d, a) for d in datasets for a in arms}, "ARFF inventory mismatch")
    paired = rows("results/loader_corrections/v1/summary/paired_summary.csv")
    for r in paired:
        if r["comparison"] == "column_correction_with_true_labels":
            a = next(x for x in arff if x["dataset"] == r["dataset"] and x["condition"] == "legacy_columns_true_labels")
            b = next(x for x in arff if x["dataset"] == r["dataset"] and x["condition"] == "corrected_columns_true_labels")
            for metric in ["roc_auc", "average_precision"]:
                close(float(r[metric + "_difference_mean"]), float(b[metric + "_mean"]) - float(a[metric + "_mean"]), "ARFF paired mean")

    for dataset in ["wine", "wbc", "satimage-2"]:
        prefix = f"results/audits/mat_reproduction_gaps/v1/{dataset}/summary/"
        trials = rows(prefix + "all_metrics.csv")
        require(len(trials) == 10 and {int(r["split_seed"]) for r in trials} == set(range(10)), "MAT seed inventory mismatch")
        for r in rows(prefix + "historical_comparison.csv"):
            metric = r["metric"]
            close(float(r["diagnostic_full_precision_mean"]), statistics.mean(float(t[metric]) for t in trials), "MAT full-precision mean")
            close(float(r["diagnostic_mean_of_trials_rounded_4dp"]), statistics.mean(float(t[metric + "_rounded_4dp"]) for t in trials), "MAT original rounding mean")

    aps = "results/aps_full_evaluation/v1/summary/"
    all_aps = rows(aps + "all_metrics.csv")
    unique_rows(all_aps, ["condition", "method", "model_seed"], "APS results")
    require(len(all_aps) == 14, "Expected 14 APS results.")
    for r in rows(aps + "performance_summary.csv"):
        group = [x for x in all_aps if (x["condition"], x["method"]) == (r["condition"], r["method"])]
        require(len(group) == int(r["n_runs"]), "APS count mismatch")
        if r["method"] == "isolation_forest":
            require({int(x["model_seed"]) for x in group} == set(range(42, 47)), "APS model seeds mismatch")
        for metric in ["roc_auc", "average_precision"]:
            values = [float(x[metric]) for x in group]
            close(float(r[metric + "_mean"]), statistics.mean(values), "APS mean")
            if len(values) > 1:
                close(float(r[metric + "_sd"]), statistics.stdev(values), "APS sample SD")

    ext = "results/synthetic_extension/v1/summary/"
    inventory = rows(ext + "verified_inventory.csv")
    expected = {(str(s), c) for s in range(100, 110) for c in CONDITIONS}
    require(len(inventory) == 70 and {(r["data_seed"], r["condition"]) for r in inventory} == expected,
            "Synthetic inventory mismatch")
    require(all(r["checks"] == "passed" and int(r["n_methods"]) == 3 for r in inventory), "Synthetic recorded inventory check failed")
    perf = rows(ext + "performance_summary.csv")
    require(len(perf) == 21 and {(r["condition"], r["method"]) for r in perf} == {(c, m) for c in CONDITIONS for m in METHODS},
            "Synthetic summary inventory mismatch")
    require(all(int(r["n_seeds"]) == 10 for r in perf), "Synthetic seed count mismatch")
    differences = rows(ext + "method_paired_differences.csv")
    unique_rows(differences, ["comparison", "condition", "data_seed"], "Synthetic paired results")
    for r in differences:
        for metric in ["roc_auc", "average_precision"]:
            close(float(r[metric + "_difference"]), float(r[metric + "_left"]) - float(r[metric + "_right"]), "Synthetic paired subtraction")
    for r in rows(ext + "method_paired_summary.csv"):
        group = [x for x in differences if (x["condition"], x["comparison"]) == (r["condition"], r["comparison"])]
        require(len(group) == int(r["n_pairs"]) == 10 and {int(x["data_seed"]) for x in group} == set(range(100, 110)), "Synthetic paired inventory mismatch")
        for metric in ["roc_auc", "average_precision"]:
            values = [float(x[metric + "_difference"]) for x in group]
            close(float(r[metric + "_difference_mean"]), statistics.mean(values), "Synthetic difference mean")
            close(float(r[metric + "_difference_sd"]), statistics.stdev(values), "Synthetic difference SD")
    return ["30-dataset benchmark inventory and aggregate ROC means",
            "four ARFF datasets, three arms, true-label column differences",
            "three MAT datasets, seeds 0-9, full and original-rounded means",
            "14 APS results, model seeds, saved means and sample SD",
            "70 synthetic realizations, 21 summaries, paired arithmetic and seed inventory"]


def parse_protocol():
    raw = git_bytes(PLAN_PATH, PLAN_COMMIT)
    require(sha(raw) == PLAN_SHA256, "Committed protocol hash mismatch")
    text = raw.decode("utf-8")
    claims = []
    for line in text.splitlines():
        if re.match(r"\| C\d{2} \|", line):
            cells = [c.strip() for c in line.strip("|").split("|")]
            claims.append({"claim_id": cells[0], "claim": cells[1]})
    require([c["claim_id"] for c in claims] == [f"C{i:02d}" for i in range(1, 11)], "Protocol claim inventory mismatch")
    instructions = text.split("```text\n", 1)[1].split("\n```", 1)[0]
    section4 = text.split("## 4. Verdict rubric\n", 1)[1].split("## 5.", 1)[0]
    rubric = section4.split("Before any responses,", 1)[0].strip()
    schema = json.loads(text.split("```json\n", 1)[1].split("\n```", 1)[0])
    citation_rule = text.split("An `INSUFFICIENT_EVIDENCE` response", 1)[1].split("\n\n## 6.", 1)[0]
    citation_rule = "An `INSUFFICIENT_EVIDENCE` response" + citation_rule
    return claims, instructions, rubric, schema, citation_rule


def save_bundle(immutable, editable):
    """Preflight every conflict. Never replace frozen inputs or edited reviews."""
    for relative, data in immutable.items():
        path = OUT / relative
        require(not path.is_symlink(), f"Refusing symbolic-link output: {relative}")
        if path.exists():
            require(path.is_file() and path.read_bytes() == data,
                    f"Existing frozen file differs: {path}. Preserve it and investigate; no overwrite is allowed.")
    created = verified = preserved = 0
    for relative, data in {**immutable, **editable}.items():
        path = OUT / relative
        if path.exists():
            if relative in immutable:
                verified += 1
            else:
                preserved += 1
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
        created += 1
    return created, verified, preserved


def main():
    print("Reading the committed protocol and fixed experiment baseline...", flush=True)
    claims, instructions, rubric, schema, citation_rule = parse_protocol()
    packet = Packet()
    build_evidence(packet)
    checks = verify_tables(packet)
    for check in checks:
        print("Verified:", check)

    glossary = """# Evidence packet

All source excerpts below come from the experiment baseline shown with each item.
E01-E06 are evidence blocks; suffixes such as E04.02 identify exact items.
For citations, give the evidence block and item plus table row/column keys,
JSON pointers or original source line numbers. CSV data-row indices start at 1
after the header. Source hashes refer to complete original files, not excerpts.
Selection records identify omitted material. Numeric CSV values retain their
saved precision. JSON excerpts are reserialized without intentional rounding.

Definitions and reading context:
- ROC AUC evaluates score ranking; higher is better and 0.5 is the random-ranking reference.
- AP means non-interpolated average precision, not classification accuracy.
  AP depends on ranking and class prevalence; prevalence is a reference, not a
  requirement that any finite null run equal it.
- SD is sample standard deviation across the specified seeds, not a confidence interval.
- ARFF arms: legacy_columns_legacy_labels uses both legacy columns and labels;
  legacy_columns_true_labels retains those columns but evaluates with true labels;
  corrected_columns_true_labels uses corrected columns and true labels.
  Column correction compares the last two arms. Label changes alter the answer key.
- MAT historical comparison follows per-trial rounding to four decimals before averaging.
- APS has one supplied train/test split. IF seeds change random model fitting on
  those same rows. Secondary-minus-primary means drop_high_missingness minus primary.
- Synthetic data seeds change generated measurements; methods in a realization
  share the saved split. Conditions within a family are paired. The two families
  have separate null datasets. A null dataset is a construction setting.
- Batch COPOD uses training plus the entire supplied test batch as its scoring reference.
  Fixed-reference COPOD uses training measurements only. Test labels are for evaluation.
- Published benchmark numbers are the project's recorded comparison values;
  this packet does not independently re-extract them from the publication.
- Protocol excerpts state planned rules. Recorded result tables supply completed
  performance; representative construction metadata is explicitly seed 100 only.

The packet is evidence to inspect, not a list of preassigned claim verdicts.
"""
    evidence = glossary + "\n" + "\n".join(packet.sections)
    prefix = (instructions + "\n\n" + rubric + "\n\n## Output format\n\n"
              + "Return one object: {\"judgments\": [one entry per requested claim]}.\n"
              + "Each entry has this schema (replace placeholders):\n\n```json\n"
              + json_text(schema).rstrip() + "\n```\n\n" + citation_rule
              + "\n\n" + evidence + "\n## Claims to assess\n\n")
    prompts = {}
    for name, ordered in [("A", claims), ("B", list(reversed(claims)))]:
        prompts[name] = prefix + json_text(ordered)
    require(prompts["A"].split("## Claims to assess\n\n")[0] == prompts["B"].split("## Claims to assess\n\n")[0],
            "Session instructions/evidence must be identical")
    require(json.loads(prompts["B"].split("## Claims to assess\n\n")[1]) == list(reversed(claims)), "Session B must reverse only the claims")
    for value in [evidence, *prompts.values()]:
        for forbidden in ["numerical-error check", "unsupported-transfer check", "expected_verdict",
                          "author_defined_expected", "Before any responses, the separate control manifest",
                          "llm_judge_evaluation_plan.md", "synthetic_extension_findings.md"]:
            require(forbidden not in value, f"Internal review material leaked into judge input: {forbidden}")

    immutable = {}
    editable = {}

    def add(relative, value, target=immutable):
        target[relative] = value.encode("utf-8")

    add("inputs/claims.csv", csv_text(claims, ["claim_id", "claim"]))
    add("inputs/evidence_packet.md", evidence)
    add("inputs/judgment_entry_schema.json", json_text(schema))
    add("inputs/source_manifest.json", json_text({"baseline_commit": BASELINE, "items": packet.items}))
    for session in ["A", "B"]:
        add(f"inputs/session_{session}_prompt.txt", prompts[session])

    controls = {"internal_only_do_not_send_to_judge": True,
                "source_protocol_commit": PLAN_COMMIT,
                "substantive_claims": [f"C{i:02d}" for i in range(1, 9)],
                "diagnostics": [
                    {"claim_id": "C09", "role": "numerical-error check", "author_defined_expected": "CONTRADICTED"},
                    {"claim_id": "C10", "role": "unsupported-transfer check", "author_defined_expected": "INSUFFICIENT_EVIDENCE"}],
                "limit": "Two author-defined checks do not validate general judge reliability."}
    add("review/diagnostic_roles.json", json_text(controls))
    human_columns = ["claim_id", "claim", "initial_verdict", "evidence_references", "reason", "reviewer", "reviewed_at_with_timezone"]
    add("review/initial_author_reference.csv", csv_text(claims, human_columns), editable)
    config = {"status": "unconfigured_no_judge_requests_made_by_this_script",
              "provider": "", "visible_model_label": "", "api_model_version_if_exposed": "",
              "interface": "", "selection_reason": "", "same_or_different_from_drafting_model": "",
              "temperature_if_exposed": "", "seed_if_exposed": "", "other_visible_settings": "",
              "browsing_tools_setting": "", "cross_chat_memory_setting": "",
              "fresh_session_method": "", "isolation_limitations": "", "recorded_by": "",
              "recorded_at_with_timezone": "", "instructions":
              "Fill before any judge request; unavailable settings must say 'not exposed'. "
              "Use the same observable model/settings in two fresh sessions. "
              "Record each session's actual timestamp and metadata separately when it occurs.",
              "input_sha256": {k: sha(v) for k, v in immutable.items() if k.startswith("inputs/")}}
    add("review/judge_configuration.json", json_text(config), editable)
    readme = """# Prepared claim-review materials

Status: inputs prepared only. No LLM requests, human judgments or agreement results
are produced by this script. Existing review files are preserved without treating
them as complete or valid. Follow notes/llm_judge_evaluation_plan.md.

Before judging:
1. Read inputs/claims.csv and inputs/evidence_packet.md.
2. Shuharto must personally fill review/initial_author_reference.csv using the
   four verdict categories, exact evidence references and a brief reason.
   Empty fields are intentional; the script does not supply substantive answers.
3. Fill review/judge_configuration.json with the chosen model/interface and
   actual visible settings. Use 'not exposed' when a setting is unavailable.
4. Review and commit these inputs, the initial author reference and configuration
   before requesting either judge response. Record that freeze commit later.

For later sessions, send ONLY the complete session_A_prompt.txt or
session_B_prompt.txt in a fresh session. Each already includes the evidence.
Do not attach this README, the plan, source manifest, whole folders, diagnostic
roles, author review, configuration, prior conversation or the other response.
Do not run the judge yet merely because these files exist.

Preserve the original raw responses before parsing. The protocol allows at most
one technical retry per session for transport/truncation/unusable format only.
Substantive disagreement, citation problems or unfavourable verdicts are not retry
reasons. The first complete format-valid response must be retained for analysis.

The source and preparation hashes document which inputs were used. This is a
bounded interpretation audit, not an independent reproduction or a report grade.
The source manifest specifies every table selection, excluded columns/row counts,
JSON selection and code range. Full source files remain at the pinned Git commit.

Rerunning preparation checks existing immutable files byte-for-byte and refuses
to replace a difference. It leaves edited human-reference/configuration files
untouched; their content needs a separate review and freeze before judging.
"""
    add("PREPARATION_README.md", readme)
    manifest = {"schema": "llm_judge_preparation_v1", "experiment_baseline_commit": BASELINE,
                "protocol_commit": PLAN_COMMIT, "protocol_path": PLAN_PATH, "protocol_sha256": PLAN_SHA256,
                "preparation_script_sha256": sha(Path(__file__).read_bytes()),
                "checks": checks + ["identical prompt prefix; only claim order reversed",
                                     "judge-input allowlist; diagnostic roles excluded",
                                     "human verdict/reason fields intentionally blank at creation"],
                "source_file_count": len(packet.cache), "evidence_item_count": len(packet.items),
                "claim_count": 10, "planned_session_count": 2,
                "preparation_performs_judging": False, "preparation_fits_models": False,
                "arithmetic_check_scope": "selected committed summary consistency; not independent rerunning of all prior validations",
                "prompt_bytes": {s: len(prompts[s].encode("utf-8")) for s in prompts},
                "immutable_output_sha256": {k: sha(v) for k, v in immutable.items()},
                "editable_template_initial_sha256": {k: sha(v) for k, v in editable.items()},
                "editable_template_policy": "create only if absent; preserve existing reviews; never treat preserved content as verified",
                "remaining_before_requests": ["review packet", "complete initial author reference",
                                               "record model/settings", "commit frozen inputs and review/configuration"]}
    add("inputs/preparation_manifest.json", json_text(manifest))
    created, verified, preserved = save_bundle(immutable, editable)
    print(f"\nPrepared six evidence blocks, {len(packet.items)} excerpts, ten claims and two prompts.")
    print("Prompt-order and internal-material separation checks: passed.")
    print(f"Created {created} files; verified {verified} frozen files; preserved {preserved} existing editable files.")
    print("Saved: results/llm_judge/v1")
    print("No detectors fitted, LLM requests sent, human verdicts filled or agreement results calculated.")
    print("Next: review the packet, complete your own initial reference, and record the judge configuration.")
    print("Commit those materials before requesting either judge response.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, json.JSONDecodeError, StopIteration) as exc:
        print(f"Preparation stopped: {exc}", file=sys.stderr)
        sys.exit(1)
