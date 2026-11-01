"""Record the completed seed-100 check and update its README status.

Run once using the temporary verification environment. This records an
author-run check in an existing checkout, not a fresh machine or full rerun.
It checks saved outputs and package metadata; it does not refit any model.
"""

import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "environment/reproducibility_check_2026-11-02.json"
MARKER = ROOT / "results/synthetic_extension/marker_seed100"
ORIGINAL = ROOT / "results/synthetic_extension/v1/seed100/evaluation_metrics.csv"
REPEATED = MARKER / "seed100/evaluation_metrics.csv"
METRICS = ("roc_auc", "average_precision")
OLD_STATUS = (
    "At this documentation checkpoint, these instructions reflect the recorded "
    "environment; a clean-install check has not yet been completed. The final "
    "report distinguishes saved-result verification from a full end-to-end rerun."
)
NEW_STATUS = (
    "On 2 November 2026, installation into a fresh Python 3.12.1 virtual "
    "environment in the existing Codespace passed, with all 25 pinned packages "
    "installed and no broken requirements. The saved LLM-summary check verified "
    "all ten existing files. Synthetic seed 100 was regenerated and evaluated "
    "across seven conditions and three methods: all 21 ROC AUC and AP pairs "
    "matched the original values exactly (maximum absolute difference 0.0). "
    "See the [verification record](environment/reproducibility_check_2026-11-02.json). "
    "This checks one seed in a fresh virtual environment, not a fresh checkout "
    "or a full rerun of every experiment. The report describes the earlier "
    "report-preparation checkpoint; this is a later supplementary check."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(arguments):
    result = subprocess.run(arguments, cwd=ROOT, capture_output=True, text=True)
    require(result.returncode == 0, result.stdout + result.stderr)
    return result.stdout.strip()


def load_metrics(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 21, f"Expected 21 metric rows: {path}")
    values = {}
    for row in rows:
        key = (row["data_seed"], row["condition"], row["method"])
        require(key not in values, f"Duplicate metric key: {key}")
        values[key] = {metric: float(row[metric]) for metric in METRICS}
        require(all(math.isfinite(v) for v in values[key].values()), "Non-finite metric")
    return values


def main():
    require(not DESTINATION.exists(), f"Record already exists: {DESTINATION}")
    readme_path = ROOT / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    require(readme.count(OLD_STATUS) == 1, "Expected README paragraph not found exactly once; nothing changed.")
    require(sys.prefix != sys.base_prefix, "Use the temporary verification virtual environment.")
    require(platform.python_version() == "3.12.1", "Expected Python 3.12.1.")
    config = (Path(sys.prefix) / "pyvenv.cfg").read_text(encoding="utf-8")
    require("include-system-site-packages = false" in config.lower(), "System site packages must be excluded.")

    requirements_path = ROOT / "environment/requirements.txt"
    requirements = dict(line.strip().split("==", 1) for line in
                        requirements_path.read_text().splitlines() if line.strip() and not line.startswith("#"))
    installed = {name: version(name) for name in requirements}
    require(len(requirements) == 25 and installed == requirements, "Installed versions differ from the 25 recorded pins.")
    install_log = Path(sys.prefix).parent / "install.log"
    install_lines = install_log.read_text(encoding="utf-8").splitlines()
    successes = [line for line in install_lines if "Successfully installed " in line]
    require(successes, "Successful installation entry missing from the temporary install.log.")

    pip_check = run([sys.executable, "-m", "pip", "check"])
    llm_check = run([sys.executable, "scripts/summarise_llm_judge_results.py"])
    require("Created 0 files; verified 10 existing files:" in llm_check, "Unexpected LLM-summary verification outcome.")

    original, repeated = load_metrics(ORIGINAL), load_metrics(REPEATED)
    require(original.keys() == repeated.keys(), "Metric keys differ.")
    differences = {metric: max(abs(original[key][metric] - repeated[key][metric])
                              for key in original) for metric in METRICS}
    require(all(value == 0.0 for value in differences.values()), "Metrics no longer match exactly; nothing recorded.")
    metadata = json.loads((MARKER / "seed100/evaluation_metadata.json").read_text())
    require(metadata["checks"] == "passed" and metadata["metric_rows"] == 21,
            "Synthetic evaluation checks are incomplete.")
    require(len(metadata["completed_conditions"]) == 7 and metadata["model_warning_count"] == 0,
            "Unexpected condition count or model warnings.")

    bound_files = [requirements_path, ORIGINAL, Path(__file__).resolve(),
                   ROOT / "scripts/summarise_llm_judge_results.py",
                   ROOT / "scripts/generate_synthetic_extension.py",
                   ROOT / "scripts/evaluate_synthetic_extension.py",
                   ROOT / "scripts/evaluate_synthetic_copod.py"]
    bound_files += sorted(p for p in MARKER.rglob("*") if p.is_file())
    record = {
        "schema": "copod_practical_reproducibility_check_v1",
        "recorded_at": datetime.now(ZoneInfo("Australia/Sydney")).isoformat(timespec="seconds"),
        "checks_performed_on": "2026-11-02",
        "timezone": "Australia/Sydney",
        "provenance": "Author executed the installation, generation and evaluation in Codespaces; this script captures the resulting environment and verifies saved outputs.",
        "scope": "Fresh virtual environment in an existing checkout; pip download cache was available. No new machine, fresh-clone test or full experiment rerun is claimed.",
        "checkout_head_before_record": run(["git", "rev-parse", "HEAD"]),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pip_version": version("pip"),
        "system_site_packages": False,
        "installed_pins": installed,
        "installation_success_entry": successes[-1].split("Successfully installed ", 1)[1],
        "installation_log_sha256": digest(install_log),
        "pip_check_output": pip_check,
        "llm_summary_check_output": llm_check,
        "synthetic_run": "marker_seed100",
        "data_seed": 100,
        "completed_conditions": metadata["completed_conditions"],
        "methods": metadata["methods"],
        "metric_rows": 21,
        "metric_values_compared": 42,
        "metrics_compared": list(METRICS),
        "maximum_absolute_difference": differences,
        "model_warning_count": metadata["model_warning_count"],
        "not_repeated": ["Full ten-seed extension", "Full APS experiments", "30-dataset benchmark", "External-data acquisition", "Formal LLM judging sessions"],
        "file_sha256": {str(path.relative_to(ROOT)): digest(path) for path in bound_files},
    }
    payload = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    with DESTINATION.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
    readme_path.write_text(readme.replace(OLD_STATUS, NEW_STATUS), encoding="utf-8")
    print(f"Saved {DESTINATION.relative_to(ROOT)}")
    print("Updated README verification status.")
    print("25 package pins matched; pip check passed; 10 LLM summary files verified.")
    print("21 ROC AUC/AP pairs matched exactly; maximum differences:", differences)
    print("No models were fitted and no LLM requests were sent by this recording step.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f"Stopped: {error}") from error
