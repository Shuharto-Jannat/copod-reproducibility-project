# Reproducibility guide

This guide accompanies the [final report](../report/COPOD_Final_Report.pdf). The [report evidence manifest](../report/evidence_manifest.json) identifies the reviewed repository checkpoint, source archive and final PDF. Dataset acquisition is described in [data_sources.md](data_sources.md).

## What has been checked

On **2 November 2026 (Australia/Sydney)**, the author performed the following checks in a newly created virtual environment inside the existing GitHub Codespaces checkout. The assistant reviewed the supplied terminal output; these were author-run commands, not a separate assistant-host execution.

| Check | Observed result |
|---|---|
| Python | 3.12.1 |
| Installation | All 25 pinned packages in `environment/requirements.txt` installed successfully; pip 23.2.1 |
| Dependency metadata | `pip check`: `No broken requirements found.` |
| Saved LLM evidence | 20 judgments checked; ten existing summary files verified; zero files created; no LLM requests |
| Synthetic construction | Seed 100, all seven conditions: validation passed; 53 files created under the separate `marker_seed100` run name |
| Synthetic evaluation | Three methods across seven conditions: 21 result rows; zero recorded model warnings; input, split, numerical and saved-score checks passed |
| Comparison with frozen seed 100 | All 21 method–condition pairs matched; maximum absolute difference was **0.0** for both ROC AUC and AP (42 metric values; comparison tolerance `1e-12`) |

The machine-readable supplementary record is [environment/reproducibility_check_2026-11-02.json](../environment/reproducibility_check_2026-11-02.json). The original `v1` results remain the report's evidence. The demonstration is a later practical check, not an extra research seed or an amendment to the original experiment.

This verifies installation in a fresh virtual environment and one complete synthetic seed in the author's existing checkout. It does **not** establish that a newly cloned checkout, all ten synthetic seeds, the full original benchmark, all APS experiments or a different operating system have been rerun. Package caches may be reused by pip. Execution times and complete output-file hashes are not expected to match because timing and environment metadata vary.

The report's Appendix C describes checks completed during report preparation. This dated supplement records the subsequent practical check; the earlier report and frozen audit records are not retrospectively rewritten.

## Clone and install

Use Bash in Linux or GitHub Codespaces. Obtain the full Git history and the authors' submodule:

```bash
git clone --recurse-submodules https://github.com/Shuharto-Jannat/copod-reproducibility-project.git
cd copod-reproducibility-project
git submodule status
```

The expected upstream COPOD commit is `3ca17400016386546d2c199886e81bdcc859b36b`. A GitHub source ZIP does not include the submodule contents or Git history. If a clone was made without submodules, initialise them with `git submodule update --init --recursive`.

In a new checkout, with Python 3.12 available:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r environment/requirements.txt
python -m pip check
```

The tested interpreter was specifically Python **3.12.1**. The dependency file preserves the historical versions; compatibility with another Python release is not claimed. Run the following commands from the repository root.

For an existing working environment, create a separate environment instead of replacing it. This is the method used for the practical check:

```bash
copod_check_dir=$(mktemp -d /tmp/copod-check.XXXXXX)
python -m venv "$copod_check_dir/venv"
"$copod_check_dir/venv/bin/python" -m pip install -r environment/requirements.txt --log "$copod_check_dir/install.log"
"$copod_check_dir/venv/bin/python" -m pip check
```

In the remaining commands, use the new environment's interpreter explicitly, or activate it. The commands below use `python` for the selected interpreter.

## Verify saved LLM evidence without new requests

```bash
python scripts/summarise_llm_judge_results.py
```

This standard-library script validates saved evidence relationships and compares generated summary content with the ten committed outputs. It does not fit detectors or call an LLM service. It refuses to replace differing outputs. The expected substantive agreement is 8/8 for each recorded comparison; the two diagnostic claims are reported separately.

Its message about final author adjudication remaining unrecorded belongs to the earlier summary checkpoint. The later [final author adjudication](../results/llm_judge/v1/review/final_author_adjudication.json) supplements that checkpoint. Preserve both records. The reference was AI-assisted and author-approved; it is not an independent human gold standard.

The separate `prepare_llm_judge_evidence.py` script reconstructs input material from Git history, including experiment commit `ed494a3b3c78ecd17f5436f6c9bcfae0050603fe` and protocol commit `3721a7dde5543c53666efa3177eaad270312e65e`. A shallow clone or archive without these commits cannot perform that task. Regeneration is unnecessary for reading or checking the committed judge responses.

## Generate and evaluate one synthetic seed

No external dataset download is needed. The completed `marker_seed100` check is now archived in the repository. To generate a new demonstration and actually refit the methods, use an unused run name such as `marker_seed100_repeat`, preserving both the report's `v1` outputs and the archived check:

```bash
python scripts/generate_synthetic_extension.py --seeds 100 --run-name marker_seed100_repeat
python scripts/evaluate_synthetic_extension.py --seeds 100 --run-name marker_seed100_repeat
```

This generates seven 10,000-row, 20-feature conditions and their annotations, with saved 6,000/4,000 train/test splits. The three evaluated methods are batch-reference COPOD, fixed-training-reference COPOD and Ledoit–Wolf covariance scoring. There are 21 result rows.

- Raw CSVs and annotations: `data_new/synthetic_extension/marker_seed100_repeat/seed100/` (Git-ignored).
- Construction evidence and evaluation: `results/synthetic_extension/marker_seed100_repeat/seed100/`.
- Combined metric table: `results/synthetic_extension/marker_seed100_repeat/seed100/evaluation_metrics.csv`.

If that run name already exists, choose another unused name consistently in both commands and the comparison below. For compatible existing outputs, the generator verifies them and the evaluator checks saved scores and metrics **without refitting the datasets**. Incompatible existing outputs are refused. Rerunning the archived `marker_seed100` name therefore verifies that record rather than demonstrating a new fit. A one-seed check cannot be passed to the ten-seed summariser.

To compare the demonstration's metrics with the preserved original seed:

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

base = Path("results/synthetic_extension")
keys = ["data_seed", "condition", "method"]
metrics = ["roc_auc", "average_precision"]

def load(run):
    frame = pd.read_csv(base / run / "seed100/evaluation_metrics.csv",
                        float_precision="round_trip")
    return frame[keys + metrics].sort_values(keys).reset_index(drop=True)

original, repeated = load("v1"), load("marker_seed100_repeat")
assert len(original) == len(repeated) == 21
pd.testing.assert_frame_equal(original, repeated, check_exact=False,
                              rtol=0, atol=1e-12)
print((original[metrics] - repeated[metrics]).abs().max())
PY
```

## Regenerate the full synthetic extension

The report uses seeds **100–109**, not the scripts' single-seed default. A separate full run is:

```bash
python scripts/generate_synthetic_extension.py --seeds 100 101 102 103 104 105 106 107 108 109 --run-name marker_extension
python scripts/evaluate_synthetic_extension.py --seeds 100 101 102 103 104 105 106 107 108 109 --run-name marker_extension
python scripts/summarise_synthetic_extension.py --run-name marker_extension
python scripts/plot_synthetic_extension_results.py --run-name marker_extension
```

The summary requires all seven conditions for all ten seeds: 210 method–condition–seed rows. Data stay beneath `data_new/synthetic_extension/marker_extension/`, evidence beneath `results/synthetic_extension/marker_extension/`, and result figures beneath `figures/synthetic_extension/marker_extension/results/`. This full sequence was inspected against the CLI definitions but was not repeated during the November one-seed check.

## APS and original benchmark prerequisites

For APS, acquire the exact provider files using the [data-source guide](data_sources.md). The frozen audit binds the training CSV, test CSV, description and original downloaded ZIP. Repacking extracted CSVs creates a different ZIP hash. `evaluate_aps_full_data.py` uses the fixed audit path `results/audits/aps_full_data/v1` even when evaluation uses a new `--run-name`. Creating an audit under another name does not redirect that evaluator. Source mismatches should be investigated and documented, not bypassed by editing frozen hashes.

The legacy `reproduce_benchmark.py` and `reproduce_benchmark_arff.py` scripts execute at top level, import `models.cod` from the authors' code, use repository-root-relative data paths, and write `ap.csv`, `roc.csv`, `prc.csv` and `time.csv` in the current directory. They do not provide a safe `--help` or output-directory interface. Use an isolated checkout and inspect the scripts before a new full benchmark run; the one-seed quick start above avoids these legacy execution constraints.

The current legacy drivers export CSV values with `float_format='%.3f'`, while the preserved historical mean tables retain greater precision. Their current exports therefore do not provide byte-exact replay of those tables. Unseeded sampling and shared randomness also limit exact historical replay. The report analyses preserved evidence and controlled diagnostic reruns; it does not claim that these limitations have been removed.

## Evidence handling

The [root README](../README.md) indexes the primary results and findings. The final report was published after the archived research checkpoint. This guide, the acquisition guide, the evidence manifest and the later practical-check record are supplementary documentation. They do not replace the original inputs, judge responses or `v1` results.

The repository's `proposal/` source belongs to the proposal. An editable LaTeX source for the final PDF is not included in the available final-report package. Do not treat the proposal source or the research source ZIP as the final report's editable source.
