# Data sources and acquisition

This guide accompanies the final COPOD report. It documents the paths expected by the saved scripts and distinguishes original benchmark data, downloaded external datasets and project-generated data. Source pages and publisher licensing statements were checked on **2 November 2026**. The historical experiment notes and frozen manifests remain unchanged.

Run commands from the repository root. Follow the [README setup instructions](../README.md) first. Downloaded and generated raw files belong under `data_new/`, which is excluded from Git; their absence from a clone is expected. Do not commit credentials or treat public availability as a blanket redistribution permission.

## Original benchmark data

The authors' [COPOD repository](https://github.com/winstonll/COPOD) is a submodule at `original_code/COPOD/`, pinned to `3ca17400016386546d2c199886e81bdcc859b36b`. A full clone with `--recurse-submodules` retrieves that pinned material; the main repository's ZIP does not include it.

| Material | Expected location |
|---|---|
| Fourteen MAT variants | `original_code/COPOD/data/*.mat` |
| Sixteen selected ARFF variants | Subdirectories of `original_code/COPOD/data/semantic/` and `original_code/COPOD/data/literature/` |
| Exact selected filenames | [MAT driver](../scripts/reproduce_benchmark.py), [ARFF driver](../scripts/reproduce_benchmark_arff.py) |
| Audited ARFF attribute and label mapping | [Mapping evidence](../results/audits/arff_dataset_mapping/), [correction findings](../notes/arff_loader_correction_findings.md) |

The same dataset name can refer to different processed variants. Preserve the pinned files and compare their recorded hashes where available; do not substitute a newer generic download. The upstream software's terms do not automatically determine the terms of every included dataset. The legacy drivers write CSVs in their working directory and have historical precision/provenance limitations; they are not a casual data-download or `--help` entry point.

## External datasets

Use the source's download button, extract the relevant CSV, and place it at the exact case-sensitive path below. Retain the source archive and accompanying documentation locally. The project scripts read local CSVs; `ucimlrepo` and the Kaggle API are not required by them.

| Dataset and official source | File expected by this project | Use in this project |
|---|---|---|
| Bank Account Fraud (BAF): [publisher repository](https://github.com/feedzai/bank-account-fraud), [publisher-linked Kaggle distribution](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022) | `data_new/baf/Base.csv` | Exploratory pilot on the Base variant, not all BAF variants. Download the CSV distribution; follow Kaggle's account/download requirements if prompted. |
| [APS Failure at Scania Trucks, UCI 421](https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks) | `data_new/aps_failure/aps_failure_training_set.csv`, `aps_failure_test_set.csv`, `aps_failure_description.txt`, and `aps_failure.zip` in that same directory | Pilot and full evaluation; see the additional provenance requirements below. |
| [RT-IoT2022, UCI 942](https://archive.ics.uci.edu/dataset/942/rt-iot2022) | `data_new/rt_iot2022/RT_IOT2022.csv` | Exploratory network-traffic pilot. The UCI file listing may display `RT_IOT2022` without an extension; if necessary, rename the extracted CSV to the expected filename without changing its contents. |
| [PhiUSIIL, UCI 967](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset) | `data_new/phiusiil/PhiUSIIL_Phishing_URL_Dataset.csv` | Exploratory phishing pilot. The script reverses the original label convention so phishing is anomaly label 1. |
| [NATICUSdroid, UCI 722](https://archive.ics.uci.edu/dataset/722/naticusdroid+android+permissions+dataset) | `data_new/naticusdroid/data.csv` | Exploratory Android-permissions pilot; malware is label 1. |

The corresponding `inspect_*.py` and `smoke_test_*.py` files are in [scripts/](../scripts/). [The feasibility notes](../notes/new_data_feasibility.md) record sampling, preprocessing, label definitions and rounded pilot outcomes. These five pilots have scripts and notes, rather than the full saved-score bundles used in the later experiments. In particular, the NATICUSdroid pilot removes conflicting duplicate permission profiles and therefore evaluates a restricted population.

## APS: preserve the exact source bytes

Save the downloaded UCI archive itself as `data_new/aps_failure/aps_failure.zip`, then extract its two CSVs and description into the same directory. Renaming the archive is acceptable; rebuilding a ZIP from extracted files changes the archive bytes and will not reproduce its recorded hash. Keep the CSV documentation preambles: the reader uses `skiprows=20`, missing token `na`, and maps `neg` to 0 and `pos` to 1. Negative observations are other truck failures, not a verified healthy population.

The committed [APS audit metadata](../results/audits/aps_full_data/v1/audit_metadata.json) contains SHA-256 values for all four raw files. Check downloaded files before evaluation:

```bash
python - <<'PY'
from pathlib import Path
import hashlib
import json

metadata = json.loads(Path(
    "results/audits/aps_full_data/v1/audit_metadata.json"
).read_text())
for name, expected in metadata["input_sha256"].items():
    if not name.startswith("data_new/aps_failure/"):
        continue
    path = Path(name)
    if not path.is_file():
        raise SystemExit(f"Missing original input: {name}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"Source differs from the recorded input: {name}")
    print("Verified:", name)
PY
```

The audit script initially treats the archive as optional, but the committed `v1` audit included it. The evaluator verifies every input bound by that audit, so replay against `v1` needs the original archive as well as the CSVs and description. Its audit directory is fixed at `results/audits/aps_full_data/v1/`, even when evaluation uses a different run name. A changed source download requires a separately documented audit/evaluation workflow; do not edit frozen hashes to force it through. The [APS findings](../notes/aps_full_evaluation_findings.md) explain the completed analysis.

## Licensing and provenance notes

- **BAF correction:** the early [feasibility note](../notes/new_data_feasibility.md) names CC BY-NC-SA 4.0. The publisher's [datasheet, question 44 and Author Statement](https://github.com/feedzai/bank-account-fraud/blob/main/documents/datasheet.pdf) instead states **CC BY-NC-ND 4.0**. This dated guide records that correction without rewriting the historical note. The publisher's repository code licence and dataset licence are different materials. Obtain the dataset from its publisher-linked distribution; this project does not provide a redistributed modified BAF dataset.
- **APS:** the [UCI source page](https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks) has a licence banner saying **CC BY 4.0**, while the original description reproduced there and included with the archive says **GPL version 3 or later**. Both source statements are recorded; this guide does not resolve their relationship or claim unrestricted redistribution. Preserve the original notice and seek clarification from the source if a new use depends on resolving it.
- **RT-IoT2022, PhiUSIIL and NATICUSdroid:** their linked UCI pages identify **CC BY 4.0**. Retain attribution, source citation and notices when using the data, and describe any transformations. Refer to the publisher terms for the intended use.

This project uses the supplied tabular data for research. It does not revisit phishing URLs, execute malware or network attacks, identify individuals, or establish production safety or fairness from the observed detection metrics. See the report's ethics and limitations sections for the project-specific analysis.

## Project-generated synthetic data

The Gaussian data are generated locally by [the initial generator](../scripts/generate_synthetic_copod_data.py) and [the extension generator](../scripts/generate_synthetic_extension.py). They are controlled numerical constructions, not copies of bank applications or truck records. The [initial design](../notes/synthetic_data_design.md) and [extension plan](../notes/synthetic_extension_plan.md) specify mechanisms, seeds, labels, null controls and paired splits.

A small reproducible extension example requires no external dataset download. Choose an unused run name, such as `marker_seed100_repeat` below. The historical installation check is retained under `marker_seed100`; using that existing name can verify its saved evaluation instead of fitting a new one.

```bash
python scripts/generate_synthetic_extension.py --seeds 100 --run-name marker_seed100_repeat
python scripts/evaluate_synthetic_extension.py --seeds 100 --run-name marker_seed100_repeat
```

Raw CSVs and annotations appear under `data_new/synthetic_extension/marker_seed100_repeat/seed100/`; construction checks and evaluation evidence appear under `results/synthetic_extension/marker_seed100_repeat/`. The [README](../README.md) also gives the full seed-100–109 commands. All ten seeds are required for the ten-seed summary. Keep fresh runs separate from `v1` and the archived `marker_seed100` check, and retain the generated metadata with the raw data when sharing a new synthetic example.

## Provenance boundary

This guide was added after the original experimental checkpoint. Web-source checks establish the documented acquisition routes and visible terms on the date above; they do not prove that a newly downloaded archive has the historical bytes. The APS audit and synthetic manifests provide stronger input bindings than the exploratory pilots. Exact replay claims must follow the actual hashes, environment and execution checks, not only matching dataset names.
