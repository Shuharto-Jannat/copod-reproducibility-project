# Reproducing and Extending COPOD

**[Read the final report](report/COPOD_Final_Report.pdf)** — *Reproducing COPOD: Data Integrity, Scoring Protocols and Limits of Generalisation*.

Shuharto Jannat · COMP8240 · Macquarie University

This project investigates which COPOD results can be reproduced, how data loading and scoring protocols affect the results, and where the method succeeds or fails on new data. It is based on:

Li, Z., Zhao, Y., Botta, N., Ionescu, C., and Hu, X. (2020). **COPOD: Copula-Based Outlier Detection.** IEEE International Conference on Data Mining (ICDM). The [authors' released code](https://github.com/winstonll/COPOD) is retained as a Git submodule.

## Completed work

- Reproduced the released benchmark procedure on 30 dataset variants: 14 MAT and 16 ARFF, using ten methods and ten trials. Compared the saved results with the paper.
- Audited ARFF feature and label selection, investigated MAT reproduction gaps, and examined a possible published AP-table alignment issue. The latter remains a hypothesis, not a confirmed correction to the paper.
- Conducted exploratory pilots on BAF, Scania APS Failure, RT-IoT2022, PhiUSIIL and NATICUSdroid. The completed extension therefore covers more than the original financial-data proposal.
- Evaluated the full APS split using batch-reference COPOD, a fixed-training-reference COPOD variant and Isolation Forest, with missingness sensitivity checks and paired bootstrap intervals.
- Created an initial Gaussian study and a larger synthetic extension: seven conditions, ten data seeds and three methods, including null controls and Ledoit–Wolf covariance scoring.
- Ran two fresh LLM judging sessions over a frozen evidence packet, audited citations and recorded a separate guided final author review.

## Main findings

- Across the 30 benchmark variants, mean COPOD ROC AUC was **0.813238**, compared with **0.824700** in the paper. Reproduction gaps and loader defects require investigation; the released results are not treated as an exact reproduction.
- On the full APS primary test set, Isolation Forest's mean AP across five seeds exceeded both COPOD procedures. These are results for this dataset and evaluation design, not a prospective deployment guarantee.
- In the synthetic dependence setting with correlation 0.8, mean AP was approximately **0.0504** for fixed-reference COPOD and **0.9763** for Ledoit–Wolf. This illustrates a limitation under the constructed mechanism, not universal superiority on real data.
- The two LLM sessions agreed on all **eight substantive verdicts**. Two additional diagnostic claims were evaluated separately. Session A had two recorded citation-locator defects. Agreement is descriptive and does not establish general judge accuracy or independent human validation.

## Evidence navigation

| Question | Start here |
|---|---|
| Where is the final report? | [Final report](report/COPOD_Final_Report.pdf) |
| How close were the benchmark results? | [Dataset comparison](results/comparison/copod_published_vs_reproduced.csv), [aggregate comparison](results/comparison/copod_comparison_summary.csv), [reproduction log](notes/reproduction_log.md) |
| What changed when ARFF loading was corrected? | [Findings](notes/arff_loader_correction_findings.md), [paired results](results/loader_corrections/v1/summary/) |
| What explains the remaining discrepancies? | [MAT diagnostics](notes/mat_reproduction_gap_findings.md), [published AP alignment audit](results/audits/published_ap_alignment/) |
| Why was APS selected after the pilots? | [New-data feasibility and pilot findings](notes/new_data_feasibility.md) |
| What happened on the full APS dataset? | [Findings](notes/aps_full_evaluation_findings.md), [performance summary](results/aps_full_evaluation/v1/summary/performance_summary.csv), [bootstrap evidence](results/aps_full_evaluation/v1/paired_bootstrap_primary/) |
| How were the synthetic data constructed? | [Initial design](notes/synthetic_data_design.md), [extension plan](notes/synthetic_extension_plan.md), [generator](scripts/generate_synthetic_extension.py) |
| What did the synthetic experiments show? | [Initial results](results/synthetic/), [extension findings](notes/synthetic_extension_findings.md), [ten-seed summary](results/synthetic_extension/v1/summary/), [figures](figures/synthetic_extension/v1/) |
| What did the LLM judges actually receive and return? | [Findings and limitations](notes/llm_judge_findings.md), [frozen inputs](results/llm_judge/v1/inputs/), [saved sessions](results/llm_judge/v1/sessions/), [summaries](results/llm_judge/v1/summary/), [final author review](results/llm_judge/v1/review/final_author_adjudication.json) |

## Obtain the repository

Use a full clone with the authors' submodule. Run the remaining commands from the repository root in Bash (for example, GitHub Codespaces or Linux).

```bash
git clone --recurse-submodules https://github.com/Shuharto-Jannat/copod-reproducibility-project.git
cd copod-reproducibility-project
```

A GitHub ZIP omits the submodule contents and Git history. Some provenance scripts, including evidence-packet preparation, require historical commits. The upstream code is pinned to commit `3ca17400016386546d2c199886e81bdcc859b36b`.

## Check the saved LLM summaries

This check uses Python's standard library and the committed evidence; it requires no API key, raw external datasets or new LLM requests:

```bash
python scripts/summarise_llm_judge_results.py
```

It verifies 20 supplied judgments and checks the ten saved summary files. It refuses to replace differing existing outputs. The summaries preserve an earlier checkpoint, so their message about final author adjudication being unrecorded is historical. The later [final author review](results/llm_judge/v1/review/final_author_adjudication.json) is a separate supplementary record.

## Set up the scientific environment

The recorded experimental environment used **Python 3.12.1**. Exact package versions are in [environment/requirements.txt](environment/requirements.txt). For a new clone with Python 3.12 available:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r environment/requirements.txt
python -m pip check
```

At this documentation checkpoint, these instructions reflect the recorded environment; a clean-install check has not yet been completed. The final report distinguishes saved-result verification from a full end-to-end rerun.

## Regenerate a small synthetic example

After setup, the following runs all seven conditions for data seed 100 and evaluates three methods per condition. No external dataset download is needed. A separate run name preserves the submitted `v1` evidence.

```bash
python scripts/generate_synthetic_extension.py --seeds 100 --run-name marker_seed100
python scripts/evaluate_synthetic_extension.py --seeds 100 --run-name marker_seed100
```

Raw datasets and annotations are written beneath `data_new/synthetic_extension/marker_seed100/seed100/`. Validation records, scores and metrics are written beneath `results/synthetic_extension/marker_seed100/`. This is a one-seed demonstration, not the report's full ten-seed analysis.

To regenerate the full extension under another separate run name:

```bash
python scripts/generate_synthetic_extension.py --seeds 100 101 102 103 104 105 106 107 108 109 --run-name marker_extension
python scripts/evaluate_synthetic_extension.py --seeds 100 101 102 103 104 105 106 107 108 109 --run-name marker_extension
python scripts/summarise_synthetic_extension.py --run-name marker_extension
python scripts/plot_synthetic_extension_results.py --run-name marker_extension
```

The ten-seed summariser requires all ten seeds. Runtime and numerical agreement depend on the environment; rerunning is distinct from inspecting the preserved results.

## Data, provenance and limitations

- Original benchmark data are accessed through the authors' submodule. Downloaded external datasets and generated raw synthetic CSVs use `data_new/`, which is Git-ignored. The project preserves generators, validation records, manifests and evaluation evidence; raw data are not all bundled with the repository.
- Third-party datasets retain their source terms. Do not assume that public access grants unrestricted redistribution. Acquisition paths and methodological limitations are recorded in the findings and experiment scripts; see also the report's ethics section.
- The five exploratory pilots are documented by their scripts and rounded findings notes. They do not have the full score bundles preserved for the later APS and synthetic experiments.
- The legacy benchmark drivers execute at top level and write CSVs into the current directory. Their current three-decimal CSV export also differs from the precision of the preserved historical averages. They are not advertised as byte-exact replay commands; use an isolated checkout for further benchmark work.
- Saved `v1` results, prompts and audit files are historical evidence. Keep new runs separate. The AI-assisted reference, two judge responses and guided author review are explicitly documented; none is presented as an independent human gold standard.

## Repository layout

- `original_code/COPOD/`: pinned upstream code and original benchmark material.
- `scripts/`: project audits, generators, evaluations, summaries and plots.
- `results/`: saved numerical results, scores, manifests and review records.
- `figures/`: generated figures.
- `environment/`: recorded dependency versions.
- `notes/`: plans, findings, limitations and protocol amendments.
- `proposal/`: original proposal and its editable source.
- `report/`: final report PDF.
- `data_new/`: local external and generated data; excluded from Git.
