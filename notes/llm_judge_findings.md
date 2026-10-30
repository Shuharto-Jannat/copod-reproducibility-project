# LLM judge evaluation: findings and limitations

Written: 31 October 2026.

Evidence checkpoints: preparation and AI-assisted reference `ed7d4a7`; judge configuration `08e41d3`; Session A `1552432`; Session B `8c3dd48`; summaries `ba01706`; final guided author review `5646a7c`.

## Purpose and scope

I used an LLM judge to assess whether selected statements about my experiments were supported by a fixed evidence packet. This was a check of claim interpretation and evidence use. The judge did not assign anomaly labels, fit detectors or generate the experimental performance results.

The packet contained six evidence blocks and 38 excerpts covering the original benchmark comparison, ARFF loader corrections, MAT reproduction gaps, full APS evaluation and synthetic extension. The underlying experiment baseline was commit `ed494a3b3c78ecd17f5436f6c9bcfae0050603fe`.

I selected ten claims: eight substantive claims (C01–C08) and two predefined diagnostic claims (C09–C10). The diagnostics tested recognition of an incorrect numerical statement and an unsupported transfer to future real-world performance. Their roles and expected categories were kept out of the judge prompts.

This small, selected set does not constitute a review of every statement in the final report.

## Protocol and reference review

The original protocol was committed before the judge sessions. A documented amendment recorded the use of an AI-assisted reference: the assistant drafted assessments and explanations, which I approved before obtaining the judge responses. I therefore describe this as an author-approved AI-assisted reference. The originally planned unaided reference file remains blank and unused.

Two fresh conversations used the same configured judge choice and evidence. Session A presented claims C01–C10; Session B presented C10–C01. The prepared prompts differed in claim order. Submitted copies were preserved separately; their recorded comparison with the frozen prompts found only LF-to-CRLF line-ending conversion.

The judge was instructed to use the supplied evidence and return structured judgments with a verdict, evidence identifiers and locators, a short rationale, any required change and any missing evidence. Internal reference verdicts and diagnostic expectations were not supplied to it.

The saved record contains one supplied response for each session, with ten judgments in each. These are two sessions with one configured judge choice, not two independently selected judges. The available records do not independently establish whether any earlier, unreported regeneration occurred in the external interface.

## Recorded model and session settings

The configuration was committed at `08e41d3` before either judge response was recorded.

| Item | Recorded observation |
|---|---|
| Provider and interface | OpenAI; ChatGPT web, Chat mode |
| Selected model-menu label | GPT-5.6 Sol |
| Composer label | 5.6 Pro |
| Conversation setup | Separate empty conversations outside a project |
| Memory and writing-style reference | Observed off during setup |
| Exact backend/API version, temperature and seed | Not exposed |
| Tools | Prompts prohibited browsing; no search/tool mode was visibly selected during setup, but a hard tool disable was not verified |

The project drafting conversation displayed a different label, GPT-6 Astra Ultra. Different visible labels do not establish independent model families, training histories or error patterns.

Exact thinking-effort equivalence between the two requests was not verified. Session A's later screenshots showed a collapsed Medium label and an expanded 5.6 Pro slider; Session B's initial screenshot showed 5.6 Pro without reopening every setting. I retain these observations without inferring an exact backend configuration.

Session A's submission time was not recorded. Session B's submission time was reported by me as 30 October 2026, 14:30 Australia/Sydney, with minute precision. Response-completion times were not recorded. These are observational records, not a complete provider execution trace.

## Verdicts and agreement

Both sessions returned the same categories for all ten claims. The table below abbreviates the claim wording; the exact statements remain in `inputs/claims.csv`.

| Claim | Statement being assessed, abbreviated | Session A | Session B |
|---|---|---|---|
| C01 | The reproduced mean ROC AUC was below the recorded published mean across the 30 datasets. | SUPPORTED | SUPPORTED |
| C02 | With correct labels in both conditions, correcting ARFF feature selection did not improve mean AP on every affected dataset. | SUPPORTED | SUPPORTED |
| C03 | The MAT diagnostics recovered project averages but did not establish the cause of the paper gaps. | SUPPORTED | SUPPORTED |
| C04 | Isolation Forest's five-seed mean AP exceeded both COPOD procedures on primary APS. | SUPPORTED | SUPPORTED |
| C05 | Removing eight additional high-missingness APS features reduced mean AP for all three methods. | SUPPORTED | SUPPORTED |
| C06 | Both COPOD procedures' ten-seed mean AP increased across the tested shift levels. | SUPPORTED | SUPPORTED |
| C07 | Ledoit-Wolf exceeded fixed-reference COPOD on both metrics in every seed at dependence correlation 0.8. | SUPPORTED | SUPPORTED |
| C08 | Within each synthetic null control, both labelled groups share the same population distribution. | SUPPORTED | SUPPORTED |
| C09 | Fixed-reference COPOD's mean AP at dependence correlation 0.8 was approximately 0.5000. | CONTRADICTED | CONTRADICTED |
| C10 | The synthetic extension establishes Ledoit-Wolf as best for future APS observations. | INSUFFICIENT_EVIDENCE | INSUFFICIENT_EVIDENCE |

For the eight substantive claims, Session A and Session B agreed on 8/8 categories. Each session also agreed with the author-approved AI-assisted initial reference on 8/8 substantive categories. All eight were SUPPORTED, so this subset provides no evidence of how well the judge distinguishes other categories across a broader claim population.

The two diagnostic outcomes are reported separately. Both sessions rejected C09: the saved mean AP is approximately 0.0504, not 0.5000. Both assigned insufficient evidence to C10: results from the synthetic experiments cannot establish best-detector status for future APS observations, and the supplied APS comparison did not evaluate Ledoit-Wolf.

Thus each session matched 2/2 predefined diagnostic expectations. These results describe the supplied responses; they are not an estimate of general judge accuracy.

## Citation checks

Separate AI-assisted citation reviews examined the evidence identifiers, locators and supporting excerpts in the supplied responses. Their recorded findings were summarised as follows.

| Session | Citation entries | Recorded locator defects | Claims affected |
|---|---:|---:|---|
| A | 29 | 2 | C09 |
| B | 31 | 0 | None identified |

Both Session A defects named data row 21 of E05.01 when the intended fixed-reference COPOD result is in data row 20, counting data rows from one and excluding the header. Row 21 contains the Ledoit-Wolf result. The quoted fixed-reference figures and intended condition were correct, but the row-number references were wrong.

These were two defective citation entries for one claim, rather than two incorrect verdicts. The recorded reviews found support for the cited findings and no other locator defects. Session B supplied 31 citation entries, with no locator defects found in its review.

I retained the original responses unchanged and preserved the locator corrections in separate audit records. No replacement response was requested in the recorded workflow to improve these findings. Agreement on verdicts therefore did not imply error-free citation details.

The citation checks were AI-assisted source reviews. The summary script validates their recorded associations and counts; it does not independently repeat the semantic review of every cited passage.

## Final guided author review

After both responses and their summaries were available, I reviewed selected evidence facts with AI-assisted explanations and recommendations. I accepted the proposed verdicts for C01, C02 and C03 individually, then accepted C04–C10 together. This produced ten decisions through four reply events.

My accepted categories were C01–C08 SUPPORTED, C09 CONTRADICTED and C10 INSUFFICIENT_EVIDENCE. Accepting the verdict for C09 rejects its false numerical statement; accepting C10's verdict does not endorse its claim about future performance.

The final record is a guided, AI-assisted post-response review. It does not establish independent inspection of every underlying source row or all 60 citation entries. I did not supply separate personal rationales or exact message timestamps; those fields remain unfilled, while assistant processing times are labelled separately.

The final decisions are preserved in `review/final_author_adjudication.csv` and `review/final_author_adjudication.json`, committed at `5646a7c`. They supplement the earlier reference without turning it into an unaided assessment.

The summaries at `ba01706` correctly recorded final author review as unrecorded at their creation. Those historical outputs remain unchanged; the later adjudication files document the subsequent decisions.

## Interpretation and limitations

The exercise showed consistent categorical interpretation of this particular packet across two supplied responses. It also identified a useful distinction: a supported finding and appropriate verdict can coexist with an incorrect evidence locator.

Several limitations constrain what I can conclude:

- The ten claims were purposefully selected and include only two diagnostic examples. They cannot establish reliability across arbitrary claims or the complete report.
- The initial reference and later author review were AI-assisted. Shared framing and exposure to recommendations limit their independence from the AI judgments.
- Reversing claim order while generating a new response does not isolate an order effect from generation variability. Agreement does not prove the absence of position bias.
- The packet was assembled from project evidence. A judge restricted to that packet cannot independently verify the publication, detect every omitted issue or establish that the experiments themselves were implemented correctly.
- Visible settings and fresh-chat procedures do not establish complete isolation from hidden provider context or identify the exact inference configuration.
- The recorded citation audits are also AI-assisted. Their findings should not be presented as an independent human audit of every citation.

I therefore report descriptive agreement and specific citation findings, without treating them as proof of experimental correctness, independent human validation or a general accuracy score.

## Reproducibility and evidence locations

The procedure is documented in `notes/llm_judge_evaluation_plan.md` and `notes/llm_judge_protocol_amendment_01.md`. The preparation script is `scripts/prepare_llm_judge_evidence.py`.

Under `results/llm_judge/v1/`, the main records are:

- `inputs/`: exact claims, evidence packet, schema, source and preparation manifests, and both frozen prompts.
- `review/`: configuration, AI draft and author-approved reference, session citation audits, and final guided author decisions.
- `sessions/session_A/attempt01/` and `sessions/session_B/attempt01/`: frozen and submitted requests, raw and parsed responses, validation results and session metadata.
- `summary/`: judgment and session inventories, claim comparison, verdict counts, agreement, diagnostic outcomes, citation checks and provenance.

The saved session summaries can be checked with `python scripts/summarise_llm_judge_results.py`. This verifies the original summary input set and recorded audit relationships without fitting detectors or sending LLM requests. The later final-author-review files remain a separate supplementary record.

For the final report, I will describe this component as a limited, documented assessment of claim support. The experimental results remain grounded in the saved data, code, metrics and audit records.