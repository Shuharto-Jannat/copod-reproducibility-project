# LLM-as-judge protocol amendment 01: AI-assisted author review

Prepared: 30 October 2026 (Australia/Sydney).

Status: prepared for author review and commit before the two planned judge sessions. No completed human review or planned judge response is recorded by this amendment.

## Reason for the change

The original protocol, committed at `3721a7dde5543c53666efa3177eaad270312e65e`, specified that the author would record an initial judgment before seeing AI assessments. After the evidence packet was prepared, Shuharto requested draft assessments and explanations to read and check. The review procedure therefore changes to an explicitly AI-assisted author review.

The assistant will propose categories and reasons using the frozen evidence. Shuharto will personally check the evidence, decide whether to accept or revise each assessment, and record any uncertainty. The resulting reference may be described as author-verified only after that review actually occurs. An AI-written draft alone is not a completed human review.

## Changes to sections 6, 8 and 9 of the original plan

1. Replace the unaided initial-author-reference step with an author review of AI-proposed assessments. The author will have seen draft categories and explanations, so the resulting judgments must not be described as unaided or independent of AI suggestions.
2. Preserve the exact AI draft separately from the author's decisions. The assistant must not fill the author's acceptance, verdict, reason, name or review time before receiving those decisions.
3. Use `results/llm_judge/v1/review/author_review_ai_assisted.csv` for the actual author review. Its author fields are initially blank. The original `initial_author_reference.csv` remains preserved and unused; do not populate it retrospectively as an unaided reference.
4. Freeze and commit the evidence, prompts, this amendment, the AI draft, the completed author review and the chosen judge configuration before either planned judge request. Record that freeze commit with the sessions.
5. Any comparison with the later judge outputs must be labelled agreement with an **author-verified, AI-assisted reference**. This is a descriptive comparison, not independent human validation or a measure of general judge accuracy.
6. Keep the original eight substantive claims and two diagnostic statements separate in reporting. The drafting assistant already knows the diagnostic roles and the project history; its proposed assessments are not an independent test of those checks.

All other protocol provisions continue to apply: the same ten claim texts, four-category rubric, frozen evidence, two isolated sessions with reversed claim order, model selection before requests, technical-retry limits, preservation of raw responses, citation checking and honest reporting of limitations.

## Separation of drafting and planned judging

The draft was prepared in the ongoing Codex project conversation with access to prior work, the protocol and diagnostic expectations. It included automated source checks and assistant-agent source review. These are drafting checks, not either of the planned fresh judge sessions and not an additional human reviewer. The exact backend model/version for this drafting interaction is not exposed in these saved materials.

Keep the draft assessments, this amendment, author review and diagnostic-role manifest outside both judge requests. The two prompt files remain unchanged. Only send the corresponding complete prompt file in each fresh session after the required preparation is committed.

The evidence baseline remains `ed494a3b3c78ecd17f5436f6c9bcfae0050603fe`. The original protocol and experimental results are preserved. This amendment changes the review method before planned judge responses; it does not revise claims or select experiments based on judge outcomes.

## Author review procedure

For each claim, Shuharto should read the exact claim, the draft explanation and the cited evidence, then record:

- `author_decision`: `accepted`, `revised` or `uncertain`.
- `author_verdict`: one of the original four categories when a judgment is reached; leave it blank if still uncertain.
- `author_reason`: a short explanation of the author's decision, including any qualification or disagreement.
- `reviewer` and `reviewed_at_with_timezone`: the actual reviewer and time of the completed review.

Acceptance is appropriate only after checking the evidence. The author may retain a draft reason after verifying it, with its AI-assisted origin preserved. There is no requirement to agree with the assistant. Resolve or explicitly report outstanding uncertainty before treating the author reference as complete.

Later adjudication after seeing the planned judge responses must be recorded separately. It must not overwrite the pre-session author review or original AI draft.

## Reporting and limitations

The final report will disclose that the author reviewed AI-generated draft assessments against saved source evidence. It will describe what review was actually completed and how disagreements were handled. It will not claim that the author independently produced the draft assessments.

Exposure to suggested categories can anchor the author's decisions. Shared model knowledge or model-family tendencies can also increase agreement. Agreement with this assisted reference is therefore weaker evidence of independent corroboration than a genuinely independent human assessment would provide. Agreement does not make a research claim correct; the recorded experimental evidence remains the basis for report wording.

No outcomes or completed review dates should be inserted until the corresponding actions occur. This amendment is additional documentation; the original plan remains available to show the change transparently.
