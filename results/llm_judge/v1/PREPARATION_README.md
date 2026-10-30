# Prepared claim-review materials

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
