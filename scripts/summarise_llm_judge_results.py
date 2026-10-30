#!/usr/bin/env python3
"""Summarise the two preserved v1 judge responses without making LLM requests.

Run from any directory: python scripts/summarise_llm_judge_results.py
Source checks are the recorded AI-assisted reviews, not new human adjudication.
Only results/llm_judge/v1/summary/ is written; conflicting outputs are refused.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = Path("results/llm_judge/v1")
LABELS = ("SUPPORTED", "PARTIALLY_SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE")
REFERENCE_LABEL = "author-approved AI-assisted reference"
PREPARATION_SHA = "427392d6051a4375d4d0e6f2cf8324ce318503a4e56fda239bc2be03f2545c8f"
CONFIG_SHA = "ced955267c98ea997034c53f6b5fb9b454a69ee6905b94a24ce52ef560d4e65e"
RAW_SHA = {
    "A": "5e54b2bdce7f5e2d1a8c73745fff6bef69d8e8b442e9bfc3de979c4d153dc13d",
    "B": "fe57f80906f891a53ac2c35235187b6b6c1f4f8a8f7b66bd7705cb9e79b9191c",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def csv_bytes(rows):
    require(bool(rows), "Unexpected empty output table")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


class Inputs:
    def __init__(self, root):
        self.root = root
        self.hashes = {}

    def read(self, relative, expected=None):
        relative = Path(relative)
        data = (self.root / relative).read_bytes()
        digest = sha(data)
        require(expected is None or digest == expected, f"Input hash changed: {relative}")
        self.hashes[relative.as_posix()] = digest
        return data

    def raw(self, relative, expected=None):
        return self.read(BASE / relative, expected)

    def json(self, relative, expected=None):
        return json.loads(self.raw(relative, expected))

    def csv(self, relative, expected=None):
        return list(csv.DictReader(io.StringIO(self.raw(relative, expected).decode("utf-8-sig"))))

    def unchanged(self):
        for name, digest in self.hashes.items():
            require(sha((self.root / name).read_bytes()) == digest, f"Input changed during run: {name}")


def unique_map(rows, key, expected):
    require(len(rows) == len(expected), f"Unexpected inventory for {key}")
    result = {r[key]: r for r in rows}
    require(len(result) == len(rows) and set(result) == set(expected), f"Duplicate or missing {key}")
    return result


def build_outputs(root=ROOT):
    """Validate preserved records and calculate deterministic output bytes."""
    inputs = Inputs(root)
    prep = inputs.json("inputs/preparation_manifest.json", PREPARATION_SHA)
    for name, digest in prep["immutable_output_sha256"].items():
        inputs.raw(name, digest)
    inputs.read(prep["protocol_path"], prep["protocol_sha256"])
    config = inputs.json("review/judge_configuration.json", CONFIG_SHA)
    for name, digest in config["input_sha256"].items():
        inputs.raw(name, digest)
    approval = inputs.json("review/author_approval_metadata.json")
    inputs.read(approval["amendment_path"], approval["amendment_sha256"])
    inputs.raw("review/ai_assisted_review_draft.md", approval["ai_draft_sha256"])
    inputs.raw("review/initial_author_reference.csv", prep["editable_template_initial_sha256"]["review/initial_author_reference.csv"])
    reference_rows = inputs.csv("review/author_review_ai_assisted.csv", approval["review_csv_after_approval_sha256"])
    claims = inputs.csv("inputs/claims.csv")
    ids = [c["claim_id"] for c in claims]
    require(ids == [f"C{i:02d}" for i in range(1, 11)], "Unexpected claim registry")
    reference = unique_map(reference_rows, "claim_id", ids)
    roles = inputs.json("review/diagnostic_roles.json")
    substantive = roles["substantive_claims"]
    diagnostics = {r["claim_id"]: r for r in roles["diagnostics"]}
    require(substantive == ids[:8] and set(diagnostics) == set(ids[8:]), "Changed claim roles")
    groups = {"substantive": substantive, "diagnostic": ids[8:]}
    role = {cid: group for group, members in groups.items() for cid in members}
    for c in claims:
        ref = reference[c["claim_id"]]
        require(ref["claim"] == c["claim"], "Reference claim text differs")
        require(ref["author_verdict"] in LABELS and ref["author_decision"] == "accepted", "Unexpected approved reference")
        require(ref["author_reason"] == ref["reviewed_at_with_timezone"] == "", "Record later author review separately")

    prompts = {s: inputs.raw(f"inputs/session_{s}_prompt.txt") for s in ("A", "B")}
    prefix_a, list_a = prompts["A"].decode().rsplit("## Claims to assess\n\n", 1)
    prefix_b, list_b = prompts["B"].decode().rsplit("## Claims to assess\n\n", 1)
    expected_claims = [{"claim_id": c["claim_id"], "claim": c["claim"]} for c in claims]
    require(prefix_a == prefix_b, "Prompt instructions/evidence differ")
    require(json.loads(list_a) == expected_claims and json.loads(list_b) == expected_claims[::-1], "Prompt claims differ")
    source_manifest = inputs.json("inputs/source_manifest.json")
    evidence_ids = {r["item_id"] for r in source_manifest["items"]}
    packet_hash = sha(inputs.raw("inputs/evidence_packet.md"))

    session_rows, judgment_rows, citations = [], [], []
    sessions, reviews = {}, {}
    for session in ("A", "B"):
        relative = f"sessions/session_{session}/attempt01"
        attempt_dirs = sorted(p.name for p in (root / BASE / f"sessions/session_{session}").iterdir() if p.is_dir())
        require(attempt_dirs == ["attempt01"], "Additional attempt folders require an explicit inventory update")
        raw = inputs.raw(f"{relative}/response_raw.txt", RAW_SHA[session])
        text = raw.decode("utf-8-sig").strip()
        fence = text.startswith("```") and text.endswith("```")
        if fence:
            text = "\n".join(text.splitlines()[1:-1])
        response = json.loads(text)
        require(isinstance(response, dict) and isinstance(response.get("judgments"), list), "Invalid response envelope")
        require(response == inputs.json(f"{relative}/response_parsed.json"), "Parsed/raw response mismatch")
        ordered = response["judgments"]
        sessions[session] = unique_map(ordered, "claim_id", ids)
        expected_order = ids if session == "A" else ids[::-1]
        word_counts = {}
        for j in ordered:
            require(all(isinstance(j[k], str) for k in ("claim_id", "verdict", "rationale", "required_change", "missing_evidence")), "Invalid judgment strings")
            require(j["verdict"] in LABELS and isinstance(j["evidence"], list), "Invalid verdict/evidence type")
            for citation in j["evidence"]:
                require(all(isinstance(citation[k], str) for k in ("evidence_id", "locator", "finding")), "Invalid citation fields")
            word_counts[j["claim_id"]] = len(j["rationale"].split())
        returned_order = [j["claim_id"] for j in ordered]
        actual_format = {
            "raw_sha256": sha(raw), "valid_json": True,
            "outer_markdown_fence_removed_for_parsing": fence,
            "judgment_count": 10, "unique_expected_claim_ids": True,
            "all_verdicts_allowed": True, "required_fields_and_types_present": True,
            "requested_order": expected_order, "returned_order": returned_order,
            "order_matches_request": returned_order == expected_order,
            "rationale_word_counts_whitespace_split": word_counts,
            "rationales_over_100_words": [k for k, n in word_counts.items() if n > 100],
            "citation_entry_count": sum(len(j["evidence"]) for j in ordered),
        }
        saved_format = inputs.json(f"{relative}/format_validation.json")
        require(all(saved_format.get(k) == v for k, v in actual_format.items()), f"Saved format check disagrees: {session}")
        frozen = inputs.raw(f"{relative}/request_frozen.txt")
        submitted = inputs.raw(f"{relative}/request_submitted.txt")
        require(frozen == prompts[session], "Frozen request differs from prepared prompt")
        require(submitted == frozen.replace(b"\n", b"\r\n"), "Submitted prompt differs beyond recorded LF-to-CRLF conversion")
        request_check = inputs.json(f"{relative}/request_validation.json")
        request_actual = {
            "frozen_bytes": len(frozen), "submitted_bytes": len(submitted),
            "frozen_sha256": sha(frozen), "submitted_sha256": sha(submitted),
            "byte_identical": submitted == frozen,
            "lf_count_in_frozen": frozen.count(b"\n"), "submitted_crlf_count": submitted.count(b"\r\n"),
            "equal_after_crlf_to_lf_normalisation": submitted.replace(b"\r\n", b"\n") == frozen,
            "normalised_submitted_sha256": sha(submitted.replace(b"\r\n", b"\n")),
        }
        require(all(request_check.get(k) == v for k, v in request_actual.items()), "Saved request comparison disagrees")
        meta = inputs.json(f"{relative}/session_metadata.json")
        for k, v in {"session": session, "attempt": 1, "raw_response_sha256": sha(raw), "frozen_request_sha256": sha(frozen), "submitted_request_sha256": sha(submitted), "configuration_sha256": CONFIG_SHA}.items():
            require(meta.get(k) == v, f"Session metadata mismatch: {session}/{k}")
        audit = inputs.json(f"review/session_{session}_ai_citation_review.json")
        require(audit["raw_response_sha256"] == sha(raw) and audit["evidence_packet_sha256"] == packet_hash, "Citation audit bound to different input")
        require(audit["session"] == session and audit["attempt"] == 1 and audit["raw_response_modified"] is False, "Wrong citation-audit scope")
        reviews[session] = unique_map(audit["claims"], "claim_id", ids)
        session_citations = []
        for j in ordered:
            cid = j["claim_id"]
            review = reviews[session][cid]
            require(review["judge_verdict_preserved"] == j["verdict"], "Review verdict differs")
            checks = review["citation_checks"]
            require(len(checks) == len(j["evidence"]), "Review citation count differs")
            for index, (citation, check) in enumerate(zip(j["evidence"], checks), 1):
                require(check["entry_number"] == index and check["original_evidence_id"] == citation["evidence_id"] and check["original_locator"] == citation["locator"], "Citation review does not match original response")
                resolved = citation["evidence_id"] in evidence_ids
                require(check["excerpt_id_resolves"] is resolved, "Evidence-ID resolution disagrees")
                require(isinstance(check["stated_finding_supported"], bool), "Support assessment must be recorded as a Boolean")
                require(check["locator_assessment"] in ("no_error_found", "inaccurate_row_number"), "Unrecognised recorded locator finding")
                row = {
                    "session": session, "claim_id": cid, "claim_group": role[cid], "citation_index": index,
                    "evidence_id": citation["evidence_id"], "original_locator": citation["locator"], "original_finding": citation["finding"],
                    "excerpt_id_resolves": resolved, "finding_supported_in_ai_review": check["stated_finding_supported"],
                    "locator_assessment_in_ai_review": check["locator_assessment"],
                    "review_correction_only": check.get("review_correction_only", ""),
                    "ai_source_review_note": check.get("source_check_note", review["source_review_note"]),
                    "review_type": audit["review_type"], "raw_response_sha256": sha(raw),
                }
                session_citations.append(row)
            judgment_rows.append({
                "session": session, "response_position": returned_order.index(cid) + 1,
                "claim_id": cid, "claim_group": role[cid], "verdict": j["verdict"],
                "rationale": j["rationale"], "rationale_word_count": word_counts[cid],
                "required_change": j["required_change"], "missing_evidence": j["missing_evidence"],
                "citation_count": len(j["evidence"]), "raw_response_sha256": sha(raw),
            })
        defects = [c for c in session_citations if c["locator_assessment_in_ai_review"] != "no_error_found"]
        counts = audit["counts"]
        require(counts["citation_entries"] == len(session_citations) and counts["excerpt_ids_resolve"] == sum(c["excerpt_id_resolves"] for c in session_citations), "Saved audit counts disagree")
        if session == "A":
            require(counts["entries_with_inaccurate_row_locator"] == len(defects) and counts["other_entries_with_no_locator_error_found"] == len(session_citations) - len(defects) and counts["claims_affected_by_row_locator_error"] == len({c["claim_id"] for c in defects}), "A audit totals disagree")
        else:
            require(counts["entries_with_inaccurate_locator_found"] == len(defects) and counts["entries_with_no_locator_error_found"] == len(session_citations) - len(defects) and counts["entries_with_stated_finding_supported"] == sum(c["finding_supported_in_ai_review"] for c in session_citations) and counts["claims_with_citation_defect_found"] == len({c["claim_id"] for c in defects}), "B audit totals disagree")
        citations.extend(session_citations)
        session_rows.append({
            "session": session, "expected_judgments": 10, "available_judgments": len(ordered), "unavailable_judgments": 0,
            "requested_order": ",".join(expected_order), "returned_order": ",".join(returned_order),
            "order_matches": returned_order == expected_order, "rationales_over_100_words": len(actual_format["rationales_over_100_words"]),
            "citation_entries": len(session_citations), "locator_defects_in_ai_review": len(defects),
            "reported_send_time": meta.get("actual_request_sent_at_with_timezone") or "not recorded",
            "send_time_provenance": meta["actual_request_time_status"],
            "response_completion_time": meta.get("actual_response_completed_at_with_timezone") or "not recorded",
            "configured_visible_model_label": meta["configured_visible_model_label"],
            "actual_model_observation": meta["actual_session_model_confirmation"],
            "tool_use_status": meta["actual_tool_use_status"], "prior_attempt_status": meta["earlier_attempts_status"],
            "configuration_sha256": CONFIG_SHA, "raw_response_sha256": sha(raw),
            "frozen_request_sha256": sha(frozen), "submitted_request_sha256": sha(submitted),
            "request_difference": "LF-to-CRLF line endings only", "author_adjudication_status": meta["author_adjudication_status"],
        })

    comparisons, verdict_counts, agreement, diagnostic_rows, citation_summary = [], [], [], [], []
    for claim in claims:
        cid = claim["claim_id"]
        comparisons.append({
            "claim_id": cid, "claim_group": role[cid], "claim": claim["claim"],
            "reference_label": REFERENCE_LABEL, "reference_verdict": reference[cid]["author_verdict"],
            "reference_decision_basis": reference[cid]["decision_basis"],
            "session_A_verdict": sessions["A"][cid]["verdict"], "session_B_verdict": sessions["B"][cid]["verdict"],
            "same_category_A_B": sessions["A"][cid]["verdict"] == sessions["B"][cid]["verdict"],
            "diagnostic_expected_category": diagnostics.get(cid, {}).get("author_defined_expected", ""),
            **{f"session_{s}_citation_defects": sum(c["session"] == s and c["claim_id"] == cid and c["locator_assessment_in_ai_review"] != "no_error_found" for c in citations) for s in ("A", "B")},
            "session_A_ai_source_review": reviews["A"][cid]["source_review_note"],
            "session_B_ai_source_review": reviews["B"][cid]["source_review_note"],
            "final_author_adjudication": "not recorded in these frozen inputs; AI checks are not author adjudication",
        })
    for group, members in groups.items():
        matching = sum(sessions["A"][cid]["verdict"] == sessions["B"][cid]["verdict"] for cid in members)
        agreement.append({"comparison": "session_A_vs_session_B", "claim_group": group, "planned_pairs": len(members), "available_pairs": len(members), "same_category": matching, "different_category": len(members) - matching, "unavailable_pairs": 0, "interpretation": "Observed category agreement across two outputs from the same configured model; no isolated causal order effect or general reliability established."})
        for session in ("A", "B"):
            for label in LABELS:
                verdict_counts.append({"session": session, "claim_group": group, "verdict": label, "count": sum(sessions[session][cid]["verdict"] == label for cid in members), "planned_claims": len(members), "available_judgments": len(members), "unavailable_judgments": 0})
            selected = [c for c in citations if c["session"] == session and c["claim_group"] == group]
            defects = [c for c in selected if c["locator_assessment_in_ai_review"] != "no_error_found"]
            citation_summary.append({"session": session, "claim_group": group, "citation_entries": len(selected), "excerpt_ids_resolve": sum(c["excerpt_id_resolves"] for c in selected), "findings_supported_in_ai_review": sum(c["finding_supported_in_ai_review"] for c in selected), "entries_with_no_locator_error_found": len(selected) - len(defects), "locator_defects": len(defects), "claims_with_locator_defect": len({c["claim_id"] for c in defects}), "review_scope": "Recorded AI-assisted source checks; not human adjudication or a new semantic check by this script."})
    for session in ("A", "B"):
        matching = sum(sessions[session][cid]["verdict"] == reference[cid]["author_verdict"] for cid in substantive)
        agreement.append({"comparison": f"session_{session}_vs_author_approved_ai_assisted_reference", "claim_group": "substantive", "planned_pairs": len(substantive), "available_pairs": len(substantive), "same_category": matching, "different_category": len(substantive) - matching, "unavailable_pairs": 0, "interpretation": "Agreement with blanket author-approved AI drafts; not independent human validation or general judge accuracy."})
        for cid, diagnostic in diagnostics.items():
            j = sessions[session][cid]
            diagnostic_rows.append({"session": session, "claim_id": cid, "diagnostic_role": diagnostic["role"], "author_defined_expected": diagnostic["author_defined_expected"], "observed_verdict": j["verdict"], "matches_expected": j["verdict"] == diagnostic["author_defined_expected"], "required_change": j["required_change"], "missing_evidence": j["missing_evidence"], "interpretation": "One of two predefined diagnostic checks; not a validated benchmark of general judge reliability."})

    tables = {
        "session_inventory.csv": session_rows, "judgment_inventory.csv": judgment_rows,
        "claim_comparison.csv": comparisons, "verdict_counts.csv": verdict_counts,
        "agreement_summary.csv": agreement, "diagnostic_outcomes.csv": diagnostic_rows,
        "citation_checks.csv": citations, "citation_summary.csv": citation_summary,
    }
    outputs = {name: csv_bytes(rows) for name, rows in tables.items()}
    limitations = [
        "Ten deliberately selected claims, split into eight substantive claims and two diagnostics; not an exhaustive report audit.",
        "Two outputs from one configured judge model, not independent judges. Order and stochastic generation are confounded.",
        "Reference is author-approved AI-assisted: blanket approval was documented, but separate claim-by-claim author source checking and personal reasons were not.",
        "Citation findings were recorded by AI-assisted source review. This script verifies their bindings and arithmetic; it does not independently validate their semantic conclusions.",
        "Final author source adjudication is not recorded in these frozen inputs; no author decisions, reasons or review timestamps are filled by this script.",
        "Session A send time is unknown; B send time is author-reported to minute precision. Both completion times are unknown.",
        "Attempt01 denotes the first response supplied to this project conversation; earlier regeneration history is unverified.",
        "Actual hidden tools, provider context, exact backend and inference-time effort equivalence are not established by the supplied UI evidence.",
        "Both submitted prompts differ from their frozen files only in LF-to-CRLF conversion; this does not establish identical tokenization or deterministic responses.",
        "Original session metadata are historical snapshots. Current session inventory is derived from both saved response bundles, without rewriting A's earlier no-B-yet observation.",
        "No significance test, confidence interval, kappa, reliability claim or submission-quality score is calculated.",
    ]
    metadata = {
        "schema": "llm_judge_descriptive_summary_v1", "experiment_baseline_commit": prep["experiment_baseline_commit"],
        "protocol_commit": prep["protocol_commit"], "pre_session_configuration_checkpoint": "08e41d3",
        "session_checkpoints_reported_in_user_git_output": {"A": "1552432", "B": "8c3dd48"},
        "checkpoint_provenance": "Commit identifiers are from user-supplied successful commit/push output; file hashes below bind the actual local inputs.",
        "reference_label": REFERENCE_LABEL, "final_author_adjudication_included": False,
        "script_path": "scripts/summarise_llm_judge_results.py", "script_sha256": sha(Path(__file__).read_bytes()),
        "input_sha256": dict(sorted(inputs.hashes.items())), "table_row_counts": {n: len(r) for n, r in tables.items()},
        "output_table_sha256": {n: sha(b) for n, b in outputs.items()},
        "validation_scope": "Frozen input hashes, exact raw/parsed equivalence, mechanical format, request content, citation-audit binding, reference provenance and descriptive arithmetic. Stored AI source reviews supply the semantic assessments.",
        "computation": "Python standard library; no LLM calls, detectors, data generation or resampling.",
        "limitations": limitations,
    }
    outputs["summary_metadata.json"] = json_bytes(metadata)
    readme = "# LLM judge descriptive summaries\n\nGenerated by `python scripts/summarise_llm_judge_results.py`.\n\n"
    readme += "The eight CSV tables retain all twenty judgments and all sixty citation entries. `claim_comparison.csv` places the exact claims and verdicts side by side; `citation_checks.csv` preserves the original locators and any separate review correction. Use `agreement_summary.csv` for counts with denominators and `diagnostic_outcomes.csv` for the two diagnostic statements.\n\n"
    readme += "These are descriptive results from the two preserved responses. Citation support is taken from the recorded AI-assisted source reviews, not newly decided by the summary script. Final author source adjudication remains a separate step; this directory is generated output and should not be edited as an author-review worksheet.\n\n"
    readme += "## Interpretation limits\n\n" + "\n".join(f"- {item}" for item in limitations) + "\n"
    outputs["SUMMARY_README.md"] = readme.encode("utf-8")
    inputs.unchanged()
    return outputs, tables, inputs


def save_outputs(root, outputs):
    destination = root / BASE / "summary"
    # Check every collision before writing any output.
    for name, data in outputs.items():
        path = destination / name
        require(not path.exists() or path.read_bytes() == data, f"Existing different output: {path}. Nothing replaced.")
    destination.mkdir(parents=True, exist_ok=True)
    created = verified = 0
    for name, data in outputs.items():
        path = destination / name
        if path.exists():
            verified += 1
        else:
            path.write_bytes(data)
            created += 1
        require(path.read_bytes() == data, f"Output verification failed: {path}")
    return created, verified


def main():
    outputs, tables, inputs = build_outputs()
    created, verified = save_outputs(ROOT, outputs)
    inputs.unchanged()
    print("Verified 20 supplied judgments across two sessions; no models or LLM requests were run.")
    for row in tables["agreement_summary.csv"]:
        if row["claim_group"] == "substantive":
            print(f"{row['comparison']}: {row['same_category']}/{row['available_pairs']} substantive categories agree.")
    for session in ("A", "B"):
        diagnostic = [r for r in tables["diagnostic_outcomes.csv"] if r["session"] == session]
        citation = [r for r in tables["citation_summary.csv"] if r["session"] == session]
        print(f"Session {session}: {sum(r['matches_expected'] for r in diagnostic)}/2 diagnostic categories match their predefined expectations; "
              f"{sum(r['citation_entries'] for r in citation)} citations, {sum(r['locator_defects'] for r in citation)} recorded locator defects.")
    print("Agreement is descriptive, not general judge accuracy or independent human validation.")
    print("Final author source adjudication remains unrecorded; no author review fields were filled.")
    print(f"Created {created} files; verified {verified} existing files: {BASE / 'summary'}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, TypeError, OSError) as exc:
        raise SystemExit(f"Stopped: {exc}") from exc