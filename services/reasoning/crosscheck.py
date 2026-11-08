"""VerifierAgent + DeliveryNotes: cross-check the answer against direct and professional requirements."""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from .intake import IntakeBrief

_STOP = {"the", "a", "an", "to", "of", "in", "on", "for", "and", "or", "it", "this", "that", "is", "are", "be", "with",
         "as", "by", "at", "from", "not", "no", "done", "answers", "still", "works", "should", "must", "that"}


class DeliveryNotes(BaseModel):
    assumptions: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    verified: List[str] = Field(default_factory=list)
    unverified: List[str] = Field(default_factory=list)
    requirements: List[Dict[str, Any]] = Field(default_factory=list)

    def to_markdown(self) -> str:
        lines = ["### Delivery notes", ""]

        def section(title: str, items: List[str]):
            if items:
                lines.append(f"**{title}**")
                lines.extend(f"- {i}" for i in items)
                lines.append("")

        section("Assumptions", self.assumptions)
        section("Limitations", self.limitations)
        section("Verified", self.verified)
        section("Not verified", self.unverified)
        if self.requirements:
            icon = {"met": "met", "partial": "partial", "unmet": "unmet", "unknown": "unknown"}
            lines.append("**Requirements cross-check**")
            for r in self.requirements:
                ev = f" - {r.get('evidence')}" if r.get("evidence") else ""
                lines.append(f"- [{icon.get(r.get('status'), 'unknown')}] ({r.get('kind', 'direct')}) {r.get('req', '')}{ev}")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"


def _keywords(text: str) -> List[str]:
    return [w for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", text.lower()) if w not in _STOP]


def _evidence(req: str, answer: str) -> tuple:
    """(status, evidence) from word overlap between the requirement and the answer text."""
    kws = list(dict.fromkeys(_keywords(req)))
    if not kws or not answer:
        return "unknown", "no answer text to compare"
    low = answer.lower()
    hit = [k for k in kws if k in low]
    ratio = len(hit) / len(kws)
    if ratio >= 0.6 and len(hit) >= 2:
        return "partial", f"answer mentions: {', '.join(hit[:5])}"      # text overlap alone never proves "met"
    if hit:
        return "partial", f"answer touches: {', '.join(hit[:5])}"
    return "unknown", "no evidence in the answer text"


class VerifierAgent:
    def __init__(self, llm_json: Optional[Callable[[str], dict]] = None):
        self.llm_json = llm_json

    def check(self, brief: IntakeBrief, answer_md: str, facts: Optional[dict]) -> DeliveryNotes:
        facts = dict(facts or {})
        answer = answer_md or ""
        tests_ran = bool(facts.get("tests_ran"))
        tests_passed = facts.get("tests_passed") is True
        changed = list(facts.get("files_changed") or [])
        is_change = brief.intent == "change"

        verified: List[str] = []
        unverified: List[str] = []
        limitations: List[str] = []

        if tests_ran and tests_passed:
            verified.append("Tests ran in the sandbox and passed")
        elif tests_ran and facts.get("tests_passed") is False:
            limitations.append("Tests ran and FAILED; the change is not verified")
            unverified.append("Behaviour (tests failed)")
        else:
            if is_change:
                limitations.append("Tests were not run for this change")
            unverified.append("Runtime behaviour (no tests executed)")
        sandbox = facts.get("sandbox")
        if sandbox:
            (verified if (tests_ran and tests_passed and str(sandbox).lower() in ("ok", "passed", "pass")) else unverified).append(
                f"Sandbox: {sandbox}")
        if changed:
            verified.append(f"Files changed: {', '.join(map(str, changed[:10]))}") if tests_ran and tests_passed else \
                unverified.append(f"Files changed but not independently verified: {', '.join(map(str, changed[:10]))}")

        mode = facts.get("retrieval_mode")
        try:
            conf = float(facts["retrieval_confidence"]) if facts.get("retrieval_confidence") is not None else None
        except (TypeError, ValueError):
            conf = None
        thr = 0.45
        if conf is not None and conf < thr:
            limitations.append(f"Code search confidence was low ({conf:.2f}); relevant files may be missing"
                               + (f" (mode: {mode})" if mode else ""))
        if facts.get("vectorize_job"):
            limitations.append(f"A background vector index job ({facts['vectorize_job']}) is pending; "
                               "semantic search will improve once it completes")

        requirements: List[Dict[str, Any]] = []
        direct = list(brief.success_criteria) + list(brief.constraints)
        for req in direct:
            status, ev = _evidence(req, answer)
            if is_change and not changed and status != "unmet":
                status, ev = "unmet", "no files were changed"
            red_green = "fails before" in req.lower()
            if red_green and status != "unmet":
                # Only proven when the tests-first stage saw the test red on stubs AND green after.
                if facts.get("tests_red_first") is True and tests_ran and tests_passed:
                    status, ev = "met", "test was red before the change and green after"
                else:
                    status, ev = "partial", "tests did not record a red-before run" if tests_passed else ev
            elif status == "partial" and tests_ran and tests_passed and is_change:
                status, ev = "met", ev + "; tests passed"
            requirements.append({"req": req, "kind": "direct", "status": status, "evidence": ev})
        for req in brief.nfrs:
            if is_change and tests_ran and tests_passed:
                status, ev = "partial", "tests passed; not independently audited"
            else:
                status, ev = "unknown", "not checked automatically"
            requirements.append({"req": req, "kind": "professional", "status": status, "evidence": ev})

        notes = DeliveryNotes(assumptions=list(brief.assumptions), limitations=limitations, verified=verified,
                              unverified=unverified, requirements=requirements)
        return self._merge_llm(notes, brief, answer, facts)

    def _merge_llm(self, notes: DeliveryNotes, brief: IntakeBrief, answer: str, facts: dict) -> DeliveryNotes:
        """Optional: LLM may add limitations/unverified and refine requirement statuses (never add 'verified')."""
        if not self.llm_json:
            return notes
        try:
            data = self.llm_json("Cross-check requirements vs answer. Reply ONLY JSON {limitations:[str], unverified:[str], "
                                 "requirements:[{req, status, evidence}]}.\nREQS: "
                                 + "; ".join(r["req"] for r in notes.requirements)[:2000] + "\nANSWER:\n" + answer[:6000])
        except Exception:
            return notes
        if not isinstance(data, dict):
            return notes
        for key in ("limitations", "unverified"):
            v = data.get(key)
            if isinstance(v, list):
                cur = getattr(notes, key)
                for s in v[:5]:
                    if isinstance(s, str) and s.strip() and s.strip() not in cur:
                        cur.append(s.strip()[:300])
        rv = data.get("requirements")
        if isinstance(rv, list):
            byreq = {r["req"]: r for r in notes.requirements}
            tests_ok = bool(facts.get("tests_ran")) and facts.get("tests_passed") is True
            for item in rv[:40]:
                if not isinstance(item, dict) or item.get("req") not in byreq:
                    continue
                st = item.get("status")
                if st not in ("met", "partial", "unmet", "unknown"):
                    continue
                if st == "met" and not (tests_ok or byreq[item["req"]]["kind"] == "direct" and facts.get("files_changed")):
                    st = "partial"
                byreq[item["req"]]["status"] = st
                if isinstance(item.get("evidence"), str):
                    byreq[item["req"]]["evidence"] = item["evidence"][:300]
        return notes
