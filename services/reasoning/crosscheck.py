"""VerifierAgent + DeliveryNotes: cross-check the answer against direct and professional requirements."""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from .intake import IntakeBrief, Lane, recommend_validation_level

_STOP = {"the", "a", "an", "to", "of", "in", "on", "for", "and", "or", "it", "this", "that", "is", "are", "be", "with",
         "as", "by", "at", "from", "not", "no", "done", "answers", "still", "works", "should", "must", "that"}


# Verification labels (skills/risk-proportional-code-dev). A label is only ever derived from facts that say the check ran.
LABEL_SANDBOX = "Verified \u2013 executed in sandbox"
LABEL_CONTAINER = "Verified \u2013 executed in container"
LABEL_BROWSER = "Verified \u2013 executed in headless browser"
LABEL_LIVE = "Verified \u2013 live env"             # + " <name>, user-assisted"
LABEL_STATIC = "Statically checked"
LABEL_UNVERIFIED = "Unverified (user must run)"


def _int(v) -> int:
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return 0


def achieved_level(t: dict) -> str:
    """The repo's own "Verification level" of a sandbox result (mirrors RepoChangeService._verification)."""
    t = t or {}
    if t.get("ran") and t.get("ok"):
        if t.get("mode") == "pytest":
            if _int(t.get("passed")) <= 0:
                return "L0-static"                                    # 0 passed verifies nothing
            return "L3-fail-to-pass" if _int((t.get("proof") or {}).get("proves")) > 0 else "L2-tests"
        return "L1-boot-smoke" if t.get("mode") == "boot" else "L1-import-smoke"
    return "L0-static"


def facts_from_sandbox(t: Optional[dict]) -> dict:
    """Cross-check facts from a sandbox result dict (SandboxVerdict.to_dict). tests_ran / tests_passed only when real tests ran
    (mode pytest with a pass/fail count); a boot or import check alone sets boot_checked / import_checked, never tests_*.
    In boot mode `passed` counts boot STEPS, not tests."""
    t = t or {}
    ran = bool(t.get("ran")) and not t.get("inconclusive")
    mode, ok = t.get("mode"), bool(t.get("ok"))
    passed, failed = _int(t.get("passed")), _int(t.get("failed"))
    tests_ran = ran and mode == "pytest" and (passed > 0 or failed > 0)
    boot = t.get("boot") or {}
    boot_checked = ran and ok and (mode == "boot" or (boot.get("ran") is True and boot.get("ok") is True))
    import_checked = ran and ok and mode == "smoke"
    if not ran:
        sandbox = None
    elif tests_ran:
        sandbox = "passed" if ok else "failed"
    elif mode == "boot":
        sandbox = "boot check passed" if ok else "boot check failed"
    elif mode == "smoke":
        sandbox = "import check passed" if ok else "import check failed"
    else:
        sandbox = "no tests ran"
    return {"tests_ran": tests_ran, "tests_passed": (ok and passed > 0) if tests_ran else None,
            "tests_count": passed if (ran and mode == "pytest") else 0, "boot_checked": boot_checked,
            "import_checked": import_checked, "sandbox": sandbox, "sandbox_backend": t.get("backend") if ran else None,
            "achieved_level": achieved_level(t) if ran else "L0-static", "sandbox_not_run": t.get("not_run"),
            "sandbox_note": t.get("note")}


def _where(facts: dict) -> str:
    return LABEL_CONTAINER if str(facts.get("sandbox_backend") or "").lower() == "docker" else LABEL_SANDBOX


def _real_tests(facts: dict) -> int:
    """Number of sandbox tests that passed, 0 unless tests ran and passed with a count."""
    try:
        n = int(facts.get("tests_count") or 0)
    except (TypeError, ValueError):
        n = 0
    return n if (facts.get("tests_ran") and facts.get("tests_passed") is True and n > 0) else 0


def verification_labels(facts: dict, proposes: bool) -> List[str]:
    """One labelled line per kind of evidence the facts support; Unverified when nothing executed. [] when no edit was proposed.
    "tests passed" needs a test count > 0; a boot or import check alone is labelled as exactly that.
    Facts used: tests_ran/tests_passed/tests_count, boot_checked, import_checked, sandbox_backend, static_checked, and (set by no
    current code path) browser_ran/browser_ok and live_env/live_ok/user_assisted."""
    if not proposes:
        return []
    out: List[str] = []
    n = _real_tests(facts)
    static = f"{LABEL_STATIC}: edits applied cleanly and passed the static checks"
    if n:
        out.append(f"{_where(facts)}: {n} sandbox test{'' if n == 1 else 's'} passed")
    elif facts.get("boot_checked") is True:
        out += [f"{_where(facts)}: boot check only (no tests cover this change)"]
    elif facts.get("import_checked") is True:
        out += [f"{_where(facts)}: import smoke check only (no tests cover this change)"]
    if facts.get("browser_ran") is True and facts.get("browser_ok") is True:
        out.append(f"{LABEL_BROWSER}: rendered and checked")
    if facts.get("live_env") and facts.get("live_ok") is True and facts.get("user_assisted") is True:
        out.append(f"{LABEL_LIVE} {facts['live_env']}, user-assisted: checked")
    if out:
        if not n and facts.get("static_checked") is True:
            out.append(static)
        return out
    if facts.get("static_checked") is True:
        out.append(static)
    ran = bool(facts.get("tests_ran"))
    if ran and facts.get("tests_passed") is False:
        why = "sandbox tests failed"
    elif ran and facts.get("tests_passed") is True:
        why = "the sandbox reported passing tests without a test count"
    elif facts.get("sandbox_not_run"):
        why = f"the sandbox did not run: {facts['sandbox_not_run']}"
    else:
        why = "nothing was executed"
    out.append(f"{LABEL_UNVERIFIED}: runtime behaviour ({why}); run the project's own tests and checks")
    return out


# ---- validation run summary: ONE scale, the ladder in skills/.../validation-ladder.md
LEVEL_NAMES = {"L0": "static", "L1": "tests", "L2": "container", "L3": "headless browser", "L4": "live env, user-assisted",
               "L5": "production"}
DASH = "\u2013"


def ladder_achieved(facts: dict) -> str:
    """Highest ladder level the facts prove, e.g. "L1 (boot smoke)". The sandbox is a container that runs the repo's tests or a
    boot / import smoke check, so it counts as L1 evidence; nothing here proves L2 (ephemeral services, rollback)."""
    if facts.get("live_env") and facts.get("live_ok") is True and facts.get("user_assisted") is True:
        return "L4 (live, user-assisted)"
    if facts.get("browser_ran") is True and facts.get("browser_ok") is True:
        return "L3 (headless browser)"
    if facts.get("tests_ran") and facts.get("tests_passed") is False:
        return "none (sandbox tests failed)"
    if _real_tests(facts):
        return "L1 (tests, fail-to-pass)" if facts.get("achieved_level") == "L3-fail-to-pass" else "L1 (tests)"
    if facts.get("boot_checked") is True:
        return "L1 (boot smoke)"
    if facts.get("import_checked") is True:
        return "L1 (import smoke)"
    if facts.get("static_checked") is True:
        return "L0 (static)"
    return "none (nothing executed)"


def validation_summary(recommended: str, facts: dict) -> tuple:
    """(headline, [one line per recommended level]): "Recommended: L1, L3 \u00b7 Achieved: L1 (boot smoke)" and, per level,
    "L3 headless browser: skipped \u2013 <reason>". Reasons come from the facts; no capability is invented."""
    levels = [x.strip() for x in (recommended or "").split(",") if x.strip()]
    lines: List[str] = []
    for lv in levels:
        name = f"{lv} {LEVEL_NAMES.get(lv, '')}".strip()
        n = _real_tests(facts)
        if lv == "L0":
            ok = facts.get("static_checked") is True
            d = "ran " + DASH + " static checks passed" if ok else "skipped " + DASH + " no static check result"
        elif lv == "L1":
            if n:
                d = f"ran {DASH} {n} sandbox test{'' if n == 1 else 's'} passed"
            elif facts.get("tests_ran") and facts.get("tests_passed") is False:
                d = "ran " + DASH + " tests FAILED"
            elif facts.get("boot_checked") or facts.get("import_checked"):
                d = "skipped " + DASH + " no test covers this change"
            elif facts.get("sandbox_not_run") or facts.get("sandbox_note"):
                d = f"skipped {DASH} sandbox did not run ({facts.get('sandbox_not_run') or facts.get('sandbox_note')})"
            else:
                d = "skipped " + DASH + " nothing was executed"
        elif lv == "L2":
            ran = n or facts.get("boot_checked") or facts.get("import_checked")
            if ran and str(facts.get("sandbox_backend") or "").lower() == "docker":
                d = "partial " + DASH + " ran in the sandbox container; no ephemeral DB / services or rollback check in this version"
            else:
                d = "skipped " + DASH + " no container run with ephemeral services or rollback in this version"
        elif lv == "L3":
            d = ("ran " + DASH + " rendered and checked" if facts.get("browser_ran") is True and facts.get("browser_ok") is True
                 else "skipped " + DASH + " no browser runner in this version")
        elif lv == "L4":
            d = ("ran " + DASH + " user-assisted check" if facts.get("live_env") and facts.get("live_ok") is True
                 else "skipped " + DASH + " needs a live environment with you in the loop; not available in this version")
        else:
            d = "not recommended"
        lines.append(f"{name}: {d}")
    return f"Recommended: {', '.join(levels) or 'n/a'} \u00b7 Achieved: {ladder_achieved(facts)}", lines


class DeliveryNotes(BaseModel):
    assumptions: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    verified: List[str] = Field(default_factory=list)
    unverified: List[str] = Field(default_factory=list)
    requirements: List[Dict[str, Any]] = Field(default_factory=list)
    lane: str = ""                    # fast|standard|strict for change requests, "" otherwise
    validation_level: str = ""        # recommended ladder level(s), e.g. "L1, L3"
    verification: List[str] = Field(default_factory=list)       # labelled evidence, see verification_labels()
    validation_summary: str = ""      # "Recommended: L1, L3 \u00b7 Achieved: L1 (boot smoke)"
    validation_steps: List[str] = Field(default_factory=list)   # one line per recommended level: ran / skipped + reason

    def _headline(self) -> str:
        return self.validation_summary or (f"Recommended: {self.validation_level}" if self.validation_level else "")

    def _fast_markdown(self) -> str:
        """Fast lane: a few lines (lane + validation run, how it was checked, one-line assumptions, anything that failed)."""
        def one(items: List[str]) -> str:
            return "; ".join(items)[:240]
        lines = ["### Delivery notes", "", "Lane: fast. " + self._headline()]
        lines += [f"- {x}" for x in self.validation_steps]
        lines.append("Checked: " + (one(self.verification) or "not verified"))
        if self.assumptions:
            lines.append("Assumptions: " + one(self.assumptions))
        if self.limitations:
            lines.append("Limitations: " + one(self.limitations))
        unmet = [r for r in self.requirements if r.get("status") == "unmet"]
        if unmet:
            lines.append(f"Unmet: {len(unmet)} requirement(s)")
        return "\n".join(lines) + "\n"

    def to_markdown(self) -> str:
        if self.lane == Lane.FAST.value:
            return self._fast_markdown()
        lines = ["### Delivery notes", ""]
        if self.lane:
            lines += [f"**Lane:** {self.lane}", ""]
            if self._headline():
                lines += ["**Validation run**", self._headline()] + [f"- {x}" for x in self.validation_steps] + [""]

        def section(title: str, items: List[str]):
            if items:
                lines.append(f"**{title}**")
                lines.extend(f"- {i}" for i in items)
                lines.append("")

        section("Verification", self.verification)
        section("Assumptions", self.assumptions)
        section("Limitations", self.limitations)
        section("Verified", self.verified)
        section("Not verified", self.unverified)
        if self.requirements:
            icon = {"met": "met", "partial": "partial", "unmet": "unmet", "unknown": "unknown"}
            lines.append("**Requirements cross-check**")
            for r in self.requirements:
                ev = f" - {r.get('evidence')}" if r.get("evidence") else ""
                # a fact-based label ("not met", "partial \u2013 boot check only", "unverified") only while the status it was made for stands
                lab = r.get("label") if r.get("label") and r.get("label_status") == r.get("status") else icon.get(r.get("status"), "unknown")
                lines.append(f"- [{lab}] ({r.get('kind', 'direct')}) {r.get('req', '')}{ev}")
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
        if "tests_count" in facts and not _real_tests(facts):
            tests_passed = False if tests_ran else tests_passed        # "passed" without a test count > 0 proves nothing
            tests_ran = tests_ran and facts.get("tests_passed") is False
        changed = list(facts.get("files_changed") or [])
        is_change = brief.intent == "change"
        proposes = is_change or facts.get("edits_proposed") is True      # an ask that proposed edits is judged like a change
        boot_only = bool(facts.get("boot_checked") or facts.get("import_checked")) and not (tests_ran and tests_passed)
        check_name = "boot check" if facts.get("boot_checked") else "import check"

        verified: List[str] = []
        unverified: List[str] = []
        limitations: List[str] = []

        if tests_ran and tests_passed:
            verified.append("Tests ran in the sandbox and passed")
        elif tests_ran and facts.get("tests_passed") is False:
            limitations.append("Tests ran and FAILED; the change is not verified")
            unverified.append("Behaviour (tests failed)")
        elif boot_only:
            # one statement, under Limitations (the Verification section already says what ran); not repeated under Not verified
            if proposes:
                limitations.append(f"No test covers this change; only the {check_name} ran")
            else:
                unverified.append(f"Runtime behaviour beyond the {check_name} (no test covers this change)")
        else:
            if proposes:
                limitations.append("Tests were not run for this change")
            unverified.append("Runtime behaviour (no tests executed)")
        sandbox = facts.get("sandbox")
        if sandbox:
            passing = str(sandbox).lower().endswith(("ok", "passed", "pass"))
            if passing and tests_ran and tests_passed:
                verified.append(f"Sandbox: {sandbox}")
            elif not passing:
                unverified.append(f"Sandbox: {sandbox}")       # a passing check is evidence (Verification), never "not verified"
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
            extra: Dict[str, Any] = {}
            if is_change and not changed and status != "unmet":
                status, ev = "unmet", "no files were changed"
            red_green = "fails before" in req.lower()
            existing = req.lower().startswith("existing behaviour")
            n_tests = _real_tests(facts)
            if red_green and status != "unmet":
                # Judged from the facts, not from words: needs tests that really ran; proven only when one was red before and green after.
                if not tests_ran:
                    status, ev = "unmet", "no tests were run for this change"
                    extra = {"label": "not met", "label_status": status}
                elif (n_tests > 0 and facts.get("achieved_level") == "L3-fail-to-pass") or (
                        facts.get("tests_red_first") is True and tests_passed):
                    status, ev = "met", "test was red before the change and green after"
                else:
                    status, ev = "partial", "tests did not record a red-before run" if tests_passed else "tests ran but did not pass"
            elif existing and is_change and status != "unmet":
                if tests_ran and tests_passed:
                    status, ev = "met", "tests passed"
                elif boot_only:
                    status, ev = "partial", f"{check_name} only; no test covers this change"
                    extra = {"label": "partial \u2013 boot check only" if facts.get("boot_checked") else "partial \u2013 import check only",
                             "label_status": status}
                else:
                    status, ev = "unknown", "nothing was executed"
                    extra = {"label": "unverified", "label_status": status}
            elif status == "partial" and tests_ran and tests_passed and is_change:
                status, ev = "met", ev + "; tests passed"
            elif status in ("partial", "unknown") and not (red_green or existing):
                # Only word overlap with the answer text: never presented as progress towards "met"
                status, ev = "unknown", f"(keyword match only, not proof) {ev}"
                extra = {"label": "unverified", "label_status": status, "basis": "keyword"}
            requirements.append({"req": req, "kind": "direct", "status": status, "evidence": ev, **extra})
        for req in brief.nfrs:
            if is_change and tests_ran and tests_passed:
                status, ev = "partial", "tests passed; not independently audited"
            else:
                status, ev = "unknown", "not checked automatically"
            requirements.append({"req": req, "kind": "professional", "status": status, "evidence": ev})

        if proposes and brief.needs_handoff:
            unverified.append("Live login / SSO / MFA flow needs your hands-on check (never share credentials in chat)")
        recommended = ""
        if proposes and brief.lane:
            recommended = brief.validation_level or recommend_validation_level(
                brief.lane, brief.change_types or [brief.change_type], brief.needs_handoff)
        headline, steps = validation_summary(recommended, facts) if recommended else ("", [])
        notes = DeliveryNotes(assumptions=list(brief.assumptions), limitations=limitations, verified=verified,
                              unverified=unverified, requirements=requirements,
                              lane=brief.lane.value if (proposes and brief.lane) else "",
                              validation_level=recommended, validation_summary=headline, validation_steps=steps,
                              verification=verification_labels(facts, proposes))
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
