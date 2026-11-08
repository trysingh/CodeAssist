"""IntakeAgent: understand the ask before acting (intent, risk, ambiguity, criteria). Pure stdlib + pydantic."""
from __future__ import annotations

import re
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field


class Risk(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


_RISK_ORDER = {Risk.LOW: 0, Risk.MEDIUM: 1, Risk.HIGH: 2}

HIGH_RISK_TERMS = [
    "auth", "authentication", "authorization", "authorisation", "login", "password", "passwd", "secret", "credential",
    "token", "jwt", "oauth", "session", "payment", "payments", "money", "billing", "invoice", "refund", "pii",
    "personal data", "delete", "drop table", "truncate", "migration", "prod", "production", "security", "encrypt",
    "encryption", "decrypt", "crypto", "compliance", "gdpr", "hipaa", "sox", "kyc", "aml", "pci", "sanctions",
    "fraud", "transaction", "transactions", "ledger", "settlement", "claim", "claims", "policyholder", "premium",
    "underwriting", "actuarial", "iban", "account balance", "audit trail", "rm -rf", "permission", "permissions",
    "rbac", "privilege",
]
MEDIUM_RISK_TERMS = [
    "schema", "database", "db", "sql", "api", "endpoint", "contract", "concurrency", "concurrent", "thread",
    "threads", "async", "lock", "race condition", "deadlock", "performance", "latency", "throughput", "cache",
    "caching", "refactor", "interface", "breaking", "backward compatible", "queue", "retry", "timeout",
    "config", "configuration", "dependency", "upgrade",
]
ENTERPRISE_TERMS = [
    "enterprise", "regulated", "bank", "banking", "insurance", "insurer", "audit", "sox", "gdpr", "hipaa", "pci",
    "kyc", "aml", "compliance", "sla", "multi-tenant", "tenant", "rbac", "sso", "ldap", "saml", "policyholder",
    "ledger", "core banking", "mainframe", "cobol", "change control", "segregation of duties", "data residency",
]

_VAGUE = ["something", "somehow", "etc", "whatever", "stuff", "things", "anything", "some way", "somewhere",
          "kind of", "sort of", "or so"]
_PRONOUNS = {"it", "this", "that", "them", "those", "these", "they"}
_ACTION_VERBS = ("fix", "change", "update", "improve", "make", "do", "handle", "refactor", "clean", "optimize",
                 "optimise", "add", "remove", "implement", "build", "create")
_STOPWORDS = {"the", "a", "an", "to", "of", "in", "on", "for", "and", "or", "it", "this", "that", "please", "can",
              "you", "we", "i", "my", "our", "is", "are", "be", "with", "so", "me", "pls"}

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:_[A-Za-z0-9]+)+|[a-z]+[A-Z][A-Za-z0-9]*|[A-Z][a-z]+[A-Z][A-Za-z0-9]*|\w+\(\)")
_PATH_RE = re.compile(r"[\w./\\-]+\.(?:py|js|ts|tsx|jsx|java|go|rs|cs|kt|rb|php|sql|yml|yaml|json|toml|md|html|css|sh|cfg|ini)\b", re.I)
_BACKTICK_RE = re.compile(r"`[^`\n]+`")


def _has_term(text: str, term: str) -> bool:
    if " " in term or "-" in term:
        return term in text
    return re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", text) is not None


def _matches(text: str, terms: List[str]) -> List[str]:
    t = text.lower()
    seen: List[str] = []
    for term in terms:
        if _has_term(t, term) and term not in seen:
            seen.append(term)
    return seen


class IntakeBrief(BaseModel):
    intent: str = "ask"
    explicit_ask: str = ""
    implicit_needs: List[str] = Field(default_factory=list)
    goal: str = ""
    success_criteria: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    nfrs: List[str] = Field(default_factory=list)
    risk: Risk = Risk.LOW
    risk_reasons: List[str] = Field(default_factory=list)
    ambiguity: float = Field(default=0.0, ge=0.0, le=1.0)
    ambiguity_reasons: List[str] = Field(default_factory=list)
    clarifying_questions: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    outcome_tasks: List[str] = Field(default_factory=list)
    enterprise: bool = False
    source: str = "heuristic"


BASELINE_NFRS = [
    "Correctness: behaves as asked, including edge cases",
    "Clear error handling: failures are reported, not swallowed",
    "Readability: follows the existing code style",
    "No obvious security holes: inputs validated, no secrets in code",
    "Reasonable performance: no needless heavy work",
]


def classify_risk(text: str) -> tuple:
    """(Risk, reasons) from keyword sets."""
    hi = _matches(text, HIGH_RISK_TERMS)
    if hi:
        return Risk.HIGH, [f"touches high-risk area: {', '.join(hi[:6])}"]
    med = _matches(text, MEDIUM_RISK_TERMS)
    if med:
        return Risk.MEDIUM, [f"touches sensitive area: {', '.join(med[:6])}"]
    return Risk.LOW, ["no sensitive area detected"]


def split_clauses(text: str) -> List[str]:
    """Split a request into outcome clauses: numbered/bulleted lists, ';', newlines, ',' and 'and'."""
    text = (text or "").strip()
    if not text:
        return []
    # protect code spans / paths so "a.py and b.py" still splits but `x, y` stays whole
    protected: Dict[str, str] = {}

    def keep(m):
        k = f"\x00{len(protected)}\x00"
        protected[k] = m.group(0)
        return k

    work = _BACKTICK_RE.sub(keep, text)
    lines = [ln.strip() for ln in work.splitlines() if ln.strip()]
    list_items = [re.sub(r"^\s*(?:\d+[.)]|[-*•])\s+", "", ln) for ln in lines
                  if re.match(r"^\s*(?:\d+[.)]|[-*•])\s+", ln)]
    if len(list_items) >= 2:
        parts = list_items
    else:
        parts = []
        for ln in lines:
            parts.extend(re.split(r";|\s+(?:and then|then|and also|also)\s+", ln, flags=re.I))
        out: List[str] = []
        for p in parts:
            # split on commas and " and " only when the right side looks like a new verb phrase
            sub = re.split(r",\s*(?:and\s+)?|\s+and\s+(?=(?:" + "|".join(_ACTION_VERBS + (
                "write", "show", "log", "return", "send", "delete", "rename", "move", "test", "validate", "ensure",
                "enable", "disable", "support", "convert", "replace", "save", "load", "display", "check")) + r")\b)", p,
                           flags=re.I)
            out.extend(sub)
        parts = out
    cleaned: List[str] = []
    for p in parts:
        for k, v in protected.items():
            p = p.replace(k, v)
        p = p.strip(" \t.;,")
        if len(p) >= 3 and p.lower() not in {c.lower() for c in cleaned}:
            cleaned.append(p)
    return cleaned[:12]


def _success_criteria(ask: str, tasks: List[str], intent: str) -> List[str]:
    crit: List[str] = []
    if intent == "change":
        for t in tasks[:6]:
            crit.append(f"Done: {t[0].lower() + t[1:] if t else t}")
        crit.append("Existing behaviour that was not asked to change still works")
        crit.append("Change is covered by a test that fails before and passes after")
    elif intent == "plan":
        crit.append("Plan covers every part of the ask, with ordered steps and named files")
        crit.append("Risks and open questions are listed")
    else:
        for t in tasks[:4]:
            crit.append(f"Answers: {t}")
        crit.append("Answer is grounded in the repository's actual code, with file references")
    return crit


def _constraints(ask: str) -> List[str]:
    out: List[str] = []
    for m in re.finditer(r"\b(?:without|don'?t|do not|never|must not|only|must|should not|avoid|keep|preserve|"
                         r"backward[- ]compatib\w+)\b[^.;\n]{0,80}", ask, flags=re.I):
        c = m.group(0).strip(" ,")
        if c and c not in out:
            out.append(c)
    return out[:6]


class IntakeAgent:
    def __init__(self, llm_json: Optional[Callable[[str], dict]] = None):
        self.llm_json = llm_json

    # ------------------------------------------------------------------ heuristics
    def analyze(self, query: str, intent: str, repo_files: Optional[List[str]] = None) -> IntakeBrief:
        q = (query or "").strip()
        intent = (intent or "ask").strip().lower() or "ask"
        low = q.lower()
        words = re.findall(r"[A-Za-z0-9_']+", q)

        risk, risk_reasons = classify_risk(q)
        ent_hits = _matches(q, ENTERPRISE_TERMS)
        enterprise = bool(ent_hits) or risk == Risk.HIGH and len(_matches(q, HIGH_RISK_TERMS)) >= 2
        if repo_files and not enterprise:
            joined = " ".join(repo_files[:2000]).lower()
            if any(_has_term(joined, t) for t in ("kyc", "aml", "ledger", "claims", "policyholder", "underwriting")):
                enterprise = True
                ent_hits = ["repo layout"]
        if enterprise and risk == Risk.LOW:
            risk, risk_reasons = Risk.MEDIUM, ["enterprise/regulated context raises the bar"]

        # ---- ambiguity
        score, reasons, questions = 0.0, [], []
        has_ident = bool(_IDENT_RE.search(q) or _PATH_RE.search(q) or _BACKTICK_RE.search(q))
        known = {p.rsplit("/", 1)[-1].lower() for p in (repo_files or [])}
        if known and any(w.lower() in known or (w.lower() + ".py") in known for w in words):
            has_ident = True
        if len(words) < 4:
            score += 0.4
            reasons.append("very short request")
            questions.append("Can you describe what you want in a sentence or two, and where in the code?")
        elif len(words) < 7:
            score += 0.15
            reasons.append("short request")
        if not has_ident:
            score += 0.2
            reasons.append("no file, function or identifier named")
            questions.append("Which file, module or function should this apply to?")
        lead = [w.lower() for w in words[:6]]
        if any(w in _PRONOUNS for w in lead) and intent in ("change", "help") and not has_ident and len(words) < 8:
            score += 0.3
            reasons.append("pronoun without a referent (e.g. 'fix it')")
            questions.append("What does 'it' refer to?")
        vague = [v for v in _VAGUE if _has_term(low, v)]
        if vague:
            score += 0.2
            reasons.append(f"vague wording: {', '.join(vague)}")
            questions.append("Can you be specific about what 'done' looks like?")
        if re.match(r"^(?:please\s+)?(?:" + "|".join(_ACTION_VERBS) + r")\b", low) and intent == "change" \
                and not re.search(r"\b(?:so that|should|must|returns?|when|if|such that|expected)\b", low) and len(words) < 12:
            score += 0.2
            reasons.append("no success criteria or expected behaviour stated")
            questions.append("What should the behaviour be afterwards (an example input and expected output)?")
        ambiguity = round(min(1.0, score), 2)

        tasks = split_clauses(q)
        if not tasks and q:
            tasks = [q]
        assumptions: List[str] = []
        if ambiguity > 0.3:
            if not has_ident:
                assumptions.append("Assumed the most relevant files found by search are the intended scope")
            if vague or ambiguity > 0.5:
                assumptions.append("Assumed the simplest reasonable interpretation of the request")
            assumptions.append("Assumed existing public behaviour and signatures must be preserved")

        implicit = ["Keep existing behaviour that was not mentioned unchanged"]
        if intent == "change":
            implicit += ["Match the repository's existing style and conventions", "Do not introduce new dependencies unless needed"]
        if risk == Risk.HIGH:
            implicit.append("Treat sensitive data and money/identity flows conservatively (validation, audit, least privilege)")
        if enterprise:
            implicit.append("Keep auditability and backward compatibility for regulated workflows")

        nfrs = list(BASELINE_NFRS)
        if risk in (Risk.MEDIUM, Risk.HIGH):
            nfrs.append("Compatibility: existing callers and data keep working")
        if risk == Risk.HIGH:
            nfrs += ["Security: validate inputs, no secrets or PII in logs, least privilege",
                     "Auditability: sensitive actions are traceable"]
        if enterprise:
            nfrs.append("Regulatory fit: no weakening of compliance controls")

        brief = IntakeBrief(
            intent=intent, explicit_ask=q, implicit_needs=implicit,
            goal=(tasks[0] if len(tasks) == 1 else (q[:200] if q else "")),
            success_criteria=_success_criteria(q, tasks, intent), constraints=_constraints(q), nfrs=nfrs,
            risk=risk, risk_reasons=risk_reasons, ambiguity=ambiguity, ambiguity_reasons=reasons,
            clarifying_questions=questions[:3] if ambiguity >= 0.3 else [],
            assumptions=assumptions, outcome_tasks=tasks, enterprise=enterprise, source="heuristic")
        return self._merge_llm(brief, q, intent, repo_files)

    # ------------------------------------------------------------------ optional LLM merge
    def _merge_llm(self, brief: IntakeBrief, q: str, intent: str, repo_files: Optional[List[str]]) -> IntakeBrief:
        if not self.llm_json:
            return brief
        prompt = (
            "Analyze this software request. Reply ONLY JSON with keys: goal (str), success_criteria (list[str]), "
            "constraints (list[str]), nfrs (list[str]), risk (low|medium|high), ambiguity (0..1), "
            "clarifying_questions (list[str], max 3), assumptions (list[str]), outcome_tasks (list[str]).\n"
            f"INTENT: {intent}\nREQUEST: {q}\nFILES: {', '.join((repo_files or [])[:60])}")
        try:
            data = self.llm_json(prompt)
        except Exception:
            return brief
        if not isinstance(data, dict):
            return brief

        def strs(key: str, cap: int) -> List[str]:
            v = data.get(key)
            if not isinstance(v, list):
                return []
            return [s.strip()[:300] for s in v if isinstance(s, str) and s.strip()][:cap]

        updates: Dict[str, Any] = {}
        changed = False
        for key, cap in (("success_criteria", 8), ("constraints", 8), ("nfrs", 10), ("assumptions", 8),
                         ("outcome_tasks", 12)):
            extra = strs(key, cap)
            cur = list(getattr(brief, key))
            have = {c.lower() for c in cur}
            add = [e for e in extra if e.lower() not in have]
            if add:
                updates[key] = (cur + add)[: cap + len(cur)]
                changed = True
        goal = data.get("goal")
        if isinstance(goal, str) and goal.strip() and len(goal.strip()) <= 300:
            updates["goal"] = goal.strip()
            changed = True
        # risk: LLM may only raise
        r = data.get("risk")
        if isinstance(r, str) and r.strip().lower() in {x.value for x in Risk}:
            lr = Risk(r.strip().lower())
            if _RISK_ORDER[lr] > _RISK_ORDER[brief.risk]:
                updates["risk"] = lr
                updates["risk_reasons"] = brief.risk_reasons + ["raised by model review"]
                changed = True
        a = data.get("ambiguity")
        if isinstance(a, (int, float)) and not isinstance(a, bool) and 0 <= a <= 1 and a > brief.ambiguity:
            updates["ambiguity"] = round(float(a), 2)
            updates["ambiguity_reasons"] = brief.ambiguity_reasons + ["raised by model review"]
            changed = True
        cq = strs("clarifying_questions", 3)
        if cq:
            merged = list(dict.fromkeys(brief.clarifying_questions + cq))[:3]
            updates["clarifying_questions"] = merged
            changed = True
        if not changed:
            return brief
        updates["source"] = "llm"
        try:
            return IntakeBrief.model_validate({**brief.model_dump(), **updates})
        except Exception:
            return brief
