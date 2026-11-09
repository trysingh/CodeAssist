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



class Lane(str, Enum):
    """Rigour lane (skills/risk-proportional-code-dev). Decides how much ceremony a request gets; it never lowers a setting."""
    FAST = "fast"
    STANDARD = "standard"
    STRICT = "strict"


_LANE_ORDER = {Lane.FAST: 0, Lane.STANDARD: 1, Lane.STRICT: 2}


class ChangeType(str, Enum):
    FE_PRESENTATIONAL = "fe_presentational"
    FE_BEHAVIOURAL = "fe_behavioural"
    FE_SECURITY = "fe_security"
    DOCS = "docs"
    SQL = "sql"
    MIGRATION = "migration"
    IAC_CONFIG = "iac_config"
    SCRIPT = "script"
    BACKEND = "backend"          # backend / other


# Lane a change type starts from; risk and the "relaxing is a trap" rules can only raise it
_TYPE_LANE = {
    ChangeType.FE_PRESENTATIONAL: Lane.FAST, ChangeType.FE_BEHAVIOURAL: Lane.STANDARD, ChangeType.FE_SECURITY: Lane.STRICT,
    ChangeType.DOCS: Lane.FAST, ChangeType.SQL: Lane.STANDARD, ChangeType.MIGRATION: Lane.STRICT,
    ChangeType.IAC_CONFIG: Lane.STRICT, ChangeType.SCRIPT: Lane.STANDARD, ChangeType.BACKEND: Lane.STANDARD,
}

# ---- lane / change-type heuristics (same matching rules as the risk lists above)
FE_SECURITY_TERMS = [
    "innerhtml", "dangerouslysetinnerhtml", "eval", "document.write", "postmessage", "cookie", "cookies", "csp",
    "content-security-policy", "content security policy", "third-party script", "third party script", "cdn", "sri",
    "subresource integrity", "payment", "payments", "kyc",
]
STORAGE_TERMS = ["localstorage", "sessionstorage"]          # a security trigger only when a token / secret goes in
STORAGE_SECRET_TERMS = ["token", "tokens", "jwt", "session", "auth", "credential", "credentials", "secret", "password",
                        "bearer", "api key"]
FE_PRESENTATIONAL_TERMS = [
    "css", "scss", "sass", "stylesheet", "styling", "colour", "color", "colours", "colors", "font", "fonts", "margin",
    "padding", "border", "spacing", "alignment", "align", "layout", "responsive", "breakpoint", "dark mode",
    "light mode", "hover", "animation", "shadow", "rounded", "html", "show/hide",
]
FE_BEHAVIOURAL_TERMS = ["javascript", "typescript", "jsx", "tsx", "event handler", "event listener", "onclick",
                        "click handler", "form validation", "fetch", "ajax", "redux", "dom", "audio", "audible", "sound",
                        "sounds", "audiocontext", "web audio", "click", "clicks", "on click", "addeventlistener", "xhr",
                        "websocket", "polling", "setinterval", "starting and stopping", "start and stop", "on submit"]
FE_GENERIC_TERMS = ["frontend", "front-end", "front end", "ui", "browser", "react", "vue", "angular", "svelte",
                    "component", "button", "modal", "navbar", "dropdown", "tooltip", "tailwind"]
_FE_PRESENTATIONAL_EXT = {"css", "scss", "sass", "less", "html", "htm"}
_FE_CODE_EXT = {"jsx", "tsx", "vue", "svelte", "js", "mjs", "cjs", "ts"}
DOCS_TERMS = ["readme", "docs", "documentation", "docstring", "docstrings", "comment", "comments", "changelog", "typo",
              "typos", "release notes"]
_DOCS_EXT = {"md", "rst", "adoc"}
SQL_TERMS = ["sql", "sqlite", "postgres", "postgresql", "mysql", "t-sql", "plsql", "stored procedure", "stored procedures"]
MIGRATION_TERMS = ["migration", "migrations", "alembic", "flyway", "liquibase", "alter table", "add column", "drop column",
                   "backfill", "schema change", "schema changes", "data migration"]
IAC_TERMS = [
    "terraform", "cloudformation", "pulumi", "ansible", "helm", "kubernetes", "k8s", "kubectl", "docker-compose",
    "dockerfile", "ci/cd", "cicd", "github actions", "gitlab ci", "jenkins", "jenkinsfile", "azure pipelines", "iam",
    "security group", "infrastructure as code", "iac", "bicep", ".github/workflows",
]
_IAC_EXT = {"tf", "tfvars", "bicep", "hcl"}
SCRIPT_TERMS = ["script", "scripts", "notebook", "notebooks", "one-off", "one off", "cron job"]
_SCRIPT_EXT = {"sh", "bash", "ps1", "bat", "ipynb"}
DISPOSABLE_TERMS = ["disposable", "throwaway", "throw-away", "one-off", "one off", "scratch", "spike", "prototype",
                    "proof of concept", "poc", "quick hack"]
STRICT_LANE_TERMS = [        # auth / payments / prod data: Strict whatever the file type
    "auth", "authentication", "authorization", "authorisation", "login", "logout", "session", "sso", "oauth", "saml",
    "mfa", "password", "rbac", "payment", "payments", "billing", "checkout", "refund", "invoice", "prod data",
    "production data", "live data", "real data", "customer data", "production database", "prod db", "live database",
]
# "relaxing is a trap": each of these keeps a request out of the Fast lane
GLOBAL_STYLE_TERMS = [
    "global css", "global style", "global styles", "global stylesheet", "design token", "design tokens", "design system",
    "design-system", "css variable", "css variables", "css custom property", "css custom properties", ":root",
    "theme token", "theme tokens", "tailwind config", "base styles",
]
CONSENT_TERMS = ["consent", "disclosure", "disclosures", "disclaimer", "terms and conditions", "terms of service",
                 "privacy notice", "privacy policy", "legal text", "legal copy", "t&c", "cookie banner", "opt-in", "opt-out"]
CONFIG_FLOOR_TERMS = ["feature flag", "feature flags", "config flag", "config flags", "flag", "flags", "timeout", "timeouts",
                      "env var", "env vars", "environment variable", "environment variables", ".env", "toggle"]
HANDOFF_TERMS = ["sso", "single sign-on", "mfa", "2fa", "multi-factor", "two-factor", "otp", "one-time password",
                 "one time password", "captcha", "recaptcha", "login", "sign-in", "sign in"]
_FAST_TEST_VERBS = ["rename", "tidy", "clean up", "cleanup", "reformat"]       # not "refactor": that is a Medium-risk term

_LANE_OVERRIDE_RE = re.compile(r"\blane\s*[:=]\s*(fast|standard|strict)\b|\buse (?:the )?(fast|standard|strict) lane\b", re.I)
_DESIGN_TOKEN_RE = re.compile(r"\b(?:design|theme|css|style|colou?r|spacing|typography)[ -]tokens?\b", re.I)
_FILE_EXT_RE = re.compile(r"(?<![\w])([\w./\\-]*?[\w-]+)\.([A-Za-z0-9]{1,8})\b")

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
    # ---- v3: rigour lane (None = not classified, e.g. the fallback brief: no lane rules apply)
    lane: Optional[Lane] = None
    lane_reasons: List[str] = Field(default_factory=list)
    change_type: ChangeType = ChangeType.BACKEND                 # primary: the type that drives the highest lane
    change_types: List[ChangeType] = Field(default_factory=list)   # every type the request touches (primary first)
    change_like: bool = False          # an "ask" that reads like an edit request (the ask agent may propose edits)
    validation_level: str = ""         # recommended ladder level(s) for change/plan intents, e.g. "L1, L3" (not the achieved "Verification level")
    needs_handoff: bool = False        # SSO / MFA / login / OTP / CAPTCHA: a live check needs the user's hands


BASELINE_NFRS = [
    "Correctness: behaves as asked, including edge cases",
    "Clear error handling: failures are reported, not swallowed",
    "Readability: follows the existing code style",
    "No obvious security holes: inputs validated, no secrets in code",
    "Reasonable performance: no needless heavy work",
]


def classify_risk(text: str) -> tuple:
    """(Risk, reasons) from keyword sets."""
    text = _DESIGN_TOKEN_RE.sub(" ", text or "")        # "design token" is styling, not an auth token
    hi = _matches(text, HIGH_RISK_TERMS)
    if hi:
        return Risk.HIGH, [f"touches high-risk area: {', '.join(hi[:6])}"]
    med = _matches(text, MEDIUM_RISK_TERMS)
    if med:
        return Risk.MEDIUM, [f"touches sensitive area: {', '.join(med[:6])}"]
    return Risk.LOW, ["no sensitive area detected"]


def _named_files(text: str, repo_files: Optional[List[str]]) -> List[str]:
    """Files the request names: paths with an extension in the text, plus repo files whose base name the text mentions."""
    names = [m.group(0) for m in _FILE_EXT_RE.finditer(text)]
    if repo_files:
        toks = set(re.findall(r"[\w.-]+\.\w+", text.lower()))
        names += [p for p in repo_files[:2000] if p.rsplit("/", 1)[-1].lower() in toks and p not in names]
    return names


def _is_test_file(path: str) -> bool:
    p = path.replace("\\", "/").lower()
    base = p.rsplit("/", 1)[-1]
    return base.startswith("test_") or ".test." in base or ".spec." in base or base.endswith("_test.py") or "/tests/" in "/" + p


_BACKEND_EXT = {"py", "java", "go", "rs", "cs", "kt", "rb", "php", "c", "cpp", "scala", "sql"}
_INLINE_DOC_TERMS = {"docstring", "docstrings", "comment", "comments", "typo", "typos"}
_CODE_SIGNAL_TERMS = ["implement", "function", "functions", "method", "methods", "class", "endpoint", "bug", "logic",
                      "algorithm", "feature", "refactor", "handler"]


_EDIT_ASK_RE = re.compile(r"\b(?:add|change|fix|implement|modify|update|refactor|rename|remove|delete|make|give|build|create|"
                          r"improve|rewrite|replace|convert|enable|disable|ensure|support|integrate|optimi[sz]e)\b", re.I)
_QUESTION_OPENER_RE = re.compile(r"^\s*(?:what|why|where|who|which|when|explain|describe|summari[sz]e|list|show|tell|walk)\b", re.I)


def _change_like(text: str) -> bool:
    """A question that asks for an edit ("do you think we can give the page a modern look?"), not for an explanation."""
    return bool(_EDIT_ASK_RE.search(text or "")) and not _QUESTION_OPENER_RE.match(text or "")


def detect_change_types(text: str, repo_files: Optional[List[str]] = None) -> List[tuple]:
    """[(ChangeType, reason)] for EVERY type the request touches, highest lane first (ties keep the order Strict types,
    front-end behaviour, SQL, script, presentational front-end, docs). Never empty: BACKEND when nothing else matches."""
    files = _named_files(text, repo_files)
    low_files = [f.replace("\\", "/").lower() for f in files]
    exts = {f.rsplit(".", 1)[-1] for f in low_files if "." in f.rsplit("/", 1)[-1]}
    backend_file = bool(exts & _BACKEND_EXT)
    found: List[tuple] = []

    hits = _matches(text, MIGRATION_TERMS)
    if hits or any("migrations/" in f or "alembic/" in f for f in low_files):
        found.append((ChangeType.MIGRATION, f"migration: {', '.join(hits[:3]) or 'migration path'}"))
    hits = _matches(text, IAC_TERMS)
    if hits or exts & _IAC_EXT or any(f.endswith(("dockerfile", "jenkinsfile")) or ".github/workflows/" in f for f in low_files):
        found.append((ChangeType.IAC_CONFIG, f"IaC / CI-CD / cloud config: {', '.join(hits[:3]) or 'file type'}"))

    beh = _matches(text, FE_BEHAVIOURAL_TERMS)
    pres = _matches(text, FE_PRESENTATIONAL_TERMS)
    gen = _matches(text, FE_GENERIC_TERMS)
    fe_context = bool(beh or pres or gen or exts & (_FE_CODE_EXT | _FE_PRESENTATIONAL_EXT))
    sec = [t for t in _matches(text, FE_SECURITY_TERMS) if fe_context or t not in ("payment", "payments", "kyc")]
    if _matches(text, STORAGE_TERMS) and _matches(text, STORAGE_SECRET_TERMS):
        sec.append("web storage holding a token/secret")
    if sec:
        found.append((ChangeType.FE_SECURITY, f"front-end security trigger: {', '.join(sec[:4])}"))
    if beh or exts & {"jsx", "tsx", "vue", "svelte", "js", "mjs", "cjs", "ts"}:
        found.append((ChangeType.FE_BEHAVIOURAL, f"front-end behaviour: {', '.join(beh[:3]) or 'JS/TS file'}"))
    hits = _matches(text, SQL_TERMS)
    if hits or "sql" in exts:
        found.append((ChangeType.SQL, f"SQL: {', '.join(hits[:3]) or '.sql file'}"))
    hits = _matches(text, SCRIPT_TERMS)
    if hits or exts & _SCRIPT_EXT:
        found.append((ChangeType.SCRIPT, f"script / notebook: {', '.join(hits[:3]) or 'file type'}"))
    if not backend_file and (pres or exts & _FE_PRESENTATIONAL_EXT):
        found.append((ChangeType.FE_PRESENTATIONAL, "presentational front-end (styling / layout / markup)"))
    elif gen and not beh and not backend_file and not any(t == ChangeType.FE_BEHAVIOURAL for t, _ in found):
        found.append((ChangeType.FE_BEHAVIOURAL, "front-end is never automatically Fast"))
    doc_terms = _matches(text, DOCS_TERMS)
    if (doc_terms or exts & _DOCS_EXT) and not _matches(text, _CODE_SIGNAL_TERMS):
        if not backend_file or (set(doc_terms) <= _INLINE_DOC_TERMS and not exts & _DOCS_EXT):
            found.append((ChangeType.DOCS, "docs / comments only"))
    if not found:
        found.append((ChangeType.BACKEND, "backend / other"))
    return sorted(found, key=lambda x: -_LANE_ORDER[_TYPE_LANE[x[0]]])        # stable: ties keep detection order


def detect_change_type(text: str, repo_files: Optional[List[str]] = None) -> tuple:
    """(primary ChangeType, [reason]): the type that drives the highest lane. See detect_change_types for all of them."""
    t, why = detect_change_types(text, repo_files)[0]
    return t, [why]


def _fast_generic(text: str, files: List[str]) -> bool:
    """Skill Fast list for plain code: a rename inside ONE named file, or a refactor that touches only test files."""
    low = text.lower()
    if re.search(r"\brenam\w*\b", low) and len(files) == 1:
        return True
    return bool(files) and all(_is_test_file(f) for f in files) and any(_has_term(low, v) for v in _FAST_TEST_VERBS)


def classify_lane(text: str, risk: Risk, change_type: Any, repo_files: Optional[List[str]] = None) -> tuple:
    """(Lane, reasons, locked). `change_type` = one ChangeType or a list of them; the lane is the highest across the types and the
    triggers. `locked` = a Strict trigger is present (an override cannot go below Strict)."""
    types = list(change_type) if isinstance(change_type, (list, tuple)) else [change_type]
    files = _named_files(text, repo_files)
    reasons = ["change type: " + ", ".join(t.value for t in types)]
    lane = Lane.FAST
    for t in types:
        tl = _TYPE_LANE[t]
        if t == ChangeType.SCRIPT and _matches(text, DISPOSABLE_TERMS):
            tl = Lane.FAST
            reasons.append("script marked disposable")
        if t == ChangeType.BACKEND and _fast_generic(text, files):
            tl = Lane.FAST
            reasons.append("rename in one file / test-only refactor")
        if _LANE_ORDER[tl] > _LANE_ORDER[lane]:
            lane = tl
    strict: List[str] = []
    if risk == Risk.HIGH:
        strict.append("high-risk triggers")
    for t in types:
        if _TYPE_LANE[t] == Lane.STRICT:
            strict.append(f"{t.value} changes are Strict")
    hits = _matches(text, STRICT_LANE_TERMS)
    if hits:
        strict.append(f"auth / payments / prod data: {', '.join(hits[:4])}")
    floors: List[str] = []
    if risk == Risk.MEDIUM:
        floors.append("medium risk")
    if _matches(text, GLOBAL_STYLE_TERMS):
        floors.append("global CSS / design-system tokens have a large blast radius")
    if _matches(text, CONSENT_TERMS):
        floors.append("consent / disclosure copy is legal text")
    if _matches(text, CONFIG_FLOOR_TERMS):
        floors.append("flags / timeouts / env vars can be prod-impacting")
    if strict:
        lane = Lane.STRICT
    elif floors and lane == Lane.FAST:
        lane = Lane.STANDARD
    reasons += strict or floors
    return lane, reasons, bool(strict)


def lane_override_from(text: str, setting: str = "") -> Optional[Lane]:
    """Explicit user choice: "lane: fast" / "use the strict lane" in the request wins over the lane_override setting."""
    m = _LANE_OVERRIDE_RE.search(text or "")
    word = (m.group(1) or m.group(2)) if m else (setting or "").strip()
    try:
        return Lane(word.strip().lower()) if word and word.strip() else None
    except ValueError:
        return None


_TYPE_LEVELS = {
    ChangeType.FE_PRESENTATIONAL: ["L3"], ChangeType.FE_BEHAVIOURAL: ["L1", "L3"], ChangeType.FE_SECURITY: ["L3"],
    ChangeType.DOCS: ["L1"], ChangeType.SQL: ["L1", "L2"], ChangeType.MIGRATION: ["L2"], ChangeType.IAC_CONFIG: ["L2"],
    ChangeType.SCRIPT: ["L1"], ChangeType.BACKEND: ["L1"],
}


def recommend_validation_level(lane: Lane, change_type: Any, needs_handoff: bool = False) -> str:
    """Ladder level(s) worth aiming for (validation-ladder.md): the lowest that gives real evidence. L5 is never recommended.
    `change_type` = one ChangeType or all of them (the levels are merged)."""
    types = list(change_type) if isinstance(change_type, (list, tuple)) else [change_type]
    if lane == Lane.STRICT and all(t in (ChangeType.BACKEND, ChangeType.SCRIPT) for t in types):
        levels = ["L2"]                               # Strict: L2-L4 where available
    else:
        levels = sorted({lv for t in types for lv in _TYPE_LEVELS[t]})
    if lane == Lane.STRICT and needs_handoff and "L4" not in levels:
        levels.append("L4")
    return ", ".join(levels)


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
    def __init__(self, llm_json: Optional[Callable[[str], dict]] = None, lane_override: str = ""):
        self.llm_json = llm_json
        self.lane_override = lane_override       # "" | fast | standard | strict (settings.lane_override)

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
        return self._with_lane(self._merge_llm(brief, q, intent, repo_files), q, repo_files)

    # ------------------------------------------------------------------ lane (after the LLM merge: it may have raised the risk)
    def _with_lane(self, brief: IntakeBrief, q: str, repo_files: Optional[List[str]]) -> IntakeBrief:
        found = detect_change_types(q, repo_files)
        types = [t for t, _ in found]
        lane, reasons, locked = classify_lane(q, brief.risk, types, repo_files)
        reasons[0] = "change type: " + "; ".join(f"{t.value} ({why})" for t, why in found[:4])
        wanted = lane_override_from(q, self.lane_override)
        if wanted and wanted != lane:
            if _LANE_ORDER[wanted] < _LANE_ORDER[lane] and locked:
                reasons.append(f"override '{wanted.value}' ignored: Strict triggers are present")
            else:
                reasons.append(f"user override: {wanted.value}")
                lane = wanted
        handoff = bool(_matches(q, HANDOFF_TERMS))
        planned = brief.intent in ("change", "plan")
        return brief.model_copy(update={
            "lane": lane, "lane_reasons": reasons, "change_type": types[0], "change_types": types, "needs_handoff": handoff,
            "validation_level": recommend_validation_level(lane, types, handoff) if planned else "",
            "change_like": brief.intent == "ask" and _change_like(q)})

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
