"""Skill pack loader (progressive disclosure): reads skills/risk-proportional-code-dev and returns prompt text for a request.
stdlib + intake only (no settings, no yaml): the directory and the size cap are injected.

SKILL.md (the core) is always included; each reference file is added only when the lane / change type calls for it
(SECTION_RULES below). A missing or broken pack gives "" and one log line; nothing here raises into the chat path."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .intake import ChangeType, IntakeBrief, Lane, Risk

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]          # a relative pack dir is resolved against this, not the cwd
CORE_FILE = "SKILL.md"
REFERENCES = {                                           # section name -> file under <pack>/references/
    "task-types-and-risk": "task-types-and-risk.md",     # read on demand; never injected (intake already classifies)
    "validation-ladder": "validation-ladder.md",
    "regulated-domains": "regulated-domains.md",
    "delivery-and-review": "delivery-and-review.md",
}
_MAX_FILE_BYTES = 256 * 1024
_TRUNC_MARK = "\n[... skill pack truncated to fit the size cap ...]"
_MIN_SECTION_CHARS = 400            # a section that would be cut below this is left out instead


def _val(x: Any) -> str:
    return str(getattr(x, "value", x) or "").strip().lower()


def sections_for(lane: Any, change_type: Any = None, risk: Any = None, intent: str = "change",
                 change_like: bool = False) -> List[str]:
    """Section names to load, in prompt order. [] = no pack (ask-only intents, or a request that was not classified).
    An ask that reads like an edit request (change_like) gets the core only: the ask agent may propose edits."""
    lane_v, type_v, risk_v = _val(lane), _val(change_type), _val(risk)
    intent = (intent or "").strip().lower()
    if lane_v not in {x.value for x in Lane}:
        return []
    if intent == "ask" and change_like:
        return ["core"]
    if intent not in ("change", "plan"):
        return []
    std_or_strict = lane_v in (Lane.STANDARD.value, Lane.STRICT.value)
    names = ["core"]
    if std_or_strict or type_v == ChangeType.FE_PRESENTATIONAL.value:        # Fast: only for presentational front-end
        names.append("validation-ladder")
    if lane_v == Lane.STRICT.value or risk_v == Risk.HIGH.value:
        names.append("regulated-domains")
    if std_or_strict:
        names.append("delivery-and-review")
    return names


# ------------------------------------------------------------------ parsing
def parse_frontmatter(text: str) -> Tuple[Dict[str, str], str]:
    """({key: value}, body) for a leading `---` block of simple `key: value` lines (indented lines continue the value).
    No leading block or an unterminated one -> ({}, text)."""
    lines = text.lstrip("﻿").splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    meta: Dict[str, str] = {}
    key = ""
    for i in range(1, len(lines)):
        ln = lines[i]
        if ln.strip() == "---":
            return meta, "\n".join(lines[i + 1:]).lstrip("\n")
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        if ln[0] in " \t" and key:
            meta[key] = (meta[key] + " " + ln.strip()).strip()
            continue
        k, sep, v = ln.partition(":")
        if not sep or not k.strip():
            continue
        key = k.strip().lower()
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
            v = v[1:-1]
        meta[key] = v
    return {}, text


def _split_sections(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    """(preamble, [(heading title, block incl. heading line)]) split on `## ` headings."""
    pre: List[str] = []
    out: List[Tuple[str, List[str]]] = []
    for ln in text.splitlines():
        if ln.startswith("## "):
            out.append((ln[3:].strip(), [ln]))
        elif out:
            out[-1][1].append(ln)
        else:
            pre.append(ln)
    return "\n".join(pre).strip(), [(t, "\n".join(b).strip()) for t, b in out]


def ladder_for_presentational_fe(text: str) -> str:
    """The part of validation-ladder.md a Fast presentational-FE change needs: the levels, plus the one profile row.
    Falls back to the whole file when the layout is not the expected one."""
    pre, secs = _split_sections(text)
    by = {t.lower(): b for t, b in secs}
    levels, profile = by.get("levels"), by.get("profile by change type")
    if not levels or not profile:
        return text.strip()
    kept = []
    for ln in profile.splitlines():
        is_row = ln.startswith("|")
        if not is_row or ln.startswith("| Change type") or ln.startswith("|---") or ln.lower().startswith("| fe presentational"):
            kept.append(ln)
    return "\n\n".join(x for x in (pre, levels, "\n".join(kept)) if x).strip()


# ------------------------------------------------------------------ loading + cache
@dataclass(frozen=True)
class Pack:
    name: str
    description: str
    core: str
    refs: Dict[str, str]


_cache: Dict[str, Tuple[tuple, Optional[Pack]]] = {}
_lock = threading.Lock()


def resolve_dir(pack_dir: Any, base: Optional[Path] = None) -> Path:
    p = Path(str(pack_dir or "")).expanduser()
    return p if p.is_absolute() else (base or REPO_ROOT) / p


def _signature(root: Path) -> tuple:
    """(file, mtime_ns, size) of the core and every reference: the cache key, so an edit is picked up on the next request."""
    sig = []
    for p in [root / CORE_FILE] + [root / "references" / f for f in REFERENCES.values()]:
        try:
            st = p.stat()
            sig.append((p.name, st.st_mtime_ns, st.st_size))
        except OSError:
            sig.append((p.name, None, None))
    return tuple(sig)


def _read(p: Path) -> str:
    if p.stat().st_size > _MAX_FILE_BYTES:
        raise ValueError(f"{p.name} is larger than {_MAX_FILE_BYTES} bytes")
    return p.read_text(encoding="utf-8-sig")


def _build(root: Path) -> Optional[Pack]:
    core_path = root / CORE_FILE
    try:
        meta, body = parse_frontmatter(_read(core_path))
    except (OSError, ValueError) as exc:        # includes UnicodeDecodeError
        log.warning("skill pack unavailable (%s): %s", core_path, exc)
        return None
    if not meta.get("name") or not body.strip():
        log.warning("skill pack broken (%s): SKILL.md needs a frontmatter `name:` and a body", core_path)
        return None
    refs: Dict[str, str] = {}
    for key, fname in REFERENCES.items():
        try:
            refs[key] = _read(root / "references" / fname).strip()
        except (OSError, ValueError) as exc:
            log.warning("skill pack reference %s skipped: %s", fname, exc)
    return Pack(meta["name"], meta.get("description", ""), body.strip(), refs)


def load_pack(pack_dir: Any, base: Optional[Path] = None) -> Optional[Pack]:
    """Parsed pack, cached until a pack file's mtime/size changes. None = missing or broken (logged once per state)."""
    try:
        root = resolve_dir(pack_dir, base)
        sig = _signature(root)
        key = str(root)
        with _lock:
            hit = _cache.get(key)
            if hit and hit[0] == sig:
                return hit[1]
        pack = _build(root)
        with _lock:
            _cache[key] = (sig, pack)
        return pack
    except Exception as exc:        # noqa: BLE001 - never raise into the chat path
        log.warning("skill pack load failed: %s: %s", type(exc).__name__, exc)
        return None


def clear_cache() -> None:
    with _lock:
        _cache.clear()


# ------------------------------------------------------------------ prompt text
def _cap(parts: List[str], max_chars: int) -> str:
    """Join parts (most important first) within max_chars. The first part that does not fit is cut at a line (the core
    even when tiny; a reference only if enough of it fits) and everything after it is dropped."""
    out: List[str] = []
    used = 0
    for i, part in enumerate(parts):
        sep = 2 if out else 0
        room = max_chars - used - sep
        if len(part) <= room:
            out.append(part)
            used += sep + len(part)
            continue
        if i == 0 or room - len(_TRUNC_MARK) >= _MIN_SECTION_CHARS:
            mark = _TRUNC_MARK if room - len(_TRUNC_MARK) >= _MIN_SECTION_CHARS else ""     # no room for a note: a bare cut
            cut = part[: max(room - len(mark), 0)]
            out.append((cut[: cut.rfind("\n")] if "\n" in cut else cut).rstrip() + mark)
        break
    return "\n\n".join(out)[:max_chars]


def load_skill_prompt(*, lane: Any, change_type: Any = None, risk: Any = None, intent: str = "change", pack_dir: Any,
                      max_chars: int = 12000, base: Optional[Path] = None, change_like: bool = False) -> str:
    """Prompt text for this lane / change type, within max_chars. "" = no pack (wrong intent, missing or broken pack)."""
    try:
        names = sections_for(lane, change_type, risk, intent, change_like)
        if not names or max_chars <= 0:
            return ""
        pack = load_pack(pack_dir, base)
        if pack is None:
            return ""
        parts = [pack.core]
        for name in names[1:]:
            text = pack.refs.get(name, "")
            if name == "validation-ladder" and text and _val(lane) == Lane.FAST.value:
                text = ladder_for_presentational_fe(text)
            if text:
                parts.append(text)
        return _cap(parts, int(max_chars))
    except Exception as exc:        # noqa: BLE001
        log.warning("skill prompt failed: %s: %s", type(exc).__name__, exc)
        return ""


def lane_statement(brief: IntakeBrief) -> str:
    lane = _val(brief.lane)
    why = "; ".join(brief.lane_reasons[:3])
    return (f"LANE FOR THIS REQUEST: {lane.upper()} ({why}). Internal: apply the {lane.capitalize()}-lane rules below and never "
            f"skip the rules under \"Never skip\"; do not state the lane in your output.")


def guidance_for(brief: IntakeBrief, *, enabled: bool, pack_dir: Any, max_chars: int, base: Optional[Path] = None) -> str:
    """The text the supervisor puts in front of change/plan prompts: lane statement + selected pack sections. "" = none."""
    if not enabled or brief.lane is None:
        return ""
    text = load_skill_prompt(lane=brief.lane, change_type=brief.change_type, risk=brief.risk, intent=brief.intent,
                             pack_dir=pack_dir, max_chars=max_chars, base=base, change_like=brief.change_like)
    return f"{lane_statement(brief)}\n\n{text}" if text else ""
