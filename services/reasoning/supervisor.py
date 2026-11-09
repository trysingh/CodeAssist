"""ReasoningSupervisor: Intake -> Baseline -> (clarify | execute under policy overlay) -> Cross-check -> Deliver."""
from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Mapping, Optional, Tuple

from services import action_log as al

from .baseline import BaselinePolicy, policy_for, policy_scope, strip_guidance_echo
from .crosscheck import VerifierAgent
from .intake import IntakeAgent, IntakeBrief
from .stacking import change_summary


def clarify_markdown(brief: IntakeBrief) -> str:
    qs = brief.clarifying_questions or ["Can you describe exactly what should change, and in which file?"]
    lines = ["Before I change anything I need a few details. This looks like a high-risk change "
             f"({'; '.join(brief.risk_reasons)}) and the request is ambiguous ({'; '.join(brief.ambiguity_reasons)}).", ""]
    lines += [f"{i}. {q}" for i, q in enumerate(qs[:3], 1)]
    return "\n".join(lines)


class ReasoningSupervisor:
    def __init__(self, intake: IntakeAgent, verifier: VerifierAgent,
                 defaults_provider: Callable[[], Mapping[str, Any]], enabled: bool = True,
                 guidance_provider: Optional[Callable[[IntakeBrief, BaselinePolicy], str]] = None):
        self.intake, self.verifier, self.defaults_provider, self.enabled = intake, verifier, defaults_provider, enabled
        self.guidance_provider = guidance_provider      # v3: (brief, policy) -> skill-pack text for change/plan prompts; "" = none

    def _guidance(self, brief: IntakeBrief, policy: BaselinePolicy) -> str:
        """Skill-pack text for this request. Any failure means no guidance: the request still runs under the baseline policy."""
        if self.guidance_provider is None:
            return ""
        try:
            return str(self.guidance_provider(brief, policy) or "")
        except Exception as exc:        # noqa: BLE001
            al.log_action("BASELINE", "SkillPack", "load skill pack", "fail", f"{type(exc).__name__}: {exc}"[:200])
            return ""

    async def run(self, *, query: str, intent: str, repo_files: list,
                  execute: Callable[[IntakeBrief, BaselinePolicy], Awaitable[Tuple[str, dict]]]) -> str:
        if not self.enabled:
            answer, _ = await execute(IntakeBrief(intent=intent, explicit_ask=query), BaselinePolicy())
            return answer

        try:
            with al.action("INTAKE", "IntakeAgent", "analyze request") as h:
                brief = await asyncio.to_thread(self.intake.analyze, query, intent, repo_files)   # worker thread: an LLM-backed intake must not block the loop
                h.note(f"risk={brief.risk.value} ambiguity={brief.ambiguity} enterprise={brief.enterprise}"
                       + (f" lane={brief.lane.value}" if brief.lane else ""),
                       brief=brief.model_dump(mode="json"))
            try:
                defaults = dict(self.defaults_provider() or {})
            except Exception:
                defaults = {}
            with al.action("BASELINE", "BaselinePolicy", "select policy") as h:
                policy = policy_for(brief, defaults)
                guidance = self._guidance(brief, policy)
                h.note(f"depth={policy.depth} overrides={policy.overrides}", policy=policy.model_dump(mode="json"),
                       lane=brief.lane.value if brief.lane else None, change_type=brief.change_type.value,
                       validation_level=brief.validation_level, guidance_chars=len(guidance))
        except Exception as exc:        # the reasoning layer must never cost the user their answer
            al.log_action("INTAKE", "ReasoningSupervisor", "fallback to plain execution", "fail",
                          f"{type(exc).__name__}: {exc}"[:200])
            answer, _ = await execute(IntakeBrief(intent=intent, explicit_ask=query), BaselinePolicy())
            return answer

        if policy.clarify_first:
            with al.action("DELIVER", "ReasoningSupervisor", "ask clarifying questions") as h:
                h.note("clarify first: high risk and high ambiguity", questions=brief.clarifying_questions)
                return clarify_markdown(brief)

        with policy_scope(policy.overrides, guidance=guidance):
            answer, facts = await execute(brief, policy)
        facts = facts or {}
        answer = strip_guidance_echo(answer) if guidance else answer      # defensive: a model may echo the internal header
        if facts.get("skip_notes"):         # e.g. a repeated request answered from an existing revision: nothing was built or checked
            return answer

        try:
            with al.action("CROSSCHECK", "VerifierAgent", "cross-check requirements") as h:
                notes = await asyncio.to_thread(self.verifier.check, brief, answer, facts)
                unmet = [r["req"] for r in notes.requirements if r.get("status") == "unmet"]
                h.note(f"{len(notes.requirements)} requirement(s), {len(unmet)} unmet",
                       facts={k: v for k, v in facts.items() if k != "files_changed"}, unmet=unmet[:10])
            with al.action("DELIVER", "ReasoningSupervisor", "append delivery notes"):
                try:        # deterministic plain summary on top of a saved change; its failure must not cost the notes
                    summary = change_summary(brief, facts, notes)
                except Exception as exc:  # noqa: BLE001
                    al.log_action("DELIVER", "ReasoningSupervisor", "summary skipped", "fail", f"{type(exc).__name__}: {exc}"[:200])
                    summary = ""
                return f"{summary}{answer}\n\n{notes.to_markdown()}"
        except Exception as exc:        # verifier/notes failed (already logged as a fail action): deliver the plain answer
            al.log_action("DELIVER", "ReasoningSupervisor", "deliver without notes", "fail",
                          f"{type(exc).__name__}: {exc}"[:200])
            return answer
