"""ReasoningSupervisor: Intake -> Baseline -> (clarify | execute under policy overlay) -> Cross-check -> Deliver."""
from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Mapping, Tuple

from services import action_log as al

from .baseline import BaselinePolicy, policy_for, policy_scope
from .crosscheck import VerifierAgent
from .intake import IntakeAgent, IntakeBrief


def clarify_markdown(brief: IntakeBrief) -> str:
    qs = brief.clarifying_questions or ["Can you describe exactly what should change, and in which file?"]
    lines = ["Before I change anything I need a few details. This looks like a high-risk change "
             f"({'; '.join(brief.risk_reasons)}) and the request is ambiguous ({'; '.join(brief.ambiguity_reasons)}).", ""]
    lines += [f"{i}. {q}" for i, q in enumerate(qs[:3], 1)]
    return "\n".join(lines)


class ReasoningSupervisor:
    def __init__(self, intake: IntakeAgent, verifier: VerifierAgent,
                 defaults_provider: Callable[[], Mapping[str, Any]], enabled: bool = True):
        self.intake, self.verifier, self.defaults_provider, self.enabled = intake, verifier, defaults_provider, enabled

    async def run(self, *, query: str, intent: str, repo_files: list,
                  execute: Callable[[IntakeBrief, BaselinePolicy], Awaitable[Tuple[str, dict]]]) -> str:
        if not self.enabled:
            answer, _ = await execute(IntakeBrief(intent=intent, explicit_ask=query), BaselinePolicy())
            return answer

        try:
            with al.action("INTAKE", "IntakeAgent", "analyze request") as h:
                brief = await asyncio.to_thread(self.intake.analyze, query, intent, repo_files)   # worker thread: an LLM-backed intake must not block the loop
                h.note(f"risk={brief.risk.value} ambiguity={brief.ambiguity} enterprise={brief.enterprise}",
                       brief=brief.model_dump(mode="json"))
            try:
                defaults = dict(self.defaults_provider() or {})
            except Exception:
                defaults = {}
            with al.action("BASELINE", "BaselinePolicy", "select policy") as h:
                policy = policy_for(brief, defaults)
                h.note(f"depth={policy.depth} overrides={policy.overrides}", policy=policy.model_dump(mode="json"))
        except Exception as exc:        # the reasoning layer must never cost the user their answer
            al.log_action("INTAKE", "ReasoningSupervisor", "fallback to plain execution", "fail",
                          f"{type(exc).__name__}: {exc}"[:200])
            answer, _ = await execute(IntakeBrief(intent=intent, explicit_ask=query), BaselinePolicy())
            return answer

        if policy.clarify_first:
            with al.action("DELIVER", "ReasoningSupervisor", "ask clarifying questions") as h:
                h.note("clarify first: high risk and high ambiguity", questions=brief.clarifying_questions)
                return clarify_markdown(brief)

        with policy_scope(policy.overrides):
            answer, facts = await execute(brief, policy)
        facts = facts or {}

        try:
            with al.action("CROSSCHECK", "VerifierAgent", "cross-check requirements") as h:
                notes = await asyncio.to_thread(self.verifier.check, brief, answer, facts)
                unmet = [r["req"] for r in notes.requirements if r.get("status") == "unmet"]
                h.note(f"{len(notes.requirements)} requirement(s), {len(unmet)} unmet",
                       facts={k: v for k, v in facts.items() if k != "files_changed"}, unmet=unmet[:10])
            with al.action("DELIVER", "ReasoningSupervisor", "append delivery notes"):
                return f"{answer}\n\n{notes.to_markdown()}"
        except Exception as exc:        # verifier/notes failed (already logged as a fail action): deliver the plain answer
            al.log_action("DELIVER", "ReasoningSupervisor", "deliver without notes", "fail",
                          f"{type(exc).__name__}: {exc}"[:200])
            return answer
