"""
Temporal activities — each one wraps a PydanticAI agent call.
Activities are the durable, retriable units of work in the workflow.
"""
from __future__ import annotations

import os
import uuid
from prompts import generator_prompt, evaluator_prompt, expander_prompt 
from dotenv import load_dotenv

load_dotenv()

import logfire
logfire.configure()
logfire.instrument_pydantic_ai()

from temporalio import activity
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

from models import (
    GenerateBranchesInput,
    GenerateBranchesOutput,
    EvaluateBranchInput,
    EvaluateBranchOutput,
    ReasoningBranch,
)

_model = OpenAIModel(os.environ["OPENAI_MODEL"])


# ---------------------------------------------------------------------------
# Activity 1 — Generate reasoning branches
# ---------------------------------------------------------------------------
_generator_agent: Agent[None, GenerateBranchesOutput] = Agent(
    model=_model,
    output_type=GenerateBranchesOutput,
    system_prompt=generator_prompt,
    output_retries=3,
)


@activity.defn(name="generate_branches")
async def generate_branches(inp: GenerateBranchesInput) -> list[ReasoningBranch]:
    """Ask the LLM to generate N distinct reasoning thoughts from the current state."""

    if inp.parent_thought:
        prompt = (
            f"Problem: {inp.problem}\n\n"
            f"Current reasoning so far:\n{inp.parent_thought}\n\n"
            f"Generate {inp.num_branches} distinct ways to continue this line of reasoning."
        )
    else:
        prompt = (
            f"Problem: {inp.problem}\n\n"
            f"Generate {inp.num_branches} distinct high-level approaches to solving this."
        )

    result = await _generator_agent.run(prompt)
    output: GenerateBranchesOutput = result.output

    branches = [
        ReasoningBranch(
            branch_id=str(uuid.uuid4()),
            thought=thought,
        )
        for thought in output.thoughts[: inp.num_branches]
    ]
    activity.logger.info(f"Generated {len(branches)} branches")
    return branches


# ---------------------------------------------------------------------------
# Activity 2 — Evaluate a single branch
# ---------------------------------------------------------------------------
_evaluator_agent: Agent[None, EvaluateBranchOutput] = Agent(
    model=_model,
    output_type=EvaluateBranchOutput,
    system_prompt=evaluator_prompt,
    output_retries=3,
)


@activity.defn(name="evaluate_branch")
async def evaluate_branch(inp: EvaluateBranchInput) -> EvaluateBranchOutput:
    """Score a reasoning branch and check if it's a terminal answer."""

    prompt = (
        f"Problem: {inp.problem}\n\n"
        f"Reasoning branch to evaluate:\n{inp.thought}"
    )

    result = await _evaluator_agent.run(prompt)
    output: EvaluateBranchOutput = result.output
    activity.logger.info(
        f"Evaluated branch — score={output.score:.2f}, terminal={output.is_terminal}"
    )
    return output


# ---------------------------------------------------------------------------
# Activity 3 — Expand a branch (generate the next thought given prior context)
# ---------------------------------------------------------------------------
_expander_agent: Agent[None, GenerateBranchesOutput] = Agent(
    model=_model,
    output_type=GenerateBranchesOutput,
    system_prompt=expander_prompt,
    output_retries=3,
)


@activity.defn(name="expand_branch")
async def expand_branch(inp: GenerateBranchesInput) -> list[ReasoningBranch]:
    """Expand an existing branch into child branches using the expander agent."""

    prompt = (
        f"Problem: {inp.problem}\n\n"
        f"Reasoning so far:\n{inp.parent_thought}\n\n"
        f"Generate {inp.num_branches} distinct next steps to continue this reasoning."
    )

    result = await _expander_agent.run(prompt)
    output: GenerateBranchesOutput = result.output

    branches = [
        ReasoningBranch(
            branch_id=str(uuid.uuid4()),
            thought=thought,
        )
        for thought in output.thoughts[: inp.num_branches]
    ]
    activity.logger.info(f"Expanded into {len(branches)} branches")
    return branches
