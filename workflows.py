"""
Tree of Thoughts Temporal Workflow
-----------------------------------
Each iteration of the loop:
  1. Fan out: generate N branches from each surviving parent (asyncio.gather)
  2. Evaluate: score every new branch in parallel (asyncio.gather)
  3. Prune: keep only the top `beam_width` branches above the score threshold
  4. Check for terminal answers — stop early if one is found
  5. Repeat until max_depth or a terminal answer

Temporal gives us:
  - Durable state across all iterations (no context window overflow)
  - Automatic retries on any activity failure
  - The ability to run this for as long as needed
"""
from __future__ import annotations

import asyncio
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from models import (
        ReasoningBranch,
        ToTConfig,
        ToTResult,
        GenerateBranchesInput,
        EvaluateBranchInput,
        EvaluateBranchOutput,
    )
    from activities import generate_branches, evaluate_branch, expand_branch

# Shared retry policy for all LLM-backed activities
_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=2),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(seconds=30),
    maximum_attempts=4,
)

_ACTIVITY_TIMEOUT = timedelta(minutes=2)


def _activity_opts(**kwargs):
    return {
        "start_to_close_timeout": _ACTIVITY_TIMEOUT,
        "retry_policy": _RETRY,
        **kwargs,
    }


@workflow.defn(name="TreeOfThoughtsWorkflow")
class TreeOfThoughtsWorkflow:

    @workflow.run
    async def run(self, config: ToTConfig) -> ToTResult:
        workflow.logger.info(f"Starting ToT for problem: {config.problem[:80]}...")

        total_explored = 0

        # ── Step 1: Generate root-level branches ──────────────────────────
        root_branches: list[ReasoningBranch] = await workflow.execute_activity(
            generate_branches,
            GenerateBranchesInput(
                problem=config.problem,
                num_branches=config.branches_per_node,
            ),
            **_activity_opts(),
        )

        # Tag root branches with depth 0
        for b in root_branches:
            b.depth = 0
        total_explored += len(root_branches)

        # ── Step 2: Evaluate root branches in parallel ─────────────────────
        scored_root = await self._evaluate_and_score(root_branches, config.problem)

        # Check for immediate terminal answer before pruning
        terminal = self._find_terminal(scored_root)
        if terminal:
            return self._build_result(config, terminal, total_explored, depth=0)

        active_branches = self._prune(scored_root, config)

        if not active_branches:
            workflow.logger.warning("All root branches pruned — stopping early")
            return self._empty_result(config, total_explored, depth=0)

        # ── Step 3: Iterative expansion loop ──────────────────────────────
        depth = 0
        for depth in range(1, config.max_depth + 1):
            workflow.logger.info(
                f"Depth {depth}: expanding {len(active_branches)} branches"
            )

            # Fan out: expand all surviving branches in parallel
            expansion_tasks = [
                workflow.execute_activity(
                    expand_branch,
                    GenerateBranchesInput(
                        problem=config.problem,
                        parent_thought=branch.thought,
                        num_branches=config.branches_per_node,
                    ),
                    **_activity_opts(),
                )
                for branch in active_branches
            ]
            expanded_lists: list[list[ReasoningBranch]] = await asyncio.gather(
                *expansion_tasks
            )

            # Flatten, tag with depth and parent
            new_branches: list[ReasoningBranch] = []
            for parent, children in zip(active_branches, expanded_lists):
                for child in children:
                    child.depth = depth
                    child.parent_id = parent.branch_id
                    # Prepend parent context so evaluator has full reasoning chain
                    child.thought = f"{parent.thought}\n\n→ {child.thought}"
                    new_branches.append(child)

            total_explored += len(new_branches)

            # Evaluate all new branches in parallel
            scored_branches = await self._evaluate_and_score(
                new_branches, config.problem
            )

            # Check for terminal answers before pruning
            terminal = self._find_terminal(scored_branches)
            if terminal:
                return self._build_result(config, terminal, total_explored, depth)

            # Prune to beam width for next iteration
            active_branches = self._prune(scored_branches, config)

            if not active_branches:
                workflow.logger.warning("All branches pruned — stopping early")
                break

        # ── No clean terminal found — return best remaining branch ─────────
        if not active_branches:
            return self._empty_result(config, total_explored, depth)

        best = max(active_branches, key=lambda b: b.score)
        best.is_terminal = True
        best.answer = (
            f"Best reasoning found (score={best.score:.2f}):\n\n{best.thought}"
        )
        return self._build_result(
            config, best, total_explored, depth=best.depth
        )

    # ── Helpers ────────────────────────────────────────────────────────────

    async def _evaluate_and_score(
        self,
        branches: list[ReasoningBranch],
        problem: str,
    ) -> list[ReasoningBranch]:
        """Evaluate all branches in parallel and attach scores."""
        eval_tasks = [
            workflow.execute_activity(
                evaluate_branch,
                EvaluateBranchInput(problem=problem, thought=b.thought),
                **_activity_opts(),
            )
            for b in branches
        ]
        evaluations: list[EvaluateBranchOutput] = await asyncio.gather(*eval_tasks)

        for branch, evaluation in zip(branches, evaluations):
            branch.score = evaluation.score
            branch.is_terminal = evaluation.is_terminal
            branch.answer = evaluation.answer

        return branches

    @staticmethod
    def _prune(
        branches: list[ReasoningBranch], config: ToTConfig
    ) -> list[ReasoningBranch]:
        """Keep top beam_width branches above the score threshold."""
        passing = [b for b in branches if b.score >= config.min_score_threshold]
        passing.sort(key=lambda b: b.score, reverse=True)
        kept = passing[: config.beam_width]
        workflow.logger.info(
            f"Pruned {len(branches)} → {len(kept)} branches "
            f"(threshold={config.min_score_threshold}, beam={config.beam_width})"
        )
        return kept

    @staticmethod
    def _find_terminal(
        branches: list[ReasoningBranch],
    ) -> ReasoningBranch | None:
        """Return the highest-scoring terminal branch, if any."""
        terminals = [b for b in branches if b.is_terminal and b.answer]
        if not terminals:
            return None
        return max(terminals, key=lambda b: b.score)

    @staticmethod
    def _empty_result(
        config: ToTConfig, total_explored: int, depth: int
    ) -> ToTResult:
        return ToTResult(
            problem=config.problem,
            answer="All branches were pruned — no viable reasoning path found.",
            winning_branch=ReasoningBranch(
                branch_id="none", thought="No branches survived pruning."
            ),
            total_branches_explored=total_explored,
            depth_reached=depth,
        )

    @staticmethod
    def _build_result(
        config: ToTConfig,
        branch: ReasoningBranch,
        total_explored: int,
        depth: int,
    ) -> ToTResult:
        return ToTResult(
            problem=config.problem,
            answer=branch.answer or branch.thought,
            winning_branch=branch,
            total_branches_explored=total_explored,
            depth_reached=depth,
        )
