from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional

TASK_QUEUE = "tree-of-thoughts"


class ReasoningBranch(BaseModel):
    """A single reasoning branch / thought in the tree."""
    branch_id: str
    parent_id: Optional[str] = None
    depth: int = 0
    thought: str  # The reasoning text produced by the LLM
    score: float = 0.0  # Evaluation score 0.0 – 1.0
    is_terminal: bool = False  # True if this branch reached a final answer
    answer: Optional[str] = None  # Populated if is_terminal


class GenerateBranchesInput(BaseModel):
    problem: str
    parent_thought: Optional[str] = None  # None for root expansion
    num_branches: int = 3


class GenerateBranchesOutput(BaseModel):
    thoughts: list[str] = Field(
        description="List of distinct reasoning steps / approaches to explore"
    )


class EvaluateBranchInput(BaseModel):
    problem: str
    thought: str


class EvaluateBranchOutput(BaseModel):
    score: float = Field(
        ge=0.0, le=1.0,
        description="How promising this reasoning branch is (0 = dead end, 1 = very promising)"
    )
    is_terminal: bool = Field(
        description="True if this thought constitutes a complete, final answer"
    )
    answer: Optional[str] = Field(
        default=None,
        description="The final answer if is_terminal is True"
    )
    rationale: str = Field(description="Brief explanation of the score")


class ToTConfig(BaseModel):
    """Configuration for a Tree of Thoughts workflow run."""
    problem: str
    max_depth: int = 3
    branches_per_node: int = 3
    beam_width: int = 2          # How many branches to keep after each pruning step
    min_score_threshold: float = 0.3  # Prune branches below this score


class ToTResult(BaseModel):
    """Final result returned by the workflow."""
    problem: str
    answer: str
    winning_branch: ReasoningBranch
    total_branches_explored: int
    depth_reached: int
