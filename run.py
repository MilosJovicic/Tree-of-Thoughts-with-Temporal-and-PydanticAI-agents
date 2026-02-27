"""
Run a Tree of Thoughts workflow.

Usage:
    python run.py
    python run.py --problem "Your custom problem here"

Make sure worker.py is running in another terminal first.
"""
import asyncio
import argparse

from temporalio.client import Client

from models import ToTConfig, ToTResult, TASK_QUEUE

DEFAULT_PROBLEM = (
    "how many number 3 are here: 3.1415926535 8979323846 2643383279 5028841971 6939937510"
    
)


async def main(problem: str):
    client = await Client.connect("localhost:7233")

    config = ToTConfig(
        problem=problem,
        max_depth=3,
        branches_per_node=3,
        beam_width=2,
        min_score_threshold=0.3,
    )

    print(f"\n{'='*60}")
    print(f"Problem:\n  {problem}")
    print(f"{'='*60}")
    print(f"Config: depth={config.max_depth}, branches={config.branches_per_node}, "
          f"beam={config.beam_width}\n")

    result: ToTResult = await client.execute_workflow(
        "TreeOfThoughtsWorkflow",
        config,
        id=f"tot-{hash(problem) & 0xFFFFFF:06x}",
        task_queue=TASK_QUEUE,
        result_type=ToTResult,
    )

    print(f"\n{'='*60}")
    print("RESULT")
    print(f"{'='*60}")
    print(f"Answer:\n{result.answer}")
    print(f"\nStats:")
    print(f"  Branches explored : {result.total_branches_explored}")
    print(f"  Depth reached     : {result.depth_reached}")
    print(f"  Winning score     : {result.winning_branch.score:.2f}")
    print(f"\nWinning reasoning chain:\n{result.winning_branch.thought}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", default=DEFAULT_PROBLEM)
    args = parser.parse_args()
    asyncio.run(main(args.problem))
