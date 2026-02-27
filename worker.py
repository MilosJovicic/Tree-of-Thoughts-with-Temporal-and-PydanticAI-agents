"""
Worker process — run this to start processing workflow and activity tasks.

Usage:
    python worker.py

Requires a local Temporal server:
    temporal server start-dev
"""
import asyncio
import logging

from temporalio.client import Client
from temporalio.worker import Worker

from activities import generate_branches, evaluate_branch, expand_branch
from workflows import TreeOfThoughtsWorkflow
from models import TASK_QUEUE

logging.basicConfig(level=logging.INFO)


async def main():
    client = await Client.connect("localhost:7233")

    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[TreeOfThoughtsWorkflow],
        activities=[generate_branches, evaluate_branch, expand_branch],
    )

    print(f"Worker started on task queue: {TASK_QUEUE}")
    print("Waiting for workflows... (Ctrl+C to stop)\n")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
