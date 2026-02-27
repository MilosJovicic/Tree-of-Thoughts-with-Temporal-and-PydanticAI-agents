# Tree of Thoughts — Temporal + PydanticAI

A production-grade implementation of **Tree of Thoughts (ToT)** reasoning using
[Temporal](https://temporal.io) for durable orchestration and
[PydanticAI](https://ai.pydantic.dev) for structured LLM calls.

## How it works

```
Problem
  │
  ├── Branch A (depth 0)    ← generate_branches activity
  ├── Branch B
  └── Branch C
        │
        ▼ evaluate_branch (parallel)
        │
   Prune to beam_width=2
        │
        ├── Branch B (score 0.8) ─┐
        └── Branch C (score 0.6) ─┤
                                  │ expand_branch (parallel fan-out)
                                  ▼
              ┌────────────────────────────┐
              │ B1  B2  B3  C1  C2  C3    │  (depth 1)
              └────────────────────────────┘
                        │
                   evaluate + prune
                        │
                    ... repeat ...
                        │
                   Terminal answer ✓
```

Each box is a **Temporal activity** — durable, retriable, and independently
logged. The workflow loop is a plain Python `while`/`for` loop inside a
Temporal workflow, meaning:

- Survives worker restarts mid-execution
- Each LLM call is automatically retried on failure
- No context window pressure — state lives in Temporal, not the LLM
- Can run for minutes or hours without issue

## Project structure

```
tot_project/
├── models.py       — Pydantic models (branches, config, result)
├── activities.py   — PydanticAI agents wrapped as Temporal activities
├── workflows.py    — ToT workflow: fan-out → evaluate → prune → loop
├── worker.py       — Temporal worker (registers workflow + activities)
├── run.py          — Client to submit a problem and get an answer
└── requirements.txt
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your environment variables

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4o        # or any OpenAI-compatible model name
```

### 3. Start the Temporal dev server

```bash
# Install CLI: https://docs.temporal.io/cli
temporal server start-dev
```

### 4. Start the worker (terminal 1)

```bash
python worker.py
```

### 5. Run a problem (terminal 2)

```bash
# Default problem (river crossing puzzle)
python run.py

# Custom problem
python run.py --problem "What is the best strategy to win at chess in 10 moves?"
```

## Configuration (ToTConfig)

| Field | Default | Description |
|-------|---------|-------------|
| `max_depth` | 3 | Maximum tree depth before forcing best answer |
| `branches_per_node` | 3 | LLM-generated branches at each expansion |
| `beam_width` | 2 | How many branches survive each pruning step |
| `min_score_threshold` | 0.3 | Branches below this score are discarded |

Increase `max_depth` and `beam_width` for harder problems. Each increase
multiplies LLM calls — Temporal handles the fan-out reliably.

## Temporal UI

While running, visit [http://localhost:8233](http://localhost:8233) to watch
the workflow execute step by step, see activity retries, and inspect state.
