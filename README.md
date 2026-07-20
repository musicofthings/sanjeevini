# Sanjeevini

Revive dead bioinformatics tools — with first-class support for long-read
sequencing (ONT/PacBio) and workflow languages (Nextflow/Snakemake/WDL).

The agent is named **Jeeva** (Sanskrit: "life"); the CLI command is `jeeva`.

📖 **[Full docs and quickstart →](https://musicofthings.github.io/sanjeevini/)**

## What it does

Jeeva reads a dead research repo, writes its own falsifiable plan, and repairs
it inside a Docker sandbox until it *provably* runs — then emits a pinned,
verified, composable brick (`contract.yaml`, `Dockerfile`, `predict.py`,
`smoke_test.sh`, `REPRODUCE.md`, `PROVENANCE.json`).

The core guarantee: **Jeeva never declares success on its own word.** A
resurrection PASSes only when a command the agent marks as the *sanity check*
actually exits 0 inside the sandbox. The model chooses commands; exit codes
decide verdicts.

## Verified resurrection

`a-slide/pycoQC` — an archived Oxford Nanopore read-QC tool:

| | |
|---|---|
| Verdict | **PASS** in 9 turns |
| Image | `sanjeevini/pycoqc:resurrected` (783 MB) |
| Output | 1.77 MB HTML report + valid JSON, exit 0 |
| Reproduced | `bash smoke_test.sh` replays the verified chain in a clean container |

Two honest caveats on this run: `bugs_fixed` was empty — pycoQC still installs
cleanly, so this is a *still-runs* case rather than a repair, and the repair
heuristics were not exercised. And `cost_usd` reads `0.0` because token pricing
is not yet wired into the provenance record.

## Test results

```
229 passed, 6 deselected  ·  84% coverage (2424 statements)
ruff check + ruff format + mypy --strict: clean across src/ and tests/
```

The 6 deselected are `-m integration` tests that need a live Docker daemon and
network; `pytest -m "not integration"` is the CI gate. Coverage by subsystem:

| Subsystem | Coverage |
|---|---|
| `repair/knowledge.py` (cross-run learning) | 99% |
| `repair/agent.py` | 85% |
| `repair/loop.py` | 84% |
| `scouts/workflow_scout.py` | 94% |
| **Total** | **84%** |

Uncovered lines are overwhelmingly Docker-daemon and optional-dependency paths
marked `# pragma: no cover`, plus CLI wiring exercised by the integration tests.

## Cross-run learning

Each resurrection that repairs something teaches the next one. The loop records
every applied patch as a lesson — the traceback it responded to (the *symptom*),
the fix, and the framework — into a JSON store under the cache root. On the next
run the agent retrieves the relevant lessons and injects them into its prompt.

Relevance is deliberately dependency-free: a framework match plus keyword overlap
between the current traceback and a stored symptom. That is enough to surface
*"last time a TensorFlow 1.x tool hit a missing-AVX abort, a non-AVX build fixed
it"* exactly when the agent is staring at that traceback, without pulling in an
embedding model.

Lessons are injected as **hints, not instructions** — the prompt tells the agent
to verify before trusting one, and PASS is still decided only by a real exit
code, so a wrong lesson cannot poison a later run.

The store starts empty. It only begins paying off once a genuinely decayed tool
gets repaired.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'

ruff check . && ruff format .
mypy src/
pytest -m "not integration" -q          # the CI gate
pytest -q                               # everything, needs Docker + network
```

See the PRD/TRD for the full implementation specification.
