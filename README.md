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

## Verified resurrections

### PyPore — dead since January 2015

[`jmschrei/PyPore`](https://github.com/jmschrei/PyPore), a nanopore signal-analysis
toolkit, last touched in January 2015. Genuinely decayed: Python 2 only (two
modules don't even parse under Python 3), Cython extensions whose pre-generated
`.c` files were never committed, `distutils`-based `setup.py`, and no test data
in the repo at all.

| | |
|---|---|
| Verdict | **PASS** in 22 turns |
| Image | `sanjeevini/pypore:resurrected` (474 MB) |
| Proof | 12/12 injected blockade events detected, every event segmented, 22,918-byte JSON written |
| Reproduced | `docker run --rm sanjeevini/pypore:resurrected bash -lc 'cd /workspace/repo && python2 prove.py'` → exit 0 |

Jeeva found and fixed four pieces of real rot on its own, with no human input:

- **`missing_toolchain`** — Debian buster's apt mirrors return 404; repointed
  `sources.list` at `archive.debian.org`. Nothing in a 2015 repo could have known
  this, and it will recur on every EOL base image.
- **`missing_toolchain`** — no compiler in `python:2.7-slim`; installed gcc.
- **`build_failure`** — gcc present but `libc6-dev` headers (`limits.h`) missing.
- **`build_failure`** — pinned `cython==0.29.37` / `scipy==1.2.3` so the Python 2
  `.pyx` sources compile (Cython 3 rejects `izip` and implicit relative imports).

Because the repo ships no `.abf` data, the sanity check proves the scientific
core on a synthetic current trace: 12 injected blockade events, requiring ≥10
recovered, ≥1 state per event, and JSON that parses to a non-empty object. The
script exits non-zero if any clause fails.

### pycoQC — a still-runs case

`a-slide/pycoQC`, an archived ONT read-QC tool: **PASS** in 9 turns,
`sanjeevini/pycoqc:resurrected` (783 MB), 1.77 MB HTML report + valid JSON.
Honest caveat: `bugs_fixed` was empty — pycoQC still installs cleanly, so this
demonstrates verification rather than repair.

`cost_usd` reads `0.0` on both runs: token pricing is not yet wired into the
provenance record.

## Test results

```
360 passed, 6 deselected  ·  86% coverage (2786 statements)
ruff check + ruff format + mypy --strict: clean across src/ and tests/
```

The 6 deselected are `-m integration` tests that need a live Docker daemon and
network; `pytest -m "not integration"` is the CI gate. Coverage by subsystem:

| Subsystem | Coverage |
|---|---|
| `repair/escalation.py` (bounded self-escalation) | 100% |
| `contracts/output_type.py` (sanity-check quality gate) | 99% |
| `repair/knowledge.py` (cross-run learning) | 99% |
| `scouts/python_scout.py` | 91% |
| `repair/agent.py` | 88% |
| `repair/loop.py` | 87% |
| **Total** | **86%** |

Uncovered lines are overwhelmingly Docker-daemon and optional-dependency paths
marked `# pragma: no cover`, plus CLI wiring exercised by the integration tests.

A large share of these tests are regressions pinned from real resurrection runs
— agent amnesia, no-op command loops, mis-typed sanity checks, and unrecorded
environment repairs were all found by watching Jeeva work, then fixed and locked
down. See the two sections below.

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

Crucially, a repair is **not only a source edit**. Repointing a dead apt mirror,
installing a missing compiler, or pinning a dependency to its commit era are all
fixes of real decay, and they are the most reusable things a run discovers. The
PyPore resurrection produced four such lessons — the first real entries in the
store.

Lessons are injected as **hints, not instructions** — the prompt tells the agent
to verify before trusting one, and PASS is still decided only by a real exit
code, so a wrong lesson cannot poison a later run.

## Within-run working memory

Cross-run learning has a within-run twin. The agent sees only the previous
command's output, so anything it read two turns ago is gone. Left unaddressed
this produces a specific, ugly failure: on the first PyPore attempt Jeeva spent
38 turns re-reading the same two files, probing past end-of-file and grepping for
a class that does not exist, with no way to know it had already looked.

Three mechanisms fix it:

- **Notes** — the agent writes durable findings each turn and they are replayed
  in every later prompt. It decides what matters; the loop just keeps them
  (bounded, deduplicated, oldest dropped first).
- **No-op suppression** — re-running a pure inspection command cannot reveal
  anything new, so the loop refuses and says so. Build and test commands are
  never suppressed: re-running those after a fix is the point.
- **Stall detection** — a run of inspect-only turns adds an explicit instruction
  to stop reading and act.

After these, the same resurrection completed in 22 turns.

## The sanity-check quality gate

A resurrection is only as meaningful as its check, and a structural check is only
correct if it names the type the tool actually emits. Checking a FASTQ-emitting
read filter with `samtools quickcheck` can pass for the wrong reason or fail for
no reason.

Output-type inference is therefore evidence-weighted rather than
first-mention-wins: each format mention is scored by the nearest directional
marker in its own sentence, so *"takes a fastq as input and writes a filtered
fastq"* scores its two mentions in opposite directions. When no type clears the
runner-up by a margin, the scout declines to guess and emits a type-agnostic
check — a weaker-but-true check beats a specific-but-wrong one.

After a PASS, the loop probes the sandbox for files matching the claimed type and
records the result. This **never overturns a verdict** — a real exit code
outranks a filesystem heuristic, since a tool may stream to stdout or write
outside the working directory — but an unsupported claim is flagged loudly in
`PROVENANCE.json`, `REPRODUCE.md`, and on the CLI.

## Bounded self-escalation

Some rot is not repairable from inside the container the run was given. Python 2
sources on a `python:3` image cannot reach a PASS no matter how many turns the
agent spends patching — the interpreter is wrong. Until now the run just died
there: the one case where a human watching would have said "try 2.7".

Jeeva now says it to itself. On a failure it reads its own error signatures and,
if a rule matches, retries on a better base image (`--escalate N`, default 1;
`0` disables). Three rules, each firing only on text the run actually printed:

| Evidence | Retarget |
|---|---|
| Python 2 markers (`Missing parentheses in call to 'print'`, py2-only stdlib imports, `except E, e:`) | `python:3.x-slim` → `python:2.7-slim` |
| No C toolchain (`gcc: command not found`, `fatal error: Python.h`) | drop `-slim` for the full image, which ships gcc |
| musl wheel failures on Alpine | leave Alpine for the glibc image |

The discipline is that there is **no blind fallback**. A run that failed for a
reason no image can fix stays failed, because that is the honest verdict. An
image is never retried; a run that died because the API was unreachable never
escalates, since it learned nothing about the image; and the extra attempts are
capped, so total work is bounded by `--turns × (1 + N)`.

Every attempt is recorded in `PROVENANCE.json` — which images were ruled out,
by which rule, on what evidence — so an escalated PASS is auditable rather than
a lucky retry.

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
