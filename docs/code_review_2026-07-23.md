# Sanjeevini / Jeeva — Code Review

**Date:** 2026-07-23 · **Scope:** `src/sanjeevini/` (~7,900 LOC), tests, packaging

## Overall

This is genuinely well-engineered code. The central design principle — *the model
never declares success; a command it marks as the sanity check must exit 0 in the
sandbox* — is enforced in tested, dependency-injected code, and the whole loop
(agent, sandbox, escalation, decay probe) is runnable with fakes and no Docker. Docstrings
are excellent, type hints are thorough, `subprocess` is always called with a fixed
argv (no `shell=True`), and the decay-check interpolated URL is guarded by a
character allowlist. Naming and module boundaries are clean. 357 test functions
across 21 files.

The findings below are mostly about *advertised-but-unenforced* behavior and one
isolation gap — not about the core logic, which is sound.

---

## High priority

### 1. `--budget-usd` is a no-op — an advertised hard cost cap that caps nothing
`--budget-usd` is exposed in the CLI (`cli.py:57`) and MCP server (`server.py:109,251`),
documented as *"Hard cost cap in USD."* It is never threaded into `RepairLoop`. The loop
accumulates `cost` but never compares it to a limit; `RepairLoop.__init__` has no budget
parameter at all. For an autonomous LLM loop that spends real money for up to
`turns × (1 + escalate)` iterations, an unenforced cap is worse than no cap — it implies a
safety rail that isn't there.
**Fix:** thread `budget_usd` into `RepairLoop.run()`; break with a `BUDGET` verdict when
cumulative `cost` crosses it, and ideally refuse the next turn if the *projected* cost would.

### 2. API-path cost is always $0.00 — cost governance is effectively absent
`_build_agent` constructs `LLMRepairAgent(spec, knowledge=...)`, which defaults to
`AnthropicClient(model)` with `price_in/out = None`, so `_cost()` returns `0.0` every turn.
Provenance and `--cost` reporting under-report to zero on the direct-API backend. Only the
subscription backend reports real cost. Combined with #1, there is no working cost signal on
the default path.
**Fix:** default to current model prices (or read from a small price table / env), and log a
warning when prices are unset so a $0.00 cost is never mistaken for "free."

### 3. Sandbox runs with unrestricted network and default privileges
Every agent-chosen command and all resurrected research code runs via `docker exec` in a
container started with default capabilities, root user, full outbound network, and no
`--pids-limit` / read-only rootfs. `memory_gb`/`cpus` are supported by `DockerSandbox` but the
**resurrect path never sets them** (`attempt()` omits them), despite the module docstring
advertising "high-RAM/CPU limits." This is the primary trust boundary of the whole system:
arbitrary LLM-selected shell plus arbitrary untrusted code with network egress.
**Fix (defense in depth):** default `--pids-limit`, `--memory`/`--cpus` caps on the resurrect
attempt, `--security-opt=no-new-privileges`, drop unneeeded capabilities, and offer a
`--network none|restricted` option (many resurrections that need net only need PyPI/apt). At
minimum, document the trust model explicitly in the README.

### 4. Pipeline execution is advertised but unimplemented
`Pipeline._execute` always raises `"pipeline execution is available through jeeva mcp"`, but
the MCP `run_pipeline` tool simply shells out to `jeeva run`, which reaches that same raise.
So a non-`--dry-run` `jeeva run` and the MCP `run_pipeline` **can never execute a pipeline** —
yet the MCP tool's description promises *"Returns pipeline run status and output artifact
paths."*
**Fix:** implement execution, or clearly mark Compose as validate/dry-run-only in both the CLI
help and the MCP tool description so callers aren't misled.

### 5. Auto-emitted contracts are typeless (`ANY → ANY`), so Compose's type-checker is a no-op on Jeeva's own output
`RepairLoop._schema()` hardcodes one `ANY` input and one `ANY` output for every resurrection.
The rich `GenomicFileType` vocabulary and `ContractSchema.compatible_with()` — the headline
"type-check the pipeline before any container runs" feature — never see a real type on anything
the resurrection loop produces; `ANY` is a wildcard on the destination side, so every edge
passes. The machinery only bites on hand-curated registry entries.
**Fix:** derive at least the output type from the sanity check (you already have
`extensions_for_check`) and the input type from the entry command / detected example input, so
emitted contracts carry meaningful ports.

---

## Medium priority

### 6. No CI
`pyproject.toml` configures a solid ruff rule set, `mypy --strict`, and pytest (asyncio + an
`integration` marker), and there are 357 tests — but there is no `.github/workflows/`. For a
project whose entire thesis is "provably runs," not gating merges on its own suite is the
conspicuous gap. (The workflows under `_lazarus_ref/` are the vendored reference, not this
repo.)
**Fix:** add a workflow running `ruff check`, `mypy --strict src`, and `pytest -m 'not
integration'`, plus a separate Docker-enabled job for the integration tests.

### 7. Reproduction smoke test isn't hermetic
`_smoke_test` replays every successful state-changing command under `set -euo pipefail`. If
those include `apt-get`/`pip`/`git clone`, the emitted "proof" needs live network and mirrors
to re-pass — which partially undercuts the reproducibility guarantee. The container-anchor
guard is a nice touch, but consider recording whether the recipe is network-dependent and
noting it in `REPRODUCE.md`, and prefer the snapshotted image as the source of truth over the
replay.

### 8. `_container_anchor` guards only the first matched path
The regex picks the first `/(workspace|work|opt|srv)/<name>` occurrence; a reproduction that
touches two different absolute anchors only gets one guarded. Low-frequency, but worth a note
or a multi-match guard.

---

## Minor / polish

- **Inline-YAML detection by newline** (`_load_pipeline_yaml`): a single-line inline pipeline
  (`{name: x, steps: []}`) with no newline is treated as a *path* and fails with a confusing
  `FileNotFoundError`. Detect more robustly (try-parse, or an explicit flag).
- **PyPI cache filename** uses `package.replace("/", "_")` but not `..`; PyPI names can't
  contain `..`, but a defensive `^[A-Za-z0-9._-]+$` validation on the package arg would close
  the theoretical path-escape and reject junk early.
- **DRY:** `_utc_now_iso` / `_utc_today` and the stdout/stderr truncation helpers are
  reimplemented in a few modules; a tiny `sanjeevini/_time.py` / `_text.py` would consolidate.
- **SSE transport** is declared as a CLI choice (`jeeva mcp --host sse`) but `serve()` exits 1
  for it. Either drop the choice or implement it, so `--help` doesn't advertise a dead option.
- **`_MAX_REPLY_ATTEMPTS` retry** is largely dead weight on the Anthropic backend (forced
  `tool_choice` makes malformed JSON near-impossible); it mainly protects the subscription/text
  path. Fine as-is, just noting.

---

## Implementation status (updated 2026-07-23)

All items above were implemented in this pass. Suite grew from 364 → 386 tests;
`ruff check`, `ruff format --check`, and `mypy --strict` all pass.

- **#1 budget** — `RepairLoop` now takes `budget_usd`, stops with a `BUDGET`
  verdict before spending past the cap, and the cap is global across escalation
  attempts (each gets only what earlier ones left). A `BUDGET` outcome never
  escalates.
- **#2 cost** — `AnthropicClient` reads `JEEVA_PRICE_IN/OUT_PER_MTOK` from the
  environment and warns once when cost tracking is off, so `$0.00` is never
  mistaken for free.
- **#6 CI** — `.github/workflows/ci.yml` runs ruff, ruff-format, `mypy --strict`,
  and pytest across 3.10–3.12. Fixed the pre-existing strict-mypy failures this
  surfaced (env-get typing, an unused ignore, missing yaml stubs).
- **#3 sandbox** — `--pids-limit` (4096) and `--security-opt no-new-privileges`
  on by default; `--memory`/`--cpus` now wired on the resurrect path; network
  modes `open`/`restricted`/`none` with `restricted` the default.
- **#3a enforced egress (follow-up)** — `restricted` is now genuinely enforced,
  not advisory. Per-run it stands up a Docker `--internal` network (no route out)
  plus a filtering Squid proxy (`sanjeevini/sandbox/egress.py`) that is the
  sandbox's only path to the internet and allowlists by domain on HTTP and HTTPS
  `CONNECT` — no TLS interception, no injected CA. Code that ignores `HTTP_PROXY`
  and dials a raw IP gets nothing, because the route does not exist. Startup is
  **fail-closed**: if the gateway can't come up, the run raises rather than
  falling back to open networking. Residual domain-fronting gap is noted in the
  module docstring, with SNI-peek splicing as the documented next hardening step.
- **#5 types** — emitted contracts now stamp a real output-port type read back
  from the sanity check (`output_type_for_check`), so Compose edges into them are
  genuinely type-checked; input stays `ANY` (nothing reliably identifies it).
- **#4 Compose execution** — real step-by-step executor: resolves
  `${params/workdir/inputs/outputs/steps.*}`, threads files between steps through
  one shared bind-mounted working dir, stops at the first non-zero step. Injectable
  `StepExecutor` keeps it unit-tested without Docker.
- **Polish** — removed the dead `--host sse`/`--port` MCP flags, made inline-YAML
  detection existence-based, and hardened the PyPI cache filename against path
  traversal.

## Suggested order of work
1. Wire up `--budget-usd` + real cost accounting (#1, #2) — safety and correctness of the
   headline governance feature.
2. Add CI (#6) — cheap, high leverage, protects everything else.
3. Harden the sandbox defaults (#3).
4. Fix the Compose honesty gaps (#4, #5) — either implement or stop advertising.
