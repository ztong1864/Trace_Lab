# TRACE

TRACE is a traceable agentic co-optimizer for scientific optimization.  This
public release focuses on **TRACE Lab**, a real-lab ask/tell workflow for
batch experimental recommendation.

TRACE Lab keeps the optimizer as the proposal backbone and wraps it with a
bounded agentic decision layer for batch composition, scoped evidence use,
recommendation rationales, and decision traces.

## Quick Start

Create the environment.  The fastest local path is a Python 3.10 virtual
environment:

```bash
bash scripts/create_venv.sh
source .venv/bin/activate
```

Alternatively, use conda:

```bash
conda create -n trace-lab python=3.10 pip -y
conda activate trace-lab
bash scripts/install_release_deps.sh
bash scripts/check_release_env.sh
```

Set your API key if you want agentic controller mode:

```bash
cp .env.example .env
# edit .env and set OPENAI_API_KEY
```

Start the local Web UI:

```bash
bash scripts/run_lab_api.sh \
  --projects-root runs/lab_projects \
  --host 127.0.0.1 \
  --port 8788
```

Open:

```text
http://127.0.0.1:8788
```

Load the demo project:

```text
oxidative_esterification_demo
```

## What This Release Supports

- Real-lab batch ask/tell workflow.
- Mixed design spaces with categorical, discrete numeric, continuous, and
  fixed/controlled conditions.
- Batch-level strategy and per-candidate role/rationale.
- Scoped evidence cards for literature or domain context.
- Local Web UI for project loading, recommendation generation, CSV export,
  result submission, and trace inspection.

## Documentation

- `docs/quickstart_lab.md`: step-by-step TRACE Lab usage.
- `docs/data_format.md`: project and CSV schemas.
- `docs/evidence_cards.md`: scoped evidence card format and policy.
- `docs/benchmark_reproduction.md`: placeholder for paper benchmark commands.

## Tests

```bash
bash scripts/check_release_env.sh
bash scripts/run_lab_tests.sh
```

## License

See `LICENSE`.

