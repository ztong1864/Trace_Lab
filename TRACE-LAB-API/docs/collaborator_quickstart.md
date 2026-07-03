# Collaborator Quickstart

This package is the collaborator pilot handoff for TRACE Lab.  It is slightly
richer than the clean public release: it includes the demo lab project, the
current descriptor CSVs, local literature PDFs, project background slides, and
a manuscript draft for context.

## 1. Install

Use Python 3.10.  The recommended path is:

```bash
bash scripts/create_venv.sh
source .venv/bin/activate
```

If `python3.10` is not available, install Python 3.10 first or use conda:

```bash
conda create -n trace-lab python=3.10 pip -y
conda activate trace-lab
bash scripts/install_release_deps.sh
bash scripts/check_release_env.sh
```

## 2. Configure API access

Create a local `.env` file:

```bash
cp .env.example .env
```

Fill in the API key field in `.env`.  If an OpenAI-compatible proxy is used,
also fill in the optional base URL field.

Keep `.env` local.  Do not commit it to GitHub.

## 3. Start TRACE Lab

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

Load:

```text
oxidative_esterification_demo
```

Then inspect the design space, click `Generate Batch`, export or copy the
recommended batch, run the real experiments, and submit measured results.

## 4. What Is Included

```text
collaborator_materials/design_inputs/
  additive_desc_datadf.csv
  solvent_desc_datadf.csv
  tempo_desc_datadf.csv

collaborator_materials/literature/Homogeneous metal-catalyzed/
  local PDFs used as pilot literature context

collaborator_materials/background/
  oxidative_esterification0526.pptx
  demo-result.pdf

docs/paper contains TRACE_manuscript_draft.pdf
  manuscript draft for method/context reading
```

The old standalone frontend demo, local environment files, bundled runtimes,
executables, and compressed archives are intentionally not included.  TRACE Lab
has its own Python API and Web UI, so those files would make the handoff larger
and easier to misconfigure.

## 5. Debug Modes

- Use `controller=bo_only` in the UI if API access is not configured.
- Use `Reset Run State` only for debugging; it clears generated
  recommendations, traces, and observations while keeping the project setup.
- If local API checks are unexpectedly routed through a proxy, set
  `NO_PROXY=127.0.0.1,localhost` or temporarily unset proxy variables before
  testing the local server.
- Run tests with:

```bash
bash scripts/run_lab_tests.sh
```

