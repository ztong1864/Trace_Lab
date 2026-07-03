# TRACE Lab Quickstart

## 1. Install

```bash
bash scripts/create_venv.sh
source .venv/bin/activate
```

If you prefer conda:

```bash
conda create -n trace-lab python=3.10 pip -y
conda activate trace-lab
bash scripts/install_release_deps.sh
bash scripts/check_release_env.sh
```

## 2. Configure LLM access

```bash
cp .env.example .env
```

Edit `.env` and set `OPENAI_API_KEY`.  For debugging without the agentic
controller, use `controller_mode=bo_only` from the UI or CLI.

## 3. Start the Web UI

```bash
bash scripts/run_lab_api.sh \
  --projects-root runs/lab_projects \
  --host 127.0.0.1 \
  --port 8788
```

Open `http://127.0.0.1:8788` and load:

```text
oxidative_esterification_demo
```

## 4. Typical workflow

1. Load a project.
2. Inspect design space.
3. Click `Generate Batch`.
4. Export recommendations as CSV.
5. Run experiments in the lab.
6. Enter completed, failed, or skipped results.
7. Generate the next batch.

TRACE Lab never reads hidden true outcomes.  Results enter the system only
through the ask/tell workflow.

