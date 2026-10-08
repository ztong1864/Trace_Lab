FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
# A writable matplotlib cache when the container runs with a non-root `user:` (see docker-compose.yml).
ENV MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

# Exact versions of the tested environment (requirements.lock.txt, what run_local.ps1 installs), not the
# ranges of requirements.txt. torch comes from PyTorch's CPU index: the Linux build on PyPI pulls in
# several GB of CUDA libraries that a CPU server never uses.
# build-essential is only needed while installing (packages without a ready-made wheel) and is removed in
# the same layer; libgomp1 (OpenMP runtime) and poppler-utils (pdftotext, used by evidence-prepare) stay.
COPY requirements.lock.txt .
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 poppler-utils \
    && python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.lock.txt \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . .

ENV PYTHONPATH=/app/third_party/atlas/src:/app/third_party/olympus/src

EXPOSE 8788

CMD ["python", "experiments/run_lab_api.py", "--projects-root", "runs/lab_projects", "--host", "0.0.0.0", "--port", "8788"]
