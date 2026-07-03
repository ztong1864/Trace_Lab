FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r requirements.txt

COPY . .

ENV PYTHONPATH=/app/third_party/atlas/src:/app/third_party/olympus/src

EXPOSE 8788

CMD ["python", "experiments/run_lab_api.py", "--projects-root", "runs/lab_projects", "--host", "0.0.0.0", "--port", "8788"]
