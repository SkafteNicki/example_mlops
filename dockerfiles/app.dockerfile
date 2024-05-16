FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN mkdir /app

WORKDIR /app

COPY requirements.txt /app/requirements.txt
COPY pyproject.toml /app/pyproject.toml
COPY src/ /app/src

RUN --mount=type=cache,target=/root/.cache/pip pip install .

EXPOSE $PORT
CMD exec uvicorn src.example_mlops.app:app --host 0.0.0.0 --port $PORT
