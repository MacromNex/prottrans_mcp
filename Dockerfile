FROM python:3.10-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    git gcc g++ wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    -r requirements.txt
RUN pip install --no-cache-dir --prefix=/install --ignore-installed fastmcp
RUN pip install --no-cache-dir --prefix=/install tiktoken

FROM python:3.10-slim AS runtime

WORKDIR /app
COPY --from=builder /install /usr/local
COPY src/ ./src/
RUN chmod -R a+r /app/src/
COPY scripts/ ./scripts/
RUN chmod -R a+r /app/scripts/
RUN mkdir -p tmp/inputs tmp/outputs .cache /tmp/.cache && \
    chmod -R 1777 /app/tmp /tmp/.cache && \
    chmod -R 777 /app/.cache

ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV HOME=/tmp
ENV XDG_CACHE_HOME=/tmp/.cache
ENV HF_HOME=/app/.cache
ENV TRANSFORMERS_CACHE=/app/.cache/transformers
ENV TORCH_HOME=/app/.cache/torch

CMD ["python", "src/server.py"]
