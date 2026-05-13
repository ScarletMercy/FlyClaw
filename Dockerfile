# Build stage
FROM python:3.12-slim AS builder

WORKDIR /app

COPY pyproject.toml uv.lock* ./
RUN pip install uv && \
    uv sync --no-dev --frozen

# Runtime stage
FROM python:3.12-slim

WORKDIR /app

RUN groupadd -r myclaw && useradd -r -g myclaw myclaw

COPY --from=builder /app/.venv /app/.venv
COPY src/ src/
COPY config.example.yaml* ./

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

USER myclaw

EXPOSE 18080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:18080/healthz')"]

CMD ["python", "-m", "src.main"]
