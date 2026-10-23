FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev

COPY . .

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8086

CMD ["python", "main.py"]
