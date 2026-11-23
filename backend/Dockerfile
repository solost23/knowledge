FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app/backend

COPY backend/pyproject.toml backend/uv.lock backend/.python-version ./
# Keep downloaded wheels across dependency-layer rebuilds without adding them to the image.
RUN --mount=type=cache,id=knowledge-uv,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

COPY backend ./

ENV PATH="/app/backend/.venv/bin:$PATH"

EXPOSE 8086

CMD ["python", "-m", "app.main"]
