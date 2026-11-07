FROM ghcr.io/astral-sh/uv:0.10.9 AS uv
FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates tar \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv
COPY bootstrap.sh run.py /opt/nautalis/
ENV HOME=/work/home UV_CACHE_DIR=/work/uv-cache UV_LINK_MODE=copy \
    JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS="" \
    JAX_COMPILATION_CACHE_DIR=/work/jax-cache PYTHONUNBUFFERED=1
USER 10001:10001
ENTRYPOINT ["/bin/sh", "/opt/nautalis/bootstrap.sh"]
