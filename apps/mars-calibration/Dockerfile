FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04
ENV PYTHONUNBUFFERED=1 JAX_ENABLE_X64=true JAX_PLATFORMS=cuda \
    XLA_FLAGS="" XLA_PYTHON_CLIENT_PREALLOCATE=false \
    JAX_COMPILATION_CACHE_DIR=/results/jax-cache MPLCONFIGDIR=/tmp/matplotlib \
    VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
ENV DEBIAN_FRONTEND=noninteractive

# Keep OS tooling in the base layer.  The CUDA runtime and cuDNN are supplied
# by the image; Nautilus supplies the NVIDIA driver through the GPU runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       python3.12 python3-pip python3.12-venv ca-certificates \
       build-essential git \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/pip3 /usr/local/bin/pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN python3.12 -m venv /opt/venv
# Install the current framework as a dependency, including uncommitted physics fixes.
COPY package /opt/framework
COPY apps/mars-calibration /opt/application
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install /opt/framework /opt/application && python -m pip check \
    && pip freeze > /opt/environment.txt \
    && groupadd --gid 10001 calibration \
    && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin calibration \
    && chown -R 10001:10001 /opt/application
WORKDIR /opt/application
USER 10001:10001
ENTRYPOINT ["mars-calibration"]
CMD ["--config", "/opt/application/experiment.json", "--inputs", "/inputs", "--output", "/results/calibration"]
