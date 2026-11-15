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
# Keep the locked environment cached when only physics source changes.
COPY package/pyproject.toml /opt/dependencies/framework.toml
COPY apps/mars-calibration/pyproject.toml /opt/dependencies/application.toml
COPY apps/mars-calibration/constraints.txt /opt/dependencies/constraints.txt
RUN --mount=type=cache,target=/root/.cache/pip python - <<'PY'
import subprocess
import sys
import tomllib
from pathlib import Path
framework=tomllib.loads(Path('/opt/dependencies/framework.toml').read_text())['project']
application=tomllib.loads(Path('/opt/dependencies/application.toml').read_text())['project']
requirements=framework['dependencies']+framework['optional-dependencies']['gcm3d']
requirements += [value for value in application['dependencies'] if not value.startswith('terraforming[')]
subprocess.check_call([sys.executable,'-m','pip','install','-c','/opt/dependencies/constraints.txt',*requirements])
PY
COPY package /opt/framework
COPY apps/mars-calibration /opt/application
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-deps /opt/framework /opt/application && python -m pip check \
    && pip freeze > /opt/environment.txt \
    && groupadd --gid 10001 calibration \
    && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin calibration \
    && chown -R 10001:10001 /opt/application
COPY scripts /opt/scripts
WORKDIR /opt/application
USER 10001:10001
ENTRYPOINT ["mars-calibration"]
CMD ["--config", "/opt/application/experiment.json", "--inputs", "/inputs", "--output", "/results/calibration"]
