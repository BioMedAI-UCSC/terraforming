"""Thread-safe console/file events and liveness during blocking JAX or I/O work."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import threading
import time

LOGGER = logging.getLogger("neural_pbl")
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False
if not LOGGER.handlers:
    LOGGER.addHandler(logging.StreamHandler(sys.stdout))


def configure(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for handler in list(LOGGER.handlers):
        if isinstance(handler, logging.FileHandler):
            LOGGER.removeHandler(handler)
            handler.close()
    LOGGER.addHandler(logging.FileHandler(output / "run.log"))


def event(name, **fields):
    LOGGER.info(json.dumps(dict(timestamp=datetime.now(timezone.utc).isoformat(),
                                event=name, **fields), default=str))


@contextmanager
def phase(name, *, heartbeat_seconds=30., **fields):
    """Elapsed-time heartbeat is liveness, not an estimate of percent complete."""
    started = time.monotonic()
    stopped = threading.Event()
    event(name + "_start", **fields)

    def heartbeat():
        while not stopped.wait(heartbeat_seconds):
            event(name + "_running", elapsed_seconds=round(time.monotonic() - started, 3), **fields)

    worker = threading.Thread(target=heartbeat, daemon=True)
    worker.start()
    try:
        yield
    except BaseException as exc:
        event(name + "_failed", error=str(exc), error_type=type(exc).__name__, **fields)
        raise
    else:
        event(name + "_complete", elapsed_seconds=round(time.monotonic() - started, 3), **fields)
    finally:
        stopped.set()
        worker.join()
