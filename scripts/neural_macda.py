"""Native-cadence, revision-pinned MACDA cache shared by staging and training."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import fsspec
import numpy as np
import xarray as xr

from neural_logging import event, phase

from stage_arco_macda import _to_dinosaur_grid, DEFAULT_VARIABLES

SPLITS = {"train": (24, 25, 26, 27, 29, 30, 31), "validation": (32, 33),
          "test": (34, 35), "storm": (28,)}
YEAR_BOUNDS = (0, 8023, 16046, 24069, 32092, 40320, 48139, 56162,
               64185, 71848, 80231, 88255, 96278)
UNITS = {"temp": "K", "uwind": "m s-1", "vwind": "m s-1", "psurf": "Pa",
         "tsurf": "K", "coldust": "1", "co2ice": "kg m-2"}


def windows(split, count=120):
    if count < 2:
        raise ValueError("chunk must contain at least two snapshots")
    return [(start, min(start + count, YEAR_BOUNDS[year - 23]))
            for year in SPLITS[split]
            for start in range(YEAR_BOUNDS[year - 24], YEAR_BOUNDS[year - 23] - 1, count - 1)]


def validate(ds, split, *, prepared=False):
    if ds.sizes.get("time", 0) < 2:
        raise ValueError("need at least two native snapshots")
    if not np.allclose(np.diff(ds.time), 1 / 12, rtol=0, atol=1e-9):
        raise ValueError("non-contiguous native MACDA time")
    if not np.isin(ds.MY_Ls, SPLITS[split]).all():
        raise ValueError("chunk crosses the requested split")
    for name, unit in UNITS.items():
        field = ds[name]
        dims = ("time", "lev", "lat", "lon") if name in ("temp", "uwind", "vwind") else ("time", "lat", "lon")
        if field.dims != dims or field.attrs.get("units") != unit:
            raise ValueError(f"invalid dimensions/units: {name}: {field.dims}, {field.attrs}")
        if not np.isfinite(field).all():
            raise ValueError(f"nonfinite {name}")
    for name in ("time", "Ls", "MY_Ls", "lev", "lat", "lon"):
        if not np.isfinite(ds[name]).all():
            raise ValueError(f"nonfinite coordinate {name}")
    if (ds.psurf <= 0).any() or (ds.temp <= 0).any() or (ds.tsurf <= 0).any():
        raise ValueError("nonpositive pressure/temperature")
    if prepared and any(ds.sizes[k] != v for k, v in {"lev": 12, "lat": 32, "lon": 64}.items()):
        raise ValueError("cache is not T21/L12")


@contextmanager
def locked(path):
    with open(path, "a") as stream:
        with phase("lock_wait", path=str(path)):
            fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def atomic_bytes(path, content):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class NativeCache:
    def __init__(self, root, revision):
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("revision must be a full immutable Hugging Face commit SHA")
        self.revision = revision
        self.root = Path(root) / revision / "native-t21-l12-v1"
        self.root.mkdir(parents=True, exist_ok=True)
        self.source = None

    def get(self, window, split):
        start, stop = window
        path = self.root / f"{start:06d}-{stop:06d}.nc"
        manifest = path.with_suffix(".json")
        event("chunk_requested", split=split, window=window, path=str(path))
        with locked(path.with_suffix(".lock")):
            if path.exists() and manifest.exists():
                event("cache_hit", path=str(path), bytes=path.stat().st_size)
                meta = json.loads(manifest.read_text())
                if meta != {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "revision": self.revision, "window": [start, stop]}:
                    raise ValueError(f"cache checksum/provenance mismatch: {path}")
            else:
                if self.source is None:
                    url = f"https://huggingface.co/datasets/ananyo01/ARCO-MACDA/resolve/{self.revision}/macda_combined.zarr"
                    with phase("dataset_metadata", revision=self.revision, url=url):
                        self.source = xr.open_zarr(fsspec.get_mapper(url), consolidated=True, decode_times=False)
                with phase("download", split=split, window=window, snapshots=stop-start, variables=DEFAULT_VARIABLES):
                    ds = self.source[[*DEFAULT_VARIABLES, "Ls", "MY_Ls"]].isel(time=slice(start, stop)).load()
                event("download_loaded", window=window, decoded_bytes=ds.nbytes)
                validate(ds, split)
                with phase("regrid", window=window, grid="T21/L12"):
                    ds = _to_dinosaur_grid(ds, "T21", 12).transpose("time", "lev", "lat", "lon", missing_dims="ignore")
                ds.time.attrs["long_name"] = "Native Martian sols since MY24 start"
                validate(ds, split, prepared=True)
                # Write in the same directory; readers only see a complete file.
                fd, temp = tempfile.mkstemp(dir=self.root, suffix=".nc")
                os.close(fd)
                try:
                    ds.to_netcdf(temp, engine="h5netcdf")
                    os.replace(temp, path)
                finally:
                    if os.path.exists(temp):
                        os.unlink(temp)
                atomic_bytes(manifest, json.dumps({"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                             "revision": self.revision, "window": [start, stop]}).encode())
            event("cache_verified", path=str(path), bytes=path.stat().st_size)
            with xr.open_dataset(path, decode_times=False) as opened:
                ds = opened.load()
            validate(ds, split, prepared=True)
            event("chunk_ready", split=split, window=window, snapshots=ds.sizes["time"], first_sol=float(ds.time[0]), last_sol=float(ds.time[-1]))
            return ds

    def iterate(self, selections, split):
        """Bounded one-chunk lookahead, overlapping I/O with device work."""
        iterator = iter(selections)
        with ThreadPoolExecutor(max_workers=1) as pool:
            first = next(iterator, None)
            pending = None if first is None else pool.submit(self.get, first, split)
            while pending is not None:
                ds = pending.result()
                following = next(iterator, None)
                if following is not None:
                    event("prefetch_queued", split=split, window=following)
                pending = None if following is None else pool.submit(self.get, following, split)
                yield ds
