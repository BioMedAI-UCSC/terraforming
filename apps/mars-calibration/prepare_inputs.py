"""Freeze the existing MY32 Ls45 demonstration inputs without new downloads."""
import argparse
import json
from pathlib import Path
import shutil
from mars_calibration.launch import sha256

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
sources = {
    "restart.npz": "outputs/branch_physical_diurnal_lw025_t21_l12_dt300/convection/checkpoints/restart_000477393.npz",
    "target.nc": "data/arco_macda/macda_my32_ls045_10sol_daily_t21_l12.nc",
    "surface.nc": "data/tes/mgs_tes_surface_1deg.nc",
    "dust.nc": "data/ames/fv3betaout1/ames_surface_reference.nc",
    "mola.img": "data/mola/meg004/megt90n000cb.img",
}
for source in sources.values():
    if not (args.repo / source).is_file():
        raise FileNotFoundError(args.repo / source)
args.output.mkdir(parents=True, exist_ok=False)
manifest = {"description": "MY32 Ls45 single-window calibration; daily-mean target index 5",
            "files": {}}
for name, source in sources.items():
    destination = args.output / name
    shutil.copyfile(args.repo / source, destination)
    manifest["files"][name] = {"source": source, "sha256": sha256(destination)}
(args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(args.output)
