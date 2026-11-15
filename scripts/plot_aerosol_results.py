"""Plot a saved minimal aerosol result bundle; does not run a simulation."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir',type=Path)
    args=parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import xarray as xr
    with xr.open_dataset(args.output_dir/'results.nc',engine='h5netcdf') as dataset:
        fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        difference='surface_temperature_active_minus_off' in dataset
        for ax,name,label in (
            (axes[0],'surface_temperature_active_minus_off' if difference else 'surface_temperature',
             'Final active − control temperature (K)' if difference else 'Final surface temperature (K)'),
            (axes[1],'aerosol_burden','Suspended aerosol (kg/m²)')):
            image=ax.pcolormesh(dataset.lon,dataset.lat,dataset[name],shading='auto')
            fig.colorbar(image,ax=ax,label=label)
            ax.set(xlabel='Longitude (°E)',ylabel='Latitude (°N)')
        fig.suptitle('Transient engineered-aerosol pilot — climate baseline unverified')
        fig.savefig(args.output_dir/'maps.png',dpi=150)
        plt.close(fig)
    rows=[json.loads(line) for line in (args.output_dir/'history.ndjson').read_text().splitlines() if line.strip()]
    paired=[row for row in rows if 'cases' in row]
    if paired:
        fig,ax=plt.subplots(figsize=(6,4),layout='constrained')
        ax.plot([r['elapsed_seconds']/3600 for r in paired],
            [r['cases']['active']['surface_temperature_k']-r['cases']['off']['surface_temperature_k'] for r in paired])
        ax.set(xlabel='Elapsed simulation hours',ylabel='Area-mean active − control temperature (K)',
            title='Transient response; no long-term warming claim')
        fig.savefig(args.output_dir/'warming.png',dpi=150)
        plt.close(fig)


if __name__=='__main__': main()
