"""Stage released 15-band CO2/H2O tables and the 800 K high-pressure option."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from stage_co2_radiation_tables import _record,GAUSS_WEIGHTS,SOLAR_WEIGHTS,WATER_MASS_FRACTIONS,ames_planck_intensities

IR_BOUNDS=np.array([10.,300.300,549.451,636.943,704.225,775.194,970.874,1282.051,2222.222])
TABLES={
    '15band':(13,33,'CO2H2O_V_2013_32','b310353edc48bcdeb66f981349ff3a21e7068cee',
        'CO2H2O_IR_2013_32','78f5d69db47ef799f9ea58e2e1cb03c2c2b785c1'),
    'extended':(16,17,'CO2H2O_V_15B_800K_v4','2659dc073452eb2a9176cd7a6272cdf7fd1c89a9',
        'CO2H2O_IR_15B_800K_v4','ba2f16b20fd9d0c2273c7676b553d512fc8c6f38'),
}
GAUSS_32=np.array([.012897418221,.029570423871,.045200293049,.059198761346,
    .071058094688,.080349346713,.086736622146,.089989039966,
    .089989039966,.086736622146,.080349346713,.071058094688,
    .059198761346,.045200293049,.029570423871,.012897418221,
    .00067881148529,.0015563380985,.0023789627921,.0031157242814,
    .0037398997204,.0042289129849,.0045650853761,.0047362652614,
    .0047362652614,.0045650853761,.0042289129849,.0037398997204,
    .0031157242814,.0023789627921,.0015563380985,.00067881148529])


def decode(path,expected,temperatures,channels,bands):
    raw=path.read_bytes()
    blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    if blob!=expected: raise ValueError(f'extended gas table differs from pinned source: {path.name}')
    shape=(temperatures,25,10,bands,channels)
    with path.open('rb') as handle:
        table=np.frombuffer(_record(handle),dtype='<f8')
        clear=np.frombuffer(_record(handle),dtype='<f8')
        if handle.read(1) or table.size!=np.prod(shape) or clear.size!=bands:
            raise ValueError(f'extended gas table dimensions are invalid: {path.name}')
    table=table.reshape(shape,order='F')[...,:-1]
    # initinterp's comparison sends NaNs, zeros and tiny entries to -200.
    finite_count=int(np.sum(~np.isfinite(table)))
    return np.log10(np.where(table>1e-200,table,1e-200)),clear,dict(git_blob=blob,
        sha256=hashlib.sha256(raw).hexdigest(),nonfinite_source_coefficients=finite_count)


def stage(data_directory,output_directory):
    # These truncated 3.162 values are literal setrad pressure coordinates.
    pressure=np.array([value*10.**exponent for exponent in range(-8,4) for value in (1.,3.162)]+[1e4])
    planck_t,planck=ames_planck_intensities(IR_BOUNDS)
    outputs=[]
    for mode,(nt,ng,sw_name,sw_blob,ir_name,ir_blob) in TABLES.items():
        sw,swclear,swprov=decode(Path(data_directory)/sw_name,sw_blob,nt,ng,7)
        ir,irclear,irprov=decode(Path(data_directory)/ir_name,ir_blob,nt,ng,8)
        arrays=dict(log10_k_sw_mixtures_ames=sw,log10_k_ir_mixtures_ames=ir,
            log10_k_sw_ames=sw[:,:,0],log10_k_ir_ames=ir[:,:,0],
            clear_fraction_sw=swclear,clear_fraction_ir=irclear,
            temperature_k=np.arange(50.,351.,25.) if mode=='15band' else np.arange(50.,801.,50.),
            pressure_mbar_ames=pressure,water_mass_fractions=WATER_MASS_FRACTIONS,
            gauss_weights=GAUSS_32 if ng==33 else GAUSS_WEIGHTS,solar_weights=SOLAR_WEIGHTS,
            ames_planck_temperature_k=planck_t,ames_planck_intensity_ir=planck,ir_bounds_cm1=IR_BOUNDS,
            source_provenance_json=json.dumps({sw_name:swprov,ir_name:irprov},sort_keys=True))
        output=Path(output_directory)/f'co2_{mode}_tables.npz'
        output.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(output,**arrays);outputs.append(output)
    return outputs


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-data',type=Path,required=True)
    parser.add_argument('--output-directory',type=Path,default=Path(__file__).parents[1]/'package/src/framework/physics')
    args=parser.parse_args()
    for path in stage(args.reference_data,args.output_directory): print(f'staged {path}')
