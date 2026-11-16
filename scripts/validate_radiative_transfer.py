"""Compare the composed JAX radiation path with original Ames routines."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
from src.framework.gcm._dinosaur import jax
from src.framework.physics.correlated_k import load_ames_co2_tables
from src.framework.physics.radiative_transfer import column_fluxes,rayleigh_per_pa
from validate_optical_depth import compile_reference as compile_optics
from validate_spectral_flux import compile_reference as compile_spectral
from validate_scattering_transfer import compile_reference as compile_grid
from validate_infrared_precision import compile_reference as compile_quad


def validate(source,compiler):
    jax.config.update('jax_enable_x64',True)
    data=load_ames_co2_tables()
    f=lambda x:np.asfortranarray(x,dtype=np.float64)
    cases=[]
    with tempfile.TemporaryDirectory(prefix='ames-radiative-driver-') as temporary:
        root=Path(temporary)
        for name in ('optics','optics-linear','spectral','grid','quad'):
            (root/name).mkdir()
        spectral,sp=compile_spectral(source,root/'spectral',compiler)
        grid_native,gp=compile_grid(source,root/'grid',compiler)
        quad,qp=compile_quad(source,root/'quad',compiler)
        ops=[]
        for linear in (False,True):
            optics,op=compile_optics(source,root/('optics-linear' if linear else 'optics'),compiler,
                linear_k_interpolation=linear)
            ops.append(op)
            optics.ames_init(f(data['log10_k_sw_ames']),f(data['log10_k_ir_ames']),
                f(data['pressure_mbar_ames']),f(data['temperature_k']),f(rayleigh_per_pa(3.72)*100.))
            optics.ames_mixtures(f(data['log10_k_sw_mixtures_ames']),f(data['log10_k_ir_mixtures_ames']))
            for n in (4,12,56):
                for top in (0.,.13):
                    half=f(np.geomspace(.13,613.,n+1));half[0]=top
                    if n==56:
                        from src.framework.physics.vertical_grid import pressure_interfaces
                        half=f(pressure_interfaces(n,705.,705.)).copy()
                        if top==0.: half[0]=0.
                        top=float(half[0])
                    for strat_temperature in (170.,200.):
                        full=f((half[1:]+half[:-1])*.5)
                        t=f(np.full(n,200.))
                        p,tl,pm,tm=[np.zeros(2*n+3) for _ in range(4)]
                        dp=np.zeros(n)
                        grid_native.ames_grid(n,half,full,t,200.,strat_temperature,p,tl,pm,tm,dp)
                        for loading in (0.,.05):
                            amount=f(np.diff(p)/100.*loading)
                            cross=[]
                            particles=[]
                            for bands in (7,5):
                                x=f(np.broadcast_to(np.linspace(.5,2.,bands),(2*n+2,bands)))
                                s=f(x*.4);g=f(np.full_like(x,.3))
                                cross.extend([x,s,g])
                                particles.append((amount[:,None]*x,amount[:,None]*s,amount[:,None]*s*g))
                            optical=[]
                            for bands in (7,5):
                                optical.extend([np.zeros((n+1,bands,17),order='F') for _ in range(3)])
                                optical.extend([np.zeros((2*n+3,bands,17),order='F'),np.zeros((bands,16),order='F')])
                            optics.optical_depth(n,f(p/100.),f(tm),f(pm/100.),amount,*cross,*optical)
                            solar=f(np.array([12.7,24.2,54.6,145.9,354.9,657.5,106.3])/1.524**2)
                            for mu,scale_surface in ((.6,False),(0.,False),(.6,True),(0.,True)):
                                up=np.zeros(n+1);down=np.zeros(n+1);topnet=np.zeros(1)
                                if mu>0.:
                                    spectral.ames_sw(n+1,mu,.25,*optical[:5],solar,
                                        f(data['clear_fraction_sw']),f(data['gauss_weights']),up,down,topnet)
                                visible_top=topnet[0]
                                irup=np.zeros(n+1);irdown=np.zeros(n+1)
                                spectral.ames_lw(n+1,.05,int(scale_surface),*optical[5:],f(p/100.),f(tl),
                                    f(data['ames_planck_intensity_ir'].T),f(np.diff(data['ir_bounds_cm1'])),
                                    f(data['clear_fraction_ir']),f(data['gauss_weights']),irup,irdown,topnet)
                                fp64_up,fp64_down=irup.copy(),irdown.copy()
                                quad.ames_lw(n+1,.05,int(scale_surface),*optical[5:],f(p/100.),f(tl),
                                    f(data['ames_planck_intensity_ir'].T),f(np.diff(data['ir_bounds_cm1'])),
                                    f(data['clear_fraction_ir']),f(data['gauss_weights']),irup,irdown,topnet)
                                net=up-down+irup-irdown
                                expected=(down,up,irup,irdown,np.diff(net),-net[-1],-net[0])
                                actual=column_fluxes(t,200.,half,full,mu,solar,.25,.95,3.72,particle_optics=particles,
                                    stratosphere_temperature=strat_temperature,scale_surface_flux=scale_surface,linear_k_interpolation=linear)
                                strat_ir=irup[0]-irdown[0]-topnet[0]
                                factor=2.2e4*(p[1]/100.)/(1.+2.2e4*(p[1]/100.))
                                strat_sw=(up[0]-down[0]-visible_top)*factor
                                errors=[]
                                interface_error=np.zeros(n+1)
                                roundoff=64.*np.finfo(np.float64).eps*max(1.,float(np.max(np.abs(net))))
                                for field,(value,ref) in enumerate(zip(actual[:7],expected,strict=True)):
                                    if field<4:
                                        np.testing.assert_allclose(value,ref,rtol=3e-10,atol=3e-9,
                                            err_msg=f'layers={n}, top={top}, loading={loading}, mu={mu}, field={field}')
                                        interface_error+=np.abs(np.asarray(value)-ref)
                                    else:
                                        # Propagate the checked interface errors through flux differences.
                                        bound=(interface_error[:-1]+interface_error[1:] if field==4 else
                                            interface_error[-1] if field==5 else interface_error[0])
                                        assert np.all(np.abs(np.asarray(value)-ref)<=bound+roundoff)
                                    errors.append(float(np.max(np.abs(np.asarray(value)-ref))))
                                sample_ir=actual.longwave_up_w_m2[0]-actual.longwave_down_w_m2[0]
                                sample_sw=(actual.shortwave_up_w_m2[0]-actual.shortwave_down_w_m2[0])*factor
                                boundary=sample_ir+sample_sw-actual.stratosphere_flux_convergence_w_m2
                                native_boundary=topnet[0]+visible_top*factor
                                np.testing.assert_allclose(boundary,native_boundary,rtol=3e-10,atol=3e-9)
                                strat_error=float(abs(actual.stratosphere_flux_convergence_w_m2-strat_ir-strat_sw))
                                # A small flux difference inherits the checked errors of both interfaces.
                                strat_bound=(abs(sample_ir-(irup[0]-irdown[0]))+
                                    abs(sample_sw-(up[0]-down[0])*factor)+abs(boundary-native_boundary)+roundoff)
                                assert strat_error<=strat_bound
                                cases.append(dict(layers=n,top_pressure_pa=top,particle_loading=loading,
                                    mu0=mu,scale_surface_flux=scale_surface,linear_k_interpolation=linear,stratosphere_temperature_k=strat_temperature,max_flux_errors_w_m2=errors,
                                    stratosphere_flux_error_w_m2=strat_error,
                                    native_fp64_ir_cancellation_error_w_m2=float(max(np.max(np.abs(fp64_up-irup)),
                                        np.max(np.abs(fp64_down-irdown))))))
    repo=Path(__file__).parents[1]
    paths=['scripts/validate_radiative_transfer.py','scripts/validate_optical_depth.py',
        'scripts/validate_spectral_flux.py','scripts/validate_scattering_transfer.py',
        'scripts/validate_infrared_precision.py']
    paths.extend(str(p.relative_to(repo)) for p in (repo/'package/src/framework/physics').glob('*.py'))
    paths.append('package/src/framework/physics/co2_12band_tables.npz')
    return dict(status='pass',reference=dict(grid=gp,optics=ops,spectral=sp,infrared_quadruple=qp),cases=cases,
        sha256={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in paths},
        limits='Composed 12-band radiation with both supported k-interpolation paths with prescribed particle cross sections. Original IR routines run at binary128 with unchanged interface-flux gates; FP64 cancellation error is recorded separately. Planck temperatures are at reference nodes; microphysics and global climate validation are separate')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ames-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    compiler=shutil.which('gfortran')
    if compiler is None: parser.error('gfortran is required')
    report=validate(args.ames_source,compiler)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],cases=len(report['cases']))))
