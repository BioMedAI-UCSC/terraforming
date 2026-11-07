import dataclasses
import math

import numpy as np
import pytest

pytest.importorskip('dinosaur')
import jax
import jax.numpy as jnp
from src.interventions.aerosols import Injection,aluminium_rod,iron_rod
from src.interventions.aerosols.data import rod_table
from src.interventions.aerosols.transport import (
    implicit_settle,remove_fraction,remap_carrier,positive_column_burden,
    injection_pattern,integrated_source,terminal_velocity)
from src.interventions.aerosols.microphysics import (
    constant_kernel_coagulation,charge_factor,settling_collision_kernel,re_loft)
from src.framework.gcm.coordinates import coordinate_system

jax.config.update('jax_enable_x64',True)


def test_primary_units_and_independent_column_mass_cross_check():
    p=aluminium_rod();table=rod_table(p)
    assert p.mass_kg==pytest.approx(2700*9e-6*(.16e-6)**2)
    # Supplement Table 2: one 0.67-um optical depth requires 212.1 mg/m2 Al.
    extinction=np.interp(.67e-6,table.wavelength_m,table.absorption_m2+table.scattering_m2)
    assert 1/(extinction/p.mass_kg)==pytest.approx(212.1e-6,rel=.015)
    assert Injection(0,10,2,'L/s').mass_between(0,10,p)==pytest.approx(54.)
    assert rod_table(iron_rod()).wavelength_m.size==75
    with pytest.raises(ValueError,match='no source-backed'):
        rod_table(dataclasses.replace(p,length_m=8e-6))
    assert p.max_mass_loading is None
    with pytest.raises(ValueError,match='mobility law'):
        terminal_velocity(dataclasses.replace(p,shape='disk'),210.,610.,3.72)


def test_source_normalization_boundaries_and_nonuniform_mask():
    coords=coordinate_system('T21',4)
    p=aluminium_rod();source=Injection(5,17,3,latitude_min_deg=0,latitude_max_deg=90,sigma_min=.5)
    pattern=injection_pattern(source,coords)
    weights=np.broadcast_to(np.asarray(coords.horizontal.quadrature_weights),coords.horizontal.nodal_shape).copy()
    weights/=weights.sum()
    total=0.
    for t,dt in [(0,10),(10,10),(20,5)]:
        mass,dose=integrated_source(t,dt,[source],[pattern],p,100.)
        assert float(jnp.sum(mass*weights)*100)==pytest.approx(float(dose),abs=1e-12)
        total+=float(dose)
    assert total==36.


def test_implicit_removal_conserves_at_arbitrarily_large_dt():
    p=aluminium_rod()
    mass=jnp.array([1e-5,2e-5,4e-5])[:,None,None]
    gas=jnp.ones_like(mass)*10
    result=jax.jit(lambda m:implicit_settle(m,m/p.mass_kg,gas,jnp.ones_like(m)*210,
                      jnp.ones_like(m)*610,1e8,p,3.72076))(mass)
    remaining,number,deposited,dn=result
    assert np.all(remaining>=0)
    np.testing.assert_allclose(remaining.sum()+deposited.sum(),mass.sum(),rtol=1e-13)
    np.testing.assert_allclose(number.sum()+dn.sum(),(mass/p.mass_kg).sum(),rtol=1e-13)
    final,_,removed,_=remove_fraction(remaining,number,1.,1e8)
    np.testing.assert_allclose(final.sum()+removed.sum()+deposited.sum(),mass.sum(),rtol=1e-13)


def test_analytic_single_layer_settling_and_refinement():
    p=aluminium_rod();t=jnp.ones((1,1,1))*210;pressure=t*0+610
    gas=t*0+10.;m=t*0+1e-5;v=terminal_velocity(p,t,pressure,3.72076)
    rate=float((v/(gas/(pressure/(188.92*t)))).squeeze())
    dt=1/rate
    result=implicit_settle(m,m/p.mass_kg,gas,t,pressure,dt,p,3.72076)
    np.testing.assert_allclose(result[0],m/2,rtol=1e-13)
    errors=[]
    for n in (4,16,64):
        current=m
        for _ in range(n):
            current=implicit_settle(current,current/p.mass_kg,gas,t,pressure,dt/n,p,3.72076)[0]
        errors.append(abs(float((current/m).squeeze())-math.exp(-1)))
    assert errors[2]<errors[1]<errors[0]


def test_carrier_remap_and_negative_repair_do_not_create_material():
    q=jnp.array([1.,2.,3.])[:,None,None]
    gas=jnp.array([10.,20.,30.])[:,None,None]
    np.testing.assert_allclose(remap_carrier(q,gas,gas*.9)*gas*.9,q*gas)
    original=jnp.array([-1.,2.,3.])[:,None,None]
    repaired,correction=positive_column_burden(original)
    assert np.all(repaired>=0) and float(correction.squeeze())>0
    assert float(repaired.sum())==pytest.approx(float(original.sum()))
    assert np.isnan(positive_column_burden(original*-1)[0]).all()


def test_constant_kernel_factor_of_two_charge_limits_and_conservative_reloft():
    mass=jnp.asarray(1.);number=jnp.asarray(1e12);k=1e-15;height=100.
    half_time=2*height/(k*number)
    m,n,events=constant_kernel_coagulation(mass,number,k,height,half_time)
    assert float(m)==1. and float(n)==pytest.approx(5e11)
    assert float(events)==pytest.approx(5e11)
    assert float(charge_factor(0.,0.,1e-6,1e-6,210))==1.
    assert float(charge_factor(10.,10.,1e-6,1e-6,210))<1
    assert float(charge_factor(10.,-10.,1e-6,1e-6,210))>1
    assert np.isfinite(jax.grad(lambda z:charge_factor(z,z,1e-6,1e-6,210))(0.))
    assert float(settling_collision_kernel(1e-6,1e-6,.1,.1,1.))==0.
    dm,dn,lm,ln=re_loft(mass,number,1.,.1,1.,100.)
    assert float(dm+lm)==1. and float(dn+ln)==float(number)
