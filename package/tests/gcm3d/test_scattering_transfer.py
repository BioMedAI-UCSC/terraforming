import numpy as np
import pytest

pytest.importorskip("dinosaur")
import jax
import jax.numpy as jnp

from src.framework.physics.scattering_transfer import infrared_fluxes,visible_fluxes

jax.config.update("jax_enable_x64",True)


def test_absorption_only_infrared_matches_exact_transfer():
    depth=jnp.array([.1,.2,.3])
    offset=depth*.5;offset=offset.at[-1].set(depth[-1])
    up,down,top=infrared_fluxes(depth,jnp.zeros(3),jnp.zeros(3),
        jnp.full(4,.2),offset,0.,0.,.3)
    absolute=jnp.cumsum(depth)-depth+offset
    expected_up=.2+.1*jnp.exp(-2.*(jnp.sum(depth)-absolute))
    expected_down=.2*(1.-jnp.exp(-2.*absolute))
    np.testing.assert_allclose(up,jnp.pi*expected_up,rtol=2e-13,atol=2e-13)
    np.testing.assert_allclose(down,jnp.pi*expected_down,rtol=2e-13,atol=2e-13)
    np.testing.assert_allclose(top,jnp.pi*(.2+.1*jnp.exp(-2.*jnp.sum(depth))),rtol=2e-13)


def test_thin_linear_source_and_gradient_match_independent_formal_integral():
    depth=1.e-9;sample=.4*depth
    nodes,weights=np.polynomial.legendre.leggauss(8)
    def integrate(a,b,fn):
        points=.5*(b-a)*nodes+.5*(b+a)
        return .5*(b-a)*np.sum(weights*fn(points))
    def source(s): return 4.+2.*s/depth
    expected_up=11.*np.exp(-2.*(depth-sample))+integrate(sample,depth,
        lambda s:2.*source(s)*np.exp(-2.*(s-sample)))
    expected_down=3.*np.exp(-2.*sample)+integrate(0.,sample,
        lambda s:2.*source(s)*np.exp(-2.*(sample-s)))
    def evaluate(top_planck):
        return infrared_fluxes(jnp.array([depth]),jnp.zeros(1),jnp.zeros(1),
            jnp.array([top_planck,6.]),jnp.array([sample]),0.,3.,11.)
    actual=jax.jit(evaluate)(4.)
    np.testing.assert_allclose(actual[0][0],np.pi*expected_up,rtol=2e-15)
    np.testing.assert_allclose(actual[1][0],np.pi*expected_down,rtol=2e-15)
    derivative=jax.jit(jax.grad(lambda b:evaluate(b)[0][0]))(4.)
    expected_derivative=np.pi*integrate(sample,depth,
        lambda s:2.*(1.-s/depth)*np.exp(-2.*(s-sample)))
    np.testing.assert_allclose(derivative,expected_derivative,rtol=2e-14,atol=1e-24)


@pytest.mark.parametrize("w",[0.,.5,.999])
def test_isothermal_infrared_equilibrium(w):
    depth=jnp.array([.1,.2,.3])
    offset=depth*.5;offset=offset.at[-1].set(depth[-1])
    up,down,top=infrared_fluxes(depth,jnp.full(3,w),jnp.full(3,.6),
        jnp.full(4,.2),offset,.1,.2,.18)
    np.testing.assert_allclose(up,.2*jnp.pi,rtol=2e-12,atol=2e-12)
    np.testing.assert_allclose(down,.2*jnp.pi,rtol=2e-12,atol=2e-12)
    np.testing.assert_allclose(top,0.,atol=2e-12)


def test_visible_zero_scattering_has_finite_absorption_limit():
    depth=jnp.array([.1,.2,.3])
    cumulative=jnp.array([0.,0.,.05,.1,.2,.3,.6])
    up,down,topup,topdown,_,_=visible_fluxes(depth,jnp.zeros(3),jnp.zeros(3),
        cumulative,.6,100.,0.,0.,0.)
    absolute=cumulative[2::2]
    np.testing.assert_allclose(up,0.,atol=1e-13)
    np.testing.assert_allclose(down,60.*jnp.exp(-absolute/.6),rtol=2e-13)
    np.testing.assert_allclose(topup,0.,atol=1e-13)
    np.testing.assert_allclose(topdown,60.,rtol=2e-13)


@pytest.mark.parametrize("kind",["infrared","visible"])
def test_radiative_transfer_reverse_derivative(kind):
    base=jnp.array([.1,.2,.3])
    def objective(scale):
        depth=base*scale
        if kind=="infrared":
            offset=depth*.5;offset=offset.at[-1].set(depth[-1])
            up,down,top=infrared_fluxes(depth,jnp.full(3,.6),jnp.full(3,.4),
                jnp.array([.1,.2,.3,.4]),offset,.1,.001,.36)
            return jnp.sum(up)+jnp.sum(down)+top
        cumulative=jnp.array([0.,0.,.05,.1,.2,.3,.6])*scale
        outputs=visible_fluxes(depth,jnp.full(3,.6),jnp.full(3,.4),cumulative,.6,100.,.2,0.,5.)
        return sum(jnp.sum(x) for x in outputs)
    auto=jax.grad(objective)(1.)
    for eps in (1e-4,1e-5,1e-6):
        finite=(objective(1.+eps)-objective(1.-eps))/(2.*eps)
        np.testing.assert_allclose(auto,finite,rtol=2e-7,atol=1e-8)


@pytest.mark.parametrize('depth',[20.,100.,2500.])
@pytest.mark.parametrize('fraction',[.1,.5,.9])
def test_capped_thick_layer_sampling_uses_the_original_single_exponent_cap(depth,fraction):
    from src.framework.physics.scattering_transfer import infrared_fluxes
    dtau=jnp.array([depth,.1]);offset=jnp.array([depth*fraction,.1])
    b=jnp.array([1.,2.,3.]);top=.2;surface=2.8
    actual=infrared_fluxes(dtau,jnp.zeros(2),jnp.zeros(2),b,offset,0.,top,surface)
    c2=.5*(3.-2.)/.1
    bottom=(surface-3.-c2)*np.exp(-.2)+2.+c2
    c1=.5/depth;middle=1.+fraction
    q=min(2.*depth*fraction,35.);full=min(2.*depth,35.)
    up=(bottom-2.-c1)*np.exp(q-full)+middle+c1
    down=(top-1.+c1)*np.exp(-q)+middle-c1
    np.testing.assert_allclose([actual[0][0],actual[1][0]],np.pi*np.array([up,down]),
        rtol=4e-13,atol=1e-13)
