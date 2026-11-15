"""Batched differentiable Thomas solve, with vertical levels on axis zero."""
from src.framework.gcm._dinosaur import jax,jnp


def solve_tridiagonal(lower,diagonal,upper,rhs):
    c0=upper[0]/diagonal[0]
    d0=rhs[0]/diagonal[0]

    def eliminate(carry,row):
        c_prev,d_prev=carry
        lo,diag,up,right=row
        pivot=diag-lo*c_prev
        c=up/pivot
        d=(right-lo*d_prev)/pivot
        return (c,d),(c,d)

    _,(c_tail,d_tail)=jax.lax.scan(eliminate,(c0,d0),
        (lower[1:],diagonal[1:],upper[1:],rhs[1:]))
    c_prime=jnp.concatenate([c0[None],c_tail],axis=0)
    d_prime=jnp.concatenate([d0[None],d_tail],axis=0)

    def substitute(next_value,row):
        c,d=row
        value=d-c*next_value
        return value,value

    _,reversed_solution=jax.lax.scan(substitute,d_prime[-1],
        (c_prime[:-1][::-1],d_prime[:-1][::-1]))
    return jnp.concatenate([reversed_solution[::-1],d_prime[-1:]],axis=0)
