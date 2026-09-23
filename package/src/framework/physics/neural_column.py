"""NeuralGCM-inspired shared-column encode/process/decode residual networks.

Reference: Kochkov et al., Nature 2024, doi:10.1038/s41586-024-07744-y,
Appendix 3.4. Mars adaptation, not pretrained NeuralGCM or an exact reproduction.
All learned output layers start at zero. There is no saturating coefficient map.
"""
from src.framework.gcm._dinosaur import jax, jnp


def initialize(key, input_size, layers, *, width=384, blocks=5):
    if min(input_size, layers, width, blocks) < 1:
        raise ValueError("network dimensions must be positive")

    def network(key):
        keys = iter(jax.random.split(key, 1 + 3 * blocks))
        def linear(source, target):
            return {"w": jax.random.normal(next(keys), (source, target)) / jnp.sqrt(float(source)),
                    "b": jnp.zeros(target)}
        encode = linear(input_size, width)
        process = tuple(tuple(linear(width, width) for _ in range(3)) for _ in range(blocks))
        return {"encode": encode, "process": process,
                "decode": {"w": jnp.zeros((width, 3 * layers)), "b": jnp.zeros(3 * layers)}}
    return dict(zip(("encoder", "physics", "decoder"),
                    (network(k) for k in jax.random.split(key, 3))))


def apply(params, features):
    def linear(x, layer):
        return x @ layer["w"] + layer["b"]
    x = linear(features, params["encode"])
    for block in params["process"]:
        residual = jax.nn.gelu(linear(x, block[0]))
        residual = jax.nn.gelu(linear(residual, block[1]))
        x = x + linear(residual, block[2]) / jnp.sqrt(float(len(params["process"])))
    return linear(x, params["decode"])
