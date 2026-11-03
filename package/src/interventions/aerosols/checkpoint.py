"""Atomic complete-experiment checkpoints, without executable pickle payloads."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np
from src.framework.gcm._dinosaur import jax, jnp


def identity(configuration):
    return hashlib.sha256(json.dumps(configuration,sort_keys=True,allow_nan=False,
        separators=(',',':')).encode()).hexdigest()


def save(path,state,configuration):
    """Configuration must include body/grid, sources, forcing and asset hashes."""
    leaves,tree=jax.tree_util.tree_flatten(state)
    arrays={f'leaf_{i}':np.asarray(x) for i,x in enumerate(leaves)}
    metadata=dict(version=1,configuration=configuration,identity=identity(configuration),
        tree=str(tree),leaves=len(leaves))
    arrays['metadata']=np.asarray(json.dumps(metadata,sort_keys=True,allow_nan=False))
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    handle,name=tempfile.mkstemp(dir=path.parent,prefix=path.name+'.',suffix='.tmp')
    try:
        with os.fdopen(handle,'wb') as stream:
            np.savez_compressed(stream,**arrays)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name): os.unlink(name)


def load(path,template,configuration):
    """Require matching experiment identity and explicit structural template."""
    expected,tree=jax.tree_util.tree_flatten(template)
    with np.load(path,allow_pickle=False) as data:
        metadata=json.loads(str(data['metadata']))
        if metadata['version']!=1 or metadata['identity']!=identity(configuration):
            raise ValueError('checkpoint configuration/assets do not match this experiment')
        if metadata['tree']!=str(tree) or metadata['leaves']!=len(expected):
            raise ValueError('checkpoint structure does not match this experiment')
        leaves=[]
        for i,example in enumerate(expected):
            value=data[f'leaf_{i}']
            if value.shape!=np.shape(example) or value.dtype!=np.asarray(example).dtype:
                raise ValueError('checkpoint array shape/dtype mismatch')
            leaves.append(jnp.asarray(value))
    return jax.tree_util.tree_unflatten(tree,leaves)
