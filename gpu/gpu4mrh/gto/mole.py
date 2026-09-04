#!/usr/bin/env python

from pyscf.gto.mole import *
from pyscf import lib

def _M(self, use_gpu=None, **kwargs):
    r'''This is a shortcut to build up Mole object.

    Args: Same to :func:`Mole.build`

    Examples:

    >>> from pyscf import gto
    >>> mol = gto.M(atom='H 0 0 0; F 0 0 1', basis='6-31g')
    '''
    
    if use_gpu is not None:
        import warnings
        warnings.warn(
            "Passing use_gpu to gto.M() is deprecated. "
            "Set lib.param.use_gpu = gpu instead.",
            DeprecationWarning, stacklevel=2
        )
        lib.param.use_gpu = use_gpu
    
    mol = Mole()
    mol.build(**kwargs)

    return mol
