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

    mol = Mole()
    mol.build(**kwargs)

    if use_gpu is None:
        use_gpu = getattr(lib.param, 'use_gpu', None)
    if use_gpu is not None:
        mol.use_gpu = use_gpu

    return mol
