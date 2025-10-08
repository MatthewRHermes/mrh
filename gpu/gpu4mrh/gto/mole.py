#!/usr/bin/env python

from pyscf.gto.mole import *

from gpu4mrh.lib.utils import patch_cpu_kernel

def _M(self, use_gpu=None, **kwargs):
    r'''This is a shortcut to build up Mole object.

    Args: Same to :func:`Mole.build`

    Examples:

    >>> from pyscf import gto
    >>> mol = gto.M(atom='H 0 0 0; F 0 0 1', basis='6-31g')
    '''
    
    print("Inside mrh::mole.py adding use_gpu flag")
    
    mol = Mole()
    mol.build(**kwargs)

    # Will generate warning; there's no use in dumping gpu handle, so can ignore
    # pyscf/gto/mole.py:1217: UserWarning: Function mol.dumps drops attribute use_gpu because it is not JSON-serializable
    # <class 'pyscf.gto.mole.Mole'> does not have attributes  use_gpu
    
    add_gpu = {"use_gpu":use_gpu}
    mol.__dict__.update(add_gpu)

    return mol
