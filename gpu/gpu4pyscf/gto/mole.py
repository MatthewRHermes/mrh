#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from pyscf.gto.mole import *

from gpu4pyscf.lib.utils import patch_cpu_kernel

#def M(use_gpu=None, **kwargs):
#    r'''This is a shortcut to build up Mole object.
#
#    Args: Same to :func:`Mole.build`
#
#    Examples:
#
#    >>> from pyscf import gto
#    >>> mol = gto.M(atom='H 0 0 0; F 0 0 1', basis='6-31g')
#    '''
#
#    print("Inside gpu4pyscf::mole.py")
#    mol = 0
#    #    mol = Mole()
##    mol.build(**kwargs)
#
#    return mol

def _M(self, use_gpu=None, **kwargs):
    r'''This is a shortcut to build up Mole object.

    Args: Same to :func:`Mole.build`

    Examples:

    >>> from pyscf import gto
    >>> mol = gto.M(atom='H 0 0 0; F 0 0 1', basis='6-31g')
    '''
    
    print("Inside mrh::mole.py")
    
    mol = Mole()
    mol.build(**kwargs)

    # Will generate warning; there's no use in dumping gpu handle, so can ignore
    # pyscf/gto/mole.py:1217: UserWarning: Function mol.dumps drops attribute use_gpu because it is not JSON-serializable
    # <class 'pyscf.gto.mole.Mole'> does not have attributes  use_gpu
    
    add_gpu = {"use_gpu":use_gpu}
    mol.__dict__.update(add_gpu)

    return mol
