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

'''
Full CI solver for spin-free Hamiltonian.  This solver can be used to compute
doublet, triplet,...

The CI wfn are stored as a 2D array [alpha,beta], where each row corresponds
to an alpha string.  For each row (alpha string), there are
total-num-beta-strings of columns.  Each column corresponds to a beta string.

Different FCI solvers are implemented to support different type of symmetry.
                    Symmetry
File                Point group   Spin singlet   Real hermitian*    Alpha/beta degeneracy
direct_spin0_symm   Yes           Yes            Yes                Yes
direct_spin1_symm   Yes           No             Yes                Yes
direct_spin0        No            Yes            Yes                Yes
direct_spin1        No            No             Yes                Yes
direct_uhf          No            No             Yes                No
direct_nosym        No            No             No**               Yes

*  Real hermitian Hamiltonian implies (ij|kl) = (ji|kl) = (ij|lk) = (ji|lk)
** Hamiltonian is real but not hermitian, (ij|kl) != (ji|kl) ...
'''

import sys
import ctypes
import numpy
import scipy.linalg
from pyscf import lib
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf.fci import cistring
from pyscf.fci import rdm
from pyscf.fci import spin_op
from pyscf.fci import addons
from pyscf.fci.spin_op import contract_ss
from pyscf.fci.addons import _unpack_nelec, civec_spinless_repr
from pyscf import __config__

libfci = cistring.libfci

def _trans_rdm1s(cibra, ciket, norb, nelec, link_index=None, use_gpu = False, gpu=None):
    r'''Spin separated transition 1-particle density matrices.
    The return values include two density matrices: (alpha,alpha), (beta,beta).
    See also function :func:`make_rdm1s`

    1pdm[p,q] = :math:`\langle q^\dagger p \rangle`
    '''
    rdm1a = rdm.make_rdm1_spin1('FCItrans_rdm1a', cibra, ciket,
                                norb, nelec, link_index)#, _use_gpu = use_gpu, _gpu = gpu)
    rdm1b = rdm.make_rdm1_spin1('FCItrans_rdm1b', cibra, ciket,
                                norb, nelec, link_index)#, _use_gpu = use_gpu, _gpu = gpu)
    return rdm1a, rdm1b

def _trans_rdm12s(cibra, ciket, norb, nelec, link_index=None, reorder=True, use_gpu = False, gpu=None):
    r'''Spin separated 1- and 2-particle transition density matrices.
    The return values include two lists, a list of 1-particle transition
    density matrices and a list of 2-particle transition density matrices.
    The density matrices are:
    (alpha,alpha), (beta,beta) for 1-particle transition density matrices;
    (alpha,alpha,alpha,alpha), (alpha,alpha,beta,beta),
    (beta,beta,alpha,alpha), (beta,beta,beta,beta) for 2-particle transition
    density matrices.

    1pdm[p,q] = :math:`\langle q^\dagger p\rangle`;
    2pdm[p,q,r,s] = :math:`\langle p^\dagger r^\dagger s q\rangle`.
    '''
    dm1a, dm2aa = rdm.make_rdm12_spin1('FCItdm12kern_a', cibra, ciket,
                                       norb, nelec, link_index, 2)#, _use_gpu = use_gpu, _gpu = gpu)
    dm1b, dm2bb = rdm.make_rdm12_spin1('FCItdm12kern_b', cibra, ciket,
                                       norb, nelec, link_index, 2)#, _use_gpu = use_gpu, _gpu = gpu)
    _, dm2ab = rdm.make_rdm12_spin1('FCItdm12kern_ab', cibra, ciket,
                                       norb, nelec, link_index, 0)#, _use_gpu = use_gpu, _gpu = gpu)
    _, dm2ba = rdm.make_rdm12_spin1('FCItdm12kern_ab', ciket, cibra,
                                       norb, nelec, link_index, 0)#, _use_gpu = use_gpu, _gpu = gpu)
    dm2ba = dm2ba.transpose(3,2,1,0)
    if reorder:
        dm1a, dm2aa = rdm.reorder_rdm(dm1a, dm2aa, inplace=True)
        dm1b, dm2bb = rdm.reorder_rdm(dm1b, dm2bb, inplace=True)
    return (dm1a, dm1b), (dm2aa, dm2ab, dm2ba, dm2bb)


