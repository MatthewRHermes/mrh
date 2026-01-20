#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
Trying to implement the FCI solver with complex Hamiltonian but with alpha-beta
degeneracy.
'''

import warnings
import ctypes
import numpy as np
import scipy.linalg
from pyscf import lib
from pyscf import ao2mo
from pyscf.fci import cistring
from pyscf.fci import direct_spin1
from pyscf.fci.addons import _unpack_nelec
from mrh.my_pyscf.pbc.fci import rdm_helper

libfci = direct_spin1.libfci

def _get_init_guess_cplx(na, nb, nroots, hdiag, nelec):
    """
    Complex-CI initial guesses: pick the nroots lowest diagonal determinants
    and return CI vectors as complex128 views.
    """
    ci0 = []
    neleca, nelecb = _unpack_nelec(nelec)

    hdiag = np.asarray(hdiag)

    if neleca == nelecb and na == nb:
        htri = lib.pack_tril(hdiag.reshape(na, na))

        if htri.size <= nroots:
            addrs = np.arange(htri.size)
        else:
            addrs = np.argpartition(htri, nroots - 1)[:nroots]
            addrs = np.sort(addrs)

        for addr in addrs:
            addra = int(((2*addr + 0.25)**0.5) - 0.5 + 1e-7)
            addrb = int(addr - addra*(addra + 1)//2)

            x = np.zeros((na, na), dtype=np.complex128)
            x[addra, addrb] = 1.0 + 0.0j
            ci0.append(x.ravel().view(direct_spin1.FCIvector))
    else:
        if hdiag.size <= nroots:
            addrs = np.arange(hdiag.size)
        else:
            addrs = np.argpartition(hdiag, nroots - 1)[:nroots]

        for addr in addrs:
            x = np.zeros((na * nb,), dtype=np.complex128)
            x[int(addr)] = 1.0 + 0.0j
            ci0.append(x.view(direct_spin1.FCIvector))

    ci0[0][0]  += (1e-3 + 1e-5j)
    ci0[0][-1] -= (1e-5 - 1e-5j)
    return ci0

def get_init_guess_cplx(norb, nelec, nroots, hdiag):
    """Complex-CI initial guess: lowest diagonal determinants."""
    neleca, nelecb = _unpack_nelec(nelec)
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    return _get_init_guess_cplx(na, nb, nroots, hdiag, nelec)

def contract_1e(h1e, fcivec, norb, nelec, link_index=None):
    '''
    Contract the 1-electron Hamiltonian with a FCI vector to get a new FCI vector.
    '''
    link_indexa, link_indexb = _unpack(norb, nelec, link_index)
    na, nlinka = link_indexa.shape[:2]
    nb, nlinkb = link_indexb.shape[:2]
    assert fcivec.size == na*nb

    if fcivec.dtype == h1e.dtype == np.float64:
        fcivec = np.asarray(fcivec, order='C')
        h1e = np.asarray(h1e, order='C')
        ci1 = np.zeros_like(fcivec)
        libfci.FCIcontract_a_1e_nosym(h1e.ctypes.data_as(ctypes.c_void_p),
                                fcivec.ctypes.data_as(ctypes.c_void_p),
                                ci1.ctypes.data_as(ctypes.c_void_p),
                                ctypes.c_int(norb),
                                ctypes.c_int(na), ctypes.c_int(nb),
                                ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
                                link_indexa.ctypes.data_as(ctypes.c_void_p),
                                link_indexb.ctypes.data_as(ctypes.c_void_p))
        libfci.FCIcontract_b_1e_nosym(h1e.ctypes.data_as(ctypes.c_void_p),
                                fcivec.ctypes.data_as(ctypes.c_void_p),
                                ci1.ctypes.data_as(ctypes.c_void_p),
                                ctypes.c_int(norb),
                                ctypes.c_int(na), ctypes.c_int(nb),
                                ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
                                link_indexa.ctypes.data_as(ctypes.c_void_p),
                                link_indexb.ctypes.data_as(ctypes.c_void_p))
        return ci1.view(direct_spin1.FCIvector)

    link_indexa, link_indexb = _unpack(norb, nelec, link_index)
    ciR = contract_1e(h1e.real, fcivec.real, norb, nelec, link_index=link_index)
    ciR -= contract_1e(h1e.imag, fcivec.imag, norb, nelec, link_index=link_index)
    ciI = contract_1e(h1e.real, fcivec.imag, norb, nelec, link_index=link_index)
    ciI += contract_1e(h1e.imag, fcivec.real, norb, nelec, link_index=link_index)

    ci1 = ciR.astype(np.complex128)
    ci1.real = ciR
    ci1.imag = ciI
    return ci1

def contract_2e(eri, fcivec, norb, nelec, link_index=None):
    r'''Contract the 2-electron Hamiltonian with a FCI vector to get a new FCI
    vector.

    Note the input arg eri is NOT the 2e hamiltonian matrix, the 2e hamiltonian is

    .. math::

        h2e &= eri_{pq,rs} p^+ q r^+ s \\
            &= (pq|rs) p^+ r^+ s q - (pq|rs) \delta_{qr} p^+ s

    So eri is defined as

    .. math::

        eri_{pq,rs} = (pq|rs) - (1/Nelec) \sum_q (pq|qs)

    to restore the symmetry between pq and rs,

    .. math::

        eri_{pq,rs} = (pq|rs) - (.5/Nelec) [\sum_q (pq|qs) + \sum_p (pq|rp)]

    See also :func:`direct_nosym.absorb_h1e`
    '''
    link_indexa, link_indexb = _unpack(norb, nelec, link_index)
    na, nlinka = link_indexa.shape[:2]
    nb, nlinkb = link_indexb.shape[:2]
    assert fcivec.size == na*nb
    if fcivec.dtype == eri.dtype == np.float64:
        fcivec = np.asarray(fcivec, order='C')
        eri = np.asarray(eri, order='C')
        ci1 = np.zeros_like(fcivec) 

        libfci.FCIcontract_2es1(eri.ctypes.data_as(ctypes.c_void_p),
                                fcivec.ctypes.data_as(ctypes.c_void_p),
                                ci1.ctypes.data_as(ctypes.c_void_p),
                                ctypes.c_int(norb),
                                ctypes.c_int(na), ctypes.c_int(nb),
                                ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
                                link_indexa.ctypes.data_as(ctypes.c_void_p),
                                link_indexb.ctypes.data_as(ctypes.c_void_p))
        return ci1.view(direct_spin1.FCIvector)

    ciR = np.asarray(fcivec.real, order='C')
    ciI = np.asarray(fcivec.imag, order='C')
    eriR = np.asarray(eri.real, order='C')
    eriI = np.asarray(eri.imag, order='C')
    link_index = (link_indexa, link_indexb)
    outR  = contract_2e(eriR, ciR, norb, nelec, link_index=link_index)
    outR -= contract_2e(eriI, ciI, norb, nelec, link_index=link_index)
    outI  = contract_2e(eriR, ciI, norb, nelec, link_index=link_index)
    outI += contract_2e(eriI, ciR, norb, nelec, link_index=link_index)
    out = outR.astype(np.complex128)
    out.real = outR
    out.imag = outI
    return out

def absorb_h1e(h1e, eri, norb, nelec, fac=1):
    '''
    I should double check the formula here.
    Modify 2e Hamiltonian to include 1e Hamiltonian contribution.
    '''
    if not isinstance(nelec, (int, np.number)):
        nelec = sum(nelec)
    if h1e.dtype == eri.dtype == np.float64:
        h2e = ao2mo.restore(1, eri.copy(), norb)
    else:
        assert eri.ndim == 4
        h2e = eri.astype(dtype=np.result_type(h1e, eri), copy=True)
    
    J = np.einsum('jiik->jk', h2e)
    J = 0.5 * (J + J.conj().T)

    f1e = h1e - J * .5
    f1e = f1e * (1./(nelec+1e-100))

    f1e = 0.5 * (f1e + f1e.conj().T)
    for k in range(norb):
        h2e[k,k,:,:] += f1e
        h2e[:,:,k,k] += f1e
    return h2e * fac

def pspace(h1e, eri, norb, nelec, hdiag=None, pspace_size=400):
    '''
    pspace Hamiltonian to improve Davidson preconditioner. See, CPL, 169, 463
    '''
    if norb >= 64:
        raise NotImplementedError('norb >= 64')
    import numpy as np
    if (h1e.dtype == np.complex128 or eri.dtype == np.complex128):
        neleca, nelecb = _unpack_nelec(nelec)
        h1e = np.ascontiguousarray(h1e)
        assert eri.ndim == 4
        eri = np.ascontiguousarray(eri)
        # eri = ao2mo.restore(1, eri, norb)
        na = cistring.num_strings(norb, neleca)
        nb = cistring.num_strings(norb, nelecb)
        if hdiag is None:
            hdiag = make_hdiag(h1e, eri, norb, nelec, compress=False)
        assert hdiag.size == na * nb
        if hdiag.size <= pspace_size:
            addr = np.arange(hdiag.size)
        else:
            try:
                addr = np.argpartition(hdiag, pspace_size-1)[:pspace_size].copy()
            except AttributeError:
                addr = np.argsort(hdiag)[:pspace_size].copy()
        addra, addrb = divmod(addr, nb)
        stra = cistring.addrs2str(norb, neleca, addra)
        strb = cistring.addrs2str(norb, nelecb, addrb)
        pspace_size = len(addr)
        h0R = np.zeros((pspace_size,pspace_size), dtype=h1e.real.dtype)

        # Real part
        h1eR = h1e.real.astype(h1e.real.dtype)
        eriR = eri.real.astype(eri.real.dtype)
        libfci.FCIpspace_h0tril(h0R.ctypes.data_as(ctypes.c_void_p),
                                h1eR.ctypes.data_as(ctypes.c_void_p),
                                eriR.ctypes.data_as(ctypes.c_void_p),
                                stra.ctypes.data_as(ctypes.c_void_p),
                                strb.ctypes.data_as(ctypes.c_void_p),
                                ctypes.c_int(norb), ctypes.c_int(pspace_size))
        
        # Now the imaginary part
        h0I = np.zeros((pspace_size,pspace_size), dtype=h1e.real.dtype)
        h1eI = h1e.imag.astype(h1e.real.dtype)
        eriI = eri.imag.astype(eri.real.dtype)
        libfci.FCIpspace_h0tril(h0I.ctypes.data_as(ctypes.c_void_p),
                                h1eI.ctypes.data_as(ctypes.c_void_p),
                                eriI.ctypes.data_as(ctypes.c_void_p),
                                stra.ctypes.data_as(ctypes.c_void_p),
                                strb.ctypes.data_as(ctypes.c_void_p),
                                ctypes.c_int(norb), ctypes.c_int(pspace_size))
        
        # Now combine real and imaginary part
        # For Hermitian Hamiltonian, we can use the hermitian property to reduce
        # the computation cost.

        h0R = lib.hermi_triu(h0R, 1)
        h0I = lib.hermi_triu(h0I, 1)
        h0 = np.zeros((pspace_size,pspace_size), dtype=np.complex128)
        h0.real = h0R
        h0.imag = h0I

        # Fill diagonal
        idx = np.arange(pspace_size)
        h0[idx,idx] = hdiag[addr].astype(np.complex128)
        return addr, h0
    else:
        # Point to direct_spin1.pspace for real Hamiltonian
        return direct_spin1.pspace(h1e, eri, norb, nelec, hdiag, pspace_size)

def energy(h1e, eri, fcivec, norb, nelec, link_index=None):
    r'''
    Compute the FCI electronic energy for given complex Hamiltonian and 
    complex FCI vector.
    sigma  = (H.real + i H.imag)(ci.real + i ci.imag)
          &= (H.real ci.real - H.imag ci.imag) + i (H.real ci.imag + H.imag ci.real)
    e = <ci|H|ci> = ci^dagger sigma
    '''
    if h1e.dtype == eri.dtype == np.complex128:
        h1e = np.asarray(h1e)
        eri = np.asarray(eri)
        ci  = np.asarray(fcivec)
        h2e_eff = absorb_h1e(h1e, eri, norb, nelec, fac=0.5)
        sigma = contract_2e(h2e_eff, ci, norb, nelec, link_index=link_index)
        e = np.vdot(ci, sigma)
    else:
        h2e = absorb_h1e(h1e, eri, norb, nelec, .5)
        ci1 = contract_2e(h2e, fcivec, norb, nelec, link_index)
        e = np.dot(fcivec.reshape(-1), ci1.reshape(-1))
    return e

def make_hdiag(h1e, eri, norb, nelec, compress=False):
    '''
    Diagonal of the FCI Hamiltonian.
    I guess, the real Hamiltonian would be a good approximation to the complex one.
    '''
    if h1e.dtype == np.complex128:
        h1e = h1e.real.copy()
    if eri.dtype == np.complex128:
        eri = eri.real.copy()
    return direct_spin1.make_hdiag(h1e, eri, norb, nelec, compress)

def make_rdm1s(fcivec, norb, nelec, link_index=None):
    make_rdm1s.__doc__ = direct_spin1.make_rdm1s.__doc__
    if link_index is None:
        neleca, nelecb = _unpack_nelec(nelec)
        link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
        link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
        link_index = (link_indexa, link_indexb)
    rdm1a = rdm_helper.make_rdm1_spin1('FCImake_rdm1a_cplx', fcivec, fcivec,
                                norb, nelec, link_index)
    rdm1b = rdm_helper.make_rdm1_spin1('FCImake_rdm1b_cplx', fcivec, fcivec,
                                norb, nelec, link_index)
    return (rdm1a, rdm1b)

def make_rdm1(fcivec, norb, nelec, link_index=None):
    make_rdm1.__doc__ = direct_spin1.make_rdm1.__doc__
    rdm1a, rdm1b = make_rdm1s(fcivec, norb, nelec, link_index)
    rdm1 = rdm1a + rdm1b
    return rdm1.conj().T

def make_rdm12s(fcivec, norb, nelec, link_index=None, reorder=True):
    make_rdm12s.__doc__ = direct_spin1.make_rdm12s.__doc__
    dm1a, dm2aa = rdm_helper.make_rdm12_spin1('FCIrdm12kern_a_cplx', fcivec, fcivec,
                                       norb, nelec, link_index, 1)
    dm1b, dm2bb = rdm_helper.make_rdm12_spin1('FCIrdm12kern_b_cplx', fcivec, fcivec,
                                       norb, nelec, link_index, 1)
    _, dm2ab = rdm_helper.make_rdm12_spin1('FCItdm12kern_ab_cplx', fcivec, fcivec,
                                    norb, nelec, link_index, 0)
    if reorder:
        dm1a, dm2aa = rdm_helper.reorder_rdm(dm1a, dm2aa, inplace=True)
        dm1b, dm2bb = rdm_helper.reorder_rdm(dm1b, dm2bb, inplace=True)
    
    return (dm1a, dm1b), (dm2aa, dm2ab, dm2bb)

def make_rdm12(fcivec, norb, nelec, link_index=None, reorder=True):
    make_rdm12.__doc__ = direct_spin1.make_rdm12.__doc__
    dm1, dm2 = rdm_helper.make_rdm12_spin1('FCIrdm12kern_sf_cplx', fcivec, fcivec,
                                    norb, nelec, link_index, 1)
    if reorder:
        dm1, dm2 = rdm_helper.reorder_rdm(dm1, dm2, inplace=True)
    return dm1, dm2

def make_rdm1_py(fcivec, norb, nelec, link_index=None):
    rdm1 = rdm_helper.make_rdm1_cplx(fcivec, norb, nelec, link_index)
    return rdm1

def make_rdm1s_py(fcivec, norb, nelec, link_index=None):
    rdm1a, rdm1b = rdm_helper.make_rdm1s_cplx(fcivec, norb, nelec, link_index)
    return rdm1a, rdm1b

def make_rdm12s_py(fcivec, norb, nelec, link_index=None, reorder=True):
    (dm1a, dm1b), (dm2aa, dm2ab, dm2bb) = rdm_helper.make_rdm12s_cplx(fcivec, norb, nelec, link_index, reorder)
    return (dm1a, dm1b), (dm2aa, dm2ab, dm2bb)

def make_rdm12_py(fcivec, norb, nelec, link_index=None, reorder=True):
    dm1, dm2 = rdm_helper.make_rdm12_cplx(fcivec, norb, nelec, link_index, reorder)
    return dm1, dm2

class FCISolver(direct_spin1.FCISolver):
    def __init__(self, *args, **kwargs):
        direct_spin1.FCISolver.__init__(self, *args, **kwargs)
        # pspace constructor only supports Hermitian Hamiltonian
        self.davidson_only = True

    def contract_1e(self, h1e, fcivec, norb, nelec, link_index=None):
        return contract_1e(h1e, fcivec, norb, nelec, link_index)

    def contract_2e(self, eri, fcivec, norb, nelec, link_index=None):
        return contract_2e(eri, fcivec, norb, nelec, link_index)

    def absorb_h1e(self, h1e, eri, norb, nelec, fac=1):
        return absorb_h1e(h1e, eri, norb, nelec, fac)

    def energy(self, h1e, eri, fcivec, norb, nelec, link_index=None):
        return energy(h1e, eri, fcivec, norb, nelec, link_index)
    
    def make_hdiag(self, h1e, eri, norb, nelec, compress=False):
        nelec = direct_spin1._unpack_nelec(nelec, self.spin)
        return make_hdiag(h1e, eri, norb, nelec, compress)

    @lib.with_doc(pspace.__doc__)
    def pspace(self, h1e, eri, norb, nelec, hdiag=None, pspace_size=400):
        nelec = _unpack_nelec(nelec, self.spin)
        return pspace(h1e, eri, norb, nelec, hdiag, pspace_size)
    
    def kernel(self, h1e, eri, norb, nelec, ci0=None,
               tol=None, lindep=None, max_cycle=None, max_space=None,
               nroots=None, davidson_only=None, pspace_size=None,
               orbsym=None, wfnsym=None, ecore=0, **kwargs):
        if isinstance(nelec, (int, np.number)):
            nelecb = nelec//2
            neleca = nelec - nelecb
        else:
            neleca, nelecb = nelec

        link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
        link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)

        e, c = direct_spin1.kernel_ms1(self, h1e, eri, norb, nelec, ci0,
                                       (link_indexa,link_indexb),
                                       tol, lindep, max_cycle, max_space, nroots,
                                       davidson_only, pspace_size, ecore=ecore,
                                       **kwargs)
        self.eci, self.ci = e, c
        return e, c

    def eig(self, op, x0=None, precond=None, **kwargs):
        if isinstance(op, np.ndarray):
            self.converged = True
            return scipy.linalg.eigh(op)

        # TODO: check the hermitian of Hamiltonian then determine whether to
        # call the non-hermitian diagonalization solver davidson_nosym1
        if False:
            warnings.warn('direct_nosym.kernel is not able to diagonalize '
                        'non-Hermitian Hamiltonian. If h1e and h2e is not '
                        'hermtian, calling symmetric diagonalization in eig '
                        'can lead to wrong results.')
            
        self.converged, e, ci = \
                lib.davidson1(lambda xs: [op(x) for x in xs],
                              x0, precond, lessio=self.lessio, **kwargs)

        if kwargs.get('nroots', 1) == 1:
            self.converged = self.converged[0]
            e = e[0]
            ci = ci[0]
        return e, ci

    def make_rdm1s(self, fcivec, norb, nelec, link_index=None):
        return make_rdm1s(fcivec, norb, nelec, link_index)
    
    def make_rdm1(self, fcivec, norb, nelec, link_index=None):
        return make_rdm1(fcivec, norb, nelec, link_index)
    
    def make_rdm12s(self, fcivec, norb, nelec, link_index=None, reorder=True):
        return make_rdm12s(fcivec, norb, nelec, link_index, reorder)
    
    def make_rdm12(self, fcivec, norb, nelec, link_index=None, reorder=True):
        return make_rdm12(fcivec, norb, nelec, link_index, reorder)

    def get_init_guess(self, norb, nelec, nroots, hdiag):
        return get_init_guess_cplx(norb, nelec, nroots, hdiag)
    
    def make_rdm1s_py(self, fcivec, norb, nelec, link_index=None):
        return make_rdm1s_py(fcivec, norb, nelec, link_index)
    
    def make_rdm1_py(self, fcivec, norb, nelec, link_index=None):
        return make_rdm1_py(fcivec, norb, nelec, link_index)
    
    def make_rdm12s_py(self, fcivec, norb, nelec, link_index=None, reorder=True):
        return make_rdm12s_py(fcivec, norb, nelec, link_index, reorder)
    
    def make_rdm12_py(self, fcivec, norb, nelec, link_index=None, reorder=True):
        return make_rdm12_py(fcivec, norb, nelec, link_index, reorder)
    
FCI = FCISolver

def _unpack(norb, nelec, link_index):
    if link_index is None:
        if isinstance(nelec, (int, np.number)):
            nelecb = nelec//2
            neleca = nelec - nelecb
        else:
            neleca, nelecb = nelec
        link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
        link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
        return link_indexa, link_indexb
    else:
        return link_index