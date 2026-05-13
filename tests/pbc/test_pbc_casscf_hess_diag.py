#!/bin/bash
import unittest
import copy
import numpy as np
from scipy.linalg import expm

from pyscf.pbc import gto as pgto
from pyscf.pbc import scf

from mrh.my_pyscf.pbc import mcscf
from mrh.my_pyscf.pbc.fci import csf_solver
from mrh.my_pyscf.pbc.mcscf.k2R import get_mo_coeff_k2R

'''
In this file, there are two unit tests for the k-CASSCF orbital hessian diagonal.
1. In the limit of single determinant.
2. More than SD case

In both of these cases, I am comparing the hdiag obtained from direct computation and 
the one reconstructed from hop.

Note: the PySCF kRHF hdiag is not exact, as in it misses the 2e response contribution. 
Therefore, I didn't consider the direct comparing to the kRHF hdiag as a reference.
'''

dtype = np.complex128

def get_cell():
    cell = pgto.Cell(
        a=np.diag([3.0, 10.0, 10.0]),
        atom="Be 0.0 0.0 0.0",
        basis="6-31G",
        unit="Angstrom",
        ke_cutoff=50,
        precision=1e-12,
        verbose=0,)
    return cell

def get_cell2():
    cell = pgto.Cell(
        a=np.diag([3.0, 10.0, 10.0]),
        atom="N 0.0 0.0 0.0; N 0.0 0.0 1.1",
        basis="STO-3G",
        unit="Angstrom",
        ke_cutoff=50,
        precision=1e-12,
        verbose=0,)
    return cell

def get_casdm12_k(kmc, ci):
    ncastot = kmc.ncas * kmc.nkpts
    nelecastot = (kmc.nelecas[0] * kmc.nkpts, kmc.nelecas[1] * kmc.nkpts)
    return kmc.fcisolver.make_rdm12(ci, ncastot, nelecastot)

def rotate_mos_kpts(kmc, mo_coeff, x, step=1.0):
    x = np.asarray(x).reshape(-1)
    nkpts = kmc.nkpts
    nvar_k = x.size // nkpts
    x = x.reshape(nkpts, nvar_k)

    mo_new = np.array(mo_coeff, copy=True)
    for k in range(nkpts):
        dr_k = kmc.unpack_uniq_var(step * x[k])
        kappa_k = 0.5 * (dr_k - dr_k.conj().T)
        mo_new[k] = mo_new[k] @ expm(kappa_k)
    return mo_new

def get_g_hop_hdiag(kmc, mo_coeff, ci):
    eris = kmc.ao2mo(mo_coeff)
    mo_phase = get_mo_coeff_k2R(kmc._scf, mo_coeff.copy(), 
                                kmc.ncore, kmc.ncas)[-1]
    casdm1, casdm2 = get_casdm12_k(kmc, ci)
    g, _, hop, hdiag = kmc.gen_g_hop(mo_coeff, mo_phase, 1, casdm1, casdm2, eris)
    return g.ravel(), hop, hdiag.ravel()

def reconstruct_hdiag_from_hop(hop, nvar):
    hdiag = np.zeros(nvar, dtype=dtype)
    for i in range(nvar):
        ei = np.zeros(nvar, dtype=dtype)
        ei[i] = 1.0
        hei = hop(ei).ravel()
        hdiag[i] = hei[i]
    return hdiag

def get_block_slices(nkpts, ncore, ncas, nmo):
    nocc = ncore + ncas
    nvir = nmo - nocc
    sizes_per_k = []
    labels_per_k = []
    if ncore > 0 and ncas > 0:
        sizes_per_k.append(ncore * ncas)
        labels_per_k.append("ca")
    if ncore > 0 and nvir > 0:
        sizes_per_k.append(ncore * nvir)
        labels_per_k.append("cv")
    if ncas > 0 and nvir > 0:
        sizes_per_k.append(ncas * nvir)
        labels_per_k.append("av")
    nvar_k = sum(sizes_per_k)
    slices = {}
    offset = 0
    for k in range(nkpts):
        local = 0
        for label, size in zip(labels_per_k, sizes_per_k):
            slices.setdefault(label, []).append(slice(offset + local, offset + local + size))
            local += size
        offset += nvar_k
    return slices

class KnownValues(unittest.TestCase):

    def assert_hdiag_blocks_allclose(self, hdiag_ref, hdiag_test, 
                                     nkpts, ncore, ncas, nmo):
        '''
        This is to compare the hdiag blocks (ca, cv, av) separately
        '''
        hdiag_ref = np.asarray(hdiag_ref).reshape(-1)
        hdiag_test = np.asarray(hdiag_test).reshape(-1)

        slices = get_block_slices(nkpts, ncore, ncas, nmo)

        for label in ("ca", "cv", "av"):
            if label not in slices: continue
            idx = np.concatenate([np.arange(s.start, s.stop, dtype=int) 
                                  for s in slices[label]])
            ref_blk = hdiag_ref[idx]
            test_blk = hdiag_test[idx]
            diff = ref_blk - test_blk

            self.assertTrue(np.allclose(ref_blk, test_blk, atol=1e-8, rtol=1e-6),
                msg=(
                    f"hdiag block '{label}' mismatch: "
                    f"size={ref_blk.size}, "
                    f"||ref||={np.linalg.norm(ref_blk):.12e}, "
                    f"||test||={np.linalg.norm(test_blk):.12e}, "
                    f"||diff||={np.linalg.norm(diff):.12e}, "
                    f"max|diff|={np.max(np.abs(diff)):.12e}"
                ),
            )

    def test_hdiag_against_kRHFLimit(self):
        '''
        Note, the kRHF hdiag is not complete hdiag, the response of 2e term is dropped in PySCF implementation.
        '''
        cell = get_cell2()
        cell.build()
        kpts = cell.make_kpts([2, 1, 1], wrap_around=True)

        kmf = scf.KRHF(cell, exxdiv=None, kpts=kpts).density_fit(auxbasis="def2-svp-jkfit").newton()
        kmf.max_cycle = 0
        kmf.exxdiv = None
        kmf.kernel()

        mo_ref = np.array(kmf.mo_coeff, copy=True)

        kmc = mcscf.CASSCF(kmf, 7, (7, 7))
        kmc.max_cycle_macro = 0
        kmc.fcisolver = csf_solver(cell, smult=1)
        kmc.kernel(mo_ref.copy())

        mo_ref = np.array(kmc.mo_coeff, copy=True)
        ci0 = copy.deepcopy(kmc.ci)

        g0, hop0, hdiag_direct = get_g_hop_hdiag(kmc, mo_ref, ci0)
        hdiag_from_hop = reconstruct_hdiag_from_hop(hop0, g0.size)

        self.assertTrue(np.allclose(hdiag_direct, hdiag_from_hop, atol=1e-8, rtol=1e-6))
        self.assert_hdiag_blocks_allclose(hdiag_direct, hdiag_from_hop, nkpts=kmc.nkpts,
                                          ncore=kmc.ncore, ncas=kmc.ncas, nmo=kmc.mo_coeff[0].shape[1])
                
    def test_hdiag(self):
        cell = get_cell()
        cell.build()
        kpts = cell.make_kpts([3, 1, 1], wrap_around=True)

        kmf = scf.KRHF(cell, exxdiv=None, kpts=kpts).density_fit(auxbasis="def2-svp-jkfit")
        kmf.max_cycle = 0
        kmf.exxdiv = None
        kmf.kernel()

        kmc = mcscf.CASSCF(kmf, 2, (1, 1))
        kmc.max_cycle_macro = 0
        kmc.fcisolver = csf_solver(cell, smult=1)
        kmc.kernel(kmf.mo_coeff.copy())

        mo_ref = np.array(kmc.mo_coeff, copy=True)
        ci0 = copy.deepcopy(kmc.ci)

        g0, hop0, hdiag_direct = get_g_hop_hdiag(kmc, mo_ref, ci0)
        hdiag_from_hop = reconstruct_hdiag_from_hop(hop0, g0.size)

        self.assertTrue(np.allclose(hdiag_direct, hdiag_from_hop, atol=1e-8, rtol=1e-6))
        self.assert_hdiag_blocks_allclose(hdiag_direct, hdiag_from_hop, nkpts=kmc.nkpts,
                                          ncore=kmc.ncore, ncas=kmc.ncas, nmo=kmc.mo_coeff[0].shape[1])

if __name__ == "__main__":
    unittest.main()