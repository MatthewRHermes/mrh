#!/bin/bash
import unittest
import numpy as np

from pyscf.pbc import gto, scf, df

from mrh.my_pyscf.pbc.mcscf.mc_ao2mo import _ERIS

# Author: Bhavnesh Jangid

class _kCASSCF:
    # Dummy class to check the _ERIS class in kCASSCF AO2MO.
    def __init__(self, kmf, ncas, nelecas):
        self._scf = kmf
        self.ncas = ncas
        self.nelecas = nelecas
        self.ncore = int((kmf.cell.nelectron - nelecas) / 2)
        self.nkpts = len(kmf.kpts)
        self.stdout = kmf.stdout
        self.verbose = kmf.verbose
        self.max_memory = kmf.max_memory

    def get_hcore(self):
        return self._scf.get_hcore()
    
def get_diamond_cell(basis='gth-szv', pseudo='gth-pade'):
    cell = gto.Cell()
    cell.atom = '''
    C 0.000000141    0.709136917   0.000147809
    C 1.228182763    0.254817207   3.507287839 '''
    cell.basis = basis
    cell.pseudo = pseudo
    cell.a = [[2.456,0.0,0.0],[-1.228, 2.127, 0.0],[0.0, -1.1635, 3.507]]
    cell.verbose = 0
    cell.build()
    return cell

def get_He_cell(basis='6-31G', pseudo=None):
    cell = gto.Cell()
    cell.atom = 'He 0 0 0'
    cell.basis = basis
    cell.pseudo = pseudo
    cell.a = np.eye(3)*2.0
    cell.verbose = 0
    cell.build()
    return cell

def _run_kcasscf_ao2mo(cell, kmesh, auxbasis='def2-svp-jkfit'):
    kpts = cell.make_kpts(kmesh)

    kmf = scf.KRHF(cell, kpts).density_fit(auxbasis=auxbasis)
    kmf.exxdiv = None
    kmf.max_cycle = 1
    kmf.kernel()

    kmc = _kCASSCF(kmf, ncas=2, nelecas=2)
    mo_kpts = kmf.mo_coeff

    nmo = mo_kpts[0].shape[1]
    ncas = kmc.ncas
    ncore = kmc.ncore
    nkpts = kmc.nkpts

    if nkpts == 1:
        mo_kpts = mo_kpts.astype(np.complex128)

    eris = _ERIS(kmc, mo_kpts, method='disk', level=1)
    
    eris2 = _ERIS(kmc, mo_kpts, method='direct', level=1)


    def compare_integrals(arr1, arr2, name, shape):
        assert arr1.dtype == arr2.dtype
        assert arr1.shape == arr2.shape == shape
        assert np.allclose(arr1, arr2), f"{name} integrals do not match between direct and disk method."
        arr1 = arr2 = None

    # Compare the j_pc and k_pc integrals
    compare_integrals(eris.j_pc, eris2.j_pc, "j_pc", (nkpts, nmo, ncore))
    compare_integrals(eris.k_pc, eris2.k_pc, "k_pc", (nkpts, nmo, ncore))

    # Compare the vhf_c integrals
    compare_integrals(eris.vhf_c, eris2.vhf_c, "vhf_c", (nkpts, nmo, nmo))

    # Compare the ppaa, papa, and paap integrals
    for k1 in range(nkpts):
        for k2 in range(nkpts):
            for k3 in range(nkpts):
                compare_integrals(eris.ppaa(k1, k2, k3), eris2.ppaa(k1, k2, k3), "ppaa", (nmo, nmo, ncas, ncas))
                compare_integrals(eris.papa(k1, k2, k3), eris2.papa(k1, k2, k3), "papa", (nmo, ncas, nmo, ncas))
                compare_integrals(eris.paap(k1, k2, k3), eris2.paap(k1, k2, k3), "paap", (nmo, ncas, ncas, nmo))

    eris = eris2 = None


class KnownValues(unittest.TestCase):
    # Unit-1: All electrons
    def test_kcasscf_ao2mo_alle(self):
        cell = get_He_cell()
        _run_kcasscf_ao2mo(cell, kmesh=[1, 1, 1], auxbasis='def2-svp-jkfit')
        _run_kcasscf_ao2mo(cell, kmesh=[2, 1, 1], auxbasis='def2-svp-jkfit')
        _run_kcasscf_ao2mo(cell, kmesh=[2, 2, 1], auxbasis='def2-svp-jkfit')
        # _run_kcasscf_ao2mo(get_diamond_cell(), kmesh=[2, 2, 2], auxbasis='def2-svp-jkfit')
    
    # Unit-2: Pseudopotential
    def test_kcasscf_ao2mo_pseudo(self):
        cell = get_diamond_cell(pseudo='gth-pade')
        auxbasis = df.aug_etb(cell, beta=1.7)
        
        _run_kcasscf_ao2mo(cell, kmesh=[1, 1, 1], auxbasis=auxbasis)
        _run_kcasscf_ao2mo(cell, kmesh=[2, 1, 1], auxbasis=auxbasis)
        _run_kcasscf_ao2mo(cell, kmesh=[2, 2, 1], auxbasis=auxbasis)
        # _run_kcasscf_ao2mo(get_diamond_cell(), kmesh=[2, 2, 2], auxbasis='def2-svp-jkfit')

if __name__ == '__main__':
    unittest.main()