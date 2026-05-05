#!/bin/bash

import unittest
import numpy as np

from pyscf.pbc import gto as pgto
from pyscf.fci import direct_spin1
from pyscf.fci import cistring

from mrh.my_pyscf.pbc.fci import direct_spin1_cplx


'''
    There are four make_rdm functions
    1. make_rdm1s (spin-separated 1-RDM)
    2. make_rdm12s (spin-separated 1-RDM and 2-RDM)
    3. make_rdm1 (spin-summed 1-RDM)
    4. make_rdm12 (spin-summed 2-RDM)

    Currently, I have these functions defined in both Python and more optimized C code.
    In the python format, I only defined the spin-separated RDMs and then construct the 
    spin-summed RDMs from them.
'''

def compute_real_space_rdm12(cell, fcivec, norb, nelec, reorder=True):
    '''
    Compute the spin-summed and spin-separated 1-RDM and 2-RDM using the 
    direct_spin1 implementation.
    '''
    cisolver = direct_spin1.FCISolver(cell)
    rdm1, rdm2 = cisolver.make_rdm12(fcivec.real, norb, nelec, reorder=reorder)
    (dm1a, dm1b), (dm2aa, dm2ab, dm2bb) = cisolver.make_rdm12s(fcivec.real, norb, nelec, reorder=reorder)
    return (rdm1, rdm2), (dm1a, dm1b), (dm2aa, dm2ab, dm2bb)

def dummy_cell():
    cell = pgto.Cell()
    cell.atom = '''
    Ne 0.000000000000 0.000000000000 0.000000000000
    Ne 0.000000000000 0.000000000000 3.500000000000
    '''
    cell.a = np.diag([10.0, 10.0, 10.0])
    cell.basis = '6-31G'
    cell.unit = 'Angstrom'
    cell.verbose = 0
    cell.build()
    return cell

class KnownValues(unittest.TestCase):

    # Unit-Test-1: In the limit of the zero imaginary part of CI coeff, the
    # RDMs from the complex implementation should match with the real implementation.
    def test_rdm12_and_rdm12s_to_real_limit(self):
        
        ncas1 = 8
        ncas2 = 5
        nelecaslst1 = [(8,0), (7,1), (6,2), (5,3), (4,4), (3,5), (2,6), (1,7), (0,8)]
        nelecaslst2 = [(5,0), (4,1), (3,2), (2,3), (1,4), (0,5)]

        ncas1lst = [ncas1,]*len(nelecaslst1) 
        ncas2lst = [ncas2,]*len(nelecaslst2)

        cell = dummy_cell()
        for ncaslst, nelecaslst in zip([ncas1lst, ncas2lst], [nelecaslst1, nelecaslst2]):
            for ncas, nelecas in zip(ncaslst, nelecaslst):
                na = cistring.num_strings(ncas, nelecas[0])
                nb = cistring.num_strings(ncas, nelecas[1])

                # Generate a random FCIvector
                fcivec = np.random.rand(na*nb) + 1j * np.random.rand(na*nb)
                fcivec.imag *= 0
                fcivec /= np.linalg.norm(fcivec)

                # Compute the RDMs using the complex implementation.
                cisolver = direct_spin1_cplx.FCISolver(cell)
                rdm1, rdm2 = cisolver.make_rdm12(fcivec, ncas, nelecas, reorder=True)
                (dm1a, dm1b), (dm2aa, dm2ab, dm2bb) = cisolver.make_rdm12s(fcivec, ncas, nelecas)
                
                # Compute the reference RDMs using realspace implementation.
                (rdm1_ref, rdm2_ref), (dm1a_ref, dm1b_ref), (dm2aa_ref, dm2ab_ref, dm2bb_ref) \
                    = compute_real_space_rdm12(cell, fcivec, ncas, nelecas)
                
                dm1 = dm1a + dm1b
                dm2 = dm2aa + dm2bb + dm2ab + dm2ab.transpose(2,3,0,1).conj()

                # Compare the spin-summed RDMs
                np.testing.assert_allclose(rdm1, rdm1_ref, atol=1e-10, rtol=1e-10)
                np.testing.assert_allclose(rdm2, rdm2_ref, atol=1e-10, rtol=1e-10)

                # Compare the spin-summed RDMs constructed from spin-separated RDMs
                np.testing.assert_allclose(dm1, rdm1, atol=1e-10, rtol=1e-10)
                np.testing.assert_allclose(dm2, rdm2, atol=1e-10, rtol=1e-10)

                # Compare the spin-separated RDMs
                np.testing.assert_allclose(dm1a, dm1a_ref, atol=1e-10, rtol=1e-10)
                np.testing.assert_allclose(dm1b, dm1b_ref, atol=1e-10, rtol=1e-10)
                np.testing.assert_allclose(dm2aa, dm2aa_ref, atol=1e-10, rtol=1e-10)
                np.testing.assert_allclose(dm2ab, dm2ab_ref, atol=1e-10, rtol=1e-10)
                np.testing.assert_allclose(dm2bb, dm2bb_ref, atol=1e-10, rtol=1e-10)
     
                # Compare the trace of 1-RDM with the number of electrons
                np.testing.assert_allclose(np.trace(dm1a), nelecas[0], atol=1e-10, rtol=1e-10)
                np.testing.assert_allclose(np.trace(dm1b), nelecas[1], atol=1e-10, rtol=1e-10)
                np.testing.assert_allclose(np.trace(rdm1), sum(nelecas), atol=1e-10, rtol=1e-10)

                # Check the hermiticity of RDMs
                np.testing.assert_allclose(dm1a, dm1a.conj().T, atol=1e-10, rtol=1e-10)
                np.testing.assert_allclose(dm1b, dm1b.conj().T, atol=1e-10, rtol=1e-10)
                np.testing.assert_allclose(rdm1, rdm1.conj().T, atol=1e-10, rtol=1e-10)
                np.testing.assert_allclose(dm2aa, dm2aa.transpose(1,0,3,2).conj(), atol=1e-10, rtol=1e-10)
                np.testing.assert_allclose(dm2bb, dm2bb.transpose(1,0,3,2).conj(), atol=1e-10, rtol=1e-10)
                np.testing.assert_allclose(dm2ab, dm2ab.transpose(1,0,3,2).conj(), atol=1e-10, rtol=1e-10)

                np.testing.assert_allclose(dm2aa - dm2aa.transpose(2,3,0,1), 0, atol=1e-8)
                np.testing.assert_allclose(dm2bb - dm2bb.transpose(2,3,0,1), 0, atol=1e-8)
                
                # Compare the trace of 2-RDM with the number of electron pairs
                np.testing.assert_allclose(np.einsum("ppqq->", dm2aa, optimize=True), nelecas[0]*(nelecas[0]-1), atol=1e-6)
                np.testing.assert_allclose(np.einsum("ppqq->", dm2bb, optimize=True), nelecas[1]*(nelecas[1]-1), atol=1e-6)
                np.testing.assert_allclose(np.einsum("ppqq->", dm2ab, optimize=True), nelecas[0]*nelecas[1], atol=1e-6)

    # Unit-Test-2:
    def test_complex_rdm12_and_rdm12s_py(self):
        ncas1 = 8
        ncas2 = 5
        nelecaslst1 = [(8,0), (7,1), (6,2), (5,3), (4,4), (3,5), (2,6), (1,7), (0,8)]
        nelecaslst2 = [(5,0), (4,1), (3,2), (2,3), (1,4), (0,5)]

        ncas1lst = [ncas1,]*len(nelecaslst1) 
        ncas2lst = [ncas2,]*len(nelecaslst2)

        cell = dummy_cell()
        for ncaslst, nelecaslst in zip([ncas1lst, ncas2lst], [nelecaslst1, nelecaslst2]):
            for ncas, nelecas in zip(ncaslst, nelecaslst):
                na = cistring.num_strings(ncas, nelecas[0])
                nb = cistring.num_strings(ncas, nelecas[1])

                # Generate a random FCIvector and normalize it.
                fcivec = np.random.rand(na*nb) + 1j * np.random.rand(na*nb)
                fcivec /= np.linalg.norm(fcivec)

                # Compute the RDMs using the complex implementation.
                cisolver = direct_spin1_cplx.FCISolver(cell)
                rdm1, rdm2 = cisolver.make_rdm12_py(fcivec, ncas, nelecas, reorder=True)
                (dm1a, dm1b), (dm2aa, dm2ab, dm2bb) = cisolver.make_rdm12s_py(fcivec, ncas, nelecas)
                rdm1_from_dm1sa, rdm1_from_dm1sb = cisolver.make_rdm1s_py(fcivec, ncas, nelecas, link_index=None)
                rdm1_from_make_rdm1 = cisolver.make_rdm1_py(fcivec, ncas, nelecas, link_index=None)
                
                dm1 = dm1a + dm1b
                dm2 = dm2aa + dm2bb + dm2ab + dm2ab.transpose(2,3,0,1).conj()

                # Check the consistency of make_rdm1s_py and make_rdm12s_py
                np.testing.assert_allclose(dm1a, rdm1_from_dm1sa, atol=1e-10, rtol=1e-10)
                np.testing.assert_allclose(dm1b, rdm1_from_dm1sb, atol=1e-10, rtol=1e-10)
                np.testing.assert_allclose(dm1, rdm1_from_make_rdm1, atol=1e-10, rtol=1e-10)
                
                # Compare the spin-summed RDMs constructed from spin-separated RDMs
                np.testing.assert_allclose(dm1.conj().T, rdm1, atol=1e-10, rtol=1e-10)
                np.testing.assert_allclose(dm2, rdm2, atol=1e-10, rtol=1e-10)
        
                # Compare the trace of 1-RDM with the number of electrons
                np.testing.assert_allclose(np.trace(dm1a), nelecas[0], atol=1e-10, rtol=1e-10)
                np.testing.assert_allclose(np.trace(dm1b), nelecas[1], atol=1e-10, rtol=1e-10)
                np.testing.assert_allclose(np.trace(rdm1), sum(nelecas), atol=1e-10, rtol=1e-10)

                # Check the hermiticity of RDMs
                np.testing.assert_allclose(dm1a, dm1a.conj().T, atol=1e-10, rtol=1e-10)
                np.testing.assert_allclose(dm1b, dm1b.conj().T, atol=1e-10, rtol=1e-10)
                np.testing.assert_allclose(rdm1, rdm1.conj().T, atol=1e-10, rtol=1e-10)
                np.testing.assert_allclose(dm2aa, dm2aa.transpose(1,0,3,2).conj(), atol=1e-10, rtol=1e-10)
                np.testing.assert_allclose(dm2bb, dm2bb.transpose(1,0,3,2).conj(), atol=1e-10, rtol=1e-10)
                np.testing.assert_allclose(dm2ab, dm2ab.transpose(1,0,3,2).conj(), atol=1e-10, rtol=1e-10)

                np.testing.assert_allclose(dm2aa - dm2aa.transpose(2,3,0,1), 0, atol=1e-8)
                np.testing.assert_allclose(dm2bb - dm2bb.transpose(2,3,0,1), 0, atol=1e-8)
                
                # Compare the trace of 2-RDM with the number of electron pairs
                np.testing.assert_allclose(np.einsum("ppqq->", dm2aa, optimize=True), nelecas[0]*(nelecas[0]-1), atol=1e-6)
                np.testing.assert_allclose(np.einsum("ppqq->", dm2bb, optimize=True), nelecas[1]*(nelecas[1]-1), atol=1e-6)
                np.testing.assert_allclose(np.einsum("ppqq->", dm2ab, optimize=True), nelecas[0]*nelecas[1], atol=1e-6)

    # Unit-Test-3:
    def test_complex_rdm12_and_rdm12s_C(self):
        pass

if __name__ == "__main__":
    print("Full Tests for RDM construction")
    unittest.main()