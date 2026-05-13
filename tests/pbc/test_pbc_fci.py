#!/bin/bash
import unittest
import numpy as np

from functools import reduce

from pyscf import ao2mo, fci
from pyscf.pbc import gto as pgto, scf
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.tools import k2gamma

from mrh.my_pyscf.pbc.fci import direct_spin1_cplx
from mrh.my_pyscf.pbc.mcscf.k2R import get_mo_coeff_k2R


'''
Unit tests for the complex FCI implementation in the PBC context.
# Test-1: Construct the random but hermitian h1, h2 and use that to compute the FCI energy using real FCI code 
#         and complex FCI code. I am expecting for the smaller imag part both the energies should match.
# Test-2: For a given kmf object compute the FCI energy using the complex FCI code and compare it to the 
#         CASCI energy computed using the real FCI code.
# Test-3: Compare different components of cplx FCI code, like energy function, contract2e, exact vs davidson solver 
#         spin-op etc.
# Test-4: Compare the FCI energies for different spin states, like singlet, triplet, quintet etc.
#         Also, compared the -ms and +ms states for the triplet. I am expecting them to be degenerate.
'''

def make_h2_2D_numpy(intraH=1.0, interH=1.5, nx=1, ny=1, vacuum=17.5):
    """
    Build a 2D periodic H2 lattice.

    Returns
    -------
    symbols : list[str]
        Atomic symbols.
    positions : np.ndarray, shape (natm, 3)
        Atomic positions in Angstrom.
    cell : np.ndarray, shape (3, 3)
        Lattice vectors in Angstrom.
    """
    period = intraH + interH

    ax = nx * period
    by = ny * period
    cz = vacuum

    cell = np.diag([ax, by, cz])

    positions = []
    symbols = []

    for ix in range(nx):
        for iy in range(ny):
            x0 = ix * period
            y0 = iy * period
            z0 = vacuum / 2.0

            positions.append([x0, y0, z0])
            positions.append([x0 + intraH, y0, z0])

            symbols += ["H", "H"]

    positions = np.array(positions, dtype=float)
    min_pos = positions.min(axis=0)
    max_pos = positions.max(axis=0)
    current_center = 0.5 * (min_pos + max_pos)
    target_center = 0.5 * np.array([ax, by, cz])

    shift = target_center - current_center
    positions += shift

    return symbols, positions, cell

def get_cell(basis="STO-6G", pseudo=None, maxMem=1000, verbose=0):
    symbols, positions, lattice = make_h2_2D_numpy( intraH=0.74, interH=1.5, nx=1, ny=1, vacuum=17.5)

    cell = pgto.Cell()
    cell.a = lattice
    cell.atom = [(symbols[i], tuple(positions[i])) 
                 for i in range(len(symbols))]
    cell.basis = basis
    cell.unit = "Angstrom"
    cell.max_memory = maxMem
    cell.ke_cutoff = 100
    cell.pseudo = pseudo
    cell.precision = 1e-10
    cell.verbose = verbose
    cell.build()
    return cell

def _basis_transformation(operator, mo):
    return reduce(np.dot, (mo.conj().T, operator, mo))

def gen_hermi_ham(norb):
    np.random.seed(12)
    h1 = np.random.random((norb,norb))
    h2 = np.random.random((norb,norb,norb,norb))

    h1 = np.asarray(h1).astype(np.complex128)
    h2 = np.asarray(h2).astype(np.complex128)

    h1.imag += 1e-4
    h2.imag += 1e-4

    # Restore symmetries
    h1 = h1 + h1.conj().T
    h2 = h2 + h2.conj().transpose(1,0,2,3)
    h2 = h2 + h2.conj().transpose(0,1,3,2)
    h2 = h2 + h2.conj().transpose(2,3,0,1)
    return h1, h2

class KnownValues(unittest.TestCase):

    # Test-1
    def test_cfci_vanilla(self):
        # Doing the calculation of (4o, 4e)
        norb = 4
        nelecas = (2, 2)
        h1, h2 = gen_hermi_ham(norb)
        # FCI from PySCF  
        cisolver = fci.direct_spin1.FCI()
        e = cisolver.kernel(h1.real, h2.real, norb, nelecas)[0]
        # FCI from complex_com_real
        cisolver = direct_spin1_cplx.FCI()
        e_com = cisolver.kernel(h1, h2, norb, nelecas)[0]
        msg = "FCI energies from vanilla and complex FCI codes do not match."
        self.assertAlmostEqual(e, e_com, places=8, msg=msg)

    # Test-2
    def test_cfci_fci(self):
        cell = get_cell()
        kmesh = (2, 2, 1)
        kpts = cell.make_kpts(kmesh)
        nkpts = len(kpts)

        kmf = scf.KRHF(cell, kpts=kpts).density_fit(auxbasis='def2-svp-jkfit')
        kmf.exxdiv = None
        kmf.conv_tol = 1e-10
        kmf.kernel()

        # Preparing the integrals for FCI        
        ncas = cell.nao_nr()
        nelecas = (1, 1)
        mo_coeff = kmf.mo_coeff
        
        scell, phase, mo_coeff_R, mo_phase = get_mo_coeff_k2R(kmf, mo_coeff, 0, ncas)
        kconserv = kpts_helper.get_kconserv(kmf.cell, kmf.kpts)
        mo_ks = mo_phase[kconserv]
        dtype = mo_coeff[0].dtype

        # h0e
        h0 = 0 + 0j
        # h1e
        h1ao_k = kmf.get_hcore()
        coredm_kpts = np.asarray([2.0 * (mo_coeff[k] @ mo_coeff[k].conj().T) 
                                  for k in range(nkpts)], dtype=dtype)
        corevhf_kpts = kmf.get_veff(cell, coredm_kpts, hermi=1)
        h1ao_k += corevhf_kpts
        h1ao_R = np.einsum('Rk,kij,Sk->RiSj', phase, h1ao_k, phase.conj())
        h1ao_R = h1ao_R.reshape(nkpts*ncas, nkpts*ncas)
        h1e = _basis_transformation(h1ao_R, mo_coeff_R)

        # h2e
        eri_k = kmf.with_df.ao2mo_7d(np.asarray(mo_coeff), kpts=kpts)
        eris = np.einsum('auR,bvS,abcuvwt,cwT,abctU->RSTU',
                         mo_phase.conj(), mo_phase, eri_k, mo_phase.conj(), mo_ks, optimize=True)
        eris *= 1/nkpts
        h2e = eris

        # Time to run complex FCI
        cisolver = direct_spin1_cplx.FCI()
        eci, civec = cisolver.kernel(h1e, h2e, ncas*nkpts, (nelecas[0]*nkpts, nelecas[1]*nkpts), ecore=h0)

        # Also compute this energy using energy function
        eci_energy_func = cisolver.energy(h1e, h2e, civec, ncas*nkpts, (nelecas[0]*nkpts, nelecas[1]*nkpts))

        h1e = h2e = eris = eris_k = h1ao_k = coredm_kpts = corevhf_kpts = mo_coeff_R = mo_phase = mo_ks = None
        civec = None

        # Generate the reference values
        mf = k2gamma.k2gamma(kmf, kmesh)
        dm0 = mf.make_rdm1()

        mf = scf.RHF(scell).density_fit(auxbasis='def2-svp-jkfit')
        mf.conv_tol = 1e-10
        mf.exxdiv = None
        mf.kernel(dm0)

        h0 = 0
        h1 = mf.get_hcore()
        coredm = 2.0 * (mf.mo_coeff @ mf.mo_coeff.conj().T)
        corevhf = mf.get_veff(cell, coredm, hermi=1)
        h1 += corevhf
        h1 = _basis_transformation(h1, mf.mo_coeff)

        h2 = mf.with_df.ao2mo(mf.mo_coeff)
        h2 = ao2mo.restore(1, h2, mf.mo_coeff.shape[1])
        h2 = h2.reshape(ncas*nkpts, ncas*nkpts, ncas*nkpts, ncas*nkpts)

        cisolver_gamma = fci.direct_spin1.FCI()
        eciref, civec = cisolver_gamma.kernel(h1, h2, ncas*nkpts, (nelecas[0]*nkpts, nelecas[1]*nkpts), ecore=h0)
        eciref_energy_func = cisolver_gamma.energy(h1, h2, civec, ncas*nkpts, (nelecas[0]*nkpts, nelecas[1]*nkpts))

        msg = "FCI energies from complex FCI and gamma-point FCI codes do not match."
        self.assertAlmostEqual(eciref, eci, places=6, msg=msg)
        msg = "FCI energies from complex FCI and gamma-point FCI codes do not match with energy function."
        self.assertAlmostEqual(eciref_energy_func, eci_energy_func, places=6, msg=msg)

        h0 = h1 = h2 = coredm = corevhf = civec = None

    def test_cfci_solver(self):
        norb = 4
        nelecas = (2, 2)
        h1, h2 = gen_hermi_ham(norb)

        # Run the exact diagonalization vs Davidson diagonalization
        cisolver = direct_spin1_cplx.FCI()
        cisolver.davidson_only = False
        eci1 = cisolver.kernel(h1, h2, norb, nelecas)[0]

        # Run the Davidson diagonalization
        cisolver2 = direct_spin1_cplx.FCI()
        cisolver2.davidson_only = True
        eci2 = cisolver2.kernel(h1, h2, norb, nelecas)[0]

        msg = "FCI energies from exact diagonalization and Davidson diagonalization do not match."
        self.assertAlmostEqual(eci1, eci2, places=7, msg=msg)

    def test_cfci_solver_with_diff_spins(self):
        norb = 4
        nelecas = (2, 2)
        h1, h2 = gen_hermi_ham(norb)

        # Run the exact diagonalization vs Davidson diagonalization
        cisolver = direct_spin1_cplx.FCI()
        cisolver.davidson_only = False
        eci1 = cisolver.kernel(h1, h2, norb, nelecas)[0]

        # Run the Davidson diagonalization
        cisolver2 = direct_spin1_cplx.FCI()
        cisolver2.davidson_only = True
        eci2 = cisolver2.kernel(h1, h2, norb, nelecas)[0]

        msg = "FCI energies from exact diagonalization and Davidson diagonalization do not match."
        self.assertAlmostEqual(eci1, eci2, places=7, msg=msg)

    def test_cfci_solver_with_diff_spins(self):
        norb = 4
        h1, h2 = gen_hermi_ham(norb)

        def run_solver(h1, h2, norb, nelecas, davidson_only=False):
            cisolver = direct_spin1_cplx.FCI()
            cisolver.davidson_only = davidson_only
            return cisolver.kernel(h1, h2, norb, nelecas)[0]
        
        eci0_sing = run_solver(h1, h2, norb, (2, 2), davidson_only=False)
        eci1_sing = run_solver(h1, h2, norb, (2, 2), davidson_only=True)
        
        eci0_trip = run_solver(h1, h2, norb, (3, 1), davidson_only=False)
        eci1_trip = run_solver(h1, h2, norb, (3, 1), davidson_only=True)

        eci0_quint = run_solver(h1, h2, norb, (4, 0), davidson_only=False)
        eci1_quint = run_solver(h1, h2, norb, (4, 0), davidson_only=True)

        # Let me also compare ms=-1 vs ms=1
        eci00_trip = run_solver(h1, h2, norb, (1, 3), davidson_only=False)
        eci01_trip = run_solver(h1, h2, norb, (1, 3), davidson_only=True)
        
        msg = "FCI energies from exact diagonalization and Davidson diagonalization do not match."
        self.assertAlmostEqual(eci0_sing, eci1_sing, places=7, msg=msg)
        self.assertAlmostEqual(eci0_trip, eci1_trip, places=7, msg=msg)
        self.assertAlmostEqual(eci0_quint, eci1_quint, places=7, msg=msg)
        self.assertAlmostEqual(eci0_trip, eci00_trip, places=7, msg=msg)
        self.assertAlmostEqual(eci1_trip, eci01_trip, places=7, msg=msg)

if __name__ == "__main__":
    unittest.main()