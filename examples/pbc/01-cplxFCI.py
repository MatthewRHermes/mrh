# !/usr/bin/env python
import numpy as np

from pyscf import gto, fci, scf, lib, ao2mo
from mrh.my_pyscf.pbc.fci import direct_spin1_cplx


# An example to run the complex FCI solver.
mol = gto.Mole(atom='H 0 0 0; F 0 0 1.1', 
               basis='STO-6G',
               spin = 0,
               charge = 0,
               verbose=lib.logger.DEBUG)
mol.build ()

mf = scf.RHF (mol)
mf.kernel()

h0 = mf.energy_nuc ()
h1 = mf.mo_coeff.conj ().T @ mf.get_hcore () @ mf.mo_coeff
h2 = ao2mo.restore (1, ao2mo.full (mf._eri, mf.mo_coeff), mol.nao_nr ())

# Defining the size of FCI problem
norb = mol.nao_nr ()
neleca = mol.nelectron // 2
nelecb = mol.nelectron - neleca
nelec = (neleca, nelecb)

# Real integrals based FCI solver
cisolver = fci.FCI(mol)
e_real, ci_real = cisolver.kernel (h1, h2, norb, nelec, ecore=h0)
e_real_check = cisolver.energy (h1, h2, ci_real, norb, nelec) + h0

assert abs(e_real - e_real_check) < 1e-10, \
    "There is some problem, these two energies should be the same!"

# Complex integrals based FCI solver
noise = 1e-3
h0cplx = h0 + noise * 1j   

h1cplx = h1 + noise * 1j
h1cplx = 0.5*(h1cplx + h1cplx.T.conj())

h2cplx = h2 + noise * 1j
h2cplx = 0.5*(h2cplx + h2cplx.transpose(2, 3, 0, 1))
h2cplx = 0.5*(h2cplx + h2cplx.transpose(1, 0, 3, 2).conj())
h2cplx = 0.5*(h2cplx + h2cplx.transpose(3, 2, 1, 0).conj())

cisolver_cplx = direct_spin1_cplx.FCISolver(mol)
e_cplx, ci_cplx = cisolver_cplx.kernel (h1cplx, h2cplx, norb, nelec, ecore=h0cplx)
e_cplx_check = cisolver_cplx.energy (h1cplx, h2cplx, ci_cplx, norb, nelec) + h0cplx
assert abs(e_cplx.real - e_cplx_check.real) < 1e-10, \
    "There is some problem, these two energies should be the same!"

# Comparison time:
print(f"Real FCI Energy     : {e_real:.12f}")
print(f"Complex FCI Energy  : {e_cplx.real:.12f}")