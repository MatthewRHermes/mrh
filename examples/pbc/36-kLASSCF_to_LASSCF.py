"""
Compare k-LASSCF and molecular LASSCF energies in Ha per primitive cell.

Also, transform the k-LASSCF object to a LASSCF object on the same Hamiltonian.
The molecular solver uses the same periodic Hamiltonian, expressed in supercell
AO coordinates, not recomputed supercell integrals.
"""

import numpy as np
from pyscf import lib
from pyscf.pbc import gto, scf

from mrh.my_pyscf.pbc import mcscf
from mrh.my_pyscf.pbc.mcscf import avas
from mrh.my_pyscf.pbc.util.klas_to_las import unpack_klas

nk = 3

cell = gto.Cell()
cell.a = np.diag([4.0, 10.0, 10.0])
cell.atom = "H 0.0 0.0 0.0; H 1.5 0.0 0.0"
cell.basis = "6-31G"
cell.unit = "Angstrom"
cell.precision = 1e-10
cell.output = f"H2Chain_LASSCF_{nk}.out"
cell.verbose = lib.logger.INFO
cell.build()

kmesh = (nk, 1, 1)
kmf = scf.KRHF(cell, kpts=cell.make_kpts(kmesh, wrap_around=True)).density_fit()
kmf.exxdiv = None
kmf.conv_tol = 1e-10
kmf.kernel()

active_labels = ["H 1s"]
mo_avas = avas.kernel(kmf, active_labels, minao=cell.basis)[2]
klas = mcscf.KLASSCF(kmf, ncas=2, nelecas=(1, 1), kmesh=kmesh)
mo_guess = klas.localize_init_guess(active_labels, mo_coeff=mo_avas,
                                    stabilize_virtuals=True)
klas.max_cycle_macro = 100
klas.kernel(mo_coeff=mo_guess)

# Transform the k-LASSCF object to a LASSCF object on the same Hamiltonian.
# Note that 2e integrals are materialized in memory; their storage grows as (nk * nao)**4.
mo_coeff, hamiltonian, ci, las = unpack_klas(klas)
las.kernel(mo_coeff=mo_coeff, ci0=ci)

# Molecular LASSCF energy is supercell energy; k-LASSCF energy is per cell.
e_klas = float(klas.e_tot.real)
e_las = float(las.e_tot) / nk

print(f"k-HF energy (per cell)      :  {float(kmf.e_tot.real)}")
print(f"k-LASSCF energy (per cell) : {e_klas: .12f}")
print(f"LASSCF energy (per cell)   : {e_las: .12f}")

