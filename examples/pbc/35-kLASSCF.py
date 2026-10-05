"""
Example script for the k-LASSCF.
"""

import sys
import numpy as np

from pyscf import lib
from pyscf.pbc import gto, scf

from mrh.my_pyscf.pbc import mcscf
from mrh.my_pyscf.pbc.mcscf import avas


nk = int(sys.argv[1])

cell = gto.Cell()
cell.a = np.diag([4.0, 10.0, 10.0])
cell.atom = "H 0.0 0.0 0.0; H 1.5 0.0 0.0"
cell.basis = "6-31G"
cell.unit = "Angstrom"
cell.precision = 1e-10
cell.output = f"H2Chain_{nk}.out"
cell.verbose = lib.logger.INFO
cell.build()

kmesh = (nk, 1, 1)
kpts = cell.make_kpts(kmesh, wrap_around=True)


kmf = scf.KRHF(cell, kpts=kpts).density_fit()
kmf.exxdiv = None
kmf.conv_tol = 1e-10
kmf.kernel()

active_labels = ["H 1s"]
mo_avas = avas.kernel(kmf, active_labels, minao=cell.basis)[2]

# Define the active space for the reference primitive cell only.
las = mcscf.KLASSCF(kmf, ncas=2, nelecas=(1, 1), kmesh=kmesh,)
mo_guess = las.localize_init_guess(active_labels, mo_coeff=mo_avas)
las.conv_tol_grad = 1e-5
las.max_cycle_macro = 100
e_lasscf, e_cas, ci, mo_coeff, mo_energy, h2eff, veff = las.kernel(
    mo_coeff=mo_guess,
)

print(f"k-RHF energy       : {kmf.e_tot.real: .12f}")
print(f"k-LASSCF energy    : {e_lasscf.real: .12f}")
print(f"k-LASSCF converged : {las.converged}")
