"""
Example script for the k-LASSCF.
"""

import numpy as np

from pyscf import lib
from pyscf.pbc import gto, scf

from mrh.my_pyscf.pbc import mcscf
from mrh.my_pyscf.pbc.mcscf import avas


nk = 5

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

if np.prod(kmesh) == 1:
    mo_avas = mo_avas[None, :, :]

# Define the active space for the reference primitive cell only.
klas = mcscf.KLASSCF(kmf, ncas=2, nelecas=(1, 1), kmesh=kmesh,)
mo_guess = klas.localize_init_guess(active_labels, mo_coeff=mo_avas)
klas.conv_tol_grad = 1e-5
klas.max_cycle_macro = 100
klas.kernel(mo_coeff=mo_guess,)[0]

kcas = mcscf.KCASCI(kmf, ncas=2, nelecas=(1, 1),)
kcas.kpts = kpts
kcas.kmesh = kmesh
kcas.kernel(mo_coeff=klas.mo_coeff,)


print(f"k-RHF energy                        : {kmf.e_tot.real: .12f}")
print(f"k-LASSCF energy                     : {klas.e_tot.real: .12f}")
print(f"k-CAS (in k-LASSCF orb.) energy     : {kcas.e_tot.real: .12f}")