#!/usr/bin/env python

"""Evaluate kLAS-PDFT on top of a completed periodic LASSCF state.

The PDFT constructor is intake-only: ``mcpdft.KLASSCF`` requires an existing
``mcscf.KLASSCF`` object and does not rerun its orbital or CI optimization.
The kLAS product-state RDMs remain in their Wannier basis until the matching
``mo_phase`` transforms them into the k blocks used by periodic MC-PDFT.

The initial implementation supports one neutral, non-translation-packed kLAS
root.  LASSI, state averaging, gradients, and translation-packed CI are outside
the present scope.
"""

import numpy as np

from pyscf import lib
from pyscf.pbc import gto, scf

from mrh.my_pyscf.pbc import mcscf, mcpdft
from mrh.my_pyscf.pbc.mcscf import avas


cell = gto.Cell()
cell.a = np.diag([4.0, 10.0, 10.0])
cell.atom = "H 0.0 0.0 0.0; H 1.5 0.0 0.0"
cell.basis = "sto-3g"
cell.unit = "Angstrom"
cell.precision = 1e-9
cell.ke_cutoff = 20
cell.verbose = lib.logger.INFO
cell.build()

kmesh = (2, 1, 1)
kpts = cell.make_kpts(kmesh, wrap_around=True)
kmf = scf.KRHF(cell, kpts=kpts).density_fit()
kmf.exxdiv = None
kmf.conv_tol = 1e-10
kmf.kernel()

active_labels = ["H 1s"]
mo_avas = avas.kernel(kmf, active_labels, minao=cell.basis)[2]

klas = mcscf.KLASSCF(
    kmf,
    ncas=2,
    nelecas=(1, 1),
    kmesh=kmesh,
    trans_sym=False,
)
mo_guess = klas.localize_init_guess(active_labels, mo_coeff=mo_avas)
klas.conv_tol_grad = 1e-5
klas.max_cycle_macro = 20
klas.max_cycle_micro = 5
klas.kernel(mo_coeff=mo_guess)

# Promote the completed k-LASSCF state to a fixed-wavefunction PDFT evaluator.
pdft = mcpdft.KLASSCF(klas, "tPBE", grids_level=2)
pdft.kernel()

print(f"k-RHF energy       : {kmf.e_tot.real: .12f}")
print(f"k-LASSCF energy    : {pdft.e_mcscf.real: .12f}")
print(f"on-top energy      : {pdft.e_ot.real: .12f}")
print(f"k-LAS-PDFT energy  : {pdft.e_tot.real: .12f}")
