#!/usr/bin/env python

"""
Example script for the k-LAS-PDFT.
"""

import numpy as np

from pyscf import lib
from pyscf.pbc import gto, scf

from mrh.my_pyscf.pbc import mcscf, mcpdft
from mrh.my_pyscf.pbc.mcscf import avas


cell = gto.Cell()
cell.a = np.diag([2.47, 17.5, 17.5])
cell.atom = [
    ("C", -0.5892731038,  0.3262391909,  0.0),
    ("H", -0.5866101958,  1.4126530287,  0.0),
    ("C",  0.5916281105, -0.3261693897,  0.0),
    ("H",  0.5889652025, -1.4125832275,  0.0)]
cell.basis = "GTH-DZVP"
cell.pseudo = "GTH-PADE"
cell.unit = "Angstrom"
cell.precision = 1e-10
cell.verbose = lib.logger.INFO
cell.build()

kmesh = (3, 1, 1)
kpts = cell.make_kpts(kmesh, wrap_around=True)
kmf = scf.KRHF(cell, kpts=kpts).density_fit()
kmf.exxdiv = None
kmf.conv_tol = 1e-10
kmf.kernel()

active_labels = ["C 2pz"]
mo_avas = avas.kernel(kmf, active_labels, minao=cell.basis)[2]

klas = mcscf.KLASSCF(kmf, ncas=2, nelecas=(1, 1), kmesh=kmesh)
mo_guess = klas.localize_init_guess(active_labels, mo_coeff=mo_avas)
klas.max_cycle_macro = 100
klas.kernel(mo_coeff=mo_guess)

klaspdft = mcpdft.KLASSCF(klas, "tPBE")
klaspdft.kernel()

kcas = mcpdft.KCASCI(kmf, 'tPBE', ncas=2, nelecas=(1, 1))
kcas.kpts = kpts
kcas.kmesh = kmesh
kcas.kernel(mo_coeff=klas.mo_coeff,)

print(f"k-RHF energy                         : {kmf.e_tot.real: .12f}")
print(f"k-LASSCF energy                      : {klaspdft.e_mcscf.real: .12f}")
print(f"k-CAS (in k-LASSCF orb.) energy      : {kcas.e_mcscf.real: .12f}")
print(f"k-LAS-PDFT energy                    : {klaspdft.e_tot.real: .12f}")
print(f"k-CAS-PDFT (in k-LASSCF orb.) energy : {kcas.e_tot.real: .12f}")
