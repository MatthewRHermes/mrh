#!/usr/bin/env python
"""
Example script to run the N-1/N+1 kLASCI calculation.
Note: N is total number of electrons in the active space for whole 
k-mesh. These states are references for subsequent state interaction.
"""

import numpy as np
from pyscf import lib
from pyscf.pbc import gto, scf

from mrh.my_pyscf.pbc import mcscf
from mrh.my_pyscf.pbc.mcscf import avas


cell = gto.Cell()
cell.a = np.diag([3.0, 10.0, 10.0])
cell.atom = "H 0 0 0; H 0.74 0 0"
cell.basis = "6-31g"
cell.unit = "Angstrom"
cell.precision = 1e-8
cell.verbose = lib.logger.INFO
cell.build()

kmesh = (3, 1, 1)
kpts = cell.make_kpts(kmesh, wrap_around=True)
kmf = scf.KRHF(cell, kpts=kpts).density_fit()
kmf.exxdiv = None
kmf.kernel()


mo_avas = avas.kernel(kmf, ["H 1s"], minao=cell.basis)[2]

if np.prod(kmesh) == 1:
    mo_avas = mo_avas[None, :, :]

# Optimize the neutral orbitals starting from localized AVAS orbitals.
klasscf = mcscf.KLASSCF(kmf, 2, (1, 1), kmesh=kmesh)
mo_guess = klasscf.localize_init_guess(["H 1s"], mo_coeff=mo_avas)
klasscf.kernel(mo_guess)

mo = klasscf.mo_coeff

# Use the optimized orbitals for all three fixed-orbital LASCI calculations.
neutral = mcscf.KLASCI(kmf, 2, (1, 1), kmesh=kmesh)
neutral.kernel(mo.copy())

# Positive charge removes an electron; negative charge adds one on ref_cell.
ref_cell = 0
hole = mcscf.KLASCI(kmf, 2, (1, 1), kmesh=kmesh, charge=+1,
                    ref_cell=ref_cell, charged_spin=1, charged_spin_mult=2)
hole.kernel(mo.copy())

particle = mcscf.KLASCI(kmf, 2, (1, 1), kmesh=kmesh, charge=-1,
                        ref_cell=ref_cell, charged_spin=1, charged_spin_mult=2)
particle.kernel(mo.copy())

# Convert per-cell energies to full-mesh electron removal/addition energy differences.
nkpts = len(kpts)
ip = nkpts * (hole.e_tot - neutral.e_tot).real
ea = nkpts * (neutral.e_tot - particle.e_tot).real


# For comparison, run kCASCI in the same orbital space (fixed orbitals, no
# localization). Neutral: N electrons in the zero-momentum sector.
ncas, nelecas = 2, 2
kcas_neutral = mcscf.KCASCI(kmf, ncas, nelecas, ncore=0, target_k=0)
kcas_neutral.kmesh = kmesh
kcas_neutral.canonicalization = False
e_kcas_neutral = kcas_neutral.kernel(mo.copy())[0]

# Charged kCASCI solves every total-momentum sector of N-1 / N+1.
kcas_hole = mcscf.KCASCI(kmf, ncas, nelecas, ncore=0, charge=+1)
kcas_hole.kmesh = kmesh
kcas_hole.canonicalization = False
kcas_hole.kernel(mo.copy())

kcas_particle = mcscf.KCASCI(kmf, ncas, nelecas, ncore=0, charge=-1)
kcas_particle.kmesh = kmesh
kcas_particle.canonicalization = False
kcas_particle.kernel(mo.copy())

# band_energies returns full-mesh poles: hole = E(N) - E(N-1), particle =
# E(N+1) - E(N). The lowest-energy states give IP = -max(hole poles) and
# EA = -min(particle poles).
hole_bands = kcas_hole.band_energies(e_kcas_neutral)
particle_bands = kcas_particle.band_energies(e_kcas_neutral)
kcas_ip = -max(b["energy"].real for b in hole_bands)
kcas_ea = -min(b["energy"].real for b in particle_bands)

kcas_hole.print_bands(e_kcas_neutral)
kcas_particle.print_bands(e_kcas_neutral)

print("")
print(f"Localized-state kLASCI IP: {ip:.12f} Ha")
print(f"kCASCI IP:                 {kcas_ip:.12f} Ha")
print("")
print(f"Localized-state kLASCI EA: {ea:.12f} Ha")
print(f"kCASCI EA:                 {kcas_ea:.12f} Ha")
