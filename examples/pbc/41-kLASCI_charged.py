#!/usr/bin/env python
"""Fixed-orbital neutral and reference-cell N-1/N+1 LAS product states.

The local charge is one electron over the whole k-mesh. These states are
references for subsequent state interaction, not momentum-resolved bands.
"""

import numpy as np
from pyscf import lib
from pyscf.pbc import gto, scf

from mrh.my_pyscf.pbc import mcscf
from mrh.my_pyscf.pbc.mcscf import avas


# Build a periodic H2 chain and a neutral mean-field reference.
cell = gto.Cell()
cell.a = np.diag([3.0, 10.0, 10.0])
cell.atom = "H 0 0 0; H 0.74 0 0"
cell.basis = "6-31g"
cell.unit = "Angstrom"
cell.precision = 1e-8
cell.ke_cutoff = 30
cell.verbose = lib.logger.INFO
cell.build()

kmesh = (3, 1, 1)
kpts = cell.make_kpts(kmesh, wrap_around=True)
kmf = scf.KRHF(cell, kpts=kpts).density_fit()
kmf.exxdiv = None
kmf.conv_tol = 1e-10
kmf.kernel()
if not kmf.converged:
    raise RuntimeError("Neutral KRHF did not converge")

# Use one common localized orbital set for all three electron sectors.
neutral = mcscf.KLASCI(kmf, 2, (1, 1), kmesh=kmesh)
mo_avas = np.asarray(avas.kernel(kmf, ["H 1s"], minao=cell.basis)[2])
mo_avas = mo_avas.reshape(len(kpts), cell.nao_nr(), -1)
mo = neutral.localize_init_guess(["H 1s"], mo_coeff=mo_avas)

# Positive charge removes an electron; negative charge adds one on ref_cell.
ref_cell = 0
hole = mcscf.KLASCI(kmf, 2, (1, 1), kmesh=kmesh, charge=+1,
                    ref_cell=ref_cell, charged_spin=1, charged_spin_mult=2)
particle = mcscf.KLASCI(kmf, 2, (1, 1), kmesh=kmesh, charge=-1,
                        ref_cell=ref_cell, charged_spin=1, charged_spin_mult=2)

# Each solver builds its own CI guess; neutral CI has the wrong charged shape.
states = {0: neutral, +1: hole, -1: particle}
for charge, las in states.items():
    las.conv_tol_grad = 1e-7
    las.conv_tol_self = 1e-11
    for box in las.fciboxes:
        box.fcisolvers[0].conv_tol = 1e-12
    las.kernel(mo.copy())
    if not las.converged:
        raise RuntimeError(f"LASCI did not converge for charge {charge:+d}")
    print(f"Charge {charge:+d}: E/cell = {las.e_tot.real:.12f} Ha")
    print("  Fragment (N_alpha, N_beta):", las.get_nelec_frs()[:, 0].tolist())

# Convert per-cell energies to full-mesh removal/addition energy differences.
nkpts = len(kpts)
ip = nkpts * (hole.e_tot - neutral.e_tot).real
ea = nkpts * (neutral.e_tot - particle.e_tot).real
print(f"Localized-state IP: {ip:.12f} Ha")
print(f"Localized-state EA: {ea:.12f} Ha")

# states[q].ci and get_nelec_frs() retain the inputs for later LAS state interaction.
