#!/usr/bin/env python
import numpy as np

from pyscf import lib
from pyscf.pbc import gto, scf

from mrh.my_pyscf.pbc import mcpdft

# Author: Bhavnesh Jangid

"""Compute momentum-resolved kCAS-PDFT quasiparticle bands."""

cell = gto.Cell()
cell.a = np.diag([2.24, 2.24, 12.0])
cell.atom = [
    ["H", (0.0, 0.0, 6.0)],
    ["H", (1.1, 0.0, 6.0)],
]
cell.basis = "CC-PVDZ"
cell.unit = "Angstrom"
cell.precision = 1e-9
cell.verbose = lib.logger.INFO
cell.build()

kmesh = [5, 1, 1]
kpts = cell.make_kpts(kmesh, wrap_around=True)
assert len(kpts) <= 5
kmf = scf.KRHF(cell, kpts=kpts).density_fit()
kmf.exxdiv = None
kmf.kernel()

mo_coeff = np.asarray(kmf.mo_coeff)
ncas = 2
nelecas = 2
reference_target_k = 0


def run_kcas_pdft(**kwargs):
    """Run one neutral sector or a charged-sector sweep."""
    mc = mcpdft.KCASCI(kmf, "tPBE", ncas, nelecas, **kwargs,)
    mc.kmesh = kmesh
    mc.canonicalization = False
    mc.kernel(mo_coeff)
    return mc


# Neutral reference and all N-1/N+1 total-momentum sectors.
neutral = run_kcas_pdft(target_k=reference_target_k)
hole = run_kcas_pdft(charge=1)
particle = run_kcas_pdft(charge=-1)

hole_bands = hole.band_energies(neutral.e_tot, reference_target_k=reference_target_k,)
particle_bands = particle.band_energies(neutral.e_tot, reference_target_k=reference_target_k)


hole_by_k = {band["momentum_index"]: band["energy"].real
             for band in hole_bands}

particle_by_k = {band["momentum_index"]: band["energy"].real
                 for band in particle_bands}

print(f"Neutral kCASCI energy    : {neutral.e_mcscf.real:16.12f}")
print(f"Neutral k-MC-PDFT energy : {neutral.e_tot.real:16.12f}")
print("\nQuasiparticle poles (Hartree)")
print(" k       kx          ky          kz          hole        particle")

for k, kpt in enumerate(kpts):
    print(
        f"{k:2d}  {kpt[0]:10.6f}  {kpt[1]:10.6f}  {kpt[2]:10.6f}  "
        f"{hole_by_k[k]:12.8f}  {particle_by_k[k]:12.8f}"
    )

gap = min(particle_by_k.values()) - max(hole_by_k.values())
print(f"\nFundamental quasiparticle gap: {gap:12.8f} Hartree")
