import numpy as np

from pyscf import lib
from pyscf.lib import misc
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf

from mrh.my_pyscf.pbc import mcscf
from mrh.my_pyscf.pbc.mcscf import avas

# Disable asynchronous prefetch it slows down the J/K calls.
misc.ASYNC_IO = False

'''
Computing the H-Cube Band Structure:
I have taken the BCC unit cell with 2-H atoms.
Currently the lattice vector is 2.794 A and the H-H distance
is sqrt(3)/2 * laticvec
'''

a = 2.794

cell = pgto.Cell()
cell.a = np.eye(3) * (a * 1.1)
cell.atom = f'''
H  0.0    0.0    0.0
H  {a/2}  {a/2}  {a/2}
'''
cell.basis = "CC-PVDZ"
cell.unit = "Angstrom"
cell.max_memory = 100000
cell.precision = 1e-12
cell.verbose = lib.logger.INFO
cell.build()

kmesh = [2, 2, 2]
nx, ny, nz = kmesh
kpts = cell.make_kpts(kmesh, wrap_around=True)

for ik, k in enumerate(cell.get_scaled_kpts(kpts)):
    print(f"{ik:2d}  {k[0]:8.4f} {k[1]:8.4f} {k[2]:8.4f}")

nkpts = len(kpts)

kmf = scf.KRHF(cell, kpts=kpts, exxdiv=None).density_fit().newton()
kmf.exxdiv = None
kmf.conv_tol = 1e-10
kmf.kernel()

print(f"k-RHF energy: {kmf.e_tot.real:12.8f}")

mo_coeff = avas.kernel(kmf, ['H 1s'], minao=cell.basis)[2]
mo_coeff = np.array(mo_coeff).copy()

klas = mcscf.KLASSCF(kmf, ncas=2, nelecas=(1, 1), kmesh=kmesh)
mo_guess = klas.localize_init_guess(
    ['H 1s'], mo_coeff=mo_coeff,

)
klas.max_cycle_macro = 100
klas.kernel(mo_coeff=mo_guess,)[0]

print(f"k-RHF energy                        : {kmf.e_tot.real: .12f}")
print(f"k-LASSCF energy                     : {klas.e_tot.real: .12f}")

np.save(f'HCube.{nx}x{ny}x{nz}.npy', klas.mo_coeff)

from mrh.my_pyscf.pbc.util import pbcmolden
pbcmolden.from_mo(kmf, f'HCube{nx}x{ny}x{nz}.molden', mo_coeff=klas.mo_coeff,
                  kmesh=kmesh, wannier=True, only_active=True,
                  ncore=klas.ncore, ncas=klas.ncas)

