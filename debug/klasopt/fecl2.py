import sys
import os
import numpy as np
from pyscf import lib
from pyscf.pbc import scf, df
from pyscf.pbc import gto

from mrh.my_pyscf.pbc.mcscf import avas
from mrh.my_pyscf.pbc import mcscf
from mrh.my_pyscf.pbc.util import pbcmolden

from pyscf.pbc.gto.cell import fromfile

a, atom = fromfile('FeCl2.vasp')

cell = gto.Cell()
cell.a = a
cell.atom = atom
cell.basis = {
    'Fe': 'gth-dzvp-molopt-sr',
    'Cl': 'gth-dzvp',
}
cell.pseudo = 'gth-pade'
cell.verbose = 4
cell.spin=4
cell.max_memory=100000

nx= 1 #int(sys.argv[1])
ny= 1 #int(sys.argv[2])
cell.output = f'fecl2.{nx}{ny}.out'
cell.build()

kmesh2D = [nx, ny, 1]
kpts = cell.make_kpts(kmesh2D, wrap_around=True)

kmf = scf.KRHF(cell, exxdiv=None, kpts=kpts).density_fit().newton()
kmf.with_df._cderi = f'FeCl2POSCAR.{nx}x{ny}.cderi'
kmf.max_cycle=100
kmf.chkfile = f'FeCl2POSCAR.{nx}x{ny}.chk'
kmf.init_guess = 'chk'
kmf.exxdiv = None
kmf.conv_tol = 1e-10
kmf.kernel()

ncas, nelecas, mo_coeff = avas.kernel(kmf, ['Fe 3d',], minao=cell.basis, threshold=0.50)[:3]

nk = nx * ny
if nk==1 and mo_coeff.ndim <= 2:
    mo_coeff = mo_coeff[None, :, :].astype(np.complex128)

klas = mcscf.KLASSCF(kmf, ncas=5, nelecas=(5, 1), kmesh=kmesh2D)
mo_guess = klas.localize_init_guess(
    ['Fe 3d'], mo_coeff=mo_coeff,)
klas.max_cycle_macro = 100
klas.kernel(mo_coeff=mo_guess,)[0]

np.save(f'FeCl2POSCAR.{nx}x{ny}.npy', klas.mo_coeff)

print(f"k-RHF energy                        : {kmf.e_tot.real: .12f}")
print(f"k-LASSCF energy                     : {klas.e_tot.real: .12f}")

pbcmolden.from_mo(kmf, f'fecl2_{nx}x{ny}.molden', mo_coeff=klas.mo_coeff,
                  kmesh=kmesh2D, wannier=True, only_active=True,
                  ncore=klas.ncore, ncas=klas.ncas)

