"""
Example script for the k-LASSCF.
"""

import sys
import numpy as np

from pyscf import lib
from pyscf.pbc import gto, scf

from mrh.my_pyscf.pbc import mcscf
from mrh.my_pyscf.pbc.util import pbcmolden


nk = 3 #int(sys.argv[1])  # number of k-points along the x direction

cell = gto.Cell()
cell.a = [[4.175, 0.0, 0.0],
          [0.0, 20.0, 0.0],
          [0.0, 0.0, 20.0]]
cell.atom = '''
Ni 2.73619  10.00000  10.00000
O 0.64880  10.00000  10.00000
O 2.73619  12.08739  10.00000
O 2.73619   7.91261  10.00000
O 2.73619  10.00000   7.91261
O 2.73619  10.00000  12.08739
H 2.73619  12.65763   9.21000
H 2.73619  12.65763  10.78999
H 1.94620  10.00000  12.65763
H 3.52619  10.00000  12.65763
H 2.73619   7.34236  10.78999
H 2.73619   7.34236   9.21000
H 1.94620   9.99999   7.34236
H 3.52619  10.00000   7.34236
'''
cell.basis = {'Ni': 'gth-szv-molopt-sr', 
              'default': 'gth-szv'}

cell.pseudo = 'gth-pade'
cell.max_memory = 100000
cell.ke_cutoff = 100
cell.precision = 1e-10
cell.verbose = lib.logger.INFO
cell.max_memory = 100000
cell.build()

# Choose the k-mesh
kmesh = [nk, 1, 1]

kpts = cell.make_kpts(kmesh, wrap_around=True)

# Use saved k-LASSCF orbitals directly, preserving their active-space ordering
# and Wannier gauge. An optional command-line argument selects another file.
mo_guess_file = sys.argv[1] if len(sys.argv) > 1 else f'niaqua_{nk}.mo_coeff.npy'
mo_guess = np.load(mo_guess_file, allow_pickle=False)
print(f"Loading k-LASSCF starting orbitals from {mo_guess_file}")

kmf = scf.KRHF(cell, kpts=kpts).density_fit()
kmf.max_cycle=1000
kmf.exxdiv = None
kmf.chkfile = f'niaqua_{nk}.chk'
kmf.init_guess = 'chk'
kmf.conv_tol = 1e-7
kmf.kernel()

expected_shape = np.asarray(kmf.mo_coeff).shape
if mo_guess.shape != expected_shape:
    raise ValueError(
        f"Saved orbitals have shape {mo_guess.shape}; expected {expected_shape}"
    )

# Define the active space for the reference primitive cell only.
las = mcscf.KLASSCF(kmf, ncas=5, nelecas=(4, 4), kmesh=kmesh,)
e_lasscf = las.kernel(mo_coeff=mo_guess,)[0]

# Preserve the starting-orbital file for reproducible restart comparisons.
np.save(f'niaqua_{nk}.restart.mo_coeff.npy', np.asarray(las.mo_coeff))

print(f"k-RHF energy       : {kmf.e_tot.real: .12f}")
print(f"k-LASSCF energy    : {e_lasscf.real: .12f}")
print(f"k-LASSCF converged : {las.converged}")

pbcmolden.from_mo(kmf, f'niaqua_{nk}.molden', mo_coeff=las.mo_coeff,
                  kmesh=kmesh, wannier=True, only_active=True,
                  ncore=las.ncore, ncas=las.ncas)
