import sys
import copy
import numpy as np
import h5py
from scipy import linalg
from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.mcscf.lasscf_sync_o0 import LASSCF
from mrh.my_pyscf.tools import molden
from mrh.tests.lasscf.c2h4n4_struct import structure as struct


lib.logger.TIMER_LEVEL = lib.logger.INFO
mol = struct (2.0, 2.0, '6-31g')
mol.output = 'approx_lassis.log'
mol.verbose = lib.logger.INFO
mol.spin = 8
mol.build ()
mf = scf.RHF (mol).run ()

las = LASSCF (mf, (4,2,4), ((2,2),(1,1),(2,2)), spin_sub=(1,1,1))
mo_coeff = las.localize_init_guess ([[0,1,2],[3,4,5,6],[7,8,9]])
las.kernel (mo_coeff)
mo_coeff = las.mo_coeff

mc = mcscf.CASCI (mf, 10, (5,5)).set (fcisolver=csf_solver(mol,smult=1))
mc.kernel (mo_coeff)

sys.stderr.flush ()
print ("LASSCF((4,4),(2,2),(4,4)) energy =", las.e_tot)
print ("CASCI(10,10) energy =", mc.e_tot, flush=True)

from mrh.my_pyscf.lassi.lassis import LASSIS
lsi = LASSIS (las).run ()
print ("LASSIS energy =", lsi.e_roots[0], "ndim =", lsi.si.shape[1],
       ("NOT converged","converged")[int(lsi.converged)], flush=True)

# Omit spin flips by setting nspin=0

lsi = LASSIS (las).run (nspin=0)
print ("LASSIS energy (no spin flips) =", lsi.e_roots[0], "ndim =", lsi.si.shape[1],
       ("NOT converged","converged")[int(lsi.converged)], flush=True)

# Pass a mask array to prevent charge hops between distant fragments

mask = np.array ([[True, True, False],
                  [True, True, True],
                  [False, True, True]], dtype=bool)

lsi = LASSIS (las).run (mask_charge_hops=mask)
print ("LASSIS energy (nearest-neighbor charge hops) =", lsi.e_roots[0], "ndim =", lsi.si.shape[1],
       ("NOT converged","converged")[int(lsi.converged)], flush=True)

lsi = LASSIS (las).run (nspin=0, mask_charge_hops=mask)
print ("LASSIS energy (nearest-neighbor charge hops and no spin flips) =", lsi.e_roots[0],
       "ndim =", lsi.si.shape[1], ("NOT converged","converged")[int(lsi.converged)], flush=True)


