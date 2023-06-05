import numpy as np
from scipy import linalg
from pyscf import gto, scf, lib, mcscf
from pyscf.tools import molden
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.mcscf import lassi
from mrh.my_pyscf.tools.molden import from_lasscf
from c2h4n4_struct import structure as struct

mol = struct (0, 0, '6-31g')
mol.output = 'c2h4n4_631g_automatic_singles.log'
mol.verbose = lib.logger.INFO
mol.build ()
mf = scf.RHF (mol).run ()

las = LASSCF (mf, (3,3), ((2,1),(1,2)))
las = las.state_average ([0.5,0.5],
    spins=[[1,-1],[-1,1]],
    smults=[[2,2],[2,2]],    
    charges=[[0,0],[0,0]],
    wfnsyms=[[1,1],[1,1]])
mo = las.sort_mo ([16,18,22,23,24,26])
mo = las.localize_init_guess ((list (range (5)), list (range (5,10))), mo)
las.kernel (mo)

mc = mcscf.CASCI (mf, 6, 6).set (fcisolver=csf_solver(mol,smult=1))
mc.kernel (las.mo_coeff)

print ("LASSCF((3,3),(3,3)) energy =", las.e_tot)
print ("CASCI(6,6) energy =", mc.e_tot)

from mrh.my_pyscf.mcscf.lassi_states import all_single_excitations
las = all_single_excitations (las)
las.lasci () # Optimize the CI vectors
las.dump_states () # prints all state tables in the output file
e_roots, si = las.lassi ()

print ("LASSI(S) energy =", e_roots[0])


