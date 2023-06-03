import sys
from pyscf import gto, scf, tools, dft, lib
from pyscf.tools import molden
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_dmet import localintegrals, dmet, fragments
from mrh.my_dmet.fragments import make_fragment_atom_list, make_fragment_orb_list
from mrh.my_pyscf import mcpdft

# Initialization
mol = gto.M()
mol.atom='C2.xyz'
mol.basis='sto3g'
mol.spin = 0
mol.charge = 0
mol.verbose = 5
mol.output = 'C2.log'
mol.build()

# Mean field calculation
mf = scf.ROHF(mol).newton().run()

# Option-1: Perform LASSCF and then pass las object

# LASSCF Calculations
#las = LASSCF(mf,(2, 2),(2, 2),spin_sub=(1, 1))
#frag_atom_list = ([1, 6] , [2, 5]) 
#mo0 = las.localize_init_guess(frag_atom_list)
#elas = las.kernel(mo0)[0]

# LAS-PDFT
#mc = mcpdft.LASSCF(las, 'tPBE', (2, 2), (2,2))
#epdft = mc.kernel()[0]

# Option-2: Feed the mean field object and fragment information to mcpdft.LASSCF
frag_atom_list = ([1, 6] , [2, 5])
mc = mcpdft.LASSCF(mf, 'tPBE', (2, 2), (2, 2), spin_sub=(1,1))
mo0 = mc.localize_init_guess (frag_atom_list)
mc.kernel(mo0)

elas = mc.e_mcscf[0]
epdft = mc.e_tot

print ("E(LASSCF) =", elas)
print ("E(tPBE) =", epdft)
