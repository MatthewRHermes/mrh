import sys
from pyscf import gto, scf, tools, dft
from pyscf.tools import molden
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_dmet import localintegrals, dmet, fragments
from mrh.my_dmet.fragments import make_fragment_atom_list, make_fragment_orb_list
from mrh.my_pyscf import mcpdft

# Initialisation
mol = gto.M()
mol.atom='C2.xyz'
mol.basis='sto3g'
mol.spin = 0
mol.charge = 0
mol.verbose = 5
mol.output = 'C2.log'
# MRH: you should generally specify an output file if verbose is set higher than 3
# just for cleanliness!
mol.build()

# Performing the mean field calculation
mf = scf.ROHF(mol)
mf.newton()
mf.run()

# Set up the localized AO basis
las = LASSCF(mf,(2, 2),(2, 2),spin_sub=(1, 1))
las.verbose = 5
frag_atom_list = ([1, 6] , [2, 5]) 
mo0 = las.localize_init_guess(frag_atom_list)
molden.from_mo(mol, f'C2.{mol.spin}.molden', mo0)
las.set(ah_level_shift=1e-5)
elas = las.kernel(mo0)[0]

# LAS-PDFT
mc = mcpdft.LASSCF(las, 'tPBE', 4, 4, verbose = 5)
epdft = mc.kernel()[0]

# MRH: this is the only place where you SHOULD use print
# A small amount of output data to summarize the calculation,
# while the more verbose intermediate output is put in a log file
print ("E(LASSCF) =", elas)
print ("E(tPBE) =", epdft)

