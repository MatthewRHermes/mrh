import sys
from pyscf import gto, scf, tools, dft, lib
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_dmet import localintegrals, dmet, fragments
from mrh.my_pyscf import mcpdft
from mrh.my_pyscf import lassi

# Initialization
mol = gto.M()
mol.atom='''H -5.10574 2.01997 0.00000; H -4.29369 2.08633 0.00000; H -3.10185 2.22603 0.00000; H -2.29672 2.35095 0.00000''' 
mol.basis='6-31g'
mol.verbose=4
mol.build()

# Mean field calculation
mf = scf.ROHF(mol).newton().run()

# LASSCF Calculations
las = LASSCF(mf,(2, 2),(2, 2),spin_sub=(1, 1))
frag_atom_list = ([0, 1], [2, 3]) 
mo0 = las.localize_init_guess(frag_atom_list)
las.max_cycle_macro=1
las.kernel(mo0)

las = lassi.spaces.all_single_excitations (las)
las.lasci ()

lsi = lassi.LASSI(las)
lsi.kernel()

# LASSI-PDFT
mc = mcpdft.LASSI(lsi, 'tPBE',states=[0, 1])
mc.kernel()

# Note: 
# mc.e_tot : LASSI-PDFT energy
# mc.e_mcscf: LASSI energy for those roots
# mc.e_roots: LASSI energies of all the roots
# mc.e_states: Energy of LAS states (basis of LASSI)

# Analyze: Takes the state no as input. By default it is for state=0
mc.analyze(state=[0,1])

# Energy decomposition analysis: Will be done on the same no of roots calculated 
# during the LASSI-PDFT
e_decomp = mc.get_energy_decomposition (split_x_c=False)
print ("e_nuc =",e_decomp[0])
print ("e_1e =",e_decomp[1])
print ("e_Coul =",e_decomp[2])
print ("e_OT =",e_decomp[3])
print ("e_ncwfn (not included in total energy) =",e_decomp[4])

