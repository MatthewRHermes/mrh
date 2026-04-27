import numpy as np
from pyscf import gto, scf, mcscf, lib
from pyscf.lib import chkfile
from pyscf.mcscf import avas
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf import lassi

xyz='''Cr    -1.320780000000   0.000050000000  -0.000070000000
Cr     1.320770000000   0.000050000000  -0.000070000000
O      0.000000000000  -0.165830000000   1.454680000000
O      0.000000000000   1.342770000000  -0.583720000000
O      0.000000000000  -1.176830000000  -0.871010000000
H      0.000020000000   0.501280000000   2.159930000000
H      0.000560000000   1.618690000000  -1.514480000000
H     -0.000440000000  -2.120790000000  -0.644130000000
N     -2.649800000000  -1.445690000000   0.711420000000
H     -2.186960000000  -2.181980000000   1.244400000000
H     -3.053960000000  -1.844200000000  -0.136070000000
H     -3.367270000000  -1.005120000000   1.287210000000
N     -2.649800000000   1.339020000000   0.896300000000
N     -2.649800000000   0.106770000000  -1.607770000000
H     -3.367270000000  -0.612160000000  -1.514110000000
H     -3.053960000000   0.804320000000   1.665160000000
N      2.649800000000  -1.445680000000   0.711420000000
N      2.649790000000   1.339030000000   0.896300000000
N      2.649800000000   0.106780000000  -1.607770000000
H     -2.186970000000   2.168730000000   1.267450000000
H     -3.367270000000   1.617370000000   0.226860000000
H     -2.186960000000   0.013340000000  -2.511900000000
H     -3.053970000000   1.039980000000  -1.529140000000
H      2.186960000000  -2.181970000000   1.244400000000
H      3.053960000000  -1.844190000000  -0.136080000000
H      3.367270000000  -1.005100000000   1.287200000000
H      2.186950000000   2.168740000000   1.267450000000
H      3.053960000000   0.804330000000   1.665160000000
H      3.367260000000   1.617380000000   0.226850000000
H      2.186960000000   0.013350000000  -2.511900000000
H      3.053960000000   1.039990000000  -1.529140000000
H      3.367270000000  -0.612150000000  -1.514110000000'''
basis = {'C': 'sto-3g','H': 'sto-3g','O': 'sto-3g','N': 'sto-3g','Cr': 'cc-pvdz'}
mol = gto.M (atom=xyz, spin=6, charge=3, basis=basis,
           verbose=4, output='lassis_chk.1.log')
mf = scf.ROHF(mol)
mf.kernel ()
las = LASSCF(mf,(6,6),((3,0),(0,3)),spin_sub=(4,4))
ncas_avas, nelecas_avas, mo_coeff = avas.kernel (mf, ['Cr 3d', 'Cr 4d'], minao=mol.basis)
mc_avas = mcscf.CASCI (mf, ncas_avas, nelecas_avas)
mo_list = mc_avas.ncore + np.array ([5,6,7,8,9,10,15,16,17,18,19,20])
mo_coeff = las.sort_mo (mo_list, mo_coeff)
mo_coeff = las.localize_init_guess (([0],[1]), mo_coeff)
las = lassi.spaces.spin_shuffle (las) # generate direct-exchange states
las.weights = [1.0/las.nroots,]*las.nroots # set equal weights
nroots_ref = las.nroots
las.kernel (mo_coeff) # optimize orbitals
assert (las.converged)
mo_coeff = las.mo_coeff
las = LASSCF(mf,(6,6),((3,0),(0,3)),spin_sub=(4,4))
las.lasci_(mo_coeff)

lsi = lassi.LASSIS (las)
lsi.sisolver.davidson_only = True
lsi.sisolver.smult = 1
lsi.sisolver.pspace_size = 0 # So you can see iterations
lsi.kernel ()
lsi.dump_chk ('lassis_chk.chk')
print ("Singlet energy:", lsi.e_roots[0])

#### Later, in a different file ####

xyz='''Cr    -1.320780000000   0.000050000000  -0.000070000000
Cr     1.320770000000   0.000050000000  -0.000070000000
O      0.000000000000  -0.165830000000   1.454680000000
O      0.000000000000   1.342770000000  -0.583720000000
O      0.000000000000  -1.176830000000  -0.871010000000
H      0.000020000000   0.501280000000   2.159930000000
H      0.000560000000   1.618690000000  -1.514480000000
H     -0.000440000000  -2.120790000000  -0.644130000000
N     -2.649800000000  -1.445690000000   0.711420000000
H     -2.186960000000  -2.181980000000   1.244400000000
H     -3.053960000000  -1.844200000000  -0.136070000000
H     -3.367270000000  -1.005120000000   1.287210000000
N     -2.649800000000   1.339020000000   0.896300000000
N     -2.649800000000   0.106770000000  -1.607770000000
H     -3.367270000000  -0.612160000000  -1.514110000000
H     -3.053960000000   0.804320000000   1.665160000000
N      2.649800000000  -1.445680000000   0.711420000000
N      2.649790000000   1.339030000000   0.896300000000
N      2.649800000000   0.106780000000  -1.607770000000
H     -2.186970000000   2.168730000000   1.267450000000
H     -3.367270000000   1.617370000000   0.226860000000
H     -2.186960000000   0.013340000000  -2.511900000000
H     -3.053970000000   1.039980000000  -1.529140000000
H      2.186960000000  -2.181970000000   1.244400000000
H      3.053960000000  -1.844190000000  -0.136080000000
H      3.367270000000  -1.005100000000   1.287200000000
H      2.186950000000   2.168740000000   1.267450000000
H      3.053960000000   0.804330000000   1.665160000000
H      3.367260000000   1.617380000000   0.226850000000
H      2.186960000000   0.013350000000  -2.511900000000
H      3.053960000000   1.039990000000  -1.529140000000
H      3.367270000000  -0.612150000000  -1.514110000000'''
basis = {'C': 'sto-3g','H': 'sto-3g','O': 'sto-3g','N': 'sto-3g','Cr': 'cc-pvdz'}
mol = gto.M (atom=xyz, spin=6, charge=3, basis=basis,
           verbose=4, output='lassis_chk.2.log')
# You don't need to actually optimize anything if you are going to load from a checkpoint
mf = scf.ROHF(mol).run (max_cycle=1)
las = LASSCF(mf,(6,6),((3,0),(0,3)),spin_sub=(4,4))
with lib.temporary_env (las, max_cycle_macro=1):
    las.lasci ()

lsi = lassi.LASSIS (las)
lsi.load_chk_('lassis_chk.chk')
lsi.sisolver.davidson_only = True
lsi.sisolver.smult = 3 # You can change the total spin now. It should project the guess.
lsi.sisolver.pspace_size = 0 # So you can see iterations
lsi.eig () # Instead of "run" or "kernel" to skip model state reoptimization
print ("Triplet energy:", lsi.e_roots[0])

