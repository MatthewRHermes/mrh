from pyscf import gto, scf, mcscf, lib, mcpdft, mrpt
from pyscf.mcscf import avas


mol = gto.Mole()
mol.atom='''
C -4.788725 -0.244718 0.223333
C -3.468433 0.436990 0.079656
C -2.294480 -0.473906 0.223333
H -5.702709 0.237458 -0.108372
H -4.878379 -1.170292 0.783591
H -3.392370 1.264779 0.826658
H -3.419456 0.970001 -0.885351
H -1.307931 -0.166365 -0.108372
H -2.375000 -1.400320 0.783591
'''

mol.charge = 0
mol.spin = 2
mol.basis = 'cc-pvdz'
mol.verbose = 4
mol.max_memory = 100000
mol.build()

mf = scf.RHF(mol).density_fit()
mf.conv_tol=1e-12
mf.max_cycle = 200
mf.kernel()

from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.tools.molden import from_lasscf
from mrh.my_pyscf import lassi
from mrh.my_pyscf import mcpdft

las = LASSCF(mf,(2,2),((1, 0),(0, 1)),spin_sub=(2,2))
mo_coeff = las.localize_init_guess (([0], [2]))
las = lassi.spaces.spin_shuffle (las)
las.max_cycle_macro=1
las.weights = [1.0/las.nroots,]*las.nroots
las.kernel(mo_coeff)

las = lassi.spaces.all_single_excitations(las)
las.lasci_()

lsi = lassi.LASSI(las)
lsi.kernel()

mc = mcpdft.LASSI(lsi,'tPBE', (2,2),((1, 0),(1, 0)),spin_sub=(2,2))
mc=mc.multi_state()
mc.kernel()


