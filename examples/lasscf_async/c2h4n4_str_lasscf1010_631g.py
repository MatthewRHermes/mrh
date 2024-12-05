from mrh.tests.lasscf.c2h4n4_struct import structure as struct
from mrh.my_pyscf.mcscf.lasscf_async import LASSCF
from pyscf.lib import logger
from pyscf import scf

mol = struct (2.0, 2.0, '6-31g', symmetry=False)
mol.spin = 8
mol.verbose = logger.DEBUG
mol.output = 'c2h4n4_str_lasscf1010_631g.log'
mol.build ()
mf = scf.RHF (mol).run ()
las = LASSCF (mf, (4,2,4), ((2,2),(1,1),(2,2)), spin_sub=(1,1,1))
mo_coeff = las.set_fragments_([[0,1,2],[3,4,5,6],[7,8,9]])
las.kernel (mo_coeff)

