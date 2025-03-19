from mrh.tests.lasscf.c2h4n4_struct import structure as struct
from mrh.my_pyscf.mcscf.lasscf_async import LASSCF
from pyscf.lib import logger
from pyscf import scf

mol = struct (0.0, 0.0, '6-31g', symmetry=False)
mol.spin = 0
mol.verbose = logger.DEBUG
mol.output = 'using_older_kernel.log'
mol.build ()
mf = scf.RHF (mol).run ()
las = LASSCF (mf, (4,2,4), ((2,2),(1,1),(2,2)), spin_sub=(1,1,1))
mo_coeff = las.sort_mo ([7,8,16,18,22,23,24,26,33,34])
mo_coeff = las.set_fragments_([[0,1,2],[3,4,5,6],[7,8,9]], mo_coeff=mo_coeff)

# Note that just importing the patch_kernel function doesn't do anything, unlike the gpu4mrh
# "patch_*" functions. I prefer not to do things in imports and I hate global variables, so
# instead, patch_kernel is a function that returns a patched version of that specific method
# instance.
from mrh.my_pyscf.mcscf.lasscf_async import old_aa_sync_kernel
las = old_aa_sync_kernel.patch_kernel (las)

# This will take fewer macrocycles to converge than c2h4n4_equil_lasscf1010_631g, to which it is
# otherwise identical.
las.kernel (mo_coeff)


