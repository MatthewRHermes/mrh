import numpy as np
from scipy import linalg
from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.mcscf.lasscf_async import LASSCF

xyz = '''H 0.0 0.0 0.0
         H 1.0 0.0 0.0
         H 0.2 3.9 0.1
         H 1.159166 4.1 -0.1'''
mol = gto.M (atom = xyz, basis = '6-31g', output='h4_631g.log',
    verbose=lib.logger.DEBUG)
mf = scf.RHF (mol).run ()
las = LASSCF (mf, (2,2), (2,2), spin_sub=(1,1))
frag_atom_list = ((0,1),(2,3))
mo_loc = las.set_fragments_(frag_atom_list, mf.mo_coeff)
las.kernel (mo_loc)

