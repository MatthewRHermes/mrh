'''
SV: test file for LASSCF analytical nuclear gradients
H2 in one full fragment, must be same as CASSCF (2,2)
'''

from pyscf import gto, scf, lib, geomopt
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF

mol = gto.Mole()
mol.atom = '''H  0 0 0; H 1 1 1'''
mol.basis = '6-31g'
mol.build(verbose = 4)
mol.output = 'h2.log'
mf = scf.RHF (mol).run ()

las = LASSCF (mf,(2,),(2,), spin_sub=(1))

frag_atom_list = [list (range (2))]

mo_coeff = las.localize_init_guess (frag_atom_list, mf.mo_coeff)

las.kernel (mo_coeff)

# Optimizing geometry using Scanner 
gs_grad = las.nuc_grad_method().as_scanner()
mol1 = gs_grad.optimizer().kernel()
