# Example to show how to perform the dkh scalar relativistic corrections in PySCF

from pyscf import gto, scf
from mrh.my_pyscf.dkh import dkh

mol = gto.Mole(atom='''Ne 0 0 0''',basis='cc-pvdz-dk',verbose=3)

# Scalar Relativisitc Effects
# dkhord = 2 or 3 or 4
# Default c(speed of light) used in PySCF is different from other softwares.

mfdkh = scf.RHF(mol)
mfdkh.get_hcore = lambda *args: dkh(mol,dkhord=2)
mfdkh.kernel()

