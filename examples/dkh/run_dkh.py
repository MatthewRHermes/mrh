import numpy as np
from pyscf import gto, scf
from mrh.my_pyscf.dkh import dkh

# Author: Bhavnesh Jangid

'''
This example shows how to use the scalar relativistic DKH hamiltonian
One can use the same with different flavors of mean-field and methods like
CASSCF and LASSCF will automatically inherit these rel. effects
from the mean-field object.
'''

mol = gto.M(atom='''Zn 0 0 0''',basis='ano@5s4p2d1f',verbose=0)

mfdkh2 = scf.RHF(mol)
mfdkh2.get_hcore = lambda *args: dkh(mol, dkhord=2)
mfdkh2.kernel()

mfdkh3 = scf.RHF(mol)
mfdkh3.get_hcore = lambda *args: dkh(mol, dkhord=3)
mfdkh3.kernel()

mfdkh4 = scf.RHF(mol)
mfdkh4.get_hcore = lambda *args: dkh(mol, dkhord=4)
mfdkh4.kernel()

mfx2c = scf.RHF(mol).sfx2c1e().run()
mfnr = scf.RHF(mol).run()

print('\nE(Non-Relativisitic) : ', mfnr.e_tot)
print('E(DKH) 2nd Order     : ', mfdkh2.e_tot)
print('E(DKH) 3rd Order     : ', mfdkh3.e_tot)
print('E(DKH) 4th Order     : ', mfdkh4.e_tot)
print('E(SFX2C1e)           : ', mfx2c.e_tot)

