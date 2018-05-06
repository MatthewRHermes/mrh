from pyscf import gto
import numpy as np

def structure( distance, basis):

    mol = gto.Mole()

    number_of_atom = 3 
		
    atomlist = []
    HeX = 1.0 + distance
    atomlist.append (["H",      0.0, 0.0, 0.0])
    atomlist.append (["H", distance, 0.0, 0.0])
    atomlist.append (["He",     HeX, 0.0, 0.0])

    mol.atom = atomlist
    mol.basis = { 'He': basis, 'H': basis }
    mol.charge = 0
    mol.spin = 0
    mol.build()
    return mol
	
#Structure test:
'''
import sys
sys.path.append('../../QC-DMET/src')
import localintegrals, dmet, qcdmet_paths
from pyscf import gto, scf, symm, future
import numpy as np
import ME2N2_struct

basis = 'sto-6g' 
distance = 3.2
mol = ME2N2_struct.structure( distance, basis)
xyz = np.asarray(mol.atom)
for atom in xyz:
	print(atom[0],atom[1],atom[2],atom[3])
'''
