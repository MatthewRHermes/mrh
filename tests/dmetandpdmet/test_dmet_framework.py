import unittest
import numpy as np
from pyscf import gto, scf
from mrh.my_pyscf.dmet import runDMET

'''
***** RHF Embedding *****
1. Consider all the atoms in embedding space
2. Consider few atoms in embedding space
3. Consider few atoms in embedding space with density fitting
***** ROHF Embedding *****
1. Consider all the atoms in embedding space
2. Consider few atoms in embedding space
3. Consider few atoms in embedding space with density fitting
'''

def get_mole1():
    mol = gto.Mole(basis='6-31G', spin=0, charge=0, verbose=0)
    mol.atom = '''
    S  -5.64983   3.02383   0.00000
    H  -4.46871   3.02383   0.00000
    H  -6.24038   2.19489   0.59928
    '''
    mol.build()
    return mol

def get_mole2():
    mol = gto.Mole(basis='6-31G', spin=1, charge=0, verbose=0)
    mol.atom = '''
    P  -5.64983   3.02383   0.00000
    H  -4.46871   3.02383   0.00000
    H  -6.24038   2.19489   0.59928
    '''
    mol.build()
    return mol

def add_hydrogen_to_water(atom, com, d):
    '''
    Here, based on the center of mass (COM) of the water molecule, 
    I am randomly adding a hydrogen atom anywhere on the surface of 
    a sphere, which is drawn using the given radius dd from 
    the COM of the water molecule.
    '''
    phi = np.random.uniform(0, 2 * np.pi)
    theta = np.random.uniform(0, np.pi)
    x = com[0] + d * np.sin(theta) * np.cos(phi)
    y = com[1] + d * np.sin(theta) * np.sin(phi)
    z = com[2] + d * np.cos(theta)
    atom.append(['H', [x, y, z]])
    return atom

def get_mole3():
    mol = gto.Mole(basis='CC-PVDZ', spin=0, charge=0, verbose=0)
    mol.atom = [['O', [0.0000, 0.0000, 0.0000]],
                ['H', [0.9572, 0.0000, 0.0000]],
                ['H', [-0.478, 0.8289, 0.0000]]]
    mol.build()
    
    # Now add Hydrogen
    dis = 4.0
    mass = mol.atom_mass_list()
    coords = mol.atom_coords()
    com = mass.dot(coords) / mass.sum()
    atom = add_hydrogen_to_water(mol.atom, com, dis)
    mol.atom = atom
    mol.spin = 1
    mol.build()
    return mol

class KnownValues(unittest.TestCase):

    # RHF Embedding
    def test_vanilla_rhf(self):
        mol = get_mole1()
        mf = scf.RHF(mol)
        mf.kernel()
        e_ref = mf.e_tot
        dmet_mf = runDMET(mf, lo_method='lowdin', bath_tol=1e-10, atmlst=[0,1,2])[0] # Considering all the atoms in embedding space
        e_check = dmet_mf.e_tot
        del mol, mf, dmet_mf
        self.assertAlmostEqual(e_ref, e_check, 6)
        
    def test_dmet_rhf(self):
        mol = get_mole1()
        mf = scf.RHF(mol)
        mf.kernel()
        e_ref = mf.e_tot
        dmet_mf = runDMET(mf, lo_method='lowdin', bath_tol=1e-10, atmlst=[0,])[0] # Only few atoms in embedding space
        e_check = dmet_mf.e_tot
        del mol, mf, dmet_mf
        self.assertAlmostEqual(e_ref, e_check, 6)

    def test_dmet_rhf_with_density_fitting(self):
        mol = get_mole1()
        mf = scf.RHF(mol).density_fit()
        mf.kernel()
        e_ref = mf.e_tot
        dmet_mf = runDMET(mf, lo_method='lowdin', bath_tol=1e-10, atmlst=[0,])[0]
        e_check = dmet_mf.e_tot
        del mol, mf, dmet_mf
        self.assertAlmostEqual(e_ref, e_check, 6)
    
    # ROHF Embedding
    def test_vanilla_rohf(self):
        mol = get_mole2()
        mf = scf.ROHF(mol)
        mf.kernel()
        e_ref = mf.e_tot
        dmet_mf = runDMET(mf, lo_method='lowdin', bath_tol=1e-10, atmlst=[0,1,2])[0]
        e_check = dmet_mf.e_tot
        del mol, mf, dmet_mf
        self.assertAlmostEqual(e_ref, e_check, 6)

    def test_dmet_rohf(self):
        mol = get_mole2()
        mf = scf.ROHF(mol)
        mf.kernel()
        e_ref = mf.e_tot
        dmet_mf = runDMET(mf, lo_method='lowdin', bath_tol=1e-10, atmlst=[0,])[0]
        e_check = dmet_mf.e_tot
        del mol, mf, dmet_mf
        self.assertAlmostEqual(e_ref, e_check, 6)
        
    def test_dmet_rohf_with_density_fitting(self):
        mol = get_mole2()
        mf = scf.ROHF(mol).density_fit()
        mf.kernel()
        e_ref = mf.e_tot
        dmet_mf = runDMET(mf, lo_method='lowdin', bath_tol=1e-10, atmlst=[0,])[0]
        e_check = dmet_mf.e_tot
        del mol, mf, dmet_mf
        self.assertAlmostEqual(e_ref, e_check, 6)

    def test_tough_rohf(self):
        '''
        In this test, I am only having H as fragment, with water as environment.
        '''
        mol = get_mole3()
        mf = scf.ROHF(mol)
        mf.kernel()
        e_ref = mf.e_tot
        dmet_mf = runDMET(mf, lo_method='lowdin', bath_tol=1e-10, atmlst=[3])[0]
        e_check = dmet_mf.e_tot
        del mol, mf, dmet_mf
        self.assertAlmostEqual(e_ref, e_check, 6)

if __name__ == "__main__":
    # See the description of the tests at the top of the file.
    unittest.main()
