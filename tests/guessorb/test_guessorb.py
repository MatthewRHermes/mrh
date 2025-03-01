import unittest
from pyscf import gto
from mrh.my_pyscf.guessorb.guessorb import get_guessorb

'''
To check the accuracy of the implementation, I am comparing the mo_energy
calculated to the OpenMolcas guessorb module.

That can be read from the file.guessorb.h5 like this:

import h5py
import numpy as np

with h5py.File(file.guessorb.h5, "r") as hdf_file:
    item = hdf_file["MO_ENERGIES"]
    print(np.asarray(item))

Note that I have used OpenMolcas
version: 082a19c42-dirty
commit: 082a19c42dded5cb87a081429237d2937ecec3fd
'''

class KnownValues(unittest.TestCase):
    def test_NAtom(self):
        mol = gto.M(atom = '''N 0 0 0''',
        basis = 'STO-3G',
        verbose = 1,
        spin=3)
        mol.output = '/dev/null'
        mol.build()

        mo_energy, mo_coeff = get_guessorb(mol)

        # Calculated from OpenMolcas: 082a19c42dded5cb87a081429237d2937ecec3fd
        mo_energy_ref = [-15.6267, -0.9432, -0.5593, -0.5593, -0.5593]
        [self.assertAlmostEqual(energy, energy_ref, 2) \
        for energy, energy_ref in zip(mo_energy, mo_energy_ref)]

        # These values are generated with this code.
        # mrh: 3ddcaf20878b0f6c64518efc42c0f70cb579fa63
        # pyscf: 6f6d3741bf42543e02ccaa1d4ef43d9bf83b3dda
        mo_energy_bench = [-15.62670557,  -0.94322069,  -0.55938518,  -0.55938518,  -0.55938518]
        [self.assertAlmostEqual(energy, energy_ref, 6) \
        for energy, energy_ref in zip(mo_energy, mo_energy_bench)]

    def test_CO2(self):
        mol = gto.M(atom ='''
        C 0.000000 0.000000 0.000000
        O 0.000000 0.000000 1.155028
        O 0.000000 0.000000 -1.155028
        ''',
        basis = 'CC-PVDZ',
        verbose = 1)
        mol.output = '/dev/null'
        mol.build()

        mo_energy, mo_coeff = get_guessorb(mol)
       
        mo_energy_ref = [-20.6900, -20.6872, -11.3490, -1.6618, 
        -1.4710, -0.8117, -0.7952, -0.7952, -0.6359, -0.6206, -0.6206, -0.2870, 
        -0.2870, -0.1837, -0.1066, 4.0399, 4.1207, 4.4700, 4.4700, 4.9189, 4.9189, 
        5.1463, 5.1463, 5.2593, 6.0021, 6.5162, 6.5162, 6.6889, 6.6889, 7.1269, 
        7.1477, 7.1477, 7.2757, 7.2757, 7.4916, 7.6046, 7.6046, 8.1617, 8.3261, 
        8.3261, 9.1172, 9.4104]

        # The virtual orbital energy difference are more than > 0.1. Therefore
        # only comparing the occupied energies.
        [self.assertAlmostEqual(energy, energy_ref, 1) 
        for energy, energy_ref in zip(mo_energy[:11], mo_energy_ref[:11])]

        # These values are generated with this code.
        # mrh: 3ddcaf20878b0f6c64518efc42c0f70cb579fa63
        # pyscf: 6f6d3741bf42543e02ccaa1d4ef43d9bf83b3dda
        mo_energy_bench = [-20.69037204, -20.68727205, -11.34790438,  -1.66511884,  -1.48326356,
        -0.80688387, -0.80105563,  -0.80105563,  -0.63126274,  -0.62290439,
        -0.62290439, -0.28263501,  -0.28263501,  -0.18607969,  -0.1034079,
        4.16176924,  4.16978493,   4.58161092,   4.58161092,   4.91891801,
        4.91891801,  5.14109756,   5.14109756,   5.27678008,   5.95580721,
        6.45332687,  6.45332687,   6.65451946,   6.65451946,   7.10580777,
        7.14766705,  7.14766705,   7.275658  ,   7.275658  ,   7.44729759,
        7.60615608,  7.60615608,   8.15368918,   8.32273446,   8.32273446,
        9.15057941,  9.43190967]
        [self.assertAlmostEqual(energy, energy_ref, 6) \
        for energy, energy_ref in zip(mo_energy, mo_energy_bench)]


if __name__ == "__main__":
    print("Full Tests for GuessOrb")
    unittest.main()
