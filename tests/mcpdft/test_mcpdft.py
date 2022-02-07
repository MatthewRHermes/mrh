# Test API:
#   0. Initialize from mol, mf, and mc
#   1. kernel
#   2. optimize_mcscf_
#   3. compute_pdft_
#   4. energy_* 
#   5. get_energy_decomposition
#   6. checkpoint stuff
#   7. get_pdft_veff (maybe this elsewhere?)
# In the context of:
#   1. CASSCF, CASCI
#   2. Symmetry, with and without
#   3. State average, state average mix w/ different spin states

# Building the rudiments of this in debug/debug_mcpdft_api.py

from pyscf import gto, scf, mcscf, lib
from mrh.my_pyscf import mcpdft
import unittest

mol_nosym = gto.M (atom = 'Li 0 0 0\nH 1.5 0 0', basis = 'sto3g',
                   output = '/dev/null', verbose = 0)
mol_sym = gto.M (atom = 'Li 0 0 0\nH 1.5 0 0', basis = 'sto3g', symmetry=True,
                 output = '/dev/null', verbose = 0)

mf_nosym = mf_sym = mc_nosym = mc_sym = None

def setUpModule():
    global mol_nosym, mf_nosym, mc_nosym, mol_sym, mf_sym, mc_sym
    mf_nosym = scf.RHF (mol_nosym).run ()
    mc_nosym = mcscf.CASSCF (mf_nosym, 5, 2).run ()
    mf_sym = scf.RHF (mol_sym).run ()
    mc_sym = mcscf.CASSCF (mf_sym, 5, 2).run ()

def tearDownModule():
    global mol_nosym, mf_nosym, mc_nosym, mol_sym, mf_sym, mc_sym
    mol_nosym.stdout.close ()
    mol_sym.stdout.close ()
    del mol_nosym, mf_nosym, mc_nosym, mol_sym, mf_sym, mc_sym

class KnownValues(unittest.TestCase):

    def test_init (self):
        ref_e = -7.924089707
        for symm in False, True:
            mol = (mol_nosym, mol_sym)[int(symm)]
            mf = (mf_nosym, mf_sym)[int(symm)]
            mc0 = (mc_nosym, mc_sym)[int(symm)]
            for i, cls in enumerate ((mcpdft.CASCI, mcpdft.CASSCF)):
                scf = bool (i)
                for my_init in (mol, mf, mc0):
                    init_name = my_init.__class__.__name__
                    if init_name == 'Mole' and symm: continue
                    # ^ The underlying PySCF modules can't do this as of 02/06/2022
                    my_kwargs = {}
                    if isinstance (my_init, gto.Mole) or (not scf):
                        my_kwargs['mo_coeff'] = mc0.mo_coeff
                    with self.subTest (symm=symm, scf=scf, init=init_name):
                        mc = cls (my_init, 'tPBE', 5, 2).run (**my_kwargs)
                        self.assertAlmostEqual (mc.e_tot, ref_e, 7)
                        self.assertTrue (mc.converged)

if __name__ == "__main__":
    print("Full Tests for MC-PDFT energy API")
    unittest.main()


