import numpy as np
from scipy import linalg
from pyscf import gto, scf, df, mcscf, lib
from pyscf.data.nist import BOHR
from mrh.my_pyscf import mcpdft
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.grad.cmspdft import sarot_response, sarot_grad, sarot_response_o0, sarot_grad_o0
from mrh.my_pyscf.grad import sipdft as sipdft_grad
from mrh.my_pyscf.df.grad import dfsacasscf, dfsipdft
import unittest, math

h2co_casscf66_631g_xyz = '''C  0.534004  0.000000  0.000000
O -0.676110  0.000000  0.000000
H  1.102430  0.000000  0.920125
H  1.102430  0.000000 -0.920125'''
mol_nosymm = gto.M (atom = h2co_casscf66_631g_xyz, basis = '6-31g', symmetry = False, output='/dev/null', verbose = 0)
mol_symm = gto.M (atom = h2co_casscf66_631g_xyz, basis = '6-31g', symmetry = True, output='/dev/null', verbose = 0)
Lis = math.pi * (np.random.rand ((1)) - 0.5)
def get_mc_ref (mol, ri=False, sam=False):
    mf = scf.RHF (mol)
    if ri: mf = mf.density_fit (auxbasis = df.aug_etb (mol))
    mc = mcscf.CASSCF (mf.run (), 6, 6)
    if sam:
        fcisolvers = [csf_solver (mol, smult=((2*i)+1)) for i in (0,1)]
        if mol.symmetry:
            fcisolvers[0].wfnsym = 'A1'
            fcisolvers[1].wfnsym = 'A2'
        mc = mcscf.addons.state_average_mix (mc, fcisolvers, [0.5,0.5])
    else:
        mc.fcisolver = csf_solver (mol, smult=1)
        if mol.symmetry:
            mc.fcisolver.wfnsym = 'A1'
        mc = mc.state_average ([0.5,0.5])
    mc.conv_tol = 1e-12
    return mc.run ()
#mc_list = [[[get_mc_ref (m, ri=i, sam=j) for i in (0,1)] for j in (0,1)] for m in (mol_nosymm, mol_symm)]
mc_list = [] # Crunch within unittest.main for accurate clock
def get_mc_list ():
    if len (mc_list) == 0:
        for m in [mol_nosymm, mol_symm]:
            mc_list.append ([[get_mc_ref (m, ri=i, sam=j) for i in (0,1)] for j in (0,1)])
    return mc_list
def diatomic (atom1, atom2, r, fnal, basis, ncas, nelecas, nstates,
  charge=None, spin=None, symmetry=False, cas_irrep=None, density_fit=False):
    xyz = '{:s} 0.0 0.0 0.0; {:s} {:.3f} 0.0 0.0'.format (atom1, atom2, r)
    mol = gto.M (atom=xyz, basis=basis, charge=charge, spin=spin, symmetry=symmetry, verbose=0, output='/dev/null')
    mf = scf.RHF (mol)
    if density_fit: mf = mf.density_fit (auxbasis = df.aug_etb (mol))
    mc = mcpdft.CASSCF (mf.run (), fnal, ncas, nelecas, grids_level=9)
    if spin is not None: smult = spin+1
    else: smult = (mol.nelectron % 2) + 1
    mc.fcisolver = csf_solver (mol, smult=smult)
    mc = mc.state_interaction ([1.0/float(nstates),]*nstates, 'cms')
    mc.conv_tol = mc.conv_tol_sarot = 1e-12
    mo = None
    if symmetry and (cas_irrep is not None):
        mo = mc.sort_mo_by_irrep (cas_irrep)
    mc.kernel (mo)
    if density_fit: return dfsipdft.Gradients (mc)
    return mc.nuc_grad_method ()

def tearDownModule():
    global mol_nosymm, mol_symm, mc_list, Lis, get_mc_list, diatomic
    mol_nosymm.stdout.close ()
    mol_symm.stdout.close ()
    del mol_nosymm, mol_symm, mc_list, Lis, get_mc_list, diatomic

class KnownValues(unittest.TestCase):

    def test_sarot_response_sanity (self):
        for mcs, stype in zip (get_mc_list (), ('nosymm','symm')):
         for mca, atype in zip (mcs, ('nomix','mix')):
          if atype == 'mix': continue # TODO: enable state-average-mix
          for mc, itype in zip (mca, ('conv', 'DF')):
            ci_arr = np.asarray (mc.ci)
            if itype == 'conv': mc_grad = mc.nuc_grad_method ()
            else: mc_grad = dfsacasscf.Gradients (mc)
            eris = mc.ao2mo (mc.mo_coeff)
            with self.subTest (symm=stype, solver=atype, eri=itype, check='energy convergence'):
                self.assertTrue (mc.converged)
            def _crunch (fn):
                dw = fn (mc_grad, Lis, mo=mc.mo_coeff, ci=mc.ci, eris=eris)
                dworb, dwci = mc_grad.unpack_uniq_var (dw)
                return dworb, dwci
            with self.subTest (symm=stype, solver=atype, eri=itype):
                dworb_test, dwci_test = _crunch (sarot_response)
                dworb_ref, dwci_ref = _crunch (sarot_response_o0)
            with self.subTest (symm=stype, solver=atype, eri=itype, check='orb'):
                self.assertAlmostEqual (lib.fp (dworb_test), lib.fp (dworb_ref), 8)
            with self.subTest (symm=stype, solver=atype, eri=itype, check='CI'):
                self.assertAlmostEqual (lib.fp (dwci_test), lib.fp (dwci_ref), 8)

    def test_sarot_grad_sanity (self):
        for mcs, stype in zip (get_mc_list (), ('nosymm','symm')):
         for mca, atype in zip (mcs, ('nomix','mix')):
          for mc, itype in zip (mca, ('conv', 'DF')):
            ci_arr = np.asarray (mc.ci)
            if itype == 'conv': mc_grad = mc.nuc_grad_method ()
            else: continue #mc_grad = dfsacasscf.Gradients (mc)
            # TODO: proper DF functionality
            eris = mc.ao2mo (mc.mo_coeff)
            with self.subTest (symm=stype, solver=atype, eri=itype):
                dh_test = sarot_grad (mc_grad, Lis, mo=mc.mo_coeff, ci=mc.ci, eris=eris)
                dh_ref = sarot_grad_o0 (mc_grad, Lis, mo=mc.mo_coeff, ci=mc.ci, eris=eris)
                self.assertAlmostEqual (lib.fp (dh_test), lib.fp (dh_ref), 8)

    def test_grad_h2_cms3ftlda22_sto3g (self):
        # z_orb:    no
        # z_ci:     no
        # z_is:     no
        mc_grad = diatomic ('H', 'H', 1.3, 'ftLDA,VWN3', 'STO-3G', 2, 2, 3)
        de_ref = [0.226842531, -0.100538192, -0.594129499] 
        # Numerical from this software
        for i in range (3):
         with self.subTest (state=i):
            de = mc_grad.kernel (state=i) [1,0] / BOHR
            self.assertAlmostEqual (de, de_ref[i], 5)

    def test_grad_h2_cms2ftlda22_sto3g (self):
        # z_orb:    no
        # z_ci:     yes
        # z_is:     no
        mc_grad = diatomic ('H', 'H', 1.3, 'ftLDA,VWN3', 'STO-3G', 2, 2, 2)
        de_ref = [0.125068648, -0.181916973] 
        # Numerical from this software
        for i in range (2):
         with self.subTest (state=i):
            de = mc_grad.kernel (state=i) [1,0] / BOHR
            self.assertAlmostEqual (de, de_ref[i], 6)

    def test_grad_h2_cms3ftlda22_631g (self):
        # z_orb:    yes
        # z_ci:     no
        # z_is:     no
        mc_grad = diatomic ('H', 'H', 1.3, 'ftLDA,VWN3', '6-31G', 2, 2, 3)
        de_ref = [0.1717391582, -0.05578044075, -0.418332932] 
        # Numerical from this software
        for i in range (3):
         with self.subTest (state=i):
            de = mc_grad.kernel (state=i) [1,0] / BOHR
            self.assertAlmostEqual (de, de_ref[i], 5)

    def test_grad_h2_cms2ftlda22_631g (self):
        # z_orb:    yes
        # z_ci:     yes
        # z_is:     no
        mc_grad = diatomic ('H', 'H', 1.3, 'ftLDA,VWN3', '6-31G', 2, 2, 2)
        de_ref = [0.1046653372, -0.07056592067] 
        # Numerical from this software
        for i in range (2):
         with self.subTest (state=i):
            de = mc_grad.kernel (state=i) [1,0] / BOHR
            self.assertAlmostEqual (de, de_ref[i], 6)

    def test_grad_lih_cms2ftlda44_sto3g (self):
        # z_orb:    no
        # z_ci:     yes
        # z_is:     yes
        mc_grad = diatomic ('Li', 'H', 1.8, 'ftLDA,VWN3', 'STO-3G', 4, 4, 2, symmetry=True, cas_irrep={'A1': 4})
        de_ref = [0.0659740768, -0.005995224082] 
        # Numerical from this software
        for i in range (2):
         with self.subTest (state=i):
            de = mc_grad.kernel (state=i) [1,0] / BOHR
            self.assertAlmostEqual (de, de_ref[i], 6)

    def test_grad_lih_cms3ftlda22_sto3g (self):
        # z_orb:    yes
        # z_ci:     no
        # z_is:     yes
        mc_grad = diatomic ('Li', 'H', 2.5, 'ftLDA,VWN3', 'STO-3G', 2, 2, 3)
        de_ref = [0.09307779491, 0.07169985876, -0.08034177097] 
        # Numerical from this software
        for i in range (3):
         with self.subTest (state=i):
            de = mc_grad.kernel (state=i) [1,0] / BOHR
            self.assertAlmostEqual (de, de_ref[i], 6)

    def test_grad_lih_cms2ftlda22_sto3g (self):
        # z_orb:    yes
        # z_ci:     yes
        # z_is:     yes
        mc_grad = diatomic ('Li', 'H', 2.5, 'ftLDA,VWN3', 'STO-3G', 2, 2, 2)
        de_ref = [0.1071803011, 0.03972321867] 
        # Numerical from this software
        for i in range (2):
         with self.subTest (state=i):
            de = mc_grad.kernel (state=i) [1,0] / BOHR
            self.assertAlmostEqual (de, de_ref[i], 6)

    def test_grad_lih_cms2ftlda22_sto3g_df (self):
        # z_orb:    yes
        # z_ci:     yes
        # z_is:     yes
        mc_grad = diatomic ('Li', 'H', 2.5, 'ftLDA,VWN3', 'STO-3G', 2, 2, 2, density_fit=True)
        de_ref = [0.1074553399, 0.03956955205] 
        # Numerical from this software
        for i in range (2):
         with self.subTest (state=i):
            de = mc_grad.kernel (state=i) [1,0] / BOHR
            self.assertAlmostEqual (de, de_ref[i], 6)

if __name__ == "__main__":
    print("Full Tests for CMS-PDFT gradient objective fn derivatives")
    unittest.main()






