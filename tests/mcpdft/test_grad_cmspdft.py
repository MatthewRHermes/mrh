import numpy as np
from scipy import linalg
from pyscf import gto, scf, df, mcscf, lib
from pyscf.data.nist import BOHR
from mrh.my_pyscf import mcpdft
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.grad.cmspdft import sarot_response, sarot_grad, sarot_response_o0, sarot_grad_o0
from mrh.my_pyscf.grad import sipdft as sipdft_grad
from mrh.my_pyscf.df.grad import dfsacasscf
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
def do_diatomic (a1='H', a2='H', r=1.6, fnal='tPBE', basis='sto-3g', ncas=2,
        nelecas=2, nroots=2, symmetry=False):
    xyz = '{:s} 0 0 0; {:s} {:.2f} 0 0'.format (a1, a2, r)
    mf = scf.RHF (gto.M (atom=xyz, basis=basis, symmetry=symmetry,
        output='/dev/null', verbose=0)).run ()
    mc = mcpdft.CASSCF (mf, fnal, ncas, nelecas, grids_level=9)
    mc.fcisolver = csf_solver (mf.mol, smult=1)
    mc = mc.state_interaction ([1.0/nroots,]*nroots, 'cms').run ()
    mc_grad = sipdft_grad.Gradients (mc)
    de = [mc_grad.kernel (state=i) [1,0] / BOHR for i in range (nroots)]
    mf.mol.stdout.close ()
    return de

def tearDownModule():
    global mol_nosymm, mol_symm, mc_list, Lis, get_mc_list, do_diatomic
    mol_nosymm.stdout.close ()
    mol_symm.stdout.close ()
    del mol_nosymm, mol_symm, mc_list, Lis, get_mc_list, do_diatomic

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
                dwis = np.einsum ('pab,qab->pq', dwci, ci_arr.conj ())
                dwci -= np.einsum ('pq,qab->pab', dwis, ci_arr)
                return dworb, dwci, dwis
            with self.subTest (symm=stype, solver=atype, eri=itype):
                dworb_test, dwci_test, dwis_test = _crunch (sarot_response)
                dworb_ref, dwci_ref, dwis_ref = _crunch (sarot_response_o0)
            with self.subTest (symm=stype, solver=atype, eri=itype, check='orb'):
                self.assertAlmostEqual (lib.fp (dworb_test), lib.fp (dworb_ref), 8)
            with self.subTest (symm=stype, solver=atype, eri=itype, check='CI'):
                self.assertAlmostEqual (lib.fp (dwci_test), lib.fp (dwci_ref), 8)
            with self.subTest (symm=stype, solver=atype, eri=itype, check='IS'):
                self.assertAlmostEqual (lib.fp (dwis_test), lib.fp (dwis_ref), 8)

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

    def test_grad_h2 (self):
        de = do_diatomic ()
        with self.subTest (root=0):
            self.assertAlmostEqual (de[0], 0.03622639, 4) # numerical
        with self.subTest (root=1):
            self.assertAlmostEqual (de[1], -0.06910341, 4) # numerical

if __name__ == "__main__":
    print("Full Tests for CMS-PDFT gradient objective fn derivatives")
    unittest.main()






