import numpy as np
from scipy import linalg
from pyscf import gto, scf, df, mcscf, lib
from mrh.my_pyscf import mcpdft
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.grad.sipdft import sipdft_heff_response, sipdft_heff_HellmanFeynman
from mrh.my_pyscf.df.grad import dfsacasscf
import unittest, math

h2co_casscf66_631g_xyz = '''C  0.534004  0.000000  0.000000
O -0.676110  0.000000  0.000000
H  1.102430  0.000000  0.920125
H  1.102430  0.000000 -0.920125'''
mol_nosymm = gto.M (atom = h2co_casscf66_631g_xyz, basis = '6-31g', symmetry = False, output='/dev/null', verbose = 0)
mol_symm = gto.M (atom = h2co_casscf66_631g_xyz, basis = '6-31g', symmetry = True, output='/dev/null', verbose = 0)
def random_si ():
    phi = math.pi * (np.random.rand (1)[0] - 0.5)
    cp, sp = math.cos (phi), math.sin (phi)
    si = np.array ([[cp,-sp],[sp,cp]])
    return si
si = random_si ()
def get_mc_ref (mol, ri=False):
    mf = scf.RHF (mol)
    if ri: mf = mf.density_fit (auxbasis = df.aug_etb (mol))
    mc = mcscf.CASSCF (mf.run (), 6, 6)
    fcisolvers = [csf_solver (mol, smult=((2*i)+1)) for i in (0,1)]
    if mol.symmetry:
        fcisolvers[0].wfnsym = 'A1'
        fcisolvers[1].wfnsym = 'A2'
    mc = mcscf.addons.state_average_mix (mc, fcisolvers, [0.5,0.5])
    mc.conv_tol = 1e-12
    return mc.run ()
mc_list = [[get_mc_ref (m, ri=i) for i in (0,1)] for m in (mol_nosymm, mol_symm)]
def tearDownModule():
    global mol_nosymm, mol_symm, mc_list, si
    mol_nosymm.stdout.close ()
    mol_symm.stdout.close ()
    del mol_nosymm, mol_symm, mc_list, si

class KnownValues(unittest.TestCase):

    def test_offdiag_response_sanity (self):
        for mcl, stype in zip (mc_list, ('nosymm','symm')):
         for mc, itype in zip (mcl, ('conv', 'DF')):
            ci_arr = np.asarray (mc.ci)
            if itype == 'conv': mc_grad = mc.nuc_grad_method ()
            else: mc_grad = dfsacasscf.Gradients (mc)
            ngorb = mc_grad.ngorb
            dw_ref = np.stack ([mc_grad.get_wfn_response (state=i) for i in (0,1)], axis=0)
            dworb_ref, dwci_ref = dw_ref[:,:ngorb], dw_ref[:,ngorb:]
            with self.subTest (symm=stype, eri=itype, check='ref CI d.f. zero'):
                self.assertLessEqual (linalg.norm (dwci_ref), 2e-6)
            ham_si = np.diag (mc.e_states)
            ham_si = si @ ham_si @ si.T
            e_mcscf = ham_si.diagonal ().copy ()
            eris = mc.ao2mo (mc.mo_coeff)
            ci = list (np.tensordot (si, ci_arr, axes=1))
            ci_arr = np.asarray (ci)
            si_diag = si * si
            dw_diag = np.stack ([mc_grad.get_wfn_response (state=i, ci=ci) for i in (0,1)], axis=0)
            dworb_diag, dwci_ref = dw_diag[:,:ngorb], dw_diag[:,ngorb:]
            dworb_ref -= np.einsum ('sc,sr->rc', dworb_diag, si_diag)
            dwci_ref = -np.einsum ('rpab,qab->rpq', dwci_ref.reshape (2,2,20,20), ci_arr)
            dwci_ref -= dwci_ref.transpose (0,2,1)
            dwci_ref = np.einsum ('spq,sr->rpq', dwci_ref, si_diag)
            for r in (0,1):
                dw_test = sipdft_heff_response (mc_grad, ci=ci, state=r, eris=eris,
                    si_bra=si[:,r], si_ket=si[:,r], ham_si=ham_si, e_mcscf=e_mcscf)
                dworb_test, dwci_test = dw_test[:ngorb], dw_test[ngorb:]
                dwci_test = np.einsum ('pab,qab->pq', dwci_test.reshape (2,20,20), ci_arr)
                with self.subTest (symm=stype, eri=itype, root=r, check='orb'):
                    self.assertAlmostEqual (lib.fp (dworb_test), lib.fp (dworb_ref[r]), 8)
                with self.subTest (symm=stype, eri=itype, root=r, check='CI'):
                    self.assertAlmostEqual (lib.fp (dwci_test), lib.fp (dwci_ref[r]), 8)

    def test_offdiag_grad_sanity (self):
        for mcl, stype in zip (mc_list, ('nosymm','symm')):
         for mc, itype in zip (mcl, ('conv', 'DF')):
            ci_arr = np.asarray (mc.ci)
            if itype == 'conv': mc_grad = mc.nuc_grad_method ()
            else: continue #mc_grad = dfsacasscf.Gradients (mc)
            # TODO: proper DF functionality
            de_ref = np.stack ([mc_grad.get_ham_response (state=i) for i in (0,1)], axis=0)
            eris = mc.ao2mo (mc.mo_coeff)
            ci = list (np.tensordot (si, ci_arr, axes=1))
            ci_arr = np.asarray (ci)
            si_diag = si * si
            de_diag = np.stack ([mc_grad.get_ham_response (state=i, ci=ci) for i in (0,1)], axis=0)
            de_ref -= np.einsum ('sac,sr->rac', de_diag, si_diag)
            for r in (0,1):
                de_test = sipdft_heff_HellmanFeynman (mc_grad, ci=ci, state=r,
                    si_bra=si[:,r], si_ket=si[:,r], eris=eris)
                with self.subTest (symm=stype, eri=itype, root=r):
                    self.assertAlmostEqual (lib.fp (de_test), lib.fp (de_ref[r]), 8)


if __name__ == "__main__":
    print("Full Tests for SI-PDFT gradient off-diagonal heff fns")
    unittest.main()






