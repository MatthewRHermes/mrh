# Test API:
#   0. Initialize from mol, mf, and mc (done)
#   1. kernel (done)
#   2. optimize_mcscf_
#   3. compute_pdft_
#   4. energy_tot (done) 
#   5. get_energy_decomposition (done)
#   6. checkpoint stuff
#   7. get_pdft_veff (maybe this elsewhere?)
# In the context of:
#   1. CASSCF, CASCI
#   2. Symmetry, with and without
#   3. State average, state average mix w/ different spin states

# Building the rudiments of this in debug/debug_mcpdft_api.py

import numpy as np
from pyscf import gto, scf, mcscf, lib, fci
from pyscf.fci.addons import fix_spin_
from mrh.my_pyscf import mcpdft
import unittest

mol_nosym = gto.M (atom = 'Li 0 0 0\nH 1.5 0 0', basis = 'sto3g',
                   output = '/dev/null', verbose = 0)
mol_sym = gto.M (atom = 'Li 0 0 0\nH 1.5 0 0', basis = 'sto3g', symmetry=True,
                 output = '/dev/null', verbose = 0)

mf_nosym = mf_sym = mc_nosym = mc_sym = None


def setUpModule():
    global mol_nosym, mf_nosym, mc_nosym, mol_sym, mf_sym, mc_sym, mcp
    mf_nosym = scf.RHF (mol_nosym).run ()
    mc_nosym = mcscf.CASSCF (mf_nosym, 5, 2).run ()
    mf_sym = scf.RHF (mol_sym).run ()
    mc_sym = mcscf.CASSCF (mf_sym, 5, 2).run ()
    mcp_ss_nosym = mcpdft.CASSCF (mc_nosym, 'tPBE', 5, 2).run ()
    mcp_ss_sym = mcpdft.CASSCF (mc_sym, 'tPBE', 5, 2).run ()
    mcp_sa_0 = mcp_ss_nosym.state_average ([1.0/5,]*5).run ()
    solver_S = fci.solver (mol_nosym, singlet=True).set (spin=0, nroots=2)
    solver_T = fci.solver (mol_nosym, singlet=False).set (spin=2, nroots=3)
    mcp_sa_1 = mcp_ss_nosym.state_average_mix (
        [solver_S,solver_T], [1.0/5,]*5).run ()
    solver_A1 = fci.solver (mol_sym).set (wfnsym='A1', nroots=3)
    solver_E1x = fci.solver (mol_sym).set (wfnsym='E1x', nroots=1, spin=2)
    solver_E1y = fci.solver (mol_sym).set (wfnsym='E1y', nroots=1, spin=2)
    mcp_sa_2 = mcp_ss_sym.state_average_mix (
        [solver_A1,solver_E1x,solver_E1y], [1.0/5,]*5).run ()
    mcp = [[mcp_ss_nosym, mcp_ss_sym], [mcp_sa_0, mcp_sa_1, mcp_sa_2]]

def tearDownModule():
    global mol_nosym, mf_nosym, mc_nosym, mol_sym, mf_sym, mc_sym, mcp
    mol_nosym.stdout.close ()
    mol_sym.stdout.close ()
    del mol_nosym, mf_nosym, mc_nosym, mol_sym, mf_sym, mc_sym, mcp

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

    def test_state_average (self): 
        # grids_level = 6
        #ref = np.array ([-7.9238958646710085,-7.7887395616498125,-7.7761692676370355,
        #                 -7.754856419853813,-7.754856419853812,]) 
        # grids_level = 5
        #ref = np.array ([-7.923895345983219,-7.788739501036741,-7.776168040902887,
        #                 -7.75485647715595,-7.7548564771559505])
        # grids_level = 4
        #ref = np.array ([-7.923894841822498,-7.788739444709943,-7.776169108993544,
        #                 -7.754856321482755,-7.754856321482756])
        # grids_level = 3
        ref = np.array ([-7.923894179700609,-7.7887396628199,-7.776172495309403,
                         -7.754856085624646,-7.754856085624647])
        # TODO: figure out why SA always needs more precision than SS to get
        # the same repeatability. Fix if possible? In the mean time, loose
        # deltas below for the sake of speed.
        for ix, mc in enumerate (mcp[1]):
            with self.subTest (symmetry=bool(ix//2), triplet_ms=(0,1,'mixed')[ix]):
                self.assertTrue (mc.converged)
                self.assertAlmostEqual (lib.fp (np.sort (mc.e_states)), lib.fp (ref), delta=1e-5)
                self.assertAlmostEqual (mc.e_tot, np.average (ref), delta=1e-6)

    def test_decomposition_ss (self): # TODO
        ref = [1.0583544218,-12.5375911135, 5.8093938665, -2.1716353580, -0.0826115115, -2.2123063329]
        terms = ['nuc', 'core', 'Coulomb', 'OT(X)', 'OT(C)', 'WFN(XC)']
        for ix, mc in enumerate (mcp[0]):
            test = mc.get_energy_decomposition ()
            for t, r, term in zip (test, ref, terms):
                with self.subTest (symmetry=bool(ix), term=term):
                    self.assertAlmostEqual (t, r, delta=1e-5)
            with self.subTest (symmetry=bool(ix), term='sanity'):
                self.assertAlmostEqual (np.sum (test[:-1]), mc.e_tot, 9)

    def test_decomposition_sa (self):
        ref_nuc = 1.0583544218
        ref_states = np.array ([[-12.5385413915, 5.8109724796, -2.1720331222, -0.0826465641, -2.2127964255],
                                [-12.1706553996, 5.5463231972, -2.1601539256, -0.0626079593, -2.1943087132],
                                [-12.1768195314, 5.5632261670, -2.1552571900, -0.0656763663, -2.1887042769],
                                [-12.1874226655, 5.5856701424, -2.1481995107, -0.0632609608, -2.1690856659],
                                [-12.1874226655, 5.5856701424, -2.1481995107, -0.0632609608, -2.1690856659]])
        terms = ['core', 'Coulomb', 'OT(X)', 'OT(C)', 'WFN(XC)']
        for ix, (mc,ms) in enumerate (zip (mcp[1], [0,1,'mixed'])):
            s = bool(ix//2)
            test = mc.get_energy_decomposition ()
            test_nuc, test_states = test[0], np.array (test[1:]).T
            # Arrange states in ascending energy order
            idx = np.argsort (mc.e_states)
            test_states = test_states[idx,:]
            e_ref = np.array (mc.e_states)[idx] 
            with self.subTest (symmetry=s, triplet_ms=ms, term='nuc'):
                self.assertAlmostEqual (test_nuc, ref_nuc, 9)
            for state, (test, ref) in enumerate (zip (test_states, ref_states)):
                for t, r, term in zip (test, ref, terms):
                    with self.subTest (symmetry=s, triplet_ms=ms, term=term, state=state):
                        self.assertAlmostEqual (t, r, delta=1e-5)
                with self.subTest (symmetry=s, triplet_ms=ms, term='sanity', state=state):
                    self.assertAlmostEqual (np.sum (test[:-1])+test_nuc, e_ref[state], 9)


    def test_energy_tot (self):
        # Test both correctness and energy_tot function purity
        def get_attr (mc):
            mo_ref = lib.fp (mc.mo_coeff)
            ci_ref = lib.fp (np.concatenate (mc.ci, axis=None))
            e_states_ref = lib.fp (getattr (mc, 'e_states', 0))
            return mo_ref, ci_ref, mc.e_tot, e_states_ref, mc.grids.level, mc.otxc
        def test_energy_tot_crunch (test_list, ref_list, casestr):
            for ix, (t, r) in enumerate (zip (test_list, ref_list)):
                with self.subTest (case=casestr, item=ix):
                    if isinstance (t, (float, np.floating)):
                        self.assertAlmostEqual (t, r, delta=1e-6)
                    else:
                        self.assertEqual (t, r)
        def test_energy_tot_loop_ss (e_ref_ss, diff, **kwargs):
            for ix, mc in enumerate (mcp[0]):
                ref_list = [e_ref_ss] + list (get_attr (mc))
                e_test = mc.energy_tot (**kwargs)[0]
                test_list = [e_test] + list (get_attr (mc))
                casestr = 'diff={}; SS; symmetry={}'.format (diff, bool(ix))
                test_energy_tot_crunch (test_list, ref_list, casestr)
        def test_energy_tot_loop_sa (e_ref_sa, diff, **kwargs):
            for ix, mc in enumerate (mcp[1]):
                ref_list = [e_ref_sa] + list (get_attr (mc))
                e_s0_test = mc.energy_tot (**kwargs)[0]
                test_list = [e_s0_test] + list (get_attr (mc))
                sym = bool(ix//2)
                tms = (0,1,'mixed')[ix]
                casestr = 'diff={}; SA; symmetry={}; triplet_ms={}'.format (diff, sym, tms)
                test_energy_tot_crunch (test_list, ref_list, casestr)
        def test_energy_tot_loop (e_ref_ss, e_ref_sa, diff, **kwargs):
            test_energy_tot_loop_ss (e_ref_ss, diff, **kwargs)
            test_energy_tot_loop_sa (e_ref_sa, diff, **kwargs)
        # tBLYP
        e_ref_ss = mcpdft.CASSCF (mcp[0][0], 'tBLYP', 5, 2).kernel ()[0]
        mc_ref = mcpdft.CASSCF (mcp[1][0], 'tBLYP', 5, 2).state_average ([1.0/5,]*5).run ()
        e_ref_sa = mc_ref.e_states[0]
        test_energy_tot_loop (e_ref_ss, e_ref_sa, 'fnal', otxc='tBLYP')
        # grids_level = 2
        e_ref_ss = mcpdft.CASSCF (mcp[0][0], 'tPBE', 5, 2, grids_level=2).kernel ()[0]
        mc_ref = mcpdft.CASSCF (mcp[1][0], 'tPBE', 5, 2, grids_level=2).state_average ([1.0/5,]*5).run ()
        e_ref_sa = mc_ref.e_states[0]
        test_energy_tot_loop (e_ref_ss, e_ref_sa, 'grids', grids_level=2)
        # CASCI wfn
        mc_ref = mcpdft.CASCI (mf_nosym, 'tPBE', 5, 2).run ()
        test_energy_tot_loop_ss (mc_ref.e_tot, 'wfn', mo_coeff=mc_ref.mo_coeff, ci=mc_ref.ci)
        fake_ci = [c.copy () for c in mcp[1][0].ci]
        fake_ci[0] = mc_ref.ci.copy ()
        test_energy_tot_loop_sa (mc_ref.e_tot, 'wfn', mo_coeff=mc_ref.mo_coeff, ci=fake_ci)

    def test_kernel_steps (self):
        ''' kernel is split into optimize_mcscf_ and compute_pdft_energy_.
            I need to make sure that they don't step on each others' toes. '''



if __name__ == "__main__":
    print("Full Tests for MC-PDFT energy API")
    unittest.main()


