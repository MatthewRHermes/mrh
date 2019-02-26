import re, sys
import numpy as np
import scipy as sp 
from pyscf import gto, scf
from pyscf.scf.hf import dot_eri_dm
from pyscf.scf.addons import project_mo_nr2nr
from pyscf.tools import molden
from mrh.my_dmet import pyscf_rhf, pyscf_mp2, pyscf_cc, pyscf_casscf, qcdmethelper, pyscf_fci #, chemps2
from mrh.util import params
from mrh.util.basis import *
from mrh.util.io import prettyprint_ndarray as prettyprint
from mrh.util.io import warnings
from mrh.util.rdm import Schmidt_decomposition_idempotent_wrapper, idempotize_1RDM, get_1RDM_from_OEI, get_2RDM_from_2CDM, get_2CDM_from_2RDM, Schmidt_decompose_1RDM
from mrh.util.tensors import symmetrize_tensor
from mrh.util.my_math import is_close_to_integer
from mrh.my_pyscf.tools.jmol import cas_mo_energy_shift_4_jmol
from functools import reduce
import traceback
import sys

#def make_fragment_atom_list (ints, frag_atom_list, solver_name, active_orb_list = np.empty (0, dtype=int), name="NONE", norbs_bath_max=None, idempotize_thresh=0.0, mf_attr={}, corr_attr={}):
def make_fragment_atom_list (ints, frag_atom_list, solver_name, **kwargs):
    assert (len (frag_atom_list) < ints.mol.natm)
    assert (np.amax (frag_atom_list) < ints.mol.natm)
    ao_offset = ints.mol.offset_ao_by_atom ()
    frag_orb_list = [orb for atom in frag_atom_list for orb in list (range (ao_offset[atom,2], ao_offset[atom,3]))]
    '''
    for atom in range (ints.mol.natm):
        print ("atom_shell_ids({}) = {}".format (atom, ints.mol.atom_shell_ids (atom)))
        print ("angular momentum = {}".format ([ints.mol.bas_angular (shell) for shell in ints.mol.atom_shell_ids (atom)]))
    norbs_in_atom = [int (np.sum ([2 * ints.mol.bas_angular (shell) + 1 for shell in ints.mol.atom_shell_ids (atom)])) for atom in range (ints.mol.natm)]
    print ("norbs_in_atom = {}".format (norbs_in_atom))
    norbs_to_atom = [int (np.sum (norbs_in_atom[:atom])) for atom in range (ints.mol.natm)]
    print ("norbs_to_atom = {}".format (norbs_to_atom))
    frag_orb_list = [i + norbs_to_atom[atom] for atom in frag_atom_list for i in range (norbs_in_atom[atom])]
    print ("frag_orb_list = {}".format (frag_orb_list))
    '''
    print ("Fragment atom list\n{0}\nproduces a fragment orbital list as {1}".format ([ints.mol.atom_symbol (atom) for atom in frag_atom_list], frag_orb_list))
    #return fragment_object (ints, np.asarray (frag_orb_list), solver_name, active_orb_list=np.asarray (active_orb_list), name=name, mf_attr=mf_attr, corr_attr=corr_attr)
    return fragment_object (ints, np.asarray (frag_orb_list), solver_name, **kwargs)

#def make_fragment_orb_list (ints, frag_orb_list, solver_name, active_orb_list = np.empty (0, dtype=int), name="NONE", norbs_bath_max=None, idempotize_thresh=0.0, mf_attr={}, corr_attr={}):
#    return fragment_object (ints, frag_orb_list, solver_name, np.asarray (active_orb_list), name=name, mf_attr=mf_attr, corr_attr=corr_attr)
def make_fragment_orb_list (ints, frag_orb_list, solver_name, **kwargs):
    return fragment_object (ints, frag_orb_list, solver_name, **kwargs)

def dummy_rhf (frag, oneRDM_imp, chempot_imp):
    ''' Skip solving the impurity problem and assume the trial and impurity wave functions are the same. '''
    assert (np.amax (np.abs (chempot_imp)) < 1e-10)
    frag.oneRDM_loc = frag.ints.oneRDM_loc
    frag.twoCDM_imp = None
    frag.E_imp = frag.ints.e_tot
    frag.loc2mo = np.dot (frag.loc2imp, 
        matrix_eigen_control_options (represent_operator_in_basis (frag.ints.activeFOCK, frag.loc2imp),
            sort_vecs=1, only_nonzero_vals=False)[1])

class fragment_object:

    def __init__ (self, ints, frag_orb_list, solver_name, **kwargs): #active_orb_list, name, norbs_bath_max=None, idempotize_thresh=0.0, mf_attr={}, corr_attr={}):

        # I really hope this doesn't copy.  I think it doesn't.
        self.ints = ints
        self.norbs_tot = self.ints.mol.nao_nr ()
        self.frag_orb_list = frag_orb_list
        self.active_orb_list = []
        self.frozen_orb_list = [] 
        self.norbs_frag = len (self.frag_orb_list)
        self.nelec_imp = 0
        self.norbs_bath_max = self.norbs_frag 
        self.solve_time = 0.0
        self.frag_name = 'None'
        self.active_space = None
        self.idempotize_thresh = 0.0
        self.bath_tol = 1e-8
        self.num_mf_stab_checks = 0
        self.target_S = 0
        self.target_MS = 0
        self.mol_output = None
        self.filehead = None
        self.debug_energy = False
        self.imp_maxiter = None # Currently only does anything for casscf solver
        self.quasidirect = True # Currently only does anything for rhf (in development)
        self.project_cderi = False
        self.mf_attr = {}
        self.corr_attr = {}
        self.cas_guess_callback = None
        for key in kwargs:
            if key in self.__dict__:
                self.__dict__[key] = kwargs[key]
        if 'name' in kwargs:
            self.frag_name = kwargs['name']

        # Assign solver function
        solver_function_map = {
            "dummy RHF" : dummy_rhf ,
            "FCI"       : pyscf_fci.solve ,
            "RHF"       : pyscf_rhf.solve ,
            "MP2"       : pyscf_mp2.solve ,
            "CC"        : pyscf_cc.solve ,
            "CASSCF"    : pyscf_casscf.solve
            }
        solver_longname_map = {
            "dummy RHF" : "dummy restricted Hartree-Fock",
            "FCI"       : "full configuration interaction",
            "RHF"       : "restricted Hartree-Fock",
            "MP2"       : "MP2 perturbation theory",
            "CC"        : "coupled-cluster with singles and doubles",
            "CASSCF"    : "complete active space SCF"
            }
        imp_solver_name = re.sub ("\([0-9,]+\)", "", solver_name)
        self.imp_solver_name = imp_solver_name
        self.imp_solver_longname = solver_longname_map[imp_solver_name]
        self.imp_solver_function = solver_function_map[imp_solver_name].__get__(self)
        active_space = re.compile ("\([0-9,]+\)").search (solver_name)
        if active_space:
            self.active_space = eval (active_space.group (0))
            if not (len (self.active_space) == 2):
                raise RuntimeError ("Active space {0} not usable; only CASSCF currently implemented".format (solver.active_space))
            self.imp_solver_longname += " with {0} electrons in {1} active-space orbitals".format (self.active_space[0], self.active_space[1])
                 
        # Set up the main basis functions. Before any Schmidt decomposition all environment states are treated as "core"
        # self.loc2emb is always defined to have the norbs_frag fragment states, the norbs_bath bath states, and the norbs_core core states in that order
        self.restore_default_embedding_basis ()
        self.oneRDMfroz_loc = None
        self.twoCDMfroz_tbc = []
        self.loc2tbc        = []
        self.E2froz_tbc     = []
        self.imp_cache      = []
        
        # Impurity Hamiltonian
        self.Ecore_frag   = 0.0  # In case this exists
        self.impham_CONST = None # Does not include nuclear potential
        self.impham_OEI   = None
        self.impham_TEI   = None
        self.impham_CDERI = None

        # Basic outputs of solving the impurity problem
        self.E_frag = 0.0
        self.E_imp  = 0.0
        self.oneRDM_loc = self.ints.oneRDM_loc
        self.twoCDM_imp = None
        self.loc2mo     = np.zeros((self.norbs_tot,0))
        self.loc2fno    = np.zeros((self.norbs_tot,0))
        self.fno_evals  = None
        self.E2_cum     = 0

        # Outputs of CAS calculations use to fix CAS-DMET
        self.loc2amo       = np.zeros((self.norbs_tot,0))
        self.loc2amo_guess = None
        self.oneRDMas_loc  = np.zeros((self.norbs_tot,self.norbs_tot))
        self.twoCDMimp_amo = np.zeros((0,0,0,0))
        self.ci_as         = None
        self.mfmo_printed  = False
        self.impo_printed  = False

        # Initialize some runtime warning bools
        self.Schmidt_done = False
        self.impham_built = False
        self.imp_solved   = False

        # Report
        print ("Constructed a fragment of {0} orbitals for a system with {1} total orbitals".format (self.norbs_frag, self.norbs_tot))
        print ("Using a {0} [{1}] calculation to solve the impurity problem".format (self.imp_solver_longname, self.imp_solver_method))
        print ("Fragment orbitals: {0}".format (self.frag_orb_list))



    # Common runtime warning checks
    ###########################################################################################################################
    def warn_check_Schmidt (self, cstr="NONE"):
        wstr = "Schmidt decomposition not performed at call to {0}. Undefined behavior likely!".format (cstr)
        return warnings.warn (wstr, RuntimeWarning) if (not self.Schmidt_done) else None

    def warn_check_impham (self, cstr="NONE"):
        wstr =  "Impurity Hamiltonian not built at call to {0} (did you redo the Schmidt decomposition".format (cstr)
        wstr += " without rebuilding impham?). Undefined behavior likely!"
        return warnings.warn (wstr, RuntimeWarning) if not self.impham_built else None

    def warn_check_imp_solve (self, cstr="NONE"):
        wstr =  "Impurity problem not solved at call to {0} (did you redo the Schmidt decomposition or".format (cstr)
        wstr += " rebuild the impurity Hamiltonian without re-solving?). Undefined behavior likely!"
        return warnings.warn (wstr, RuntimeWarning) if not self.imp_solved else None



    # Dependent attributes, never to be modified directly
    ###########################################################################################################################
    @property
    def loc2imp (self):
        return self.loc2emb[:,:self.norbs_imp]

    @property
    def loc2core (self):
        return self.loc2emb[:,self.norbs_imp:]

    @property
    def frag2loc (self):
        return self.loc2frag.conjugate ().T

    @property
    def amo2loc (self):
        return self.loc2amo.conjugate ().T

    @property
    def emb2loc (self):
        return self.loc2emb.conjugate ().T

    @property
    def imp2loc (self):
        return self.loc2imp.conjugate ().T

    @property
    def core2loc (self):
        return self.loc2core.conjugate ().T

    @property
    def mo2loc (self):
        return self.loc2mo.conjugate ().T

    @property
    def imp2frag (self):
        return np.dot (self.imp2loc, self.loc2frag)

    @property
    def frag2imp (self):
        return np.dot (self.frag2loc, self.loc2imp)

    @property
    def amo2imp (self):
        return np.dot (self.amo2loc, self.loc2imp)

    @property
    def imp2amo (self):
        return np.dot (self.imp2loc, self.loc2amo)

    @property
    def mo2imp (self):
        return np.dot (self.mo2loc, self.loc2imp)

    @property
    def imp2mo (self):
        return np.dot (self.imp2loc, self.loc2mo)

    @property
    def amo2frag (self):
        return np.dot (self.amo2loc, self.loc2frag)

    @property
    def frag2amo (self):
        return np.dot (self.frag2loc, self.loc2amo)

    @property
    def is_frag_orb (self):
        r = np.zeros (self.norbs_tot, dtype=bool)
        r[self.frag_orb_list] = True
        return r

    @property
    def is_env_orb (self):
        return np.logical_not (self.is_frag_orb)

    @property
    def env_orb_list (self):
        return np.flatnonzero (self.is_env_orb)

    @property
    def is_active_orb (self):
        r = np.zeros (self.norbs_tot, dtype=bool)
        r[self.active_orb_list] = True
        return r

    @property
    def is_inactive_orb (self):
        return np.logical_not (self.is_active_orb)

    @property
    def inactive_orb_list (self):
        return np.flatnonzero (self.is_inactive_orb)

    @property
    def norbs_as (self):
        return self.loc2amo.shape[1]

    @property
    def nelec_as (self):
        result = np.trace (self.oneRDMas_loc)
        if is_close_to_integer (result, params.num_zero_atol) == False:
            raise RuntimeError ("Somehow you got a non-integer number of electrons in your active space!")
        return int (round (result))

    @property
    def norbs_tbc (self):
        return loc2tbc.shape[1]

    @property
    def norbs_core (self):
        self.warn_check_Schmidt ("norbs_core")
        return self.norbs_tot - self.norbs_imp

    @property
    def imp_solver_method (self):
        if self.active_space:
            return self.imp_solver_name + str (self.active_space)
        else:
            return self.imp_solver_name

    @property
    def loc2tb_all (self):
        if self.imp_solver_name != 'RHF':
            return [self.loc2imp] + self.loc2tbc
        else:
            return self.loc2tbc

    @property
    def twoCDM_all (self):
        if self.imp_solver_name != 'RHF':
            if self.norbs_as > 0:
                return [self.twoCDMimp_amo] + self.twoCDMfroz_tbc
            return [self.twoCDM_imp] + self.twoCDMfroz_tbc
        return self.twoCDMfroz_tbc

    def get_loc2bath (self):
        ''' Don't use this too much... I don't know how it's gonna behave under ofc_emb'''
        loc2nonbath = orthonormalize_a_basis (np.append (self.loc2frag, self.loc2core, axis=1))
        loc2bath = get_complementary_states (loc2nonbath)
        return loc2bath

    def get_true_loc2frag (self):
        return np.eye (self.norbs_tot)[:,self.frag_orb_list]

    ############################################################################################################################




    # The Schmidt decomposition
    ############################################################################################################################
    def restore_default_embedding_basis (self):
        idx                  = np.append (self.frag_orb_list, self.env_orb_list)
        self.norbs_frag      = len (self.frag_orb_list)
        self.norbs_imp       = self.norbs_frag
        self.loc2frag        = self.get_true_loc2frag ()
        self.loc2emb         = np.eye (self.norbs_tot)[:,idx]
        self.E2_frag_core    = 0
        self.twoCDMfroz_tbc  = []
        self.loc2tbc         = []

    def set_new_fragment_basis (self, loc2frag):
        self.loc2frag = loc2frag
        self.loc2emb = np.eye (self.norbs_tot)
        self.norbs_frag = loc2frag.shape[1]
        self.norbs_imp = self.norbs_frag
        self.E2_frag_core = 0
        self.twoCDM_froz_tbc = []
        self.loc2tbc = []

    def do_Schmidt (self, oneRDM_loc, all_frags, loc2wmcs, doLASSCF):
        self.imp_cache = []
        if doLASSCF:
            self.do_Schmidt_LASSCF (oneRDM_loc, all_frags, loc2wmcs)
        else:
            print ("DMET Schmidt decomposition of {0} fragment".format (self.frag_name))
            self.loc2emb, norbs_bath, self.nelec_imp, self.oneRDMfroz_loc = Schmidt_decomposition_idempotent_wrapper (oneRDM_loc, 
                self.loc2frag, self.norbs_bath_max, idempotize_thresh=self.idempotize_thresh, bath_tol=self.bath_tol, num_zero_atol=params.num_zero_atol)
            self.norbs_imp = self.norbs_frag + norbs_bath
            self.Schmidt_done = True
            self.impham_built = False
            self.imp_solved = False
            print ("Final impurity for {0}: {1} electrons in {2} orbitals".format (self.frag_name, self.nelec_imp, self.norbs_imp))
        #if self.impo_printed == False:
        #    self.impurity_molden ('imporb_begin')
        #    self.impo_printed = True
        sys.stdout.flush ()

    def do_Schmidt_LASSCF (self, oneRDM_loc, all_frags, loc2wmcs):
        print ("LASSCF Schmidt decomposition of {0} fragment".format (self.frag_name))
        # First, I should add as many "quasi-fragment" states as there are last-iteration active orbitals, just so I don't
        # lose bath states.
        # How many do I need?
        frag2wmcs = np.dot (self.frag2loc, loc2wmcs)
        proj = np.dot (frag2wmcs.conjugate ().T, frag2wmcs)
        norbs_wmcsf = np.trace (proj)
        norbs_xtra = int (round (self.norbs_frag - norbs_wmcsf))
        assert (norbs_xtra == self.norbs_as), "{} active orbitals, {} extra fragment orbitals".format (self.norbs_as, norbs_xtra)

        # Now get them. (Make sure I don't add active-space orbitals by mistake!)
        if norbs_xtra:
            loc2qfrag, _, svals = get_overlapping_states (loc2wmcs, self.get_true_loc2frag ())
            loc2qenv = get_complementary_states (loc2qfrag, already_complete_warning=False)
            loc2wmas = get_complementary_states (loc2wmcs, already_complete_warning=False)
            loc2qfrag = get_complementary_states (np.concatenate ([self.loc2frag, loc2qenv, loc2wmas], axis=1), already_complete_warning=False)
            norbs_qfrag = min (loc2qfrag.shape[1], norbs_xtra)
            if norbs_qfrag > 0:
                print ("Add {} of {} possible quasi-fragment orbitals ".format (
                    norbs_qfrag, loc2qfrag.shape[1])
                    + "to compensate for {} active orbitals which cannot generate bath states".format (self.norbs_as))
            loc2wfrag = np.append (self.loc2frag, loc2qfrag[:,:norbs_qfrag], axis=1)
            assert (is_basis_orthonormal (loc2wfrag)), prettyprint (np.dot (loc2wfrag.conjugate ().T, loc2wfrag))
        else:
            norbs_qfrag = 0
            loc2wfrag = self.loc2frag
            print ("NO quasi-fragment orbitals constructed")

        # This will RuntimeError on me if I don't have even integer.
        # For safety's sake, I'll project into wmcs subspace and add wmas part back to self.oneRDMfroz_loc afterwards.
        oneRDMi_loc = project_operator_into_subspace (oneRDM_loc, loc2wmcs)
        oneRDMa_loc = oneRDM_loc - oneRDMi_loc
        self.loc2emb, norbs_bath, self.nelec_imp, self.oneRDMfroz_loc = Schmidt_decomposition_idempotent_wrapper (oneRDMi_loc, 
            loc2wfrag, self.norbs_bath_max, bath_tol=self.bath_tol, idempotize_thresh=self.idempotize_thresh, num_zero_atol=params.num_zero_atol)
        self.norbs_imp = self.norbs_frag + norbs_qfrag + norbs_bath
        self.Schmidt_done = True
        oneRDMacore_loc = project_operator_into_subspace (oneRDMa_loc, self.loc2core)
        nelec_impa = compute_nelec_in_subspace (oneRDMa_loc, self.loc2imp)
        nelec_impa_target = 0 if self.active_space is None else self.active_space[0]
        print ("Adding {} active-space electrons to impurity and {} active-space electrons to core".format (nelec_impa, np.trace (oneRDMacore_loc)))
        self.oneRDMfroz_loc += oneRDMacore_loc
        self.nelec_imp += int (round (nelec_impa))

        # Core 2CDMs
        active_frags = [frag for frag in all_frags if frag is not self and frag.norbs_as > 0]
        self.twoCDMfroz_tbc = [np.copy (frag.twoCDMimp_amo) for frag in active_frags]
        self.loc2tbc        = [np.copy (frag.loc2amo) for frag in active_frags]
        self.E2froz_tbc     = [frag.E2_cum for frag in active_frags]

        self.analyze_ao_imp (oneRDM_loc, loc2wmcs)
        self.impham_built = False
        sys.stdout.flush ()

    def analyze_ao_imp (self, oneRDM_loc, loc2wmcs):
        ''' See how much of the atomic-orbitals corresponding to the true fragment ended up in the impurity and how much ended 
            up in the virtual-core space '''
        loc2ao = orthonormalize_a_basis (self.ints.ao2loc[self.frag_orb_list,:].conjugate ().T)
        svals = get_overlapping_states (loc2ao, self.loc2core)[2]
        lost_aos = np.sum (svals)

        oneRDM_wmcs = represent_operator_in_basis (oneRDM_loc, loc2wmcs)
        mo_occ, mo_evec = matrix_eigen_control_options (oneRDM_wmcs, sort_vecs=1, only_nonzero_vals=False)
        idx_virtunac = np.isclose (mo_occ, 0)
        loc2virtunac = np.dot (loc2wmcs, mo_evec[:,idx_virtunac])
        ''' These orbitals have to be splittable into purely on the impurity/purely in the core for the same reason that nelec_imp has to
            be an integer, I think. '''
        loc2virtunaccore, loc2corevirtunac, svals = get_overlapping_states (loc2virtunac, self.loc2core)
        assert (np.all (np.logical_or (np.isclose (svals, 0), np.isclose (svals, 1)))), svals
        idx_virtunaccore = np.isclose (svals, 1)
        loc2virtunaccore = loc2virtunaccore[:,idx_virtunaccore]
        loc2virtbath, svals = get_overlapping_states (loc2ao, loc2virtunaccore)[1:]
        aos_in_virt_core = np.count_nonzero (np.logical_not (np.isclose (svals, 0))) 
        print (("For this Schmidt decomposition, the impurity basis loses {} of {} atomic orbitals, accounted for by"
        " {} of {} virtual core orbitals").format (lost_aos, self.norbs_frag, aos_in_virt_core, loc2virtunaccore.shape[1]))
        check_nocc = np.trace (represent_operator_in_basis (oneRDM_loc, loc2virtbath))
        check_bath = np.amax (np.abs (np.dot (self.imp2loc, loc2virtbath)))
        print ("Are my `virtual bath' orbitals unoccupied and currently in the core? check_nocc = {}, check_bath = {}".format (check_nocc, check_bath))
        idx = np.logical_not (np.isclose (svals, 0))
        print ("Nonzero `virtual bath' singular values: {}".format (svals[idx]))


    ##############################################################################################################################





    # Impurity Hamiltonian
    ###############################################################################################################################
    def construct_impurity_hamiltonian (self, xtra_CONST=0.0):
        self.warn_check_Schmidt ("construct_impurity_hamiltonian")
        if self.imp_solver_name == "dummy RHF":
            self.E2_frag_core = 0
            self.impham_built = True
            self.imp_solved   = False
            return
        self.impham_OEI = self.ints.dmet_fock (self.loc2emb, self.norbs_imp, self.oneRDMfroz_loc)
        if self.imp_solver_name == "RHF" and self.quasidirect:
            ao2imp = np.dot (self.ints.ao2loc, self.loc2imp)
            def my_jk (mol, dm, hermi=1):
                dm_ao        = represent_operator_in_basis (dm, ao2imp.T)
                vj_ao, vk_ao = self.ints.get_jk_ao (dm_ao, hermi)
                vj_basis     = represent_operator_in_basis (vj_ao, ao2imp)
                vk_basis     = represent_operator_in_basis (vk_ao, ao2imp)
                return vj_basis, vk_basis
            self.impham_TEI = None 
            #self.impham_TEI_fiii = None # np.empty ([self.norbs_frag] + [self.norbs_imp for i in range (3)], dtype=np.float64)
            self.impham_get_jk = my_jk
        elif self.project_cderi:
            self.impham_TEI = None
            self.impham_get_jk = None
            self.impham_CDERI = self.ints.dmet_cderi (self.loc2emb, self.norbs_imp)
        else:
            f = self.loc2frag
            i = self.loc2imp
            self.impham_TEI = self.ints.dmet_tei (self.loc2emb, self.norbs_imp) 
            #self.impham_TEI_fiii = self.ints.general_tei ([f, i, i, i])
            self.impham_get_jk = None

        # Constant contribution to energy from core 2CDMs
        self.impham_CONST = (self.ints.dmet_const (self.loc2emb, self.norbs_imp, self.oneRDMfroz_loc)
                             + self.ints.const () + xtra_CONST + sum (self.E2froz_tbc))
        self.E2_frag_core = 0
        '''
        for loc2tb, twoCDM in zip (self.loc2tbc, self.twoCDMfroz_tbc):
            # Impurity energy
            V      = self.ints.dmet_tei (loc2tb)
            L      = twoCDM
            Eimp   = 0.5 * np.tensordot (V, L, axes=4)
            # Fragment energy is zero by construction (old version had nonzero Efrag from core)
            f      = self.loc2frag
            c      = loc2tb
            V      = self.ints.general_tei ([f, c, c, c])
            L      = reduce (lambda x,y: np.tensordot (x, y, axes=1), [self.frag2loc, loc2tb, twoCDM])
            Efrag  = 0 #0.5 * np.tensordot (V, L, axes=4)
            self.impham_CONST += Eimp
            self.E2_frag_core += Efrag
            if self.debug_energy:
                print ("construct_impurity_hamiltonian {0}: Eimp = {1:.5f}, Efrag = {2:.5f} from this 2CDM".format (
                    self.frag_name, float (Eimp), float (Efrag)))
        '''

        self.impham_built = True
        self.imp_solved   = False
        sys.stdout.flush ()
    ###############################################################################################################################


    

    # Solving the impurity problem
    ###############################################################################################################################
    def get_guess_1RDM (self, chempot_imp):
        FOCK = represent_operator_in_basis (self.ints.activeFOCK, self.loc2imp) - chempot_imp
        return (get_1RDM_from_OEI (FOCK, (self.nelec_imp // 2) + self.target_MS)
              + get_1RDM_from_OEI (FOCK, (self.nelec_imp // 2) - self.target_MS))

    def solve_impurity_problem (self, chempot_frag):
        self.warn_check_impham ("solve_impurity_problem")

        # Make chemical potential matrix and guess_1RDM
        chempot_imp = represent_operator_in_basis (chempot_frag * np.eye (self.norbs_frag), self.frag2imp)
        guess_1RDM = self.get_guess_1RDM (chempot_imp)

        # Execute solver function
        self.imp_solver_function (guess_1RDM, chempot_imp)
        self.imp_solved = True

        # Main results: oneRDM in local basis and nelec_frag
        self.nelec_frag = self.get_nelec_frag ()
        self.E_frag     = self.get_E_frag ()
        self.S2_frag    = self.get_S2_frag ()
        print ("Impurity results for {0}: E_imp = {1}, E_frag = {2}, nelec_frag = {3}, S2_frag = {4}".format (self.frag_name,
            self.E_imp, self.E_frag, self.nelec_frag, self.S2_frag))

        # In order to comply with ``NOvecs'' bs, let's get some pseudonatural orbitals
        self.fno_evals, frag2fno = sp.linalg.eigh (self.get_oneRDM_frag ())
        self.loc2fno = np.dot (self.loc2frag, frag2fno)

        # Testing
        '''
        oneRDMimp_loc = represent_operator_in_basis (self.oneRDMimp_imp, self.imp2loc)
        idx = np.ix_(self.frag_orb_list, self.frag_orb_list)
        print ("Number of electrons on {0} from the impurity model: {1}; from the core: {2}".format (
            self.frag_name, np.trace (oneRDMimp_loc[idx]), np.trace (self.oneRDMfroz_loc[idx])))
        '''


    def load_amo_guess_from_casscf_molden (self, moldenfile, norbs_cmo, norbs_amo):
        ''' Use moldenfile from whole-molecule casscf calculation to guess active orbitals '''
        print ("Attempting to load guess active orbitals from {}".format (moldenfile))
        mol, _, mo_coeff, mo_occ = molden.load (moldenfile)[:4]
        print ("Difference btwn self mol coords and moldenfile mol coords: {}".format (sp.linalg.norm (mol.atom_coords () - self.ints.mol.atom_coords ())))
        norbs_occ = norbs_cmo + norbs_amo
        amo_coeff = mo_coeff[:,norbs_cmo:norbs_occ]
        amo_coeff = scf.addons.project_mo_nr2nr (mol, amo_coeff, self.ints.mol)
        self.loc2amo = reduce (np.dot, [self.ints.ao2loc.conjugate ().T, self.ints.ao_ovlp, amo_coeff])
        self.loc2amo = self.retain_fragonly_guess_amo (self.loc2amo)

    def load_amo_guess_from_casscf_npy (self, npyfile, norbs_cmo, norbs_amo):
        ''' Use npy from whole-molecule casscf calculation to guess active orbitals. Must have identical geometry orientation and basis! 
        npyfile must contain an array of shape (norbs_tot+1,norbs_active), where the first row contains natural-orbital occupancies
        for the active orbitals, and subsequent rows contain active natural orbital coefficients.'''
        matrix = np.load (npyfile)
        ano_occ = matrix[0,:]
        ano_coeff = matrix[1:,:]
        loc2ano = reduce (np.dot, (self.ints.ao2loc.conjugate ().T, self.ints.ao_ovlp, ano_coeff))
        oneRDMwm_ano = np.diag (ano_occ)
        frag2ano = loc2ano[self.frag_orb_list,:]
        oneRDMano_frag = represent_operator_in_basis (oneRDMwm_ano, frag2ano.conjugate ().T)
        evals, evecs = matrix_eigen_control_options (oneRDMano_frag, sort_vecs=-1, only_nonzero_vals=False)
        self.loc2amo = np.zeros ((self.norbs_tot, self.active_space[1]))
        self.loc2amo[self.frag_orb_list,:] = evecs[:,:self.active_space[1]]
        #norbs_occ = norbs_cmo + norbs_amo
        #mo_coeff = np.load (npyfile)
        #amo_coeff = mo_coeff[:,norbs_cmo:norbs_occ]
        #self.loc2amo = reduce (np.dot, [self.ints.ao2loc.conjugate ().T, self.ints.ao_ovlp, amo_coeff])
        #self.loc2amo = self.retain_fragonly_guess_amo (self.loc2amo)

    def save_amo_guess_for_pes_scan (self, npyfile):
        no_occ, no_coeff = matrix_eigen_control_options (self.oneRDMas_loc, sort_vecs=-1, only_nonzero_vals=True)
        no_coeff = np.dot (self.ints.ao2loc, no_coeff)
        matrix = np.insert (no_coeff, 0, no_occ, axis=0)
        np.save (npyfile, matrix)

    def load_amo_guess_for_pes_scan (self, npyfile):
        print ("Loading amo guess from npyfile")
        matrix = np.load (npyfile)
        no_occ = matrix[0,:]
        print ("NO occupancies: {}".format (no_occ))
        no_coeff = matrix[1:,:]
        loc2ano = reduce (np.dot, (self.ints.ao2loc.conjugate ().T, self.ints.ao_ovlp, no_coeff))
        ovlp = np.dot (loc2ano.conjugate ().T, loc2ano)
        print ("Active orbital overlap matrix:\n{}".format (prettyprint (ovlp, fmt='{:5.2f}')))
        evals, evecs = matrix_eigen_control_options (ovlp, sort_vecs=-1)
        print ("Overlap eigenvalues: {}".format (evals))
        oneRDM_ano = represent_operator_in_basis (np.diag (no_occ), evecs)
        print ("1RDM_ano (trace = {}):\n{}".format (np.trace (oneRDM_ano), prettyprint (oneRDM_ano, fmt='{:5.2f}')))
        loc2ano = np.dot (loc2ano, evecs) / np.sqrt (evals)
        print ("New overlap matrix:\n{}".format (np.dot (loc2ano.conjugate ().T, loc2ano)))
        m = loc2ano.shape[1]
        self.loc2amo = loc2ano
        self.oneRDMas_loc = represent_operator_in_basis (oneRDM_ano, self.loc2amo.conjugate ().T)
        self.twoCDMimp_amo = np.zeros ((m,m,m,m), dtype=self.oneRDMas_loc.dtype)

    def retain_projected_guess_amo (self, loc2amo_guess):
        print ("Diagonalizing fragment projector in guess amo basis and retaining highest {} eigenvalues".format (self.active_space[1]))
        frag2amo = loc2amo_guess[self.frag_orb_list,:]
        proj = np.dot (frag2amo.conjugate ().T, frag2amo)
        evals, evecs = matrix_eigen_control_options (proj, sort_vecs=-1, only_nonzero_vals=False)
        print ("Projector eigenvalues: {}".format (evals))
        return np.dot (loc2amo_guess, evecs[:,:self.active_space[1]])

    def retain_fragonly_guess_amo (self, loc2amo_guess):
        print ("Diagonalizing guess amo projector in fragment basis and retaining highest {} eigenvalues".format (self.active_space[1]))
        frag2amo = loc2amo_guess[self.frag_orb_list,:]
        proj = np.dot (frag2amo, frag2amo.conjugate ().T)
        evals, evecs = matrix_eigen_control_options (proj, sort_vecs=-1, only_nonzero_vals=False)
        print ("Projector eigenvalues: {}".format (evals))
        return np.dot (self.get_true_loc2frag (), evecs[:,:self.active_space[1]])
        
    def load_amo_guess_from_dmet_molden (self, moldenfile):
        ''' Use moldenfile of impurity from another DMET calculation with the same active space (i.e., at a different geometry) to guess active orbitals '''
        print ("Attempting to load guess active orbitals from {}".format (moldenfile))
        mol, _, mo_coeff, mo_occ = molden.load (moldenfile)[:4]
        nelec_amo, norbs_amo = self.active_space
        nelec_tot = int (round (np.sum (mo_occ)))
        norbs_cmo = (nelec_tot - nelec_amo) // 2
        norbs_occ = norbs_cmo + norbs_amo
        amo_coeff = mo_coeff[:,norbs_cmo:norbs_occ]
        #amo_coeff = scf.addons.project_mo_nr2nr (mol, amo_coeff, self.ints.mol)
        self.loc2amo = reduce (np.dot, [self.ints.ao2loc.conjugate ().T, self.ints.ao_ovlp, amo_coeff])
        self.loc2amo = self.retain_fragonly_guess_amo (self.loc2amo)

    def load_amo_from_aobasis (self, ao2amo, dm, twoRDM=None, twoCDM=None):
        print ("Attempting to load the provided active orbitals to fragment {}".format (self.frag_name))
        self.loc2amo = reduce (np.dot, (self.ints.ao2loc.conjugate ().T, self.ints.ao_ovlp, ao2amo))
        self.oneRDMas_loc = represent_operator_in_basis (dm, self.loc2amo.conjugate ().T)
        if twoRDM is not None:
            self.twoCDMimp_amo = get_2CDM_from_2RDM (twoRDM, dm)
        elif twoCDM is not None:
            self.twoCDMimp_amo = twoCDM
        else:
            self.twoCDMimp_amo = np.zeros ([self.norbs_as for i in range (4)])
              
        

    ###############################################################################################################################





    # Convenience functions and properties for results
    ###############################################################################################################################
    def get_nelec_frag (self):
        self.warn_check_imp_solve ("get_nelec_frag")
        return np.trace (self.get_oneRDM_frag ())

    def get_E_frag (self):
        self.warn_check_imp_solve ("get_E_frag")

        # E_f = H_fi G_fi + 1/2 V_fiii G_fiii                            + E2_frag_core
        #     = H_fi G_fi + 1/2 V_fijkl (G_fi*G_jk - G_fk*G_ji + L_fijk) + E2_frag_core
        #     = (H_fi + 1/2 JK_fi[G]) G_fi + 1/2 V_fiii L_fiii           + E2_frag_core
        #     = (H_fi + JK_fi[1/2 G]) G_fi + 1/2 V_fiii L_fiii           + E2_frag_core
        #     = F[1/2 G]_fi G_fi           + 1/2 V_fiii L_fiii           + E2_frag_core

        if self.imp_solver_name == 'dummy RHF':
            fock = (self.ints.activeFOCK + self.ints.activeOEI)/2
            F_fi = np.dot (self.frag2loc, fock)
            G_fi = np.dot (self.frag2loc, self.oneRDM_loc)
        elif isinstance (self.impham_TEI, np.ndarray):
            vj, vk = dot_eri_dm (self.impham_TEI, self.get_oneRDM_imp (), hermi=1)
            fock = (self.ints.dmet_oei (self.loc2emb, self.norbs_imp) + (self.impham_OEI + vj - vk/2))/2
            F_fi = np.dot (self.frag2imp, fock)
            G_fi = np.dot (self.frag2imp, self.get_oneRDM_imp ())
        else:
            fock = self.ints.loc_rhf_fock_bis (self.oneRDM_loc/2)        
            F_fi = np.dot (self.frag2loc, fock)
            G_fi = np.dot (self.frag2loc, self.oneRDM_loc)

        E1 = np.tensordot (F_fi, G_fi, axes=2)
        if self.debug_energy:
            print ("get_E_frag {0} :: E1 = {1:.5f}".format (self.frag_name, float (E1)))

        E2 = 0
        # Remember that non-overlapping fragments are now, by necessity, ~contained~ within the impurity!
        if self.norbs_as > 0:
            L_fiii = np.tensordot (self.frag2amo, self.twoCDMimp_amo, axes=1)
            if isinstance (self.impham_TEI, np.ndarray):
                mo_coeffs = [self.imp2frag, self.imp2amo, self.imp2amo, self.imp2amo]
                norbs = [self.norbs_frag, self.norbs_as, self.norbs_as, self.norbs_as]
                V_fiii = ao2mo.incore.general (self.impham_TEI, mo_coeffs, compact=False).reshape (*norbs)
            elif isinstance (self.impham_CDERI, np.ndarray):
                with_df = copy.copy (self.ints.with_df)
                with_df._cderi = self.impham_CDERI
                mo_coeffs = [self.imp2frag, self.imp2amo, self.imp2amo, self.imp2amo]
                norbs = [self.norbs_frag, self.norbs_as, self.norbs_as, self.norbs_as]
                V_fiii = with_df.ao2mo (mo_coeffs, compact=False).reshape (*norbs)
            else:
                V_fiii = self.ints.general_tei ([self.loc2frag, self.loc2amo, self.loc2amo, self.loc2amo])
            E2 = 0.5 * np.tensordot (V_fiii, L_fiii, axes=4)
        elif isinstance (self.twoCDM_imp, np.ndarray):
            L_iiif = np.tensordot (self.twoCDM_imp, self.imp2frag, axes=1)
            if isinstance (self.impham_TEI, np.ndarray):
                V_iiif = ao2mo.restore (1, self.impham_TEI, self.norbs_imp)
                V_iiif = np.tensordot (V_iiif, self.imp2frag, axes=1) 
                E2 = 0.5 * np.tensordot (V_iiif, L_iiif, axes=4)
            elif isinstance (self.impham_CDERI, np.ndarray):
                raise NotImplementedError ("No opportunity to test this yet.")
                # It'll be something like:
                # (P|ii) * L_iiif -> R^P_if
                # (P|ii) * u^i_f -> (P|if)
                # (P|if) * R^P_if -> E2
                # But factors of 2 abound especially with the orbital-pair compacting of CDERI
            else:
                V_iiif = self.ints.general_tei ([self.loc2frag, self.loc2imp, self.loc2imp, self.loc2imp])
                E2 = 0.5 * np.tensordot (V_iiif, L_iiif, axes=4)
        if self.debug_energy:
            print ("get_E_frag {0} :: E2 = {1:.5f}".format (self.frag_name, float (E2)))

        if self.debug_energy:
            print ("get_E_frag {0} :: E2_frag_core = {1:.5f}".format (self.frag_name, float (self.E2_frag_core)))

        return float (E1 + E2 + self.E2_frag_core)

    def get_S2_frag (self):
        self.warn_check_imp_solve ("get_S2_frag")
        # S2_f = Tr_f [G - (G**2)/2] - 1/2 sum_fi L_fiif

        dm = self.get_oneRDM_imp ()
        exc_mat = dm - np.dot (dm, dm)/2
        if self.norbs_as > 0:
            exc_mat -= represent_operator_in_basis (np.einsum ('prrq->pq', self.twoCDMimp_amo)/2, self.amo2imp)
        elif isinstance (self.twoCDM_imp, np.ndarray):
            exc_mat -= np.einsum ('prrq->pq', self.twoCDM_imp)/2
        return np.einsum ('fp,pq,qf->', self.frag2imp, exc_mat, self.imp2frag) 

    def get_twoRDM (self, *bases):
        bases = bases if len (bases) == 4 else (basis[0] for i in range[4])
        oneRDM_pq = represent_operator_in_basis (self.oneRDM_loc, bases[0], bases[1])
        oneRDM_rs = represent_operator_in_basis (self.oneRDM_loc, bases[2], bases[3])
        oneRDM_ps = represent_operator_in_basis (self.oneRDM_loc, bases[0], bases[3])
        oneRDM_rq = represent_operator_in_basis (self.oneRDM_loc, bases[2], bases[1])
        twoRDM  =       np.einsum ('pq,rs->pqrs', oneRDM_pq, oneRDM_rs)
        twoRDM -= 0.5 * np.einsum ('ps,rq->pqrs', oneRDM_ps, oneRDM_rq)
        return twoRDM + self.get_twoCDM (*bases)

    def get_twoCDM (self, *bases):
        bases = bases if len (bases) == 4 else (bases[0] for i in range[4])
        bra1_basis, ket1_basis, bra2_basis, ket2_basis = bases
        twoCDM = np.zeros (tuple(basis.shape[1] for basis in bases))
        for loc2tb, twoCDM_tb in zip (self.loc2tb_all, self.twoCDM_all):
            tb2loc = np.conj (loc2tb.T)
            tb2bs = (np.dot (tb2loc, basis) for basis in bases)
            twoCDM += represent_operator_in_basis (twoCDM_tb, *tb2bs)
        return twoCDM

    def get_oneRDM_frag (self):
        return represent_operator_in_basis (self.oneRDM_loc, self.loc2frag)

    def get_oneRDM_imp (self):
        self.warn_check_Schmidt ("oneRDM_imp")
        return represent_operator_in_basis (self.oneRDM_loc, self.loc2imp)

    def impurity_molden (self, tag=None, canonicalize=False, natorb=False, molorb=False, ene=None, occ=None):
        tag = '.' if tag == None else '_' + str (tag) + '.'
        filename = self.filehead + self.frag_name + tag + 'molden'
        mol = self.ints.mol.copy ()
        mol.nelectron = self.nelec_imp
        mol.spin = int (round (2 * self.target_MS))

        oneRDM = self.get_oneRDM_imp ()
        FOCK = represent_operator_in_basis (self.ints.loc_rhf_fock_bis (self.oneRDM_loc), self.loc2imp)
        ao2imp = np.dot (self.ints.ao2loc, self.loc2imp) 

        ao2molden = ao2imp
        if molorb:
            assert (not natorb)
            assert (not canonicalize)
            ao2molden = np.dot (self.ints.ao2loc, self.loc2mo)
            occ = np.einsum ('ip,ij,jp->p', self.imp2mo.conjugate (), oneRDM, self.imp2mo)
            ene = np.einsum ('ip,ij,jp->p', self.imp2mo.conjugate (), FOCK, self.imp2mo)
            if self.norbs_as > 0:
                ene = cas_mo_energy_shift_4_jmol (ene, self.norbs_imp, self.nelec_imp, self.norbs_as, self.nelec_as)
        if natorb:
            assert (not canonicalize)
            # Separate active and external (inactive + virtual) orbitals and pry their energies apart by +-1e4 Eh so Jmol sets them 
            # in the proper order
            if self.norbs_as > 0:
                imp2xmo = get_complementary_states (self.imp2amo)
                FOCK_amo = represent_operator_in_basis (FOCK, self.imp2amo)
                FOCK_xmo = represent_operator_in_basis (FOCK, imp2xmo)
                oneRDM_xmo = represent_operator_in_basis (oneRDM, imp2xmo)
                oneRDM_amo = represent_operator_in_basis (oneRDM, self.imp2amo)
                ene_xmo, xmo2molden = matrix_eigen_control_options (FOCK_xmo, sort_vecs=1, only_nonzero_vals=False)
                occ_amo, amo2molden = matrix_eigen_control_options (oneRDM_amo, sort_vecs=-1, only_nonzero_vals=False)
                occ_xmo = np.einsum ('ip,ij,jp->p', xmo2molden.conjugate (), oneRDM_xmo, xmo2molden)
                ene_amo = np.einsum ('ip,ij,jp->p', amo2molden.conjugate (), FOCK_amo, amo2molden)
                norbs_imo = (self.nelec_imp - self.nelec_as) // 2
                occ = np.concatenate ((occ_xmo[:norbs_imo], occ_amo, occ_xmo[norbs_imo:]))
                ene = np.concatenate ((ene_xmo[:norbs_imo], ene_amo, ene_xmo[norbs_imo:]))
                ene = cas_mo_energy_shift_4_jmol (ene, self.norbs_imp, self.nelec_imp, self.norbs_as, self.nelec_as)
                imp2molden = np.concatenate ((np.dot (imp2xmo, xmo2molden[:,:norbs_imo]),
                                              np.dot (self.imp2amo, amo2molden),
                                              np.dot (imp2xmo, xmo2molden[:,norbs_imo:])), axis=1)
            else:
                occ, imp2molden = matrix_eigen_control_options (oneRDM, sort_vecs=-1, only_nonzero_vals=False)
                ene = np.einsum ('ip,ij,jp->p', imp2molden.conjugate (), FOCK, imp2molden)
            ao2molden = np.dot (ao2imp, imp2molden)
        elif canonicalize:
            # Separate active and external (inactive + virtual) orbitals and pry their energies apart by +-1e4 Eh so Jmol sets them 
            # in the proper order
            if self.norbs_as > 0:
                imp2xmo = get_complementary_states (self.imp2amo)
                FOCK_amo = represent_operator_in_basis (FOCK, self.imp2amo)
                FOCK_xmo = represent_operator_in_basis (FOCK, imp2xmo)
                ene_xmo, xmo2molden = matrix_eigen_control_options (FOCK_xmo, sort_vecs=1, only_nonzero_vals=False)
                ene_amo, amo1molden = matrix_eigen_control_options (FOCK_amo, sort_vecs=1, only_nonzero_vals=False)
                norbs_imo = (self.nelec_imp - self.nelec_as) // 2
                ene = np.concatenate ((ene_xmo[:norbs_imo], ene_amo, ene_xmo[norbs_imo:]))
                ene = cas_mo_energy_shift_4_jmol (ene, self.norbs_imp, self.nelec_imp, self.norbs_as, self.nelec_as)
                imp2molden = np.concatenate ((np.dot (imp2xmo, xmo2molden[:,:norbs_imo]),
                                              np.dot (self.imp2amo, amo2molden),
                                              np.dot (imp2xmo, xmo2molden[:,norbs_imo:])), axis=1)
            else:
                ene, imp2molden = matrix_eigen_control_options (FOCK, sort_vecs=-1, only_nonzero_vals=False)
            occ = np.einsum ('ip,ij,jp->p', imp2molden.conjugate (), oneRDM, imp2molden)
            ao2molden = np.dot (ao2imp, imp2molden)

        molden.from_mo (mol, filename, ao2molden, ene=ene, occ=occ)


    ###############################################################################################################################




    # For interface with DMET
    ###############################################################################################################################
    def get_errvec (self, dmet, mf_1RDM_loc):
        self.warn_check_imp_solve ("get_errvec")
        # Fragment natural-orbital basis matrix elements needed
        if dmet.doDET_NO:
            mf_1RDM_fno = represent_operator_in_basis (mf_1RDM_loc, self.loc2fno)
            return np.diag (mf_1RDM_fno) - self.fno_evals
        # Bath-orbital matrix elements needed
        if dmet.incl_bath_errvec:
            mf_err1RDM_imp = represent_operator_in_basis (mf_1RDM_loc, self.loc2imp) - self.get_oneRDM_imp ()
            return mf_err1RDM_imp.flatten (order='F')
        # Only fragment-orbital matrix elements needed
        mf_err1RDM_frag = represent_operator_in_basis (mf_1RDM_loc, self.loc2frag) - self.get_oneRDM_frag ()
        if dmet.doDET:
            return np.diag (mf_err1RDM_frag)
        elif dmet.altcostfunc:
            return mf_err1RDM_frag[np.triu_indices(self.norbs_frag)]
        else:
            return self.get_oneRDM_frag ().flatten (order='F')


    def get_rsp_1RDM_elements (self, dmet, rsp_1RDM):
        self.warn_check_imp_solve ("get_rsp_1RDM_elements")
        if dmet.altcostfunc:
            raise RuntimeError("You shouldn't have gotten in to get_rsp_1RDM_elements if you're using the constrained-optimization cost function!")
        # If the error function is working in the fragment NO basis, then rsp_1RDM will already be in that basis. Otherwise it will be in the local basis
        if dmet.doDET_NO:
            return np.diag (rsp_1RDM)[self.frag_orb_list]
        # Bath-orbital matrix elements needed
        if dmet.incl_bath_errvec:
            rsp_1RDM_imp = represent_operator_in_basis (rsp_1RDM, self.loc2imp)
            return rsp_1RDM_imp.flatten (order='F')
        # Only fragment-orbital matrix elements needed
        rsp_1RDM_frag = represent_operator_in_basis (rsp_1RDM, self.loc2frag)
        if dmet.doDET:
            return np.diag (rsp_1RDM_frag)
        else:
            return rsp_1RDM_frag.flatten (order='F')






