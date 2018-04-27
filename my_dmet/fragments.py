# Matt, DON'T do any hairy bullshit until you can just reproduce the results of the existing code!


import re
import numpy as np
from pyscf import gto
from . import chemps2, pyscf_rhf, pyscf_mp2, pyscf_cc, pyscf_casscf, qcdmethelper
from mrh.util import params
from mrh.util.basis import *
from mrh.util.rdm import Schmidt_decomposition_idempotent_wrapper, idempotize_1RDM, get_1RDM_from_OEI, get_2RDM_from_2RDMR, get_2RDMR_from_2RDM
from mrh.util.tensors import symmetrize_tensors
from mrh.util.my_math import is_close_to_integer
import warnings
import traceback
import sys

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

warnings.showwarning = warn_with_traceback

def make_fragment_atom_list (ints, frag_atom_list, solver_name, active_orb_list = np.empty (0, dtype=int), name="NONE", norbs_bath_max=None, idempotize_thresh=0.0):
    assert (len (frag_atom_list) < ints.mol.natm)
    assert (np.amax (frag_atom_list) < ints.mol.natm)
    norbs_in_atom = [int (np.sum ([2 * ints.mol.bas_angular (shell) + 1 for shell in ints.mol.atom_shell_ids (atom)])) for atom in range (ints.mol.natm)]
    norbs_to_atom = [int (np.sum (norbs_in_atom[:atom])) for atom in range (ints.mol.natm)]
    frag_orb_list = [i + norbs_to_atom[atom] for atom in frag_atom_list for i in range (norbs_in_atom[atom])]
    print ("Fragment atom list\n{0}\nproduces a fragment orbital list as {1}".format ([ints.mol.atom[atom] for atom in frag_atom_list], frag_orb_list))
    return fragment_object (ints, np.asarray (frag_orb_list), solver_name, active_orb_list=np.asarray (active_orb_list), name=name)

def make_fragment_orb_list (ints, frag_orb_list, solver_name, active_orb_list = np.empty (0, dtype=int), name="NONE", norbs_bath_max=None, idempotize_thresh=0.0):
    return fragment_object (ints, frag_orb_list, solver_name, np.asarray (active_orb_list), name=name)


class fragment_object:

    def __init__ (self, ints, frag_orb_list, solver_name, active_orb_list, name, norbs_bath_max=None, idempotize_thresh=0.0):

        # I really hope this doesn't copy.  I think it doesn't.
        self.ints = ints
        self.norbs_tot = self.ints.mol.nao_nr ()
        self.frag_orb_list = frag_orb_list
        self.active_orb_list = active_orb_list
        self.norbs_frag = len (self.frag_orb_list)
        self.nelec_imp = 0
        self.norbs_bath_max = self.norbs_frag if norbs_bath_max == None else norbs_bath_max
        self.solve_time = 0.0
        self.frag_name = name
        self.active_space = None
        self.idempotize_thresh = abs (idempotize_thresh)

        # Assign solver function
        solver_function_map = {
            "ED"     : chemps2.solve ,
            "RHF"    : pyscf_rhf.solve ,
            "MP2"    : pyscf_mp2.solve ,
            "CC"     : pyscf_cc.solve ,
            "CASSCF" : pyscf_casscf.solve
            }
        solver_longname_map = {
            "ED"     : "exact diagonalization",
            "RHF"    : "restricted Hartree-Fock",
            "MP2"    : "MP2 perturbation theory",
            "CC"     : "coupled-cluster with singles and doubles",
            "CASSCF" : "complete active space SCF"
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
        self.loc2frag = np.eye(self.norbs_tot)[:,self.frag_orb_list]
        self.restore_default_embedding_basis ()
        self.oneRDMfroz_loc  = None
        self.twoRDMRfroz_tbc = []
        self.loc2tbc         = []
        
        # Impurity Hamiltonian
        self.Ecore_frag   = 0.0  # In case this exists
        self.impham_CONST = None # Does not include nuclear potential
        self.impham_OEI   = None
        self.impham_FOCK  = None
        self.impham_TEI   = None

        # Basic outputs of solving the impurity problem
        self.E_frag = 0.0
        self.E_imp  = 0.0
        self.oneRDM_loc     = None
        self.twoRDMRimp_imp = None
        self.loc2fno        = None
        self.fno_evals      = None

        # Outputs of CAS calculations use to fix CAS-DMET
        self.loc2as        = np.zeros((self.norbs_tot,0))
        self.oneRDMas_loc  = np.zeros((self.norbs_tot,self.norbs_tot))
        self.twoRDMRimp_as = np.zeros((0,0,0,0))

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
        loc2frag = np.asmatrix (self.loc2frag)
        return np.asarray (loc2frag.H)

    @property
    def as2loc (self):
        loc2as = np.asmatrix (self.loc2as)
        return np.asarray (loc2as.H)

    @property
    def emb2loc (self):
        loc2emb = np.asmatrix (self.loc2emb)
        return np.asarray (loc2emb.H)

    @property
    def imp2loc (self):
        loc2imp = np.asmatrix (self.loc2imp)
        return np.asarray (loc2imp.H)

    @property
    def core2loc (self):
        loc2core = np.asmatrix (self.loc2core)
        return np.asarray (loc1core.H)

    @property
    def imp2frag (self):
        return np.dot (self.imp2loc, self.loc2frag)

    @property
    def frag2imp (self):
        return np.dot (self.frag2loc, self.loc2imp)

    @property
    def as2imp (self):
        return np.dot (self.as2loc, self.loc2imp)

    @property
    def imp2as (self):
        return np.dot (self.imp2loc, self.loc2as)

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
        return self.loc2as.shape[1]

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
    def TEI_tbc (self):
        return [self.ints.dmet_tei (loc2tb, loc2tb.shape[1]) for loc2tb in self.loc2tbc]

    @property
    def TEI_all (self):
        if self.imp_solver_name != 'RHF':
            return [self.impham_TEI] + self.TEI_tbc
        else:
            return self.TEI_tbc

    @property
    def twoRDMR_all (self):
        if self.imp_solver_name != 'RHF':
            return [self.twoRDMRimp_imp] + self.twoRDMRfroz_tbc
        else:
            return self.twoRDMRfroz_tbc
    ############################################################################################################################




    # The Schmidt decomposition
    ############################################################################################################################
    def restore_default_embedding_basis (self):
        idx                  = np.append (self.frag_orb_list, self.env_orb_list)
        self.norbs_imp       = self.norbs_frag
        self.loc2frag        = np.eye (self.norbs_tot)[:,self.frag_orb_list]
        self.loc2emb         = np.eye (self.norbs_tot)[:,idx]
        self.twoRDMRfroz_tbc = []
        self.loc2tbc         = []

    def do_Schmidt (self, oneRDM_loc, all_frags, do_ofc_embedding):
        if do_ofc_embedding:
            self.do_Schmidt_ofc_embedding (oneRDM_loc, all_frags)
        else:
            self.do_Schmidt_normal (oneRDM_loc)

    def do_Schmidt_normal (self, oneRDM_loc):
        print ("Normal Schmidt decomposition of {0} fragment".format (self.frag_name))
        self.restore_default_embedding_basis ()
        self.loc2emb, norbs_bath, self.nelec_imp, self.oneRDMfroz_loc = Schmidt_decomposition_idempotent_wrapper (oneRDM_loc, 
            self.loc2frag, self.norbs_bath_max, idempotize_thresh=self.idempotize_thresh, num_zero_atol=params.num_zero_atol)
        self.norbs_imp = self.norbs_frag + norbs_bath
        self.Schmidt_done = True
        self.impham_built = False
        self.imp_solved = False
        print ("Final impurity for {0}: {1} electrons in {2} orbitals".format (self.frag_name, self.nelec_imp, self.norbs_imp))

    def do_Schmidt_ofc_embedding (self, oneRDM_loc, all_frags):
        print ("Other-fragment-core Schmidt decomposition of {0} fragment".format (self.frag_name))
        self.restore_default_embedding_basis ()
        other_frags = [frag for frag in all_frags if (frag is not self) and (frag.norbs_as > 0)]

        # (Re)build the whole-molecule active and core spaces
        loc2wmas = np.concatenate ([frag.loc2as for frag in all_frags], axis=1)
        loc2wmcs = get_complementary_states (loc2wmas)

        # For now, let's not mix loc2wmcs up.  Let's just find the best choices for this fragment
        '''
        frag2wmcs = loc2wmcs[self.frag_orb_list,:]
        Pfrag_wmcs = np.diag (np.dot (frag2wmcs.conjugate ().T, frag2wmcs))
        print ("Pfrag_wmcs diagonals for {0} fragment is: {1}".format (self.frag_name, Pfrag_wmcs))
        idx = Pfrag_wmcs.argsort ()[::-1]
        print ("The best wmcs states for {0} fragment are: {1}".format (self.frag_name, idx[:self.norbs_frag-self.norbs_as]))
        '''


        # Starting with Schmidt decomposition in the idempotent subspace
        oneRDMwmcs_loc      = project_operator_into_subspace (oneRDM_loc, loc2wmcs)
        loc2ifrag, _, svals = get_overlapping_states (loc2wmcs, self.loc2frag)
        norbs_ifrag         = loc2ifrag.shape[1]
        assert (not np.any (svals > 1.0 + params.num_zero_atol)), "{0}".format (svals)
        print ("{0} fragment orbitals becoming {1} pseudo-fragment orbitals in idempotent subspace".format (self.norbs_frag, norbs_ifrag))
        loc2iemb, norbs_ibath, nelec_iimp, self.oneRDMfroz_loc = Schmidt_decomposition_idempotent_wrapper (oneRDMwmcs_loc, 
            loc2ifrag, self.norbs_bath_max, idempotize_thresh=self.idempotize_thresh, num_zero_atol=params.num_zero_atol)
        norbs_iimp = norbs_ifrag + norbs_ibath
        loc2iimp   = loc2iemb[:,:norbs_iimp]
        assert (is_matrix_zero (np.dot (loc2wmas.T, loc2iimp))), "{0}".format (np.dot (loc2wmas.T, loc2iimp))

        # Add this-fragment active-space orbitals from last iteration to the impurity
        print ("Adding {0} this-fragment active-space orbitals and {1} this-fragment active-space electrons to the impurity".format (self.norbs_as, self.nelec_as))
        self.nelec_imp = nelec_iimp + self.nelec_as
        self.norbs_imp = norbs_iimp + self.norbs_as
        loc2imp        = np.append (loc2iimp, self.loc2as, axis=1)
        assert (is_basis_orthonormal (loc2imp))
        self.loc2emb   = get_complete_basis (loc2imp)

        # Keep track of where the fragment orbitals went
        olap_mag  = measure_basis_olap (self.loc2frag, self.loc2imp)[0]
        print ("Final impurity model overlap with raw fragment states: {0} / {1}".format (olap_mag, self.norbs_frag))
        olap_mag += measure_basis_olap (self.loc2frag, loc2wmas)[0]
        olap_mag -= measure_basis_olap (self.loc2frag, self.loc2as)[0]
        if (abs (olap_mag - self.norbs_frag) > params.num_zero_atol):
            raise RuntimeError ("Fragment states have vanished? olap_mag = {0} / {1}".format (olap_mag, self.norbs_frag))

        # Add other-fragment active-space RDMs to core RDMs
        self.oneRDMfroz_loc += sum([ofrag.oneRDMas_loc for ofrag in other_frags])
        self.twoRDMRfroz_tbc = [np.copy (ofrag.twoRDMRimp_as) for ofrag in other_frags]
        self.loc2tbc         = [np.copy (ofrag.loc2as) for ofrag in other_frags]

        nelec_bleed = compute_nelec_in_subspace (self.oneRDMfroz_loc, self.loc2imp)
        assert (nelec_bleed < params.num_zero_atol), "Core electrons on the impurity! Overlapping active states?"
        print ("Final impurity for {0}: {1} electrons in {2} orbitals".format (self.frag_name, self.nelec_imp, self.norbs_imp))
        self.Schmidt_done = True
        self.impham_built = False

    ##############################################################################################################################





    # Impurity Hamiltonian
    ###############################################################################################################################
    def construct_impurity_hamiltonian (self, xtra_CONST=0.0):
        self.warn_check_Schmidt ("construct_impurity_hamiltonian")
        self.impham_CONST  = self.ints.dmet_const (self.loc2emb, self.norbs_imp, self.oneRDMfroz_loc) + self.ints.const () + xtra_CONST
        self.impham_OEI    = self.ints.dmet_fock (self.loc2emb, self.norbs_imp, self.oneRDMfroz_loc)
        self.impham_TEI    = self.ints.dmet_tei (self.loc2emb, self.norbs_imp)
        self.impham_CONST += sum ([0.5 * np.einsum ('ijkl,ijkl->', TEI, twoRDMR) for TEI, twoRDMR in zip (
                                    self.TEI_tbc, self.twoRDMRfroz_tbc)])
        self.impham_built  = True
        self.imp_solved    = False
    ###############################################################################################################################


    

    # Solving the impurity problem
    ###############################################################################################################################
    def get_guess_1RDM (self, chempot_frag):
        return self.ints.dmet_init_guess_rhf (self.loc2emb, self.norbs_imp, self.nelec_imp // 2, self.norbs_frag, chempot_frag)

    def solve_impurity_problem (self, chempot_frag):
        self.warn_check_impham ("solve_impurity_problem")

        # Make chemical potential matrix and guess_1RDM
        chempot_imp = represent_operator_in_basis (chempot_frag * np.eye (self.norbs_frag), self.frag2imp)
        guess_1RDM = 2.0 * get_1RDM_from_OEI (self.impham_OEI + chempot_imp, self.nelec_imp // 2)

        # Execute solver function
        self.imp_solver_function (guess_1RDM, chempot_imp)
        self.imp_solved = True

        # Main results: oneRDM in local basis and nelec_frag
        self.nelec_frag = self.get_nelec_frag ()
        self.E_frag     = self.get_E_frag ()
        print ("Impurity results for {0}: E_imp = {1}, E_frag = {2}, nelec_frag = {3}".format (self.frag_name,
            self.E_imp, self.E_frag, self.nelec_frag))

        # In order to comply with ``NOvecs'' bs, let's get some pseudonatural orbitals
        self.fno_evals, frag2fno = np.linalg.eigh (self.get_oneRDM_frag ())
        self.loc2fno = np.dot (self.loc2frag, frag2fno)

        # Testing
        '''
        oneRDMimp_loc = represent_operator_in_basis (self.oneRDMimp_imp, self.imp2loc)
        idx = np.ix_(self.frag_orb_list, self.frag_orb_list)
        print ("Number of electrons on {0} from the impurity model: {1}; from the core: {2}".format (
            self.frag_name, np.trace (oneRDMimp_loc[idx]), np.trace (self.oneRDMfroz_loc[idx])))
        '''
    ###############################################################################################################################





    # Convenience functions and properties for results
    ###############################################################################################################################
    def get_nelec_frag (self):
        self.warn_check_imp_solve ("get_nelec_frag")
        return np.trace (self.get_oneRDM_frag ())

    def get_E_frag (self):
        self.warn_check_imp_solve ("get_E_frag")
        E1_loc  = 0.5 * np.einsum ('ij,ij->i', self.ints.activeOEI, self.oneRDM_loc)
        E1_loc += 0.5 * np.einsum ('ij,ij->j', self.ints.activeOEI, self.oneRDM_loc)
        E1 = np.sum (np.dot (E1_loc, self.loc2frag))
        if self.debug_energy:
            print ("get_E_frag {0} :: E1 = {1:.5f}".format (self.frag_name, E1))

        JK_loc     = self.ints.loc_rhf_jk_bis (0.5 * self.oneRDM_loc)
        E1_JK_loc  = 0.5 * np.einsum ('ij,ij->i', JK_loc, self.oneRDM_loc)
        E1_JK_loc += 0.5 * np.einsum ('ij,ij->j', JK_loc, self.oneRDM_loc)
        E1_JK = np.sum (np.dot (E1_JK_loc, self.loc2frag))
        if self.debug_energy:
            print ("get_E_frag {0} :: E_JK_frag = {1:.5f}".format (self.frag_name, E1_JK))

        E2 = 0.0
        for loc2bas, twoRDMR in zip (self.loc2tb_all, self.twoRDMR_all):
            E2_t = 0.0
            frag2bas = np.dot (self.frag2loc, loc2bas)
            contractions = ['ip,pqrs->iqrs',
                            'iq,pqrs->pirs',
                            'ir,pqrs->pqis',
                            'is,pqrs->pqri']
            for idx, contraction in zip (range(4), contractions):
                bases      = [loc2bas for i in range(4)]
                bases[idx] = self.loc2frag
                V = self.ints.general_tei (bases)
                G = np.einsum (contraction, frag2bas, twoRDMR) 
                E2_t += 0.125 * np.einsum ('pqrs,pqrs->', V, G)
            if self.debug_energy:
                print ("get_E_frag {0} :: E2 from this 2RDMR = {1:.5f}".format (self.frag_name, E2_t))
            E2 += E2_t
        if self.debug_energy:
            print ("get_E_frag {0} :: E2 = {1:.5f}".format (self.frag_name, E2))

        return E1 + E1_JK + E2

    def get_twoRDM (self, *bases):
        bases = bases if len (bases) == 4 else (basis[0] for i in range[4])
        oneRDM_pq = represent_operator_in_basis (self.oneRDM_loc, bases[0], bases[1])
        oneRDM_rs = represent_operator_in_basis (self.oneRDM_loc, bases[2], bases[3])
        oneRDM_ps = represent_operator_in_basis (self.oneRDM_loc, bases[0], bases[3])
        oneRDM_rq = represent_operator_in_basis (self.oneRDM_loc, bases[2], bases[1])
        twoRDM  =       np.einsum ('pq,rs->pqrs', oneRDM_pq, oneRDM_rs)
        twoRDM -= 0.5 * np.einsum ('ps,rq->pqrs', oneRDM_ps, oneRDM_rq)
        return twoRDM + self.get_twoRDMR (*bases)

    def get_twoRDMR (self, *bases):
        bases = bases if len (bases) == 4 else (basis[0] for i in range[4])
        bra1_basis, ket1_basis, bra2_basis, ket2_basis = bases
        twoRDMR = np.zeros (tuple(basis.shape[1] for basis in bases))
        for loc2tb, twoRDMR_tb in zip (self.loc2tb_all, self.twoRDMR_all):
            tb2loc = np.conj (loc2tb.T)
            tb2bs = (np.dot (tb2loc, basis) for basis in bases)
            twoRDMR += represent_operator_in_basis (twoRDMR_tb, *tb2bs)
        return twoRDMR

    def get_oneRDM_frag (self):
        self.warn_check_imp_solve ("oneRDM_frag")
        return represent_operator_in_basis (self.oneRDM_loc, self.loc2frag)

    def get_oneRDM_imp (self):
        self.warn_check_imp_solve ("oneRDM_imp")
        return represent_operator_in_basis (self.oneRDM_loc, self.loc2imp)
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


