# Matt, DON'T do any hairy bullshit until you can just reproduce the results of the existing code!


import re
import numpy as np
from pyscf import gto
from . import chemps2, pyscf_rhf, pyscf_mp2, pyscf_cc, pyscf_casscf, qcdmethelper
from mrh.util.basis import *
from mrh.util.rdm import Schmidt_decomposition_idempotent_wrapper, idempotize_1RDM, get_1RDM_from_OEI, get_2RDM_from_2RDMR, get_2RDMR_from_2RDM
from mrh.util.tensors import symmetrize_tensor
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
        self.num_zero_atol = 1.0e-8
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
        self.TEI_tbc         = []
        
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
        self.loc2as       = np.zeros((self.norbs_tot,0))
        self.oneRDMas_loc = np.zeros((self.norbs_tot,self.norbs_tot))
        self.twoRDMR_as   = np.zeros((0,0,0,0))

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
        if is_close_to_integer (result, self.num_zero_atol) == False:
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
    ############################################################################################################################




    # The Schmidt decomposition
    ############################################################################################################################
    def restore_default_embedding_basis (self):
        idx                  = np.append (self.frag_orb_list, self.env_orb_list)
        self.norbs_imp       = self.norbs_frag
        self.loc2emb         = np.eye (self.norbs_tot)[:,idx]
        self.twoRDMRfroz_tbc = []
        self.loc2tbc         = []
        self.TEI_tbc         = []

    def do_Schmidt (self, oneRDM_loc, all_frags, do_ofc_embedding):
        if do_ofc_embedding:
            self.do_Schmidt_ofc_embedding (oneRDM_loc, all_frags)
        else:
            self.do_Schmidt_normal (oneRDM_loc)

    def do_Schmidt_normal (self, oneRDM_loc):
        print ("Normal Schmidt decomposition of {0} fragment".format (self.frag_name))
        self.restore_default_embedding_basis ()
        self.loc2emb, norbs_bath, self.nelec_imp, self.oneRDMfroz_loc = Schmidt_decomposition_idempotent_wrapper (oneRDM_loc, 
            self.loc2frag, self.norbs_bath_max, idempotize_thresh=self.idempotize_thresh, num_zero_atol=self.num_zero_atol)
        self.norbs_imp = self.norbs_frag + norbs_bath
        self.Schmidt_done = True
        self.impham_built = False
        self.imp_solved = False
        print ("Final impurity for {0}: {1} electrons in {2} orbitals".format (self.frag_name, self.nelec_imp, self.norbs_imp))

    def do_Schmidt_ofc_embedding (self, oneRDM_loc, all_frags):
        print ("Other-fragment-core Schmidt decomposition of {0} fragment".format (self.frag_name))
        self.restore_default_embedding_basis ()
        other_frags = [frag for frag in all_frags if frag is not self]

        # (Re)build the whole-molecule active and core spaces
        loc2wmas = np.concatenate ([frag.loc2as for frag in all_frags], axis=1)
        loc2wmcs = get_complementary_states (loc2wmas)

        # Starting with Schmidt decomposition in the idempotent subspace
        oneRDMwmcs_loc      = project_operator_into_subspace (oneRDM_loc, loc2wmcs)
        loc2ifrag, _, svals = get_overlapping_states (loc2wmcs, self.loc2frag)
        norbs_ifrag         = loc2ifrag.shape[1]
        assert (not np.any (svals > 1.0 + self.num_zero_atol)), "{0}".format (svals)
        print ("{0} fragment orbitals becoming {1} pseudo-fragment orbitals in idempotent subspace".format (self.norbs_frag, norbs_ifrag))
        loc2iemb, norbs_ibath, nelec_iimp, self.oneRDMfroz_loc = Schmidt_decomposition_idempotent_wrapper (oneRDMwmcs_loc, 
            loc2ifrag, self.norbs_bath_max, idempotize_thresh=self.idempotize_thresh, num_zero_atol=self.num_zero_atol)
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
        if (abs (olap_mag - self.norbs_frag) > self.num_zero_atol):
            raise RuntimeError ("Fragment states have vanished? olap_mag = {0} / {1}".format (olap_mag, self.norbs_frag))

        # Add other-fragment active-space RDMs to core RDMs
        ofrags_w_as          = [ofrag for ofrag in other_frags if ofrag.norbs_as > 0]
        ofrags_1RDM_as       = [represent_operator_in_basis (ofrag.oneRDM_loc, ofrag.loc2as) for ofrag in ofrags_w_as]
        ofrags_2RDM_as       = [get_2RDM_from_2RDMR (ofrag.twoRDMR_as, oneRDM) for ofrag, oneRDM in zip (ofrags_w_as, ofrags_1RDM_as)]
        self.oneRDMfroz_loc  += sum([ofrag.oneRDMas_loc for ofrag in ofrags_w_as])
        oneRDMfroz_ofas      = [represent_operator_in_basis (self.oneRDMfroz_loc, ofrag.loc2as) for ofrag in ofrags_w_as]
        self.twoRDMRfroz_tbc = [get_2RDMR_from_2RDM (twoRDM, oneRDM) for oneRDM, twoRDM in zip (oneRDMfroz_ofas, ofrags_2RDM_as)]
        self.loc2tbc         = [np.copy (ofrag.loc2as) for ofrag in ofrags_w_as]
        self.TEI_tbc         = [self.ints.dmet_tei (ofrag.loc2as, ofrag.norbs_as) for ofrag in ofrags_w_as]
        nelec_bleed          = compute_nelec_in_subspace (self.oneRDMfroz_loc, self.loc2imp)
        print ("Found {0} electrons from the core bleeding onto impurity states".format (nelec_bleed))
        print ("(If this number is large, you either are dealing with overlapping fragment active spaces or you made an error)")

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
        chempot_loc = chempot_frag * np.diag (self.is_frag_orb).astype (int)
        chempot_imp = represent_operator_in_basis (chempot_loc, self.loc2imp)
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
        E1 = np.sum (E1_loc[self.frag_orb_list])
        if self.debug_energy:
            print ("get_E_frag {0} :: E1 = {1:.5f}".format (self.frag_name, E1))

        JK_loc  = self.ints.loc_rhf_jk_bis (0.5 * self.oneRDM_loc)
        E2_loc  = 0.5 * np.einsum ('ij,ij->i', JK_loc, self.oneRDM_loc)
        E2_loc += 0.5 * np.einsum ('ij,ij->j', JK_loc, self.oneRDM_loc)

        '''
        This brute-force recovery of the full twoRDM ~works~, which means the problem ~must~ somehow
        be in the code below. I can't for the life of me fucking figure out what's wrong though!
        TEI_loc     = np.copy (self.ints.activeERI)
        twoRDMR_loc = self.get_twoRDMR (np.eye (self.norbs_tot))
        E2_loc += 0.125 * np.einsum ('ijkl,ijkl->i', TEI_loc, twoRDMR_loc)
        E2_loc += 0.125 * np.einsum ('ijkl,ijkl->j', TEI_loc, twoRDMR_loc)
        E2_loc += 0.125 * np.einsum ('ijkl,ijkl->k', TEI_loc, twoRDMR_loc)
        E2_loc += 0.125 * np.einsum ('ijkl,ijkl->l', TEI_loc, twoRDMR_loc)
        '''
        
        E2_imp  = 0.125 * np.einsum ('iqrs,jqrs->ij', self.impham_TEI, self.twoRDMRimp_imp)
        E2_imp += 0.125 * np.einsum ('aipq,ajpq->ij', self.impham_TEI, self.twoRDMRimp_imp)
        E2_imp += 0.125 * np.einsum ('abip,abjp->ij', self.impham_TEI, self.twoRDMRimp_imp)
        E2_imp += 0.125 * np.einsum ('abci,abcj->ij', self.impham_TEI, self.twoRDMRimp_imp)
        E2_loc += np.diag (represent_operator_in_basis (E2_imp, self.imp2loc))
        for loc2bas, TEI, twoRDMR in zip (self.loc2tbc, self.TEI_tbc, self.twoRDMRfroz_tbc):
            E2_bas  = 0.125 * np.einsum ('iqrs,jqrs->ij', TEI, twoRDMR)
            E2_bas += 0.125 * np.einsum ('aipq,ajpq->ij', TEI, twoRDMR)
            E2_bas += 0.125 * np.einsum ('abip,abjp->ij', TEI, twoRDMR)
            E2_bas += 0.125 * np.einsum ('abci,abcj->ij', TEI, twoRDMR)
            E2_loc += np.diag (represent_operator_in_basis (E2_bas, loc2bas.T))
        E2 = np.sum (E2_loc[self.frag_orb_list])
        if self.debug_energy:
            print ("get_E_frag {0} :: E2 = {1:.5f}".format (self.frag_name, E2))

        return E1 + E2

    def get_twoRDM (self, bra1_basis=None, ket1_basis=None, bra2_basis=None, ket2_basis=None):
        all_bases = [basis for basis in [bra1_basis, ket1_basis, bra2_basis, ket2_basis] if basis is not None]
        bra1_basis = all_bases[0]
        ket1_basis = ket1_basis if np.any (ket1_basis) else bra1_basis
        bra2_basis = bra2_basis if np.any (bra2_basis) else bra1_basis
        ket2_basis = ket2_basis if np.any (ket2_basis) else bra2_basis
        oneRDM_b1k1 = represent_operator_in_basis (self.oneRDM_loc, bra1_basis, ket1_basis)
        oneRDM_b2k2 = represent_operator_in_basis (self.oneRDM_loc, bra2_basis, ket2_basis)
        oneRDM_b1k2 = represent_operator_in_basis (self.oneRDM_loc, bra1_basis, ket2_basis)
        oneRDM_b2k1 = represent_operator_in_basis (self.oneRDM_loc, bra2_basis, ket1_basis)
        twoRDM  =       np.einsum ('pq,rs->pqrs', oneRDM_b1k1, oneRDM_b2k2)
        twoRDM -= 0.5 * np.einsum ('ps,rq->pqrs', oneRDM_b1k2, oneRDM_b2k1)
        return twoRDM + self.get_twoRDMR (bra1_basis, ket1_basis, bra2_basis, ket2_basis)

    def get_twoRDMR (self, bra1_basis=None, ket1_basis=None, bra2_basis=None, ket2_basis=None):
        all_bases = [basis for basis in [bra1_basis, ket1_basis, bra2_basis, ket2_basis] if basis is not None]
        bra1_basis = all_bases[0]
        ket1_basis = ket1_basis if np.any (ket1_basis) else bra1_basis
        bra2_basis = bra2_basis if np.any (bra2_basis) else bra1_basis
        ket2_basis = ket2_basis if np.any (ket2_basis) else bra2_basis
        i2b1 = np.dot (self.imp2loc, bra1_basis)
        i2k1 = np.dot (self.imp2loc, ket1_basis)
        i2b2 = np.dot (self.imp2loc, bra2_basis)
        i2k2 = np.dot (self.imp2loc, ket2_basis)
        twoRDMR = represent_operator_in_basis (self.twoRDMRimp_imp, i2b1, i2k1, i2b2, i2k2)
        for loc2froz, twoRDMR_froz in zip (self.loc2tbc, self.twoRDMRfroz_tbc):
            froz2loc = np.conj (loc2froz.T)
            f2b1 = np.dot (froz2loc, bra1_basis)
            f2k1 = np.dot (froz2loc, ket1_basis)
            f2b2 = np.dot (froz2loc, bra2_basis)
            f2k2 = np.dot (froz2loc, ket2_basis)
            twoRDMR += represent_operator_in_basis (twoRDMR_froz, f2b1, f2k1, f2b2, f2k2)
        return twoRDMR

    def get_oneRDM_frag (self):
        self.warn_check_imp_solve ("oneRDM_frag")
        return self.oneRDM_loc [np.ix_(self.frag_orb_list, self.frag_orb_list)]

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


