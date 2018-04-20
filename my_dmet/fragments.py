# Matt, DON'T do any hairy bullshit until you can just reproduce the results of the existing code!


import re
import numpy as np
from pyscf import gto
from . import chemps2, pyscf_rhf, pyscf_mp2, pyscf_cc, pyscf_casscf, qcdmethelper
from mrh.util.basis import *
from mrh.util.rdm import Schmidt_decomposition_idempotent_wrapper, idempotize_1RDM
from mrh.util.rdm import electronic_energy_orbital_decomposition as get_E_orbs
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
        self.norbs_bath = 0
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
        self.imp_solver_function = solver_function_map[imp_solver_name]
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
        
        # Impurity Hamiltonian
        self.impham_CONST = None # Does not include nuclear potential
        self.impham_OEI   = None
        self.impham_FOCK  = None
        self.impham_TEI   = None

        # Basic outputs of solving the impurity problem
        self.E_frag = 0
        self.E_imp = 0
        self.oneRDM_imp = None
        self.oneRDM_loc = None
        self.twoRDMR_imp = None
        self.loc2fno = None
        self.fno_evals = None

        # Outputs of CAS calculations use to fix CAS-DMET
        self.loc2as = np.zeros((self.norbs_tot,0))
        self.oneRDMas_loc = np.zeros((self.norbs_tot,self.norbs_tot))
        self.E2cas_loc = np.zeros (self.norbs_tot)

        # Initialize some runtime warning bools
        self.Schmidt_done = False
        self.impham_built = False
        self.imp_solved = False

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
    def frag2loc (self):
        return np.asarray (np.asmatrix (self.loc2frag).H)

    @property
    def norbs_core (self):
        self.warn_check_Schmidt ("norbs_core")
        return self.norbs_tot - self.norbs_imp

    @property
    def loc2imp (self):
        return self.loc2emb[:,:self.norbs_imp]

    @property
    def imp2loc (self):
        return np.asarray (np.asmatrix (self.loc2imp).H)

    @property
    def imp2frag (self):
        imp2loc = np.asmatrix (self.imp2loc)
        loc2frag = np.asmatrix (self.loc2frag)
        return np.asarray (imp2loc * loc2frag)

    @property
    def frag2imp (self):
        return np.asarray (np.asmatrix (self.imp2frag).H)

    @property
    def loc2core (self):
        return self.loc2emb[:,self.norbs_imp:]

    @property
    def imp_solver_method (self):
        if self.active_space:
            return self.imp_solver_name + str (self.active_space)
        else:
            return self.imp_solver_name

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
        return np.trace (self.oneRDMas_loc)
    ############################################################################################################################




    # The Schmidt decomposition
    ############################################################################################################################
    def restore_default_embedding_basis (self):
        idx            = np.append (self.frag_orb_list, self.env_orb_list)
        self.norbs_imp = self.norbs_frag
        self.loc2emb   = np.eye (self.norbs_tot)[:,idx]
        self.Efroz_imp  = 0.0
        self.Efroz_frag = 0.0

    def do_Schmidt_normal (self, oneRDM_loc):
        print ("Normal Schmidt decomposition of {0} fragment".format (self.frag_name))
        self.restore_default_embedding_basis ()
        self.loc2emb, norbs_bath, self.nelec_imp, self.oneRDMcore_loc = Schmidt_decomposition_idempotent_wrapper (oneRDM_loc, 
            self.loc2frag, self.norbs_bath_max, idempotize_thresh=self.idempotize_thresh, num_zero_atol=self.num_zero_atol)
        self.norbs_imp = self.norbs_frag + norbs_bath
        self.Schmidt_done = True
        self.impham_built = False
        self.imp_solved = False
        print ("Final impurity for {0}: {1} electrons in {2} orbitals".format (self.frag_name, self.nelec_imp, self.norbs_imp))

    def do_Schmidt_ofc_embedding (self, oneRDM_loc, loc2idem):
        print ("Other-fragment-core Schmidt decomposition of {0} fragment".format (self.frag_name))
        self.restore_default_embedding_basis ()
        print ("Starting with Schmidt decomposition in the idempotent subspace")
        oneRDMidem_loc = project_operator_into_subspace (oneRDM_loc, loc2idem)
        loc2ifrag      = get_overlapping_states (loc2idem, self.loc2frag)
        norbs_ifrag    = loc2ifrag.shape[1]
        print ("{0} fragment orbitals becoming {1} pseudo-fragment orbitals in idempotent subspace".format (self.norbs_frag, norbs_ifrag))
        loc2iemb, norbs_ibath, nelec_iimp, oneRDMidemcore_loc = Schmidt_decomposition_idempotent_wrapper (oneRDMidem_loc, 
            loc2ifrag, self.norbs_bath_max, idempotize_thresh=self.idempotize_thresh, num_zero_atol=self.num_zero_atol)
        norbs_iimp = norbs_ifrag + norbs_ibath
        loc2iimp = loc2iemb[:,:norbs_iimp]
        print ("Adding {0} this-fragment active-space orbitals and {1} this-fragment active-space electrons to the impurity".format (self.norbs_as, self.nelec_as))
        self.nelec_imp = nelec_iimp + self.nelec_as
        self.norbs_imp = norbs_iimp + self.norbs_as
        loc2imp = orthonormalize_a_basis (np.append (loc2iimp, self.loc2as, axis=1))
        self.loc2emb = get_complete_basis (loc2imp)
        print ("Adding other-fragment active-space 1RDMs to core1RDM")
        self.oneRDMcore_loc = oneRDMidemcore_loc + (oneRDM_loc - oneRDMidem_loc - self.oneRDMas_loc)
        nelec_bleed = compute_nelec_in_subspace (self.oneRDMcore_loc, self.loc2imp)
        print ("Found {0} electrons from the core bleeding onto impurity states".format (nelec_bleed))
        print ("(If this number is large, you either are dealing with overlapping fragment active spaces or you made an error)")
        print ("Final impurity for {0}: {1} electrons in {2} orbitals".format (self.frag_name, self.nelec_imp, self.norbs_imp))

    ##############################################################################################################################





    # Impurity Hamiltonian
    ###############################################################################################################################
    def construct_impurity_hamiltonian (self):
        self.warn_check_Schmidt ("construct_impurity_hamiltonian")
        self.impham_CONST = self.ints.dmet_const (self.loc2emb, self.norbs_imp, self.oneRDMcore_loc) + self.Efroz_imp
        self.impham_OEI   = self.ints.dmet_fock (self.loc2emb, self.norbs_imp, self.oneRDMcore_loc)
        self.impham_TEI   = self.ints.dmet_tei (self.loc2emb, self.norbs_imp)
        self.impham_built = True
        self.imp_solved   = False
    ###############################################################################################################################


    

    # Solving the impurity problem
    ###############################################################################################################################
    def get_guess_1RDM (self, chempot_frag):
        return self.ints.dmet_init_guess_rhf (self.loc2emb, self.norbs_imp, self.nelec_imp // 2, self.norbs_frag, chempot_frag)

    def solve_impurity_problem (self, chempot_frag):
        self.warn_check_impham ("solve_impurity_problem")
        guess_1RDM = self.get_guess_1RDM (chempot_frag)
        self.imp_solver_function (self, guess_1RDM, chempot_frag)
        self.imp_solved = True

        # Main results: oneRDM in local basis and nelec_frag
        self.oneRDM_loc = self.get_oneRDM_loc ()
        self.nelec_frag = self.get_nelec_frag ()
        self.E_frag     = self.get_E_frag ()
        print ("Impurity results for {0}: E_imp (elec) = {1}, E_frag (elec) = {2}, nelec_frag = {3}".format (self.frag_name,
            self.E_imp, self.E_frag, self.nelec_frag))

        # In order to comply with ``NOvecs'' bs, let's get some pseudonatural orbitals
        self.fno_evals, frag2fno = np.linalg.eigh (self.oneRDM_frag)
        self.loc2fno = np.dot (self.loc2frag, frag2fno)

    ###############################################################################################################################





    # Convenience functions and properties for results
    ###############################################################################################################################
    def get_oneRDM_loc (self):
        self.warn_check_imp_solve ("oneRDM_loc")
        oneRDMimp_loc = represent_operator_in_basis (self.oneRDM_imp, self.imp2loc)
        return oneRDMimp_loc + self.oneRDMcore_loc

    def get_nelec_frag (self):
        return np.trace (self.oneRDM_frag)

    def get_E_frag (self):
        OEIeff_loc = self.ints.loc_rhf_fock_bis (0.5 * self.oneRDM_loc) # 0.5 to compensate for overcounting by Fock 
        E_imp_loc  =         get_E_orbs (self.norbs_tot, OEI=OEIeff_loc,      oneRDM=self.oneRDM_loc)
        E_imp_loc += np.dot (get_E_orbs (self.norbs_imp, TEI=self.impham_TEI, twoRDM=self.twoRDMR_imp), self.imp2loc)
        return np.sum (E_imp_loc[self.frag_orb_list]) + self.Efroz_frag

    @property
    def oneRDM_frag (self):
        self.warn_check_imp_solve ("oneRDM_frag")
        return self.oneRDM_loc [np.ix_(self.frag_orb_list, self.frag_orb_list)]
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
            mf_err1RDM_imp = represent_operator_in_basis (mf_1RDM_loc, self.loc2imp) - self.oneRDM_imp
            return mf_err1RDM_imp.flatten (order='F')
        # Only fragment-orbital matrix elements needed
        mf_err1RDM_frag = represent_operator_in_basis (mf_1RDM_loc, self.loc2frag) - self.oneRDM_frag
        if dmet.doDET:
            return np.diag (mf_err1RDM_frag)
        elif dmet.altcostfunc:
            return mf_err1RDM_frag[np.triu_indices(self.norbs_frag)]
        else:
            return self.oneRDM_frag.flatten (order='F')


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


