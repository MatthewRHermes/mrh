# Matt, DON'T do any hairy bullshit until you can just reproduce the results of the existing code!


import re
import numpy as np
from pyscf import gto
from . import chemps2, pyscf_rhf, pyscf_mp2, pyscf_cc, pyscf_casscf, testing, qcdmethelper
import mrh.util.basis
from mrh.util.la import matrix_eigen_control_options
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
        self.norbs_active = len (self.active_orb_list)
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
        idx = np.append (self.frag_orb_list, self.env_orb_list)
        self.loc2emb = np.eye (self.norbs_tot, dtype=float)[:,idx]
        
        # Impurity Hamiltonian
        self.impham_OEI = None
        self.impham_FOCK = None
        self.impham_TEI = None

        # Basic outputs of solving the impurity problem
        self.E_frag = 0
        self.E_imp = 0
        self.oneRDM_imp = None
        self.twoRDM_imp = None
        self.loc2fno = None
        self.fno_evals = None

        # Outputs of CAS calculations use to fix CAS-DMET
        self.loc2mcno = None
        self.mcno_evals = None

        # Legacy variables of Hung's hackery. Clean this up as soon as possible
        self.CASMOmf = None
        self.CASMO = None
        self.CASMOnat = None
        self.CASOccNum = None
        
        # Initialize some runtime warning bools
        self.schmidt_done = False
        self.impham_built = False
        self.imp_solved = False


        # Report
        print ("Constructed a fragment of {0} orbitals for a system with {1} total orbitals".format (self.norbs_frag, self.norbs_tot))
        print ("Using a {0} [{1}] calculation to solve the impurity problem".format (self.imp_solver_longname, self.imp_solver_method))
        print ("Fragment orbitals: {0}".format (self.frag_orb_list))



    # Common runtime warning checks
    ###########################################################################################################################
    def warn_check_schmidt (self, cstr="NONE"):
        wstr = "Schmidt decomposition not performed at call to {0}. Undefined behavior likely!".format (cstr)
        return warnings.warn (wstr, RuntimeWarning) if (not self.schmidt_done) else None

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
    def norbs_imp (self):
        self.warn_check_schmidt ("norbs_imp")
        return self.norbs_frag + self.norbs_bath

    @property
    def norbs_core (self):
        self.warn_check_schmidt ("norbs_core")
        return self.norbs_tot - self.norbs_frag - self.norbs_bath

    @property
    def loc2frag (self):
        return self.loc2emb[:,:self.norbs_frag]

    @property
    def loc2imp (self):
        return self.loc2emb[:,:self.norbs_imp]

    @property
    def loc2bath (self):
        return self.loc2emb[:,self.norbs_frag:self.norbs_imp]

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
    ############################################################################################################################




    # The Schmidt decomposition
    ############################################################################################################################
    def do_Schmidt_decomposition (self, guide_1RDM):
        self.loc2emb, self.norbs_bath, self.nelec_imp = testing.schmidt_decompose_1RDM (guide_1RDM, self.loc2frag, self.norbs_bath_max)
        self.schmidt_done = True
        print ("Schmidt decomposition found a total of {0} bath orbitals for this fragment, of an allowed total of {1}".format (self.norbs_bath, self.norbs_bath_max))
        print ("Schmidt decomposition found {0} electrons in this impurity".format (self.nelec_imp))
        self.oneRDMcore_loc = mrh.util.basis.project_operator_into_subspace (guide_1RDM, self.loc2core)
        if abs (self.idempotize_thresh) > self.num_zero_atol:
            self.oneRDMcore_loc, nelec_core_diff = self.idempotize_1RDM (self.oneRDMcore_loc, self.idempotize_thresh)
            self.nelec_imp -= nelec_core_diff
            print ("After attempting to idempotize the core (part of the putatively idempotent guide) 1RDM with a threshold of " 
                + "{0}, {1} electrons were found in the impurity".format (self.idempotize_thresh, self.nelec_imp))
        nelec_imp_eint_err = round (self.nelec_imp/2) - (self.nelec_imp/2)
        if (abs (nelec_imp_eint_err) > self.num_zero_atol):
            raise RuntimeError ("Can't solve impurity problems without even-integer number of electrons! err={0}".format (nelec_imp_eint_err))
        self.nelec_imp = int (round (self.nelec_imp))
        self.impham_built = False
        self.imp_solved = False
        return self.loc2emb, self.norbs_bath, self.nelec_imp

    def idempotize_1RDM (self, oneRDM, thresh):
        evals, evecs = np.linalg.eigh (oneRDM)
        diff_evals = (2.0 * np.around (evals / 2.0)) - evals
        # Only allow evals to change by at most +-thresh
        idx_floor = np.where (diff_evals < -abs (thresh))[0]
        idx_ceil  = np.where (diff_evals >  abs (thresh))[0]
        diff_evals[idx_floor] = -abs(thresh)
        diff_evals[idx_ceil]  =  abs(thresh)
        nelec_diff = np.sum (diff_evals)
        new_evals = evals + diff_evals
        new_oneRDM = mrh.util.basis.represent_operator_in_basis (np.diag (new_evals), evecs.T)
        return new_oneRDM, nelec_diff
    ##############################################################################################################################





    # Impurity Hamiltonian
    ###############################################################################################################################
    def construct_impurity_hamiltonian (self):
        self.warn_check_schmidt ("construct_impurity_hamiltonian")
        self.impham_OEI = self.ints.dmet_oei (self.loc2emb, self.norbs_imp)
        self.impham_FOCK = self.ints.dmet_fock (self.loc2emb, self.norbs_imp, self.oneRDMcore_loc)
        self.impham_TEI = self.ints.dmet_tei (self.loc2emb, self.norbs_imp)
        self.impham_built = True
        self.imp_solved = False

    def get_impham_CONST (self):
        return self.ints.dmet_electronic_const (self.loc2emb, self.norbs_imp, self.oneRDMcore_loc)
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
        print ("{0} energy results for {1}: E_frag = {2}; E_imp - CONST = {3}".format (self.imp_solver_longname, self.frag_name, self.E_frag, self.E_imp))

        # In order to comply with ``NOvecs'' bs, let's get some pseudonatural orbitals
        self.fno_evals, frag2fno = np.linalg.eigh (self.oneRDM_imp[:self.norbs_frag,:self.norbs_frag])
        self.loc2fno = np.dot (self.loc2frag, frag2fno)
    ###############################################################################################################################





    # Convenience functions and properties for results
    ###############################################################################################################################
    @property
    def oneRDM_loc (self):
        self.warn_check_imp_solve ("oneRDM_loc")
        oneRDMimp_loc = mrh.util.basis.represent_operator_in_basis (self.oneRDM_imp, self.loc2imp.T)
        return oneRDMimp_loc + self.oneRDMcore_loc


    @property
    def oneRDM_frag (self):
        self.warn_check_imp_solve ("oneRDM_frag")
        return self.oneRDM_imp[:self.norbs_frag,:self.norbs_frag]


    @property
    def nelec_frag (self):
        return np.trace (self.oneRDM_frag)



    # For interface with DMET
    ###############################################################################################################################
    def get_errvec (self, dmet, mf_1RDM_loc):
        self.warn_check_imp_solve ("get_errvec")
        # Fragment natural-orbital basis matrix elements needed
        if dmet.doDET_NO:
            mf_1RDM_fno = mrh.util.basis.represent_operator_in_basis (mf_1RDM_loc, self.loc2fno)
            return np.diag (mf_1RDM_fno) - self.fno_evals
        # Bath-orbital matrix elements needed
        if dmet.incl_bath_errvec:
            mf_err1RDM_imp = mrh.util.basis.represent_operator_in_basis (mf_1RDM_loc, self.loc2imp) - self.oneRDM_imp
            return mf_err1RDM_imp.flatten (order='F')
        # Only fragment-orbital matrix elements needed
        mf_err1RDM_frag = mrh.util.basis.represent_operator_in_basis (mf_1RDM_loc, self.loc2frag) - self.oneRDM_frag
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
            rsp_1RDM_imp = mrh.util.basis.represent_operator_in_basis (rsp_1RDM, self.loc2imp)
            return rsp_1RDM_imp.flatten (order='F')
        # Only fragment-orbital matrix elements needed
        rsp_1RDM_frag = mrh.util.basis.represent_operator_in_basis (rsp_1RDM, self.loc2frag)
        if dmet.doDET:
            return np.diag (rsp_1RDM_frag)
        else:
            return rsp_1RDM_frag.flatten (order='F')


