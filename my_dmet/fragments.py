# Matt, DON'T do any hairy bullshit until you can just reproduce the results of the existing code!


import re
import numpy as np
from pyscf import gto
from . import chemps2, pyscf_rhf, pyscf_mp2, pyscf_cc, pyscf_casscf, testing, qcdmethelper
import mrh.util.basis

def make_fragment_atom_list (ints, frag_atom_list, solver_name, active_orb_list = np.empty (0, dtype=int)):
    assert (len (atom_list) < ints.mol.natm)
    assert (np.amax (atom_list) < ints.mol.natm)
    norbs_in_atom = [np.sum ([2 * ints.mol.bas_angular (shell) + 1 for shell in ints.mol.atom_shell_ids (atom)]) for atom in range (ints.mol.natm)]
    norbs_thru_atom = [np.sum (norbs_in_atom[:atom]) for atom in range (ints.mol.natm)]
    frag_orb_list = [range (norbs_thru_atom[atom-1],norbs_thru_atom[atom]) for atom in range (ints.mol.natm)]
    print ("Fragment atom list\n{0}\nproduces a fragment orbital list as:".format (ints.mol.atom[frag_atom_list]))
    print (' '.join (frag_orb_list))
    return fragment_object (ints, np.asarray (frag_orb_list), solver_name, active_orb_list=active_orb_list)

def make_fragment_orb_list (ints, frag_orb_list, solver_name, active_orb_list = np.empty (0, dtype=int)):
    return fragment_object (ints, frag_orb_list, solver_name, active_orb_list)


class fragment_object:

    def __init__ (self, ints, frag_orb_list, solver_name, active_orb_list):

        # I really hope this doesn't copy.  I think it doesn't.
        self.ints = ints
        self.norbs_tot = self.ints.nao_nr ()
        self.frag_orb_list = frag_orb_list
        self.active_orb_list = active_orb_list
        self.norbs_frag = len (self.frag_orb_list)
        self.norbs_bath = 0
        self.nelec_imp = 0
        self.norbs_active = len (self.active_orb_list)
        self.norbs_bath_max = self.norbs_frag

        # Assign solver function
        solver_function_map = {
            "ED"     : chemps2 ,
            "RHF"    : pyscf_rhf ,
            "MP2"    : pyscf_mp2 ,
            "CC"     : pyscf_cc ,
            "CASSCF" : pyscf_casscf
            }
        solver_longname_map = {
            "ED"     : "exact diagonalization"
            "RHF"    : "restricted Hartree-Fock"
            "MP2"    : "MP2 perturbation theory"
            "CC"     : "coupled-cluster with singles and doubles"
            "CASSCF" : "complete active space SCF"
            }
        solver_name = re.sub ("\([0-9,]+\)", "", imp.solver_name)
        self.imp_solver_name = solver_name
        self.imp_solver_longname = solver_longname_map[solver_name]
        self.imp_solver_function = solver_function_map[solver_name]
        active_space = re.compile ("\([0-9,]+\)").search (solver_name)
        if active_space:
            self.active_space = eval (active_space.group (0))
            if not (len (self.active_space) == 2):
                raise RuntimeError ("Active space {0} not usable; only CASSCF currently implemented".format (solver.active_space))
            self.imp_solver_longname += " with {0} electrons in {1} active-space orbitals".format (self.active_space[0], self.active_space[1])
                 
        # Set up the main basis functions. Before any Schmidt decomposition all environment states are treated as "core"
        # self.loc2emb is always defined to have the norbs_frag fragment states, the norbs_bath bath states, and the norbs_core core states in that order
        idx = np.append (self.frag_orb_list, self.env_orb_list)
        self.loc2emb = np.eye (norbs_tot, dtype=float)[:,idx]
        
        # Impurity Hamiltonian
        self.impham_CONST = None
        self.impham_OEI = None
        self.impham_FOCK = None
        self.impham_TEI = None

        # Basic outputs of solving the impurity problem
        self.E_frag = 0
        self.E_imp = 0
        self.oneRDM_imp = None
        self.twoRDM_imp = None
        self.nelec_frag = None

        # Legacy variables of Hung's hackery. Clean this up as soon as possible
        self.CASMOmf = None
        self.CASMO = None
        self.CASMOnat = None
        self.CASOccNum = None

`       # Report
        print ("Constructed a fragment of {0} orbitals for a system with {1} total orbitals".format (self.norbs_frag, self.norbs_tot))
        print ("Using a {0} [{1}] calculation to solve the impurity problem".format (self.imp_solver_longname, self.imp_solver_method))
        print ("Fragment orbitals: " + ' '.join (self.frag_orb_list))
        print ("Testing loc2emb matrix:\n{0}".format (self.loc2emb))


    # Dependent attributes, never to be modified directly
    ###########################################################################################################################
    @property norbs_imp (self):
    def norbs_imp (self):
        return self.norbs_frag + self.norbs_bath

    @property
    def norbs_core (self):
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
        try:
            return self.imp_solver_name + str (self.active_space)
        except AttributeError:
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
        return np.flatnonzero (self.is_env_orb))

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
        print ("Schmidt decomposition found a total of {0} bath orbitals for this fragment, of an allowed total of {1}".format (self.norbs_bath, self.norbs_bath_max))
        self.oneRDM_core = mrh.util.basis.represent_operator_in_basis (guide_1RDM, self.loc2core)
        return self.loc2emb, self.norbs_bath, self.nelec_imp
    ##############################################################################################################################




    # Impurity Hamiltonian
    ###############################################################################################################################
    def construct_impurity_hamiltonian (self, core1RDM, get_core2RDM=None):
        self.impham_CONST = self.ints.dmet_electronic_const (self.loc2emb, self.norbs_imp, core1RDM, get_core2RDM=get_core2RDM)
        self.impham_OEI = self.ints.dmet_oei (self.loc2emb, self.norbs_imp)
        self.impham_FOCK = self.ints.dmet_fock (self.loc2emb, self.norbs_imp, core1RDM)
        self.impham_TEI = self.ints.dmet_tei (self.loc2emb, self.norbs_imp)
    ###############################################################################################################################

    

    # Solving the impurity problem
    ###############################################################################################################################
    def get_guess_1RDM (self, chempot_frag):
        return self.ints.dmet_init_guess_rhf (self.loc2emb, self.norbs_imp, self.nelec_imp // 2, self.norbs_frag, chempot_frag)

    def solve_impurity_problem (self, chempot_frag, CC_E_TYPE):

        # For all solvers, the input arguments begin as:
        # CONST, OEI, FOCK, TEI, norbs_imp, nelec_imp, norbs_frag
        inputlist = [self.impham_CONST, self.impham_OEI, self.impham_FOCK,
                     self.impham_TEI, self.norbs_imp, self.nelec_imp,
                     self.norbs_frag]
        # For the CASSCF solver, the next items are the definitions of the active space
        if self.imp_solver_name == "CASSCF":
            inputlist.extend ([(self.nelec_active, self.norbs_active), self.is_active_orb])
        # For everything except exact diagonalization, the next item is the guess 1RDM
        if not (self.imp_solver_name == "ED"):
            inputlist.append (self.get_guess_1RDM (chempot_frag))
        # For the CC and CASSCF methods, the next item is CC_E_TYPE
        if (self.imp_solver_name == "CC") or (self.imp_solver_name == "CASSCF"):
            inputlist.append (CC_E_TYPE)
        # For all solvers, the final argument is chempot
        inputlist.append (self.chempot_frag)
            
        outputtuple = self.imp_solver_function (*inputlist)

        # The first item in the output tuple is "IMP_energy," which is the *impurity* energy for CC or CASSCF with CC_E_TYPE=CASCI and the *fragment* energy otherwise
        if ((self.imp_solver_name == "CC") or (self.imp_solver_name == "CASSCF")) and CC_E_TYPE == "CASCI":
            self.E_imp = outputtuple[0]
        else:
            self.E_frag = outputtuple[0]

        # The second item in the output tuple is the impurity 1RDM.
        self.oneRDM_imp = outputtuple[1]
        self.oneRDM_loc = mrh.util.basis.represent_operator_in_basis (np.diag (self.oneRDM_imp, self.oneRDM_core), self.loc2emb.T) 

        # CASSCF additionally has MOmf, MO, MOnat, and OccNum as a legacy of Hung's hackery
        if (self.imp_solver_name == "CASSCF"):
            self.CASMOmf, self.CASMO, self.CASMOnat, self.CASOccNum = outputtuple[3:]

        # In order to comply with ``NOvecs'' bs, let's get some pseudonatural orbitals
        self.fno_evals, self.frag2fno = np.linalg.eigh (self.oneRDM_imp[:norbs_frag,:norbs_frag])
        self.loc2fno = np.dot (self.loc2frag, self.frag2fno)




