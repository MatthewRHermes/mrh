# Matt, DON'T do any hairy bullshit until you can just reproduce the results of the existing code!


import re
import numpy as np
from pyscf import gto
from . import chemps2, pyscf_rhf, pyscf_mp2, pyscf_cc, pyscf_casscf, testing
import mrh.util.basis.get_complementary_states

def make_fragment_atomlist (ints, frag_atom_list, solver_name, active_orb_list = None):
    assert (len (atom_list) < ints.mol.natm)
    assert (np.amax (atom_list) < ints.mol.natm)
    norbs_in_atom = [np.sum ([2 * ints.mol.bas_angular (shell) + 1 for shell in ints.mol.atom_shell_ids (atom)]) for atom in range (ints.mol.natm)]
    norbs_thru_atom = [np.sum (norbs_in_atom[:atom]) for atom in range (ints.mol.natm)]
    frag_orb_list = [range (norbs_thru_atom[atom-1],norbs_thru_atom[atom]) for atom in range (ints.mol.natm)]
    print ("Fragment atom list\n{0}\nproduces a fragment orbital list as:\n{1}".format (ints.mol.atom[frag_atom_list], frag_orb_list))
    return fragment_object (ints, frag_orb_list, solver_name, active_orb_list=active_orb_list)

def make_fragment_orblist (ints, frag_orb_list, solver_name, active_orb_list = None):
    return fragment_object (ints, frag_orb_list, solver_name, active_orb_list=active_orb_list)


class fragment_impurity_dependent_variables:

    # This object will contain variables that need to be voided every time anything changes that would alter the outcome of an impurity calculation
    # Essentially, the Schmidt decomposition or the chemical potential

    def __init__ (self):

        # Outputs of solving the impurity problem
        self._E_frag = None
        self._E_imp = None
        self._oneRDM_imp = None
        self._twoRDM_imp = None

        # Legacy variables of Hung's hackery. Clean this up as soon as possible
        self._MOmf = None
        self._MO = None
        self._MOnat = None
        self._OccNum = None


class fragment_Schmidt_decomp_dependent_variables:

    # This object will contain variables that need to be voided when the Schmidt decomposition changes

    def __init__ (self):

        # Scmidt-decomposition-related objects
        self._loc2emb = None
        self._norbs_bath = None
        self._nelec_imp = None

        # Impurity-Hamiltonian-related objects
        self._impham_CONST = None
        self._impham_OEI = None
        self._impham_FOCK = None
        self._impham_TEI = None

class fragment_object:

    def __init__ (self, ints, frag_orb_list, solver_name, active_orb_list=None):

        # I really hope this doesn't copy.  I think it doesn't.
        self.ints = ints
        self.frag_orb_list = frag_orb_list

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


        # Strip active space specification out of solver name
        active_space_regexp = re.compile ("\(([0-9]+),([0-9]+)\)$")
        active_space = active_space_regexp.search (solver_name)
        self.nelec_active = None
        self.norbs_active = None
        if active_space:
            self.nelec_active = int (active_space.group (1))
            self.norbs_active = int (active_space.group (2))
            solver_name = solver_name[:active_space.start ()]

        # Set impurity solver function and name        
        self.imp_solver_function = solver_function_map[solver_name]
        self.imp_solver_name = solver_name
        self.imp_solver_longname= solver_longname_map[solver_name]

        # Return active space to the name if any
        if active_space:
            self.imp_solver_name += active_space.group (0)
            self.imp_solver_longname = "{0} with {1} electrons in {2} orbitals".format (
                self.imp_solver_longname, self.nelec_active, self.norbs_active)
        self.active_orbs = active_orb_list

        # Fragment orbital data
        self.loc2frag = np.eye (self.ints.mol.nao_nr (), dtype=float)[:,frag_orb_list]

        # Internally modified things that I want to be able to set once and then forget about
        # Never ever ever use these names outside of the getter/setter/property definitions below

        # The Schmidt 1RDM and chemical potential
        self._oneRDM_Schmidt = np.eye (self.ints.mol.nao_nr (), dtype=float)
        self._chempot_frag = 0.0
        self._CC_E_TYPE = 'LAMBDA'
        self._init_Schmidt_decomp_dependent_vars ()
        self._init_chempot_frag_dependent_vars ()

    def _init_Schmidt_decomp_dependent_vars (self):
        # Variables that need to be voided whenever the Schmidt decomposition (i.e., oneRDM_Schmidt) changes

        # Scmidt-decomposition outputs
        self._loc2emb = None
        self._norbs_bath = None
        self._nelec_imp = None

        # Impurity Hamiltonian
        self._impham_CONST = None
        self._impham_OEI = None
        self._impham_FOCK = None
        self._impham_TEI = None

    def _init_impurity_outputs (self):
        # Variables that depend on the entire definition of the impurity model, including the chemical potential.
        # Need to be voided whenever the Schmidt decomposition, the chemical potential, or CC_E_TYPE changes

        # Basic outputs of solving the impurity problem
        self._E_frag = None
        self._E_imp = None
        self._oneRDM_imp = None
        self._twoRDM_imp = None

        # Legacy variables of Hung's hackery. Clean this up as soon as possible
        self._MOmf = None
        self._MO = None
        self._MOnat = None
        self._OccNum = None

    @property
    def oneRDM_Schmidt (self):
        return self._oneRDM_Schmidt
    @oneRDM_Schmidt.setter 
    def oneRDM_Schmidt (self, value):
        self._init_Schmidt_decomp_dependent_vars ()
        self._init_impurity_outputs ()
        self._oneRDM_Schmidt = value

    @property
    def chempot_frag (self):
        return self._chempot_frag
    @chempot_frag.setter
    def chempot_frag (self, value):
        self._init_impurity_outputs ()
        self._chempot_frag = value

    @property
    def CC_E_TYPE (self):
        return self._CC_E_TYPE
    @chempot_frag.setter
    def CC_E_TYPE (self, value):
        self._init_impurity_outputs ()
        self._CC_E_TYPE = value



    # Getters for properties of loc2frag
    ############################################################################################################################
    def get_norbs_tot (self):
        return self.loc2frag.shape[0]

    def get_norbs_frag (self):
        return self.loc2frag.shape[1]

    def get_norbs_active (self):
        return None if self.loc2active is None else self.loc2active.shape[1]
    ############################################################################################################################



    # I'm gonna hide the Schmidt decomposition in here
    ############################################################################################################################
    def get_loc2emb (self):
        if self._loc2emb is None:
            self.do_Schmidt_decomposition ()
        return self._loc2emb

    def get_norbs_bath (self):
        if self._norbs_bath is None:
            self.do_Schmidt_decomposition ()
        return self._norbs_bath

    def get_nelec_imp (self):
        if self._nelec_imp is None:
            self.do_Schmidt_decomposition ()
        return self._nelec_imp

    def do_Schmidt_decomposition (self):
        if self.oneRDM_Schmidt is None:
            raise RuntimeError ("Can't do the Schmidt decomposition without self.oneRDM_Schmidt!")
        else:
            self._loc2emb, self._norbs_bath, self._nelec_imp = testing.schmidt_decompose_1RDM (self.oneRDM_Schmidt, self.loc2frag)
    ##############################################################################################################################



    # Orbital coefficients and numbers directly dependent on Schmidt decomposition
    ##############################################################################################################################
    def get_norbs_imp (self):
        return self.get_norbs_frag () + self.get_norbs_bath ()

    def get_norbs_core (self):
        return self.get_norbs_tot () - self.get_norbs_imp ()

    def get_loc2bath (self):
        return self.get_loc2emb ()[:,self.get_norbs_frag ():self.get_norbs_imp ()]

    def get_loc2imp (self):
        return self.get_loc2emb ()[:,:self.get_norbs_imp ()]

    def get_loc2core (self):
        return self.get_loc2emb ()[:,self.get_norbs_imp ():]
    ###############################################################################################################################



    # Impurity Hamiltonian parameters
    ###############################################################################################################################
    def get_impham_CONST (self):
        if self._impham_CONST is None:
            self._impham_CONST = self.ints.dmet_electronic_const (self.get_loc2emb (), self.get_norbs_imp (), self.oneRDM_Schmidt)
        return self._impham_CONST

    def get_impham_OEI (self):
        if self._impham_OEI is None:
            self._impham_OEI = self.ints.dmet_oei (self.get_loc2emb (), self.get_norbs_imp ())
        return self._impham_OEI

    def get_impham_FOCK (self):
        if self._impham_FOCK is None:
            self._impham_FOCK = self.ints.dmet_fock (self.get_loc2emb (), self.get_norbs_imp (), self.oneRDM_Schmidt)
        return self._impham_FOCK

    def get_impham_TEI (self):
        if self._impham_TEI is None:
            self._impham_TEI = self.ints.dmet_tei (self.get_loc2emb (), self.get_norbs_imp ())
        return self._impham_TEI
    ###############################################################################################################################

    

    # Solving the impurity problem
    ###############################################################################################################################
    def get_guess_1RDM (self):
        return self.ints.dmet_init_guess_rhf (self.get_loc2emb (), self.get_norbs_imp (), self.get_nelec_imp () // 2, self.get_norbs_frag (), self.chempot_frag)

    def solve_impurity_problem (self):

        # For all solvers, the input arguments begin as:
        # CONST, OEI, FOCK, TEI, norbs_imp, nelec_imp, norbs_frag
        inputlist = [self.get_impham_CONST (), self.get_impham_OEI (), self.get_impham_FOCK (),
                     self.get_impham_TEI (), self.get_norbs_imp (), self.get_nelec_imp (),
                     self.get_norbs_frag ()]
        # For the CASSCF solver, the next items are the definitions of the active space
        if self.imp_solver_name == "CASSCF":
            inputlist.extend ([(self.nelec_active, self.norbs_active), self.active_orbs)
        # For everything except exact diagonalization, the next item is the guess 1RDM
        if not (self.imp_solver_name == "ED"):
            inputlist.append (self.get_guess_1RDM ())
        # For the CC and CASSCF methods, the next item is CC_E_TYPE
        if (self.imp_solver_name == "CC") or (self.imp_solver_name == "CASSCF"):
            inputlist.append (self.CC_E_TYPE)
        # For all solvers, the final argument is chempot
        inputlist.append (self.chempot_frag)
            
        outputtuple = self.imp_solver_function (*inputlist)

        # The first item in the output tuple is "IMP_energy," which is the *impurity* energy for CC or CASSCF with CC_E_TYPE=CASCI and the *fragment* energy otherwise
        if ((self.imp_solver_name == "CC") or (self.imp_solver_name == "CASSCF")) and self.CC_E_TYPE == "CASCI":
            self._E_imp = outputtuple[0]
        else:
            self._E_frag = outputtuple[0]

        # The second item in the output tuple is the impurity 1RDM
        self._oneRDM_imp = outputtuple[1]

        # CASSCF additionally has MOmf, MO, MOnat, and OccNum as a legacy of Hung's hackery
        if (self.imp_solver_name == "CASSCF"):
            self._MOmf, self_MO, self._MOnat, self._OccNum = outputtuple[3:]

    def get_fragment_energy (self):
        if self._E_frag is None:
            self.solve_impurity_problem ()
        return self._E_frag

    def get_impurity_energy (self):
        if (self.imp_solver_name != "CASSCF") and (self.imp_solver_name != "CC"):
            raise RuntimeError ("calculating the ~impurity~ energy with any solver other than CASSCF or CC is not yet implemented")
        if self._E_imp is None:
            self.solve_impurity_problem ()
        return self._E_imp

    def get_impurity_1RDM (self):
        if self._oneRDM_imp is None:
            self.solve_impurity_problem ()
        return self._oneRDM_imp





