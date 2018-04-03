
import re
import numpy as np
from pyscf import gto
from . import chemps2, pyscf_rhf, pyscf_mp2, pyscf_cc, pyscf_casscf
import mrh.util.basis.get_complementary_states

def make_fragment_atomlist (ints, frag_atom_list, solver_name, active_orb_list = None):
        assert (len (atom_list) < ints.mol.natm)
        assert (np.amax (atom_list) < ints.mol.natm)
        norbs_in_atom = [np.sum ([2 * ints.mol.bas_angular (shell) + 1 for shell in ints.mol.atom_shell_ids (atom)]) for atom in range (ints.mol.natm)]
        norbs_thru_atom = [np.sum (norbs_in_atom[:atom]) for atom in range (ints.mol.natm)]
        frag_orb_list = [range (norbs_thru_atom[atom-1],norbs_thru_atom[atom]) for atom in range (ints.mol.natm)]
        return fragment_object (ints, frag_orb_list, solver_name, active_orb_list=active_orb_list)

def make_fragment_orblist (ints, frag_orb_list, solver_name, active_orb_list = None):
        return fragment_object (ints, frag_orb_list, solver_name, active_orb_list=active_orb_list)

class fragment_object:

        def __init__ (self, ints, frag_orb_list, solver_name, active_orb_list=None):

                self.norbs_tot = ints.mol.nao_nr ()
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
                self.loc2frag = np.eye (self.norbs_tot, dtype=float)[:,frag_orb_list]
                self.loc2fragbanned = None
                self.loc2envbanned = None

                # Initialize other data
                self.fragment_energy = 0
                self.impurity_energy = 0
                self.norbs_bath_tot = self.loc2frag.shape[0] - self.loc2frag.shape[1]
                self.loc2bath = None
                self.loc2active = None
                self.oneRDM = None
                self.twoRDM = None

        @property
        def norbs_frag (self):
                return self.loc2frag.shape[1]

        @property
        def norbs_bath (self):
                return None if self.loc2bath is None else self.loc2bath.shape[1]

        @property
        def norbs_active (self):
                return None if self.loc2active is None else self.loc2active.shape[1]

        @property
        def loc2fragallowed (self):
                return None if self.loc2fragbanned is None else mrh.util.basis.get_complementary_states (self.loc2fragbanned)
        @loc2fragallowed.setter
        def loc2fragallowed (self, states):
                self.loc2fragbanned = mrh.util.basis.get_complementary_states (states)

        @property
        def loc2envallowed (self):
                if self.loc2envbanned is None:
                        return None
                loc2notenv = np.append (self.loc2frag, self.loc2envbanned, axis=1)
                return mrh.util.basis.get_complementary_states (loc2notenv)
        @loc2envallowed.setter
        def loc2envallowed (self, states):
                print ("Warning: orthogonalizing loc2envallowed states against fragment orbitals")
                loc2envallowed_notfrag = mrh.orthonormalize_a_basis (states - mrh.util.basis.get_overlapping_states (loc2frag, states))
                loc2envbanned_incfrag = mrh.get_complementary_states (loc2nevallowed_notfrag)
                self.loc2envbanned = mrh.orthonormalize_a_basis (loc2envbanned_incfrag - mrh.util.basis.get_overlapping_states (loc2frag, states))

        @property
        def loc2imp (self):
                return None if self.loc2bath is None else np.append (self.loc2frag, self.loc2bath, axis=1)

        @property
        def loc2core (self):
                return None if self.loc2bath is None else mrh.util.basis.get_complementary_states (self.loc2imp)

        @property
        def loc2emb (self):
                return None if self.loc2bath is 


