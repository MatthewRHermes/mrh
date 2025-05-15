import numpy as np
from scipy import linalg
from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.tools import molden
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.lassi import lassi
from c2h4n4_struct import structure as struct

mol = struct (3.0, 3.0, '6-31g')
mol.symmetry = 'Cs'
mol.output = 'c2h4n4_631g_w_symmetry.log'
mol.verbose = lib.logger.INFO
mol.build ()
mf = scf.RHF (mol).run ()

# SA-LASSCF object
# The first positional argument of "state_average" is the orbital weighting function
# Note that there are two states and two fragments and the weights sum to 1
# "Spins" is neleca - nelecb (= 2m for the sake of being an integer)
# "Smults" is the desired local spin quantum *MULTIPLICITY* (2s+1)
# "Wfnsyms" can also be the names of the irreps but I got lazy
# "Charges" modifies the number of electrons in ncas_sub (third argument of LASSCF constructor)
#   For fragment i in state j:
#       neleca = (sum(las.ncas_sub[i]) - charges[j][i] + spins[j][i]) / 2
#       nelecb = (sum(las.ncas_sub[i]) - charges[j][i] - spins[j][i]) / 2
# If your molecule doesn't have point-group symmetry turned on then don't pass "wfnsyms"
las2 = LASSCF (mf, (5,5), ((3,2),(2,3)))
las2 = las2.state_average ([0.5,0.5],
    spins=[[1,-1],[-1,1]],
    smults=[[2,2],[2,2]],    
    charges=[[0,0],[0,0]],
    wfnsyms=[[1,1],[1,1]])
las2.conv_tol = 1e-12
mo_loc = las2.localize_init_guess ((list (range (5)), list (range (5,10))), mf.mo_coeff)
las2.kernel (mo_loc)

print ("\n--- SA(2)-LASSCF ---")
print ("Energy:", las2.e_states)

# Now I will add "spectator" states, which don't affect the state-averaged
# energy for the purposes of orbital optimization. (You can do this in one
# step, but then the orbitals would not optimize as quickly.) 
print (("\nThe orbitals and CI vectors are preserved as much as possible when\n"
    "going from a completed smaller SA-LASSCF to a larger one.\n\n"
    "The 'lasci ()' function call optimizes the CI vectors with the orbitals all\n"
    "frozen, so the energies of the first two states below should be unchanged\n"
    "to many digits:"))
las4 = las2.state_average ([0.5,0.5,0.0,0.0],
    spins=[[1,-1],[-1,1],[0,0],[0,0]],
    smults=[[2,2],[2,2],[1,1],[1,1]],    
    charges=[[0,0],[0,0],[-1,1],[1,-1]],
    wfnsyms=[[1,1],[1,1],[0,0],[0,0]])   
las4.lasci ()
print ("\n--- LASCI(4) @ SA(2)-LASSCF orbitals ---")
print ("Energy:", las4.e_states)

# An example of input error handling
print (("\nThe LASSCF kernel and lasci functions check for duplicate states\n"
    "in the state list by default, using this AssertionError:"))
try:
    las5 = las2.state_average ([0.5,0.5,0.0,0.0,0.0],
        spins=[[1,-1],[-1,1],[0,0],[0,0],[0,0]],
        smults=[[2,2],[2,2],[1,1],[1,1],[1,1]],
        charges=[[0,0],[0,0],[-1,1],[1,-1],[-1,1]],
        wfnsyms=[[1,1],[1,1],[0,0],[0,0],[0,0]])
    las5.lasci ()
except AssertionError as e:
    print (e)

print ("\n--- LASSI(4) @ SA(2)-LASSCF orbitals ---")

lsi = lassi.LASSI (las4)
e_roots, si = lsi.kernel ()

print ("Energy:", e_roots)
print ("<S**2>:",lsi.s2)
print ("(neleca, nelecb):", lsi.nelec)
print ("Symmetry:", lsi.wfnsym)

print (("\nIn this example, the triplet eigenvector is determined by symmetry\n"
          "to within phase factors because I only have 1 triplet in this\n"
          "space. This also means that the triplet natural orbitals are still\n"
          "localized and the interfragment entanglement is second-order, only\n"
          "appearing in the 2RDM. On the other hand, the singlets do interact\n"
          "leading to first-order entanglement which is visible in the NOs:\n"))
print ("--- LASSI eigenvectors ---")
print (si)

# You can get the 1-RDMs of the SA-LASSCF states like this
states_casdm1s = las4.states_make_casdm1s ()

# You can get the 1- and 2-RDMs of the LASSI solutions like this
roots_casdm1s, roots_casdm2s = lsi.make_casdm12s ()

# By default orbitals are state-averaged natural-orbitals at the end
# of the SA-LASSCF calculation
# But you can recanonicalize
print ("\nlasscf_state_0-3.molden: single LAS state NOs, (strictly) unentangled")
for iroot in range (4): 
    molden.from_lasscf (las4, 'lasscf_state_{}.molden'.format (iroot), state=iroot)
print ("lassi_root_0-3.molden: LASSI eigenstate NOs, (generally) entangled")
for iroot in range (4): 
    molden.from_lassi (lsi, 'lassi_root_{}.molden'.format (iroot), state=iroot)


