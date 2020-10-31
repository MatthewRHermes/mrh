import numpy as np
from scipy import linalg
from pyscf import gto, scf, lib, mcscf
from pyscf.tools import molden
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.mcscf.lasscf_testing import LASSCF
from mrh.my_pyscf.mcscf import lassi
from c2h4n4_struct import structure as struct

mol = struct (3.0, 3.0, '6-31g')
mol.symmetry = 'Cs'
mol.output = 'c2h4n4_631g.log'
mol.verbose = lib.logger.INFO
mol.build ()
mf = scf.RHF (mol).run ()

# SA-LASSCF object
# The first positional argument of "state_average" is the orbital weighting function
# Note that there are four states and two fragments and the weights sum to 1
# "Spins" is neleca - nelecb (= 2m for the sake of being an integer)
# "Smults" is the desired local spin quantum *MULTIPLICITY* (2s+1)
# "Wfnsyms" can also be the names of the irreps but I got lazy
# "Charges" should be self-explanatory
las = LASSCF (mf, (5,5), ((3,2),(2,3)))
las = las.state_average ([0.5,0.5,0.0,0.0],
    spins=[[1,-1],[-1,1],[0,0],[0,0]],
    smults=[[2,2],[2,2],[1,1],[1,1]],    
    charges=[[0,0],[0,0],[-1,1],[1,-1]],
    wfnsyms=[[1,1],[1,1],[0,0],[0,0]])   
mo_loc = las.localize_init_guess ((list (range (5)), list (range (5,10))), mf.mo_coeff)
las.kernel (mo_loc)
print ("\n---SA-LASSCF energies---")
print (las.e_states)

# For now, the LASSI diagonalizer is just a post-hoc function call
# It returns eigenvalues (energies) in the first position and
# eigenvectors (here, a 4-by-4 vector)
e_roots, si = las.lassi ()

# Symmetry information about the LASSI solutions is "tagged" on the si array
print ("\n---LASSI solutions---")
print ("Energy:", e_roots)
print ("<S**2>:",si.s2)
print ("(neleca, nelecb):", si.nelec)
print ("Symmetry:", si.wfnsym)

# The triplet eigenvector in this space is determined by symmetry to within a
# phase factor. The singlet eigenvectors depend on the Hamiltonian because
# the singlet states can interact with each other.
print ("\n---LASSI eigenvectors---")
print (si)

# You can get the 1-RDMs of the SA-LASSCF states like this
states_casdm1s = las.states_make_casdm1s ()

# You can get the 1- and 2-RDMs of the LASSI solutions like this
roots_casdm1s, roots_casdm2s = lassi.roots_make_rdm12s (las, las.ci, si)

# Beware! Don't do ~~~anything~~ to the si array before you pass it to the
# function above or grab the important data from its attachments!
print ("\nSurely I can type si = si * 1 without any consequences")
si = si * 1
try:
    print ("<S**2>:",si.s2)
except AttributeError as e:
    print ("Oh no! <S**2> disappeared and all I have now is this error message:")
    print ("AttributeError:", str (e))
try:
    roots_casdm1s, roots_casdm2s = lassi.roots_make_rdm12s (las, las.ci, si)
except AttributeError as e:
    print ("Oh no! I can't make rdms anymore either because:")
    print ("AttributeError:", str (e))
print ("(Yes, dear user, I will have to make this less stupid in future)")

# No super-convenient molden API yet
# By default orbitals are state-averaged natural-orbitals at the end
# of the SA-LASSCF calculation
# But you can manipulate them using the RDMs if you have patience
# Example: making moldens to compare the four roots
occ = np.zeros (las.mo_coeff.shape[1])
ncore, nocc = las.ncore, las.ncore+las.ncas
occ[:ncore] = 2
for iroot, dm1 in enumerate (roots_casdm1s.sum (1)):
    occ[ncore:nocc], umat = linalg.eigh (-dm1)
    occ[ncore:nocc] *= -1 # Just a sorting trick
    no_coeff = las.mo_coeff.copy ()
    no_coeff[:,ncore:nocc] = np.dot (no_coeff[:,ncore:nocc], umat)
    molden.from_mo (las.mol, 'lassi_root_{}.molden'.format (iroot),
        no_coeff, occ=occ)

# Remember that LASSI is a post-hoc diagonalization step if you want to do a
# potential energy scan
las = las.as_scanner ()
new_mol = struct (2.9, 2.9, '6-31g', symmetry='Cs')
new_mol.symmetry = 'Cs'
new_mol.build ()
print ("\n\nPotential energy scan to dr = 2.9 Angs")
e = las (new_mol)
print (e, "<- this is just the state-average energy!")
print ("(Which happens to be identical to the first two LAS state energies because I chose a bad example, but shhhh)")
print ("You need to interrogate the LAS object to get the interesting parts!")
print ("New state energies:", las.e_states)
e_roots, si = las.lassi ()
print ("New LASSI root energies:", e_roots)

