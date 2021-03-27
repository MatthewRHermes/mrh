import numpy as np
from scipy import linalg
from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.mcscf.lasscf_testing import LASSCF
from mrh.exploratory.citools import fockspace
from mrh.exploratory.unitary_cc import lasuccsd 

xyz = '''H 0.0 0.0 0.0
         H 1.0 0.0 0.0
         H 0.2 3.9 0.1
         H 1.159166 4.1 -0.1'''
mol = gto.M (atom = xyz, basis = 'sto-3g', output='h4_sto3g.log',
    verbose=lib.logger.DEBUG)
mf = scf.RHF (mol).run ()
ref = mcscf.CASSCF (mf, 4, 4).run () # = FCI
las = LASSCF (mf, (2,2), (2,2), spin_sub=(1,1))
frag_atom_list = ((0,1),(2,3))
mo_loc = las.localize_init_guess (frag_atom_list, mf.mo_coeff)
las.kernel (mo_loc)

# LASUCC is implemented as a FCI solver for MC-SCF
# It's compatible with CASSCF as well as CASCI, but it's really slow
mc = mcscf.CASCI (mf, 4, 4)
mc.mo_coeff = las.mo_coeff
mc.fcisolver = lasuccsd.FCISolver (mol)
mc.fcisolver.nlas = [2,2] # Number of orbitals per fragment
mc.kernel ()

print ("FCI energy:      {:.9f}".format (ref.e_tot))
print ("LASSCF energy:   {:.9f}".format (las.e_tot))
print ("LASUCCSD energy: {:.9f}\n".format (mc.e_tot))

# It's a bit opaque but this is an object I use to set up the BFGS
# and store intermediates. I cache it at the end.
# All of the CI vecs below are in Fock space <s>because I'm an idiot</s>
obj_fn = mc.fcisolver._obj_fn

res = obj_fn.res # OptimizeResult object returned by scipy.optimize.minimize
                 # see docs.scipy.org for more documentation about this
x = res.x # Amplitude vector for the BFGS problem 
energy, gradient = obj_fn (x) # obj_fn is callable!
print (("Recomputing LASUCCSD total energy with cached objective "
    "function: {:.9f}").format (energy, mc.e_tot))
print (("At convergence, the gradient norm of the LASUCCSD "
    "energy-minimization problem was {:.9e}").format (
    linalg.norm (gradient)))
print ("If that seems too high to you, consider: BFGS sucks.\n")

fcivec = obj_fn.get_fcivec (x) # |LASUCC> itself as a CI vector
ss, multip = mc.fcisolver.spin_square (fcivec, 4, 'ThisArgDoesntMatter')
print ("<LASUCC|S^2|LASUCC> = {:.3f}; apparent S = {:.1f}".format (
    ss, 0.5*(multip-1)))
print ("But is that really the case?")
print ("Singlet weight: {:.2f}".format (fockspace.hilbert_sector_weight(
    fcivec, 4, (2,2), 1)))
print ("Triplet weight: {:.2f}".format (fockspace.hilbert_sector_weight(
    fcivec, 4, (2,2), 3)))
print ("Quintet weight: {:.2f}".format (fockspace.hilbert_sector_weight(
    fcivec, 4, (2,2), 5)))
print ("Oh well, I guess it couldn't have been anything else.\n")

ci_f = obj_fn.get_ci_f (x) # list of optimized CI vectors for each fragment
ci_h = [fockspace.fock2hilbert (c, 2, (1,1)) for c in ci_f]
w_nb = [linalg.norm (c_f)**2 - linalg.norm (c_h)**2 for (c_f, c_h) in 
    zip (ci_f, ci_h)]
print (("If the two numbers below are nonzero, then my implementation "
        "of this definitely doesn't pointlessly waste memory."))
for ix in range (2):
    print (("Weight of fragment-{} wfn outside of the singlet "
            "2-electron Hilbert space: {:.1e}".format (ix, w_nb[ix])))

# U'HU for a single fragment can be retrieved as a
# LASUCCEffectiveHamiltonian object, which is just the ndarray (in 
# the member "full") and some convenience functions
heff = obj_fn.get_dense_heff (x, 0)
print ("\nThe shape of the dense matrix U'HU for the first fragment is",
    heff.full.shape)
hc_f = np.dot (heff.full, ci_f[0].ravel ())
chc_f = np.dot (ci_f[0].ravel ().conj (), hc_f)
g_f = hc_f - (ci_f[0].ravel ()*chc_f)
print (("Recomputing LASUCCSD total energy from the effective "
        "Hamiltonian of the first fragment and its optimized CI vector"
        ": {:.9f}").format (chc_f))
print (("Gradient norm according to the effective Hamiltonian of the "
        "first fragment and its optimized CI vector: {:.9e}".format (
        linalg.norm (g_f))))
heff_non0, idx = heff.get_nonzero_elements ()
print (("The effective Hamiltonian of the first fragment has {} nonzero"
        " elements.").format (len (heff_non0)))
print ("They are:")
ix, jx = np.where (idx)
print ("{:>8s}  {:>3s}  {:>3s}  {:>13s}".format ("Index",
    "Bra", "Ket", "Value"))
for i, j, el in zip (ix, jx, heff_non0):
    idx = "({},{})".format (i, j)
    ia, ib = divmod (i, 4) # 4 determinants possible for 2 spinorbitals
    ja, jb = divmod (j, 4)
    print ("{:>8s}  {:>3s}  {:>3s}  {:13.6e}".format (idx, 
           fockspace.pretty_str (ia, ib, 2),
           fockspace.pretty_str (ja, jb, 2), el))
heff_11, idx = heff.get_number_block ((1,1),(1,1))
print (("The diagonal 2-electron singlet block of the first effective "
        "Hamiltonian of the first fragment is:"))
print (heff_11)
print ("The eigenspectrum of this block is:")
print (linalg.eigh (heff_11)[0])

