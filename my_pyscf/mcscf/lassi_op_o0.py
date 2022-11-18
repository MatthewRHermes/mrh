import sys
import numpy as np
from scipy import linalg
from pyscf.fci import cistring
from pyscf import fci, lib
from pyscf.fci.direct_spin1 import _unpack_nelec
from pyscf.fci.spin_op import contract_ss, spin_square
from pyscf.data import nist
from itertools import combinations
from mrh.my_pyscf.mcscf import soc_int as soc_int
from mrh.my_pyscf.mcscf import lassi_dms as lassi_dms 

def memcheck (las, ci):
    '''Check if the system has enough memory to run these functions!'''
    nfrags = len (ci)
    nroots = len (ci[0])
    assert (all ([len (c) == nroots for c in ci]))
    mem = sum ([np.prod ([c[iroot].size for c in ci]) 
        * np.amax ([c[iroot].dtype.itemsize for c in ci]) 
        for iroot in range (nroots)]) / 1e6
    max_memory = las.max_memory - lib.current_memory ()[0]
    lib.logger.debug (las, 
        "LASSI op_o0 memory check: {} MB needed of {} MB available ({} MB max)".format (mem,\
        max_memory, las.max_memory))
    return mem < max_memory

def addr_outer_product (norb_f, nelec_f):
    '''Build index arrays for reshaping a direct product of LAS CI
    vectors into the appropriate orbital ordering for a CAS CI vector'''
    norb = sum (norb_f)
    nelec = sum (nelec_f)
    # Must skip over cases where there are no electrons of a specific spin in a particular subspace
    norbrange = np.cumsum (norb_f)
    addrs = []
    for i in range (0, len (norbrange)):
        irange = range (norbrange[i]-norb_f[i], norbrange[i])
        new_addrs = cistring.sub_addrs (norb, nelec, irange, nelec_f[i]) if nelec_f[i] else []
        if len (addrs) == 0:
            addrs = new_addrs
        elif len (new_addrs) > 0:
            addrs = np.intersect1d (addrs, new_addrs)
    if not len (addrs): addrs=[0] # No beta electrons edge case
    return addrs

def _ci_outer_product (ci_f, norb_f, nelec_f):
    '''Compute ONE outer-product CI vector from fragment LAS CI vectors.
    See "ci_outer_product"'''
    # The two steps here are:
    #   1. Multiply the CI vectors together using np.multiply.outer, and
    #   2. Reshape and transpose the product so that the orbitals appear
    #      in the correct order.
    # There may be an ambiguous factor of -1, but it should apply to the
    # entire product CI vector so maybe it doesn't matter?
    neleca_f = [ne[0] for ne in nelec_f]
    nelecb_f = [ne[1] for ne in nelec_f]
    ndet_f = [(cistring.num_strings (norb, neleca), cistring.num_strings (norb, nelecb))
              for norb, neleca, nelecb in zip (norb_f, neleca_f, nelecb_f)]
    ci_dp = ci_f[-1].copy ().reshape (ndet_f[-1])
    for ci_r, ndet in zip (ci_f[-2::-1], ndet_f[-2::-1]):
        ndeta, ndetb = ci_dp.shape
        ci_dp = np.multiply.outer (ci_dp, ci_r.reshape (ndet))
        ci_dp = ci_dp.transpose (0,2,1,3).reshape (ndeta*ndet[0], ndetb*ndet[1])
    addrs_a = addr_outer_product (norb_f, neleca_f)
    addrs_b = addr_outer_product (norb_f, nelecb_f)
    ndet_a = cistring.num_strings (sum (norb_f), sum (neleca_f))
    ndet_b = cistring.num_strings (sum (norb_f), sum (nelecb_f))
    ci = np.zeros ((ndet_a,ndet_b), dtype=ci_dp.dtype)
    ci[np.ix_(addrs_a,addrs_b)] = ci_dp[:,:] / linalg.norm (ci_dp)
    if not np.isclose (linalg.norm (ci), 1.0):
        errstr = 'CI norm = {}\naddrs_a = {}\naddrs_b = {}'.format (
            linalg.norm (ci), addrs_a, addrs_b)
        raise RuntimeError (errstr)
    return ci

def ci_outer_product (ci_fr, norb_f, nelec_fr):
    '''Compute outer-product CI vectors from fragment LAS CI vectors.
    TODO: extend to accomodate states o different ms being addressed
    together. I think the only thing this entails is turning "nelec"
    into a list of length (nroots)

    Args:
        ci_fr : nested list of shape (nfrags, nroots)
            Contains CI vectors; element [i,j] is ndarray of shape
            (ndeta[i,j],ndetb[i,j])
        norb_f : list of length (nfrags)
            Number of orbitals in each fragment
        nelec_fr : ndarray-like of shape (nfrags, nroots, 2)
            Number of spin-up and spin-down electrons in each fragment
            and root

    Returns:
        ci_r : list of length (nroots)
            Contains full CAS CI vector
        nelec : tuple of length 2
            (neleca, nelecb) for this batch of states
    '''

    ci_r = []
    nelec_r = []
    for state in range (len (ci_fr[0])):
        ci_f = [ci[state] for ci in ci_fr]
        nelec_f = [nelec[state] for nelec in nelec_fr]
        ci_r.append (_ci_outer_product (ci_f, norb_f, nelec_f))
        nelec_r.append ((sum ([ne[0] for ne in nelec_f]),
                       sum ([ne[1] for ne in nelec_f])))
    # Teffanie: rather than doing one thing for SOC calc'ns and another thing for spin-pure
    # calc'ns, I think it makes good sense to just return the full list of nelec_r for every calc'n
    # regardless of whether or not it's needed; calling functions can be easily modified to account
    # for this.
    return ci_r, nelec_r

# Teffanie: I would rather "hso" be computed by the caller (lassi.py) and passed as part of h1 or in
# addition to h1, because that makes it more general for later on if we want to do X2C, other kinds
# of SOMF, etc. 
def si_soc (las, ci, nelec, norb):

### function adapted from github.com/hczhai/fci-siso/blob/master/fcisiso.py ###

    au2cm = nist.HARTREE2J / nist.PLANCK / nist.LIGHT_SPEED_SI * 1e-2
    nroots = len(ci)
    hsiso = np.zeros((nroots, nroots), dtype=complex)

    dm0 = las.mol.make_rdm1() # potentially replace with pre-calculated atomic densities used in AMFI
    hsoao = soc_int.compute_hso(las.mol, dm0, amfi=True)
    hso = np.einsum('rij, ip, jq -> rpq', hsoao, las.mo_coeff[:, las.ncore:las.ncore + norb],
            las.mo_coeff[:, las.ncore:las.ncore + norb])

    for istate, (ici, inelec) in enumerate(zip(ci, nelec)):
        for jstate, (jci, jnelec) in enumerate(zip(ci, nelec)):
            if jstate > istate:
                continue

            tp1 = lassi_dms.make_trans(1, ici, jci, norb, inelec, jnelec)
            tze = lassi_dms.make_trans(0, ici, jci, norb, inelec, jnelec)
            tm1 = lassi_dms.make_trans(-1, ici, jci, norb, inelec, jnelec)

            t = np.zeros((3, norb, norb), dtype=complex)
            t[0] = (0.5 + 0j) * (tm1 - tp1)
            t[1] = (0.5j + 0) * (tm1 + tp1)
            t[2] = (np.sqrt(0.5) + 0j) * tze

            somat = np.einsum('rij, rij ->', t, hso)
            hsiso[istate, jstate] = somat

            if istate!= jstate:
                hsiso[jstate, istate] = somat.conj()
            somat *= au2cm

    heigso, hvecso = np.linalg.eigh(hsiso)

    return hsiso

# Teffanie: when you add an argument or keyword argument to a function, you should describe it in
# the docstring. I changed soc from a boolean to an integer, on the idea that later on we can
# distinguish between soc=1 for SOMF calculations and soc=2 for 2-electron spin-breaking.
# However, you can still pass "soc=True" or "soc=False" and it should work.
def ham (las, h1, h2, ci_fr, idx_root, soc=0, orbsym=None, wfnsym=None):
    '''Build LAS state interaction Hamiltonian, S2, and ovlp matrices
    TODO: extend to accomodate states of different ms being addressed
    together, and then spin-orbit coupling.

    Args:
        las : instance of class LASSCF
        h1 : ndarray of shape (ncas, ncas)
            Spin-orbit-free one-body CAS Hamiltonian
        h2 : ndarray of shape (ncas, ncas, ncas, ncas)
            Spin-orbit-free two-body CAS Hamiltonian
        ci_fr : nested list of shape (nfrags, count_nonzero (idx_root))
            Contains CI vectors; element [i,j] is ndarray of shape
            (ndeta[i,j],ndetb[i,j])
        idx_root : mask index array of shape (las.nroots)
            Maps the states included in ci_fr to the states in "las"

    Kwargs:
        soc : integer
            Order of spin-orbit coupling included in the Hamiltonian
        orbsym : list of int of length (ncas)
            Irrep ID for each orbital
        wfnsym : int
            Irrep ID for target matrix block

    Returns:
        ham_eff : square ndarray of length (count_nonzero (idx_root))
            Spin-orbit-free Hamiltonian in state-interaction basis
        s2_eff : square ndarray of length (count_nonzero (idx_root))
            S2 operator matrix in state-interaction basis
        ovlp_eff : square ndarray of length (count_nonzero (idx_root))
            Overlap matrix in state-interaction basis
    '''
    mol = las.mol
    norb_f = las.ncas_sub
    nelec_fr = [[_unpack_nelec (fcibox._get_nelec (solver, nelecas))
                 for solver, ix in zip (fcibox.fcisolvers, idx_root) if ix]
                for fcibox, nelecas in zip (las.fciboxes, las.nelecas_sub)]

    # The function below is the main workhorse of this whole implementation
    ci_r, nelec_r = ci_outer_product (ci_fr, norb_f, nelec_fr)

    # Teffanie: I got rid of the conditional here doing one thing with soc and another thing w/out
    # it, because the former works for both cases. However, see the note below.
    solver = fci.solver (mol).set (orbsym=orbsym, wfnsym=wfnsym)
    norb = sum (norb_f)
    nroots = len(ci_r)
    # Teffanie: note that absorb_h1e here required knowledge of nelec.
    ham_ci = []
    for ci, nelec in zip (ci_r, nelec_r):
        h2eff = solver.absorb_h1e (h1, h2, norb, nelec, 0.5)
        ham_ci.append (solver.contract_2e (h2eff, ci, norb, nelec))
    s2_ci = [contract_ss (c, norb, ne) for c, ne in zip(ci_r, nelec_r)]

    ham_eff, s2_eff, ovlp_eff = [ np.zeros((nroots, nroots)) for i in range (3) ]
    for i, hc, s2c, ket, nelec_ket in zip(range(nroots), ham_ci, s2_ci, ci_r, nelec_r):
        for j, c, nelec_bra in zip(range(nroots), ci_r, nelec_r):
            if nelec_ket == nelec_bra:
                # Teffanie: you had 'if len(hc) == len(c):', which I'm not sure is general enough.
                # For example, in six orbitals, nelec=(4,2) and nelec=(2,4) have the same array
                # shapes because of particle-hole symmetry, but the matrix elements coupling them
                # should be zero.
                ham_eff[i, j] = c.ravel ().dot (hc.ravel ())
                s2_eff[i, j] = c.ravel ().dot (s2c.ravel ())
                ovlp_eff[i,j] = c.ravel ().dot (ket.ravel ())
    if soc: # Teffanie: conveniently, this still works if soc is an integer
        hso = si_soc(las, ci_r, nelec_r, norb)
        ham_eff = ham_eff + hso
    
    return ham_eff, s2_eff, ovlp_eff

# Teffanie: in these functions, we don't need to know why or how we have multiple distinct spin
# sectors; the "ci_outer_product" function itself tells us this. So there's no reason to pass
# "soc" here or in the next function.
def make_stdm12s (las, ci_fr, idx_root, orbsym=None, wfnsym=None):
    '''Build LAS state interaction transition density matrices
    TODO: extend to accomodate states of different ms being addressed
    together, and then spin-orbit coupling.

    Args:
        las : instance of class LASSCF
        ci_fr : nested list of shape (nfrags, count_nonzero (idx_root))
            Contains CI vectors; element [i,j] is ndarray of shape
            (ndeta[i,j],ndetb[i,j])
        idx_root : mask index array of shape (las.nroots)
            Maps the states included in ci_fr to the states in "las"
            (Below, "nroots" means "count_nonzero (idx_root)")

    Kwargs:
        orbsym : list of int of length (ncas)
            Irrep ID for each orbital
        wfnsym : int
            Irrep ID for target matrix block

    Returns:
        stdm1s : ndarray of shape (nroots, nroots, 2, ncas, ncas)
            One-body transition density matrices between LAS states
        stdm2s : ndarray of shape [nroots,]*2 + [2,ncas,ncas,]*2
            Two-body transition density matrices between LAS states
    '''
    # Teffanie: we need to extend this to the spin-breaking blocks of the transition density
    # matrices
    mol = las.mol
    norb_f = las.ncas_sub
    nelec_fr = [[_unpack_nelec (fcibox._get_nelec (solver, nelecas))
                 for solver, ix in zip (fcibox.fcisolvers, idx_root) if ix]
                for fcibox, nelecas in zip (las.fciboxes, las.nelecas_sub)]
    ci_r, nelec_r = ci_outer_product (ci_fr, norb_f, nelec_fr)
    norb = sum (norb_f) 
    solver = fci.solver (mol).set (orbsym=orbsym, wfnsym=wfnsym)
    nroots = len (ci_r)
    stdm1s = np.zeros ((nroots, nroots, 2, norb, norb),
        dtype=ci_r[0].dtype).transpose (0,2,3,4,1)
    stdm2s = np.zeros ((nroots, nroots, 2, norb, norb, 2, norb, norb),
        dtype=ci_r[0].dtype).transpose (0,2,3,4,5,6,7,1)
    # Teffanie: I removed the conditional "if soc:" below because both blocks work for the
    # spin-pure case
    for i, (ci, ne) in enumerate (zip (ci_r, nelec_r)):
        rdm1s, rdm2s = solver.make_rdm12s (ci, norb, ne)
        stdm1s[i,0,:,:,i] = rdm1s[0]
        stdm1s[i,1,:,:,i] = rdm1s[1]
        stdm2s[i,0,:,:,0,:,:,i] = rdm2s[0]
        stdm2s[i,0,:,:,1,:,:,i] = rdm2s[1]
        stdm2s[i,1,:,:,0,:,:,i] = rdm2s[1].transpose (2,3,0,1)
        stdm2s[i,1,:,:,1,:,:,i] = rdm2s[2]

    for (i, (ci_bra, ne_bra)), (j, (ci_ket, ne_ket)) in combinations (enumerate (zip (ci_r, nelec_r)), 2):
        spin_bra = ne_bra[1] - ne_bra[0]
        spin_ket = ne_ket[0] - ne_ket[1]
        if ne_bra == ne_ket:
            tdm1s, tdm2s = solver.trans_rdm12s (ci_bra, ci_ket, norb, ne_bra)
            # Transpose for 1TDM is backwards because of stupid PySCF convention
            stdm1s[i,0,:,:,j] = tdm1s[0].T
            stdm1s[i,1,:,:,j] = tdm1s[1].T
            stdm1s[j,0,:,:,i] = tdm1s[0]
            stdm1s[j,1,:,:,i] = tdm1s[1]
            for spin, tdm2 in enumerate (tdm2s):
                p = spin // 2
                q = spin % 2
                stdm2s[i,p,:,:,q,:,:,j] = tdm2
                stdm2s[j,p,:,:,q,:,:,i] = tdm2.transpose (1,0,3,2)
        # Teffanie: the below is unnecessary; the whole arrays were initialized to zero anyway
        #else:
        #   stdm1s[i,:,:,:,j] =  stdm1s[j,:,:,:,i] = np.array(0)
        #   stdm2s[i,:,:,:,:,:,:,j] = stdm2s[j,:,:,:,:,:,:,i] = np.array(0)

    return stdm1s, stdm2s 

def roots_make_rdm12s (las, ci_fr, idx_root, si, orbsym=None, wfnsym=None):
    '''Build LAS state interaction reduced density matrices for final
    LASSI eigenstates.
    TODO: extend to accomodate states of different ms being addressed
    together, and then spin-orbit coupling.

    Args:
        las : instance of class LASSCF
        ci_fr : nested list of shape (nfrags, count_nonzero (idx_root))
            Contains CI vectors; element [i,j] is ndarray of shape
            (ndeta[i,j],ndetb[i,j])
        idx_root : mask index array of shape (las.nroots)
            Maps the states included in ci_fr to the states in "las"
            (Below, "nroots" means "count_nonzero (idx_root)")
        si : ndarray of shape (nroots, nroots)
            Unitary matrix defining final LASSI states in terms of
            non-interacting LAS states

    Kwargs:
        orbsym : list of int of length (ncas)
            Irrep ID for each orbital
        wfnsym : int
            Irrep ID for target matrix block

    Returns:
        rdm1s : ndarray of length (nroots, 2, ncas, ncas)
            One-body transition density matrices between LAS states
        rdm2s : ndarray of length (nroots, 2, ncas, ncas, 2, ncas, ncas)
            Two-body transition density matrices between LAS states
    '''
    # Teffanie: there are whole other sectors of transition density matrices that should be
    # considered with soc: the <I|ap'bq|J> and <I|bp'aq|J> sectors, etc. Also, this will/should
    # crash when different spin sectors are considered, so I put a NotImplementedError below.
    mol = las.mol
    norb_f = las.ncas_sub
    nelec_fr = [[_unpack_nelec (fcibox._get_nelec (solver, nelecas))
                 for solver, ix in zip (fcibox.fcisolvers, idx_root) if ix]
                for fcibox, nelecas in zip (las.fciboxes, las.nelecas_sub)]
    ci_r, nelec_r = ci_outer_product (ci_fr, norb_f, nelec_fr)
    if len (set (nelec_r)) > 1:
        raise NotImplementedError ("Reduced density matrices for spin-impure states")
    norb = sum (norb_f)
    solver = fci.solver (mol).set (orbsym=orbsym, wfnsym=wfnsym)
    nroots = len (ci_r)
    ci_r = np.tensordot (si.conj ().T, np.stack (ci_r, axis=0), axes=1)
    rdm1s = np.zeros ((nroots, 2, norb, norb), dtype=ci_r.dtype)
    rdm2s = np.zeros ((nroots, 2, norb, norb, 2, norb, norb), dtype=ci_r.dtype)
    # Teffanie: again, you don't need the conditional if both blocks are nearly identical
    # like this. Just leave the more general one.
    for ix, (ci, ne) in enumerate (zip (ci_r, nelec_r)):
        d1s, d2s = solver.make_rdm12s (ci, norb, ne)
        rdm1s[ix,0,:,:] = d1s[0]
        rdm1s[ix,1,:,:] = d1s[1]
        rdm2s[ix,0,:,:,0,:,:] = d2s[0]
        rdm2s[ix,0,:,:,1,:,:] = d2s[1]
        rdm2s[ix,1,:,:,0,:,:] = d2s[1].transpose (2,3,0,1)
        rdm2s[ix,1,:,:,1,:,:] = d2s[2]
    return rdm1s, rdm2s

if __name__ == '__main__':
    from pyscf import scf, lib
    from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
    import os
    class cd:
        """Context manager for changing the current working directory"""
        def __init__(self, newPath):
            self.newPath = os.path.expanduser(newPath)

        def __enter__(self):
            self.savedPath = os.getcwd()
            os.chdir(self.newPath)

        def __exit__(self, etype, value, traceback):
            os.chdir(self.savedPath)
    from mrh.examples.lasscf.c2h6n4.c2h6n4_struct import structure as struct
    with cd ("/home/herme068/gits/mrh/examples/lasscf/c2h6n4"):
        mol = struct (2.0, 2.0, '6-31g', symmetry=False)
    mol.verbose = lib.logger.DEBUG
    mol.output = 'sa_lasscf_slow_ham.log'
    mol.build ()
    mf = scf.RHF (mol).run ()
    tol = 1e-6 if len (sys.argv) < 2 else float (sys.argv[1])
    las = LASSCF (mf, (4,4), (4,4)).set (conv_tol_grad = tol)
    mo = las.localize_init_guess ((list(range(3)),list(range(9,12))), mo_coeff=mf.mo_coeff)
    las.state_average_(weights = [0.5, 0.5], spins=[[0,0],[2,-2]])
    h2eff_sub, veff = las.kernel (mo)[-2:]
    e_states = las.e_states

    ncore, ncas, nocc = las.ncore, las.ncas, las.ncore + las.ncas
    mo_coeff = las.mo_coeff
    mo_core = mo_coeff[:,:ncore]
    mo_cas = mo_coeff[:,ncore:nocc]
    e0 = las._scf.energy_nuc () + 2 * (((las._scf.get_hcore () + veff.c/2) @ mo_core) * mo_core).sum () 
    h1 = mo_cas.conj ().T @ (las._scf.get_hcore () + veff.c) @ mo_cas
    h2 = h2eff_sub[ncore:nocc].reshape (ncas*ncas, ncas * (ncas+1) // 2)
    h2 = lib.numpy_helper.unpack_tril (h2).reshape (ncas, ncas, ncas, ncas)
    nelec_fr = []
    for fcibox, nelec in zip (las.fciboxes, las.nelecas_sub):
        ne = sum (nelec)
        nelec_fr.append ([_unpack_nelec (fcibox._get_nelec (solver, ne)) for solver in fcibox.fcisolvers])
    ham_eff = slow_ham (las.mol, h1, h2, las.ci, las.ncas_sub, nelec_fr)[0]
    print (las.converged, e_states - (e0 + np.diag (ham_eff)))


