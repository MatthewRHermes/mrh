import sys
import numpy as np
from scipy import linalg
from pyscf.fci import cistring
from pyscf import fci, lib
from pyscf.fci.direct_nosym import contract_1e as contract_1e_nosym
from pyscf.fci.direct_spin1 import _unpack_nelec
from pyscf.fci.spin_op import contract_ss, spin_square
from pyscf.data import nist
from itertools import combinations
from mrh.my_pyscf.mcscf import soc_int as soc_int
from mrh.my_pyscf.lassi import dms as lassi_dms

def memcheck (las, ci, soc=None):
    '''Check if the system has enough memory to run these functions!'''
    nfrags = len (ci)
    nroots = len (ci[0])
    assert (all ([len (c) == nroots for c in ci]))
    if soc: # Complex numbers and spinless CI vectors
        itemsize = np.dtype (complex).itemsize
        nelec_fr = [[sum (_unpack_nelec (fcibox._get_nelec (solver, nelecas)))
                     for solver in fcibox.fcisolvers]
                    for fcibox, nelecas in zip (las.fciboxes, las.nelecas_sub)]
        nelec_r = np.asarray (nelec_fr).sum (0)
        mem = sum ([cistring.num_strings (2*las.ncas, nelec) for nelec in nelec_r])
        mem *= itemsize / 1e6
    else:
        mem = sum ([np.prod ([c[iroot].size for c in ci]) 
            * np.amax ([c[iroot].dtype.itemsize for c in ci]) 
            for iroot in range (nroots)]) / 1e6
    max_memory = las.max_memory - lib.current_memory ()[0]
    lib.logger.debug (las, 
        "LASSI op_o0 memory check: {} MB needed of {} MB available ({} MB max)".format (mem,\
        max_memory, las.max_memory))
    return mem < max_memory

def civec_spinless_repr (ci0_r, norb, nelec_r):
    ''' Put CI vectors in the spinless representation; i.e., map
            norb -> 2 * norb
            (neleca, nelecb) -> (neleca+nelecb, 0)
        This permits linear combinations of CI vectors with different
        M == neleca-nelecb at the price of higher memory cost. This function
        does NOT change the datatype.
    '''
    nroots = len (ci0_r)
    nelec_r_tot = [sum (n) for n in nelec_r]
    if len (set (nelec_r_tot)) > 1:
        raise NotImplementedError ("Different particle-number subspaces")
    nelec = nelec_r_tot[0]
    ndet = cistring.num_strings (2*norb, nelec)
    ci1_r = np.zeros ((nroots, ndet), dtype=ci0_r[0].dtype)
    for ci0, ci1, ne in zip (ci0_r, ci1_r, nelec_r):
        neleca, nelecb = _unpack_nelec (ne)
        ndeta = cistring.num_strings (norb, neleca)
        ndetb = cistring.num_strings (norb, nelecb)
        strsa = cistring.addrs2str (norb, neleca, list(range(ndeta)))
        strsb = cistring.addrs2str (norb, nelecb, list(range(ndetb)))
        strs = np.add.outer (strsa, np.left_shift (strsb, norb)).ravel ()
        addrs = cistring.strs2addr (2*norb, nelec, strs)
        ci1[addrs] = ci0[:,:].ravel ()
        if abs(neleca*nelecb)%2: ci1[:] *= -1
        # Sign comes from changing representation:
        # ... a2' a1' a0' ... b2' b1' b0' |vac>
        # ->
        # ... b2' b1' b0' .. a2' a1' a0' |vac>
        # i.e., strictly decreasing from left to right
        # (the ordinality of spin-down is conventionally greater than spin-up)
    return ci1_r[:,:,None]

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
    '''Compute outer-product CI vector for one space table from fragment LAS CI vectors.
    See "ci_outer_product"'''
    neleca_f = [ne[0] for ne in nelec_f]
    nelecb_f = [ne[1] for ne in nelec_f]
    lroots_f = [1 if ci.ndim<3 else ci.shape[0] for ci in ci_f]
    shape_f = [(lroots, cistring.num_strings (norb, neleca), cistring.num_strings (norb, nelecb))
              for lroots, norb, neleca, nelecb in zip (lroots_f, norb_f, neleca_f, nelecb_f)]
    ci_dp = ci_f[-1].copy ().reshape (shape_f[-1])
    for ci_r, shape in zip (ci_f[-2::-1], shape_f[-2::-1]):
        lroots, ndeta, ndetb = ci_dp.shape
        ci_dp = np.multiply.outer (ci_dp, ci_r.reshape (shape))
        ci_dp = ci_dp.transpose (0,3,1,4,2,5).reshape (
            lroots*shape[0], ndeta*shape[1], ndetb*shape[2]
        )
    addrs_a = addr_outer_product (norb_f, neleca_f)
    addrs_b = addr_outer_product (norb_f, nelecb_f)
    nroots = ci_dp.shape[0]
    ndet_a = cistring.num_strings (sum (norb_f), sum (neleca_f))
    ndet_b = cistring.num_strings (sum (norb_f), sum (nelecb_f))
    ci = np.zeros ((nroots,ndet_a,ndet_b), dtype=ci_dp.dtype)
    idx = np.ix_(np.arange (nroots,dtype=int),addrs_a,addrs_b)
    ci[idx] = ci_dp[:,:,:] / linalg.norm (ci_dp, axis=(1,2))[:,None,None]
    if not np.allclose (linalg.norm (ci, axis=(1,2)), 1.0):
        errstr = 'CI norm = {}\naddrs_a = {}\naddrs_b = {}'.format (
            linalg.norm (ci, axis=(1,2)), addrs_a, addrs_b)
        raise RuntimeError (errstr)
    return list (ci)

def ci_outer_product (ci_fr, norb_f, nelec_fr):
    '''Compute outer-product CI vectors from fragment LAS CI vectors.

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
    for space in range (len (ci_fr[0])):
        ci_f = [ci[space] for ci in ci_fr]
        nelec_f = [nelec[space] for nelec in nelec_fr]
        nelec = (sum ([ne[0] for ne in nelec_f]), sum ([ne[1] for ne in nelec_f]))
        ci = _ci_outer_product (ci_f, norb_f, nelec_f)
        ci_r.extend (ci)
        nelec_r.extend ([nelec,]*len(ci))
    return ci_r, nelec_r

#def si_soc (las, h1, ci, nelec, norb):
#
#### function adapted from github.com/hczhai/fci-siso/blob/master/fcisiso.py ###
#
##    au2cm = nist.HARTREE2J / nist.PLANCK / nist.LIGHT_SPEED_SI * 1e-2
#    nroots = len(ci)
#    hsiso = np.zeros((nroots, nroots), dtype=complex)
#    ncas = las.ncas
#    hso_m1 = h1[ncas:2*ncas,0:ncas]
#    hso_p1 = h1[0:ncas,ncas:2*ncas]
#    hso_ze = (h1[0:ncas,0:ncas] - h1[ncas:2*ncas,ncas:2*ncas])/2 
#
#    for istate, (ici, inelec) in enumerate(zip(ci, nelec)):
#        for jstate, (jci, jnelec) in enumerate(zip(ci, nelec)):
#            if jstate > istate:
#                continue
#
#            tp1 = lassi_dms.make_trans(1, ici, jci, norb, inelec, jnelec)
#            tze = lassi_dms.make_trans(0, ici, jci, norb, inelec, jnelec)
#            tm1 = lassi_dms.make_trans(-1, ici, jci, norb, inelec, jnelec)
#
#            if tp1.shape == ():
#                tp1 = np.zeros((ncas,ncas))
#            if tze.shape == ():
#                tze = np.zeros((ncas,ncas))
#            if tm1.shape == ():
#                tm1 = np.zeros((ncas,ncas))
#
#            somat = np.einsum('ri, ir ->', tm1, hso_m1)
#            somat += np.einsum('ri, ir ->', tp1, hso_p1)
#            #somat = somat/2
#            somat += np.einsum('ri, ir ->', tze, hso_ze)
#
#            hsiso[jstate, istate] = somat
#            if istate!= jstate:
#                hsiso[istate, jstate] = somat.conj()
##            somat *= au2cm
#
#    #heigso, hvecso = np.linalg.eigh(hsiso)
#
#    return hsiso

def ham (las, h1, h2, ci_fr, nelec_frs, soc=0, orbsym=None, wfnsym=None):
    '''Build LAS state interaction Hamiltonian, S2, and ovlp matrices

    Args:
        las : instance of class LASSCF
        h1 : ndarray of shape (ncas, ncas)
            Spin-orbit-free one-body CAS Hamiltonian
        h2 : ndarray of shape (ncas, ncas, ncas, ncas)
            Spin-orbit-free two-body CAS Hamiltonian
        ci_fr : nested list of shape (nfrags, nroots)
            Contains CI vectors; element [i,j] is ndarray of shape
            (ndeta[i,j],ndetb[i,j])
        nelec_frs : ndarray of shape (nfrags,nroots,2)
            Number of electrons of each spin in each rootspace in each
            fragment

    Kwargs:
        soc : integer
            Order of spin-orbit coupling included in the Hamiltonian
        orbsym : list of int of length (ncas)
            Irrep ID for each orbital
        wfnsym : int
            Irrep ID for target matrix block

    Returns:
        ham_eff : square ndarray of length (nroots)
            Spin-orbit-free Hamiltonian in state-interaction basis
        s2_eff : square ndarray of length (nroots)
            S2 operator matrix in state-interaction basis
        ovlp_eff : square ndarray of length (nroots)
            Overlap matrix in state-interaction basis
    '''
    if soc>1:
        raise NotImplementedError ("Two-electron spin-orbit coupling")
    mol = las.mol
    norb_f = las.ncas_sub
    norb = sum (norb_f)

    # The function below is the main workhorse of this whole implementation
    ci_r, nelec_r = ci_outer_product (ci_fr, norb_f, nelec_frs)
    nroots = len(ci_r)
    nelec_r_spinless = [tuple((n[0] + n[1], 0)) for n in nelec_r]
    if not len (set (nelec_r_spinless)) == 1:
        raise NotImplementedError ("States with different numbers of electrons")
    # S2 best taken care of before "spinless representation"
    s2_ci = [contract_ss (c, norb, ne) for c, ne in zip(ci_r, nelec_r)]
    s2_eff = np.zeros ((nroots,nroots))
    for i, s2c, nelec_ket in zip(range(nroots), s2_ci, nelec_r):
        for j, c, nelec_bra in zip(range(nroots), ci_r, nelec_r):
            if nelec_ket == nelec_bra:
                s2_eff[i, j] = c.ravel ().dot (s2c.ravel ())
    # Hamiltonian may be complex
    h1_re = h1.real
    h2_re = h2.real
    h1_im = None
    if soc:
        h1_im = h1.imag
        h2_re = np.zeros ([2,norb,]*4, dtype=h1_re.dtype)
        h2_re[0,:,0,:,0,:,0,:] = h2[:]
        h2_re[1,:,1,:,0,:,0,:] = h2[:]
        h2_re[0,:,0,:,1,:,1,:] = h2[:]
        h2_re[1,:,1,:,1,:,1,:] = h2[:]
        h2_re = h2_re.reshape ([2*norb,]*4)
        ci_r = civec_spinless_repr (ci_r, norb, nelec_r)
        nelec_r = nelec_r_spinless
        norb = 2 * norb
        if orbsym is not None: orbsym *= 2

    solver = fci.solver (mol, symm=(wfnsym is not None)).set (orbsym=orbsym, wfnsym=wfnsym)
    ham_ci = []
    for ci, nelec in zip (ci_r, nelec_r):
        h2eff = solver.absorb_h1e (h1_re, h2_re, norb, nelec, 0.5)
        ham_ci.append (solver.contract_2e (h2eff, ci, norb, nelec))
    if h1_im is not None:
        for i, (ci, nelec) in enumerate (zip (ci_r, nelec_r)):
            ham_ci[i] = ham_ci[i] + 1j*contract_1e_nosym (h1_im, ci, norb, nelec)

    ham_eff = np.zeros ((nroots, nroots), dtype=ham_ci[0].dtype)
    ovlp_eff = np.zeros ((nroots, nroots))
    for i, hc, ket, nelec_ket in zip(range(nroots), ham_ci, ci_r, nelec_r):
        for j, c, nelec_bra in zip(range(nroots), ci_r, nelec_r):
            if nelec_ket == nelec_bra:
                ham_eff[i, j] = c.ravel ().dot (hc.ravel ())
                ovlp_eff[i,j] = c.ravel ().dot (ket.ravel ())
    
    return ham_eff, s2_eff, ovlp_eff

def make_stdm12s (las, ci_fr, nelec_frs, orbsym=None, wfnsym=None):
    '''Build LAS state interaction transition density matrices

    Args:
        las : instance of class LASSCF
        ci_fr : nested list of shape (nfrags, nroots)
            Contains CI vectors; element [i,j] is ndarray of shape
            (ndeta[i,j],ndetb[i,j])
        nelec_frs : ndarray of shape (nfrags,nroots,2)
            Number of electrons of each spin in each rootspace in each
            fragment

    Kwargs:
        orbsym : list of int of length (ncas)
            Irrep ID for each orbital
        wfnsym : int
            Irrep ID for target matrix block

    Returns:
        stdm1s : ndarray of shape (nroots,2,ncas,ncas,nroots) OR (nroots,2*ncas,2*ncas,nroots)
            One-body transition density matrices between LAS states.
            If states with different spin projections (i.e., neleca-nelecb) are present, the 4d
            spinorbital array is returned. Otherwise, the 5d spatial-orbital array is returned.
        stdm2s : ndarray of shape [nroots,]+ [2,ncas,ncas,]*2 + [nroots,]
            Two-body transition density matrices between LAS states
    '''
    mol = las.mol
    norb_f = las.ncas_sub
    norb = sum (norb_f) 
    ci_r, nelec_r = ci_outer_product (ci_fr, norb_f, nelec_frs)
    nelec_r_spinless = [tuple((n[0] + n[1], 0)) for n in nelec_r]
    nroots = len (ci_r)
    if not len (set (nelec_r_spinless)) == 1:
        raise NotImplementedError ("States with different numbers of electrons")
    spin_pure = len (set (nelec_r)) == 1
    if not spin_pure:
        # Map to "spinless electrons": 
        ci_r = civec_spinless_repr (ci_r, norb, nelec_r)
        nelec_r = nelec_r_spinless
        norb = 2 * norb
        if orbsym is not None: orbsym *= 2

    solver = fci.solver (mol).set (orbsym=orbsym, wfnsym=wfnsym)
    stdm1s = np.zeros ((nroots, nroots, 2, norb, norb),
        dtype=ci_r[0].dtype).transpose (0,2,3,4,1)
    stdm2s = np.zeros ((nroots, nroots, 2, norb, norb, 2, norb, norb),
        dtype=ci_r[0].dtype).transpose (0,2,3,4,5,6,7,1)
    for i, (ci, ne) in enumerate (zip (ci_r, nelec_r)):
        rdm1s, rdm2s = solver.make_rdm12s (ci, norb, ne)
        stdm1s[i,0,:,:,i] = rdm1s[0]
        stdm1s[i,1,:,:,i] = rdm1s[1]
        stdm2s[i,0,:,:,0,:,:,i] = rdm2s[0]
        stdm2s[i,0,:,:,1,:,:,i] = rdm2s[1]
        stdm2s[i,1,:,:,0,:,:,i] = rdm2s[1].transpose (2,3,0,1)
        stdm2s[i,1,:,:,1,:,:,i] = rdm2s[2]

    spin_sector_offset = np.zeros ((nroots,nroots))
    for (i,(ci_bra,ne_bra)), (j,(ci_ket,ne_ket)) in combinations(enumerate(zip(ci_r,nelec_r)),2):
        M_bra = ne_bra[1] - ne_bra[0]
        M_ket = ne_ket[0] - ne_ket[1]
        N_bra = sum (ne_bra)
        N_ket = sum (ne_ket)
        if ne_bra == ne_ket:
            tdm1s, tdm2s = solver.trans_rdm12s (ci_bra, ci_ket, norb, ne_bra)
            stdm1s[i,0,:,:,j] = tdm1s[0]
            stdm1s[i,1,:,:,j] = tdm1s[1]
            stdm1s[j,0,:,:,i] = tdm1s[0].T
            stdm1s[j,1,:,:,i] = tdm1s[1].T
            for spin, tdm2 in enumerate (tdm2s):
                p = spin // 2
                q = spin % 2
                stdm2s[i,p,:,:,q,:,:,j] = tdm2
                stdm2s[j,p,:,:,q,:,:,i] = tdm2.transpose (1,0,3,2)

    if not spin_pure: # cleanup the "spinless mapping"
        stdm1s = stdm1s[:,0,:,:,:]
        # TODO: 2e- spin-orbit coupling support in caller
        n = norb // 2
        stdm2s_ = np.zeros ((nroots, nroots, 2, n, n, 2, n, n),
            dtype=ci_r[0].dtype).transpose (0,2,3,4,5,6,7,1)
        stdm2s_[:,0,:,:,0,:,:,:] = stdm2s[:,0,:n,:n,0,:n,:n,:]
        stdm2s_[:,0,:,:,1,:,:,:] = stdm2s[:,0,:n,:n,0,n:,n:,:]
        stdm2s_[:,1,:,:,0,:,:,:] = stdm2s[:,0,n:,n:,0,:n,:n,:]
        stdm2s_[:,1,:,:,1,:,:,:] = stdm2s[:,0,n:,n:,0,n:,n:,:]
        stdm2s = stdm2s_

    return stdm1s, stdm2s 

def roots_make_rdm12s (las, ci_fr, nelec_frs, si, orbsym=None, wfnsym=None):
    '''Build LAS state interaction reduced density matrices for final
    LASSI eigenstates.

    Args:
        las : instance of class LASSCF
        ci_fr : nested list of shape (nfrags, nroots)
            Contains CI vectors; element [i,j] is ndarray of shape
            (ndeta[i,j],ndetb[i,j])
        nelec_frs : ndarray of shape (nfrags,nroots,2)
            Number of electrons of each spin in each rootspace in each
            fragment
        si : ndarray of shape (nroots, nroots)
            Unitary matrix defining final LASSI states in terms of
            non-interacting LAS states

    Kwargs:
        orbsym : list of int of length (ncas)
            Irrep ID for each orbital
        wfnsym : int
            Irrep ID for target matrix block

    Returns:
        rdm1s : ndarray of shape (nroots, 2, ncas, ncas) OR (nroots, 2*ncas, 2*ncas)
            One-body transition density matrices between LAS states
            If states with different spin projections (i.e., neleca-nelecb) are present, the 3d
            spinorbital array is returned. Otherwise, the 4d spatial-orbital array is returned.
        rdm2s : ndarray of length (nroots, 2, ncas, ncas, 2, ncas, ncas)
            Two-body transition density matrices between LAS states
    '''
    mol = las.mol
    norb_f = las.ncas_sub
    ci_r, nelec_r = ci_outer_product (ci_fr, norb_f, nelec_frs)
    nelec_r_spinless = [tuple((n[0] + n[1], 0)) for n in nelec_r]
    nroots = len (ci_r)
    norb = sum (norb_f)
    if not len (set (nelec_r_spinless)) == 1:
        raise NotImplementedError ("States with different numbers of electrons")
    spin_pure = len (set (nelec_r)) == 1
    if not spin_pure:
        # Map to "spinless electrons": 
        ci_r = civec_spinless_repr (ci_r, norb, nelec_r)
        nelec_r = nelec_r_spinless
        norb = 2 * norb
        if orbsym is not None: orbsym *= 2

    ci_r = np.tensordot (si.T, np.stack (ci_r, axis=0), axes=1)
    ci_r_real = np.ascontiguousarray (ci_r.real)
    rdm1s = np.zeros ((nroots, 2, norb, norb), dtype=ci_r.dtype)
    rdm2s = np.zeros ((nroots, 2, norb, norb, 2, norb, norb), dtype=ci_r.dtype)
    is_complex = np.iscomplexobj (ci_r)
    if is_complex:
        #solver = fci.fci_dhf_slow.FCISolver (mol)
        #for ix, (ci, ne) in enumerate (zip (ci_r, nelec_r)):
        #    d1, d2 = solver.make_rdm12 (ci, norb, sum(ne))
        #    rdm1s[ix,0,:,:] = d1[:]
        #    rdm2s[ix,0,:,:,0,:,:] = d2[:]
        # ^ this is WAY too slow!
        ci_r_imag = np.ascontiguousarray (ci_r.imag)
    else:
        ci_r_imag = [0,]*nroots
        #solver = fci.solver (mol).set (orbsym=orbsym, wfnsym=wfnsym)
    solver = fci.solver (mol).set (orbsym=orbsym, wfnsym=wfnsym)
    for ix, (ci_re, ci_im, ne) in enumerate (zip (ci_r_real, ci_r_imag, nelec_r)):
        d1s, d2s = solver.make_rdm12s (ci_re, norb, ne)
        d2s = (d2s[0], d2s[1], d2s[1].transpose (2,3,0,1), d2s[2])
        if is_complex:
            d1s = np.asarray (d1s, dtype=complex)
            d2s = np.asarray (d2s, dtype=complex)
            d1s2, d2s2 = solver.make_rdm12s (ci_im, norb, ne)
            d2s2 = (d2s2[0], d2s2[1], d2s2[1].transpose (2,3,0,1), d2s2[2])
            d1s += np.asarray (d1s2)
            d2s += np.asarray (d2s2)
            d1s2, d2s2 = solver.trans_rdm12s (ci_re, ci_im, norb, ne)
            d1s2 -= np.asarray (d1s2).transpose (0,2,1)
            d2s2 -= np.asarray (d2s2).transpose (0,2,1,4,3)
            d1s -= 1j * d1s2 
            d2s += 1j * d2s2
        rdm1s[ix,0,:,:] = d1s[0]
        rdm1s[ix,1,:,:] = d1s[1]
        rdm2s[ix,0,:,:,0,:,:] = d2s[0]
        rdm2s[ix,0,:,:,1,:,:] = d2s[1]
        rdm2s[ix,1,:,:,0,:,:] = d2s[2]
        rdm2s[ix,1,:,:,1,:,:] = d2s[3]

    if not spin_pure: # cleanup the "spinless mapping"
        rdm1s = rdm1s[:,0,:,:]
        # TODO: 2e- SOC
        n = norb // 2
        rdm2s_ = np.zeros ((nroots, 2, n, n, 2, n, n), dtype=ci_r.dtype)
        rdm2s_[:,0,:,:,0,:,:] = rdm2s[:,0,:n,:n,0,:n,:n]
        rdm2s_[:,0,:,:,1,:,:] = rdm2s[:,0,:n,:n,0,n:,n:]
        rdm2s_[:,1,:,:,0,:,:] = rdm2s[:,0,n:,n:,0,:n,:n]
        rdm2s_[:,1,:,:,1,:,:] = rdm2s[:,0,n:,n:,0,n:,n:]
        rdm2s = rdm2s_

    return rdm1s, rdm2s

if __name__ == '__main__':
    from pyscf import scf, lib
    from mrh.my_pyscf.mcscf.lasscf_sync_o0 import LASSCF
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

