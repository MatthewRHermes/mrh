import numpy as np
from pyscf.lib import logger
from scipy import linalg
from mrh.util.la import safe_svd_warner

class LASKeyframe (object):
    '''Shallow struct for various intermediates. DON'T put complicated code in here Matt!!!'''

    def __init__(self, las, mo_coeff, ci):
        self.las = las
        self.mo_coeff = mo_coeff
        self.ci = ci
        self._dm1s = self._veff = self._fock1 = self._h1eff_sub = self._h2eff_sub = None
        self.frags = set ()

    @property
    def dm1s (self):
        if self._dm1s is None:
            self._dm1s = self.las.make_rdm1s (mo_coeff=self.mo_coeff, ci=self.ci)
        return self._dm1s

    @property
    def veff (self):
        if self._veff is None:
            self._veff = self.las.get_veff (dm=self.dm1s, spin_sep=True)
        return self._veff

    @property
    def fock1 (self):
        if self._fock1 is None:
            self._fock1 = self.las.get_grad_orb (
                mo_coeff=self.mo_coeff, ci=self.ci, h2eff_sub=self.h2eff_sub, veff=self.veff,
                dm1s=self.dm1s, hermi=0)
        return self._fock1

    @property
    def h2eff_sub (self):
        if self._h2eff_sub is None:
            self._h2eff_sub = self.las.get_h2eff (self.mo_coeff)
        return self._h2eff_sub

    @property
    def h1eff_sub (self):
        if self._h1eff_sub is None:
            self._h1eff_sub = self.las.get_h1eff (self.mo_coeff, ci=self.ci, veff=self.veff,
                h2eff_sub=self.h2eff_sub)
        return self._h1eff_sub

    def copy (self):
        ''' MO coefficients deepcopy; CI vectors shallow copy. Everything else, drop. '''
        mo1 = self.mo_coeff.copy ()
        ci1_fr = []
        ci0_fr = self.ci
        for ci0_r in ci0_fr:
            ci1_r = []
            for ci0 in ci0_r:
                ci1 = ci0.view ()
                ci1_r.append (ci1)
            ci1_fr.append (ci1_r)
        return LASKeyframe (self.las, mo1, ci1_fr)


def approx_keyframe_ovlp (las, kf1, kf2):
    '''Evaluate the similarity of two keyframes in terms of orbital and CI vector overlaps.

    Args:
        las : object of :class:`LASCINoSymm`
        kf1 : object of :class:`LASKeyframe`
        kf2 : object of :class:`LASKeyframe`

    Returns:
        mo_ovlp : float
            Products of the overlaps of the rotationally-invariant subspaces across the two
            keyframes; i.e.: prod (svals (inactive orbitals)) * prod (svals (virtual orbitals))
            * prod (svals (active 1)) * prod (svals (active 2)) * ...
        ci_ovlp : list of length nfrags of list of length nroots of floats
            Overlaps of the CI vectors, assuming that prod (svals (active n)) = 1. Meaningless
            if mo_ovlp deviates significantly from 1.
    '''

    u, svals, vh = orbital_block_svd (las, kf1, kf2)
    mo_ovlp = np.prod (svals)

    ci_ovlp = []
    for ifrag, (fcibox, c1_r, c2_r) in enumerate (zip (las.fciboxes, kf1.ci, kf2.ci)):
        nlas, nelelas = las.ncas_sub[ifrag], las.nelecas_sub[ifrag]
        i = las.ncore + sum (las.ncas_sub[:ifrag])
        j = i + las.ncas_sub[ifrag]
        umat = u[i:j,i:j] @ vh[i:j,i:j]
        c1_r = fcibox.states_transform_ci_for_orbital_rotation (c1_r, nlas, nelelas, umat)
        ci_ovlp.append ([abs (c1.conj ().ravel ().dot (c2.ravel ()))
                         for c1, c2 in zip (c1_r, c2_r)])

    return mo_ovlp, ci_ovlp
    
def orbital_block_svd (las, kf1, kf2):
    '''Evaluate the block-SVD of the orbitals of two keyframes. Blocks are inactive (core), active
    of each fragment, and virtual.

    Args:
        las : object of :class:`LASCINoSymm`
        kf1 : object of :class:`LASKeyframe`
        kf2 : object of :class:`LASKeyframe`

    Returns:
        u : array of shape (nao,nmo)
            Block-diagonal unitary matrix of orbital rotations for kf1, keeping each subspace
            unchanged but aligning the orbitals to identify the spaces the two keyframes have in
            common, if any
        svals : array of shape (nmo)
            Singular values.
        vh: array of shape (nmo,nao)
            Transpose of block-diagonal unitary matrix of orbital rotations for kf2, keeping each
            subspace unchanged but aligning the orbitals to identify the spaces the two keyframes
            have in common, if any
    '''
    log = logger.new_logger (las, las.verbose)
    svd = safe_svd_warner (log.warn)
    nao, nmo = kf1.mo_coeff.shape    
    ncore, ncas = las.ncore, las.ncas
    nocc = ncore + ncas
    nvirt = nmo - nocc

    u = []
    svals = []
    vh = []

    s0 = las._scf.get_ovlp ()
    if ncore:
        mo1 = kf1.mo_coeff[:,:ncore]
        mo2 = kf2.mo_coeff[:,:ncore]
        s1 = mo1.conj ().T @ s0 @ mo2
        u_core, svals_core, vh_core = svd (s1)
        u.append (u_core)
        svals.append (svals_core)
        vh.append (vh_core)

    for ifrag, (fcibox, c1_r, c2_r) in enumerate (zip (las.fciboxes, kf1.ci, kf2.ci)):
        nlas, nelelas = las.ncas_sub[ifrag], las.nelecas_sub[ifrag]
        i = ncore + sum (las.ncas_sub[:ifrag])
        j = i + las.ncas_sub[ifrag]
        mo1 = kf1.mo_coeff[:,i:j]
        mo2 = kf2.mo_coeff[:,i:j]
        s1 = mo1.conj ().T @ s0 @ mo2
        u_i, svals_i, vh_i = svd (s1)
        u.append (u_i)
        svals.append (svals_i)
        vh.append (vh_i)

    if nvirt:
        mo1 = kf1.mo_coeff[:,nocc:]
        mo2 = kf2.mo_coeff[:,nocc:]
        s1 = mo1.conj ().T @ s0 @ mo2
        u_virt, svals_virt, vh_virt = svd (s1)
        u.append (u_virt)
        svals.append (svals_virt)
        vh.append (vh_virt)

    u = linalg.block_diag (*u)
    svals = np.concatenate (svals)
    vh = linalg.block_diag (*vh)

    return u, svals, vh

def count_common_orbitals (las, kf1, kf2, verbose=None):
    '''Evaluate how many orbitals in each subspace two keyframes have in common

    Args:
        las : object of :class:`LASCINoSymm`
        kf1 : object of :class:`LASKeyframe`
        kf2 : object of :class:`LASKeyframe`

    Kwargs:
        verbose: integer or None

    Returns:
        ncommon_core : int
        ncommon_active : list of length nfrags
        ncommon_virt : int
    '''
    if verbose is None: verbose=las.verbose
    nao, nmo = kf1.mo_coeff.shape    
    ncore, ncas = las.ncore, las.ncas
    nocc = ncore + ncas
    nvirt = nmo - nocc
    log = logger.new_logger (las, verbose)

    u, svals, vh = orbital_block_svd (las, kf1, kf2)

    fmt_str = '{:s} orbitals: {:d}/{:d} in common'
    def _count (lbl, i, j):
        ncommon = np.count_nonzero (np.isclose (svals[i:j], 1))
        log.info (fmt_str.format (lbl, ncommon, j-i))
        return ncommon

    ncommon_core = _count ('Inactive', 0, ncore)
    ncommon_active = []
    j_list = np.cumsum (las.ncas_sub) + ncore
    i_list = j_list - np.asarray (las.ncas_sub)
    for ifrag, (i, j) in enumerate (zip (i_list, j_list)):
        lbl = 'Active {:d}'.format (ifrag)
        ncommon_active.append (_count (lbl, i, j))
    ncommon_virt = _count ('Virtual', nocc, nmo)

    return ncommon_core, ncommon_active, ncommon_virt

def gradient_analysis (las, kf, log):
    ncore, ncas = las.ncore, las.ncas
    nocc = ncore + ncas
    gorb = kf.fock1 - kf.fock1.conj ().T
    gci = las.get_grad_ci (mo_coeff=kf.mo_coeff, ci=kf.ci, h2eff_sub=kf.h2eff_sub, veff=kf.veff)
    log.debug ('Inactive-virtual |g_orb|: %.15g', linalg.norm (gorb[:ncore,nocc:]))
    for ifrag, gc in enumerate (gci):
        i = ncore + sum (las.ncas_sub[:ifrag])
        j = i + las.ncas_sub[ifrag]
        log.debug ('Active fragment %d |g_orb|: %.15g ; |g_ci|: %.15g',
                   ifrag, linalg.norm (gorb[i:j,:]), linalg.norm (gc))
    return


# Function from failed algorithm. May have a future use.
def get_kappa (las, kf1, kf2):
    '''Decompose unitary matrix of orbital rotations between two keyframes as

      <kf1|kf2>         = exp ( kappa )               *   rmat

    | U11 U12 U13 ... |       | 0   -K'21 -K'31 ... |   | R11 0   0   ... |
    | U21 U22 U23 ... | = exp | K21 0     -K'32 ... | * | 0   R22 0   ... |
    | U31 U32 U33 ... |       | K31 K32   0     ... |   | 0   0   R33 ... |
    | ... ... ... ... |       | ... ...   ...   ... |   | ... ... ... ... |

    Where the first block is inactive orbitals, the next blocks are the active
    orbitals of individual fragments, and the final block is virtual orbitals.
    The skew-symmetric kappa matrix has zero diagonal blocks because the LASSCF
    energy is invariant to those degrees of freedom, but it is not generally
    possible to transform between any arbitrary pair of orbital bases without
    them, so instead they are factorized via repeated BCH expansions:

    kappa = lim n->infty kappa[n]
    rmat = ... @ rmat[3] @ rmat[2] @ rmat[1] 

    ovlp[0] = (kf1.mo_coeff|kf2.mo_coeff)
    log (ovlp[n-1]) = kappa[n] + log (rmat[n])
    ovlp[n] = ovlp[n-1] @ rmat[n].conj ().T

    The first-order correction to log (rmat[n]) vanishes because the commutator
    [kappa, log (rmat)] diagonal blocks are zero. So this should converge fast.
    If it doesn't, maybe try solving for rmat[n] to second order in each cycle?

    Args:
        las : object of :class:`LASCINoSymm`
        kf1 : object of :class:`LASKeyframe`
        kf2 : object of :class:`LASKeyframe`

    Returns:
        kappa : ndarray of shape (nmo, nmo)
            Skew-symmetric matrix of orbital rotation amplitudes whose lower
            triangle gives the unitary generator amplitudes for transforming
            from kf1 to kf2
        rmat : ndarray of shape (nmo, nmo)
            Block-diagonal unitary matrix. The overall unitary transformation
            to go from the orbitals of kf1 to those of kf2 is expm(kappa)@rmat
    '''
    log = logger.new_logger (las, las.verbose)

    # Work in orbital block svd basis for numerical stability
    u, svals, vh = orbital_block_svd (las, kf1, kf2)

    # Iteration parameters
    tol_strict = 1e-6
    tol_target = 1e-10
    max_cycle = 1000

    # Indexing
    nao, nmo = kf1.mo_coeff.shape
    ncore, ncas = las.ncore, las.ncas
    nocc = ncore + ncas
    nvirt = nmo - nocc
    nblk = []
    if ncore: nblk.append (ncore)
    nblk += list (las.ncas_sub)
    if nvirt: nblk.append (nvirt)
    blkoff = np.cumsum (nblk)

    # Iteration
    mo1 = kf1.mo_coeff @ u
    mo2 = kf2.mo_coeff @ vh.conj ().T
    s0 = las._scf.get_ovlp ()
    ovlp = mo1.conj ().T @ s0 @ mo2
    rmat = np.eye (nmo)
    lasterr = 1
    log.debug ('get_kappa: iterating BCH expansion until maximum diagonal element is less than %e',
               tol_target)
    for it in range (max_cycle):
        kappa = linalg.logm (ovlp @ rmat.conj ().T)
        rmat1 = np.zeros_like (kappa)
        skewerr = linalg.norm (kappa + kappa.conj ().T) 
        if (skewerr/nmo)>tol_strict:
            log.error ('get_kappa matrix logarithm failed (skewerr = %e)', skewerr)
        kappa = .5 * (kappa - kappa.conj ().T)
        diagerr = 0
        for i in range (len (nblk)):
            i1 = blkoff[i]
            i0 = i1 - nblk[i]
            diagerr = max (diagerr, np.amax (np.abs (kappa[i0:i1,i0:i1])))
            rmat1[i0:i1,i0:i1] = linalg.expm (kappa[i0:i1,i0:i1])
        log.debug ('get_kappa iter %d diagerr: %e', it, diagerr)
        if (diagerr < tol_target) or ((lasterr<tol_strict) and (diagerr>lasterr)): break
        # If you run this for infinity cycles it will always diverge. I'd like to get to
        # 1e-10 but if 1e-8 is the best it can do then it should stop there.
        lasterr = diagerr
        rmat = rmat1 @ rmat
    if diagerr > tol_strict:
        log.warn ('get_kappa iteration failed after %d cycles with err = %e',
                  it, diagerr)

    # Rollback from orbital_block_svd basis into original basis
    kappa = u @ kappa @ u.conj ().T
    rmat = u @ rmat @ vh

    # Final check
    mo1 = kf1.mo_coeff @ linalg.expm (kappa) @ rmat
    fovlp = mo1.conj ().T @ s0 @ kf2.mo_coeff
    finalerr = linalg.norm ((fovlp) - np.eye (nmo))
    log.debug ('get_kappa final error = %e', finalerr)
    assert (finalerr < tol_strict), '{}'.format (finalerr)

    return kappa, rmat

# Function from failed algorithm. May have a future use.
def democratic_matrix (las, mat, frags, mo_coeff):
    '''Weight a matrix in the "democratic DMET" way

    Args:
        las : object of :class:`LASCINoSymm`
        mat : ndarray of shape (nmo, nmo)
            In basis of mo_coeff
        frags : sequence of integers
            Identify fragments
        mo_coeff : ndarray of shape (nao, nmo)
            MO basis of mat

    Returns:
        mat : ndarray of shape (nmo, nmo)
            Diagonal environment block eliminated; off-diagonal frag-env block halved
    '''
    assert (len (frags))
    frag_orbs = []
    for ifrag in frags:
        frag_orbs.extend (las.frags_orbs[ifrag])
    frag_orbs = list (set (frag_orbs))

    s0 = las._scf.get_ovlp ()[frag_orbs,:][:,frag_orbs]
    mo = mo_coeff[frag_orbs,:]
    s1 = mo.conj ().T @ s0 @ mo
    w, u = linalg.eigh (-s1)

    mat = u.conj ().T @ mat @ u
    n = len (frag_orbs)
    mat[n:,:n] *= .5
    mat[:n,n:] *= .5
    mat[n:,n:] = 0

    return u @ mat @ u.conj ().T

# Thought I might need this; realize I don't. Might still be useful later.
def fock_cycle (las, kf1):
    '''For the inactive-virtual orbital rotations only, build and diagonalize the fock
    matrix once'''
    nao, nmo = kf1.mo_coeff.shape
    ncore, ncas = las.ncore, las.ncas
    nocc = ncore + ncas
    nvirt = nmo - nocc
    mo = np.append (kf1.mo_coeff[:,:ncore], kf1.mo_coeff[:,nocc:])
    if not mo.shape[1]: return kf1
    kf2 = kf1.copy ()
    fock = las.get_hcore ()[None,:,:] + kf1.veff
    fock = get_roothaan_fock (fock, kf1.dm1s, las._scf.get_ovlp())
    orbsym = None # TODO: symmetry
    fock = mo.conj ().T @ fock @ mo
    ene, umat = las._eig (fock, 0, 0, orbsym)
    if ncore: kf2.mo_coeff[:,:ncore] = mo @ umat[:,:ncore]
    if nvirt: kf2.mo_coeff[:,nocc:] = mo @ umat[:,ncore:]
    return kf2


