import numpy as np
from pyscf.lib import logger
from scipy import linalg

class LASKeyframe (object):
    '''Shallow struct for various intermediates. DON'T put complicated code in here Matt!!!'''

    def __init__(self, las, mo_coeff, ci):
        self.las = las
        self.mo_coeff = mo_coeff
        self.ci = ci
        self._dm1s = self._veff = self._fock1 = self._h1eff_sub = self._h2eff_sub = None

    @property
    def dm1s (self):
        if self._dm1s is None:
            self._dm1s = self.las.make_rdm1s (mo_coeff=self.mo_coeff, ci=self.ci)
        return self._dm1s

    @property
    def veff (self):
        if self._veff is None:
            self._veff = self.las.get_veff (dm1s=self.dm1s, spin_sep=True)
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
    nao, nmo = kf1.mo_coeff.shape    
    ncore, ncas = las.ncore, las.ncas
    nocc = ncore + ncas
    nvirt = nmo - nocc

    s0 = las._scf.get_ovlp ()
    mo1 = kf1.mo_coeff[:,:ncore]
    mo2 = kf2.mo_coeff[:,:ncore]
    s1 = mo1.conj ().T @ s0 @ mo2
    u_core, svals_core, vh_core = linalg.svd (s1)

    u = [u_core,]
    svals = [svals_core,]
    vh = [vh_core,]
    for ifrag, (fcibox, c1_r, c2_r) in enumerate (zip (las.fciboxes, kf1.ci, kf2.ci)):
        nlas, nelelas = las.ncas_sub[ifrag], las.nelecas_sub[ifrag]
        i = ncore + sum (las.ncas_sub[:ifrag])
        j = i + las.ncas_sub[ifrag]
        mo1 = kf1.mo_coeff[:,i:j]
        mo2 = kf2.mo_coeff[:,i:j]
        s1 = mo1.conj ().T @ s0 @ mo2
        u_i, svals_i, vh_i = linalg.svd (s1)
        u.append (u_i)
        svals.append (svals_i)
        vh.append (vh_i)

    mo1 = kf1.mo_coeff[:,nocc:]
    mo2 = kf2.mo_coeff[:,nocc:]
    s1 = mo1.conj ().T @ s0 @ mo2
    u_virt, svals_virt, vh_virt = linalg.svd (s1)
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
    ncore, ncas = las.ncore, las.ncas
    nocc = ncore + ncas
    nvirt = nmo - nocc
    log = logger.new_logger (las, verbose)

    u, svals, vh = orbital_block_svd (las, kf1, kf2)

    fmt_str = '{:s} orbitals: {:d}/{:d} in common'
    def _count (lbl, i, j):
        ncommon = np.count_nonzero (np.isclose (svals[i:j], 1))
        log.info (fmt_string.format (lbl, ncommon, j-i))
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










