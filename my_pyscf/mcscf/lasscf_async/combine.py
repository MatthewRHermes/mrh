import os
import numpy as np
from scipy import linalg
from pyscf import lib
from pyscf.lo import orth
from pyscf.scf.rohf import get_roothaan_fock
from mrh.my_pyscf.mcscf import lasci, _DFLASCI
from mrh.my_pyscf.mcscf.lasscf_async import keyframe

# TODO: symmetry
def orth_orb (las, kf2_list, kf_ref=None):
    ncore, ncas = las.ncore, las.ncas
    nocc = ncore + ncas
    nao, nmo = las.mo_coeff.shape
    nfrags = las.nfrags
    log = lib.logger.new_logger (las, las.verbose)

    # orthonormalize active orbitals
    if kf_ref is not None:
        ci = [c for c in kf_ref.ci]
        mo_cas = kf_ref.mo_coeff[:,ncore:nocc].copy ()
    else:
        ci = [None for i in range (las.nfrags)]
        mo_cas = np.empty ((nao, ncas), dtype=las.mo_coeff.dtype)
    for kf2 in kf2_list:
        for ifrag in kf2.frags:
            i = sum (las.ncas_sub[:ifrag])
            j = i + las.ncas_sub[ifrag]
            k, l = i + ncore, j + ncore
            mo_cas[:,i:j] = kf2.mo_coeff[:,k:l]
            ci[ifrag] = kf2.ci[ifrag]
    mo_cas_preorth = mo_cas.copy ()
    s0 = las._scf.get_ovlp ()
    mo_cas = orth.vec_lowdin (mo_cas_preorth, s=s0)
    
    # reassign orthonormalized active orbitals
    proj = []
    for ifrag in range (nfrags):
        i = sum (las.ncas_sub[:ifrag])
        j = i + las.ncas_sub[ifrag]
        proj.append (mo_cas_preorth[:,i:j] @ mo_cas_preorth[:,i:j].conj ().T)
    smo1 = s0 @ mo_cas
    frag_weights = np.stack ([((p @ smo1) * smo1.conjugate ()).sum (0)
                              for p in proj], axis=-1)
    idx = np.argsort (frag_weights, axis=1)[:,-1]
    mo_las = []
    for ifrag in range (nfrags):
        mo = mo_cas[:,(idx == ifrag)]
        i = sum (las.ncas_sub[:ifrag])
        j = i + las.ncas_sub[ifrag]
        s1 = mo.conj ().T @ s0 @ mo_cas_preorth[:,i:j]
        u, svals, vh = linalg.svd (s1)
        mo_las.append (mo @ u @ vh)
    mo_cas = np.concatenate (mo_las, axis=1)
    
    # non-active orbitals
    ucas = las.mo_coeff.conj ().T @ s0 @ mo_cas
    u, R = linalg.qr (ucas)
    # Isn't it weird that you do Gram-Schmidt by doing QR?
    errmax = np.amax (np.abs (np.abs (R[:ncas,:ncas]) - np.eye (ncas)))
    if errmax>1e-8:
        log.warn ('Active orbital orthogonalization may have failed: %e', errmax)
    mo1 = las.mo_coeff @ u
    errmax = np.amax (np.abs (np.abs (mo_cas.conj ().T @ s0 @ mo1[:,:ncas]) - np.eye (ncas)))
    if errmax>1e-8:
        log.warn ('Active orbitals leaking into non-active space: %e', errmax)
    errmax = np.amax (np.abs ((mo1.conj ().T @ s0 @ mo1) - np.eye (mo1.shape[1])))
    if errmax>1e-8:
        log.warn ('Non-orthogonal AOs in lasscf_async.combine.orth_orb: %e', errmax)
    mo1 = mo1[:,ncas:]
    if mo1.size:
        veff = sum ([kf2.veff for kf2 in kf2_list]) / len (kf2_list)
        dm1s = sum ([kf2.dm1s for dm1s in kf2_list]) / len (kf2_list)
        fock = las.get_hcore ()[None,:,:] + veff
        fock = get_roothaan_fock (fock, dm1s, s0)
        orbsym = None # TODO: symmetry
        fock = mo1.conj ().T @ fock @ mo1
        ene, umat = las._eig (fock, 0, 0, orbsym)
        mo_core = mo1 @ umat[:,:ncore]
        mo_virt = mo1 @ umat[:,ncore:]
        mo_coeff = np.concatenate ([mo_core, mo_cas, mo_virt], axis=1)
    else:
        mo_coeff = mo_cas

    return las.get_keyframe (mo_coeff, ci)

class flas_stdout_env (object):
    def __init__(self, las, flas_stdout):
        self.las = las
        self.flas_stdout = flas_stdout
        self.las_stdout = las.stdout
    def __enter__(self):
        self.las.stdout = self.flas_stdout
        self.las._scf.stdout = self.flas_stdout
        self.las.fcisolver.stdout = self.flas_stdout
        for fcibox in self.las.fciboxes:
            fcibox.stdout = self.flas_stdout
            for fcisolver in fcibox.fcisolvers:
                fcisolver.stdout = self.flas_stdout
        if getattr (self.las, 'with_df', None):
            self.las.with_df.stdout = self.flas_stdout
    def __exit__(self, type, value, traceback):
        self.las.stdout = self.las_stdout
        self.las._scf.stdout = self.las_stdout
        self.las.fcisolver.stdout = self.las_stdout
        for fcibox in self.las.fciboxes:
            fcibox.stdout = self.las_stdout
            for fcisolver in fcibox.fcisolvers:
                fcisolver.stdout = self.las_stdout
        if getattr (self.las, 'with_df', None):
            self.las.with_df.stdout = self.las_stdout

def relax (las, kf, freeze_inactive=False, unfrozen_frags=None):
    if unfrozen_frags is None: frozen_frags = []
    else:
        frozen_frags = [i for i in range (las.nfrags) if i not in unfrozen_frags]
    log = lib.logger.new_logger (las, las.verbose)
    flas_stdout = getattr (las, '_flas_stdout', None)
    if flas_stdout is None:
        output = getattr (las.mol, 'output', None)
        if not ((output is None) or (output=='/dev/null')):
            flas_output = output + '.flas'
            if las.verbose > lib.logger.QUIET:
                if os.path.isfile (flas_output):
                    print('overwrite output file: %s' % flas_output)
                else:
                    print('output file: %s' % flas_output)
            flas_stdout = open (flas_output, 'w')
            las._flas_stdout = flas_stdout
        else:
            flas_stdout = las.stdout
    with flas_stdout_env (las, flas_stdout):
        flas = lasci.LASCI (las._scf, las.ncas_sub, las.nelecas_sub)
        flas.__dict__.update (las.__dict__)
        flas.frozen = []
        flas.frozen_ci = frozen_frags
        if freeze_inactive:
            flas.frozen.extend (list (range (las.ncore)))
        for ifrag in frozen_frags:
            i0 = las.ncore + sum (las.ncas_sub[:ifrag])
            i1 = i0 + las.ncas_sub[ifrag]
            flas.frozen.extend (list (range (i0,i1)))
        if freeze_inactive:
            nocc = las.ncore + las.ncas
            nmo = kf.mo_coeff.shape[1]
            flas.frozen.extend (list (range (nocc,nmo)))
        # Default: scale down conv_tol_grad according to size of subproblem
        scale = np.sqrt (flas.get_ugg ().nvar_tot / las.get_ugg ().nvar_tot)
        flas.conv_tol_grad = scale * las.conv_tol_grad
        flas.min_cycle_macro = 1
        params = getattr (las, 'relax_params', {})
        glob = {key: val for key, val in params.items () if isinstance (key, str)}
        glob = {key: val for key, val in glob.items () if key not in ('frozen', 'frozen_ci')}
        flas.__dict__.update (glob)
        if unfrozen_frags is not None:
            loc = params.get (tuple (unfrozen_frags), {})
            loc = {key: val for key, val in loc.items () if key not in ('frozen', 'frozen_ci')}
            flas.__dict__.update (loc)
        e_tot, e_cas, ci, mo_coeff, mo_energy, h2eff_sub, veff = \
            flas.kernel (kf.mo_coeff, ci0=kf.ci)
    ovlp = mo_coeff.conj ().T @ las._scf.get_ovlp () @ mo_coeff
    errmat = ovlp - np.eye (ovlp.shape[0])
    errmax = np.amax (np.abs (errmat))
    if errmax>1e-8:
        log.warn ('Non-orthogonal AOs in lasscf_async.combine.relax: max ovlp error = %e', errmax)
    return las.get_keyframe (mo_coeff, ci)

def combine_o0 (las, kf2_list):
    kf1 = orth_orb (las, kf2_list)
    kf1 = relax (las, kf1)
    return kf1

def select_aa_block (las, frags1, frags2, fock1):
    '''Identify from two lists of candidate fragments the single active-active orbital-rotation
    gradient block with the largest norm

    Args:
        las : object of :class:`LASCINoSymm`
        frags1 : sequence of integers
        frags2 : sequence of integers
        fock1 : ndarray of shape (nmo,nmo)

    Returns:
        i : integer
            From frags1.
        j : integer
            From frags2.
'''
    frags1 = list (frags1)
    frags2 = list (frags2)
    g_orb = fock1 - fock1.conj ().T
    ncore = las.ncore
    nocc = ncore + las.ncas
    g_orb = g_orb[ncore:nocc,ncore:nocc]
    gblk = []
    for i in frags1:
        i0 = sum (las.ncas_sub[:i])
        i1 = i0 + las.ncas_sub[i]
        for j in frags2:
            j0 = sum (las.ncas_sub[:j])
            j1 = j0 + las.ncas_sub[j]
            gblk.append (linalg.norm (g_orb[i0:i1,j0:j1]))
    gmax = np.argmax (gblk)
    i = frags1[gmax // len (frags2)]
    j = frags2[gmax % len (frags2)]
    return i, j

def combine_pair (las, kf1, kf2, kf_ref=None):
    '''Combine two keyframes and relax one specific block of active-active orbital rotations
    between the fragments assigned to each with the inactive and virtual orbitals frozen.'''
    if kf_ref is None: kf_ref=kf1
    if len (kf1.frags.intersection (kf2.frags)):
        errstr = ("Cannot combine keyframes that are responsible for the same fragments "
                  "({} {})").format (kf1.frags, kf2.frags)
        raise RuntimeError (errstr)
    kf3 = orth_orb (las, [kf1, kf2], kf_ref=kf_ref)
    i, j = select_aa_block (las, kf1.frags, kf2.frags, kf3.fock1)
    kf3 = relax (las, kf3, freeze_inactive=True, unfrozen_frags=(i,j))
    kf3.frags = kf1.frags.union (kf2.frags)
    return kf3

# Function from failed algorithm. Retained for reference
def combine_o1_kappa_rigid (las, kf1, kf2, kf_ref):
    '''Combine two keyframes (without relaxing the active orbitals) by weighting the kappa matrices
    with respect to a third reference keyframe democratically

    Args:
        las : object of :class:`LASCINoSymm`
        kf1 : object of :class:`LASKeyframe`
        kf2 : object of :class:`LASKeyframe`
        kf_ref : object of :class:`LASKeyframe`
            Reference point for the kappa matrices

    Returns:
        kf3 : object of :class:`LASKeyframe`
    '''
    log = lib.logger.new_logger (las, las.verbose)
    nmo = las.mo_coeff.shape[1]
    kf3 = kf_ref.copy ()
    kappa1, rmat1 = keyframe.get_kappa (las, kf1, kf_ref)
    kappa2, rmat2 = keyframe.get_kappa (las, kf2, kf_ref)
    kappa1 = keyframe.democratic_matrix (las, kappa1, kf1.frags, kf_ref.mo_coeff)
    kappa2 = keyframe.democratic_matrix (las, kappa2, kf2.frags, kf_ref.mo_coeff)
    kappa = kappa1 + kappa2
    rmat = np.eye (nmo) + np.zeros_like (rmat1) + np.zeros_like (rmat2) # complex safety

    offs = np.cumsum (las.ncas_sub) + las.ncore
    for i in kf1.frags:
        i1 = offs[i]
        i0 = i1 - las.ncas_sub[i]
        kf3.ci[i] = kf1.ci[i]
        rmat[i0:i1,i0:i1] = rmat1[i0:i1,i0:i1]
    for i in kf2.frags:
        i1 = offs[i]
        i0 = i1 - las.ncas_sub[i]
        kf3.ci[i] = kf2.ci[i]
        rmat[i0:i1,i0:i1] = rmat2[i0:i1,i0:i1]

    # set orbitals and frag associations
    umat = linalg.expm (kappa) @ rmat
    if np.iscomplexobj (umat):
        log.warn ('Complex umat constructed. Discarding imaginary part; norm: %e',
                  linalg.norm (umat.imag))
        umat = umat.real
    kf3.mo_coeff = kf_ref.mo_coeff @ umat
    kf3.frags = kf1.frags.union (kf2.frags)
    
    return kf3



    
