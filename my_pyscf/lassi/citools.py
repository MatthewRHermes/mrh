import numpy as np
import functools
from scipy.sparse import linalg as sparse_linalg
from scipy import linalg
from pyscf.scf.addons import canonical_orth_
from pyscf import __config__
from itertools import combinations

LINDEP_THRESH = getattr (__config__, 'lassi_lindep_thresh', 1.0e-5)
SCREEN_THRESH = getattr (__config__, 'lassi_frag_screen_thresh', 1e-10)

def get_lroots (ci):
    '''Generate a table showing the number of states contained in a (optionally nested) list
    of CI vectors.

    Args:
        ci: list (of list of) ndarray of shape (ndeta,ndetb) or (nroots,ndeta,ndetb)
            CI vectors. The number of roots in each entry is 1 for each 1-d or 2d-ndarray and
            shape[0] for each 3d- or higher ndarray. The nesting depth equals the number of
            dimensions of the returned lroots array. An exception is raised if the sequence
            is ragged (e.g., if ci[i] is a list but ci[j] is an ndarray, or likewise at any
            depth).

    Returns:
        lroots: ndarray of integers
            Number of states stored as CI vectors in each individual ndarray within ci. The shape
            is given by the length and nesting depth of the ci lists (so a single list of ndarrays
            results in a 1d lroots; a list of lists of ndarrays results in a 2d lroots, etc.)
    '''
    lroots = []
    didrecurse = False
    dideval = False
    raggedarray = RuntimeError ("ragged argument to get_lroots")
    for c in ci:
        if isinstance (c, np.ndarray):
            if didrecurse: raise raggedarray
            dideval = True
            lroots.append (1 if c.ndim<3 else c.shape[0])
        else:
            if dideval: raise raggedarray
            didrecurse = True
            lroots.append (get_lroots (c))
    return np.asarray (lroots)

def get_rootaddr_fragaddr (lroots):
    '''Generate an index array into a compressed fragment basis for a state in the LASSI model
    space

    Args:
        lroots: ndarray of shape (nfrags, nroots)
            Number of local roots in each fragment in each rootspace

    Returns:
        rootaddr: ndarray of shape (nstates,)
            The rootspace associated with each state
        fragaddr: ndarray of shape (nfrags, nstates)
            The ordinal designation local to each fragment of each LAS state.
    '''
    nfrags, nroots = lroots.shape
    nprods = np.prod (lroots, axis=0)
    fragaddr = np.zeros ((nfrags, sum(nprods)), dtype=int)
    rootaddr = np.zeros (sum(nprods), dtype=int)
    offs = np.cumsum (nprods)
    for iroot in range (nroots):
        j = offs[iroot]
        i = j - nprods[iroot]
        for ifrag in range (nfrags):
            prods_before = np.prod (lroots[:ifrag,iroot], axis=0)
            prods_after = np.prod (lroots[ifrag+1:,iroot], axis=0)
            addrs = np.repeat (np.arange (lroots[ifrag,iroot]), prods_before)
            addrs = np.tile (addrs, prods_after)
            fragaddr[ifrag,i:j] = addrs
        rootaddr[i:j] = iroot
    return rootaddr, fragaddr

def umat_dot_1frag_(target, umat, lroots, ifrag, iroot, axis=0):
    '''Apply a unitary transformation for 1 fragment in 1 rootspace to a tensor
    whose target axis spans all model states.

    Args:
        target: ndarray whose length on axis 'axis' is nstates
            The object to which the unitary transformation is to be applied.
            Modified in-place.
        umat: ndarray of shape (lroots[ifrag,iroot],lroots[ifrag,iroot])
            A unitary matrix; the row axis is contracted
        lroots: ndarray of shape (nfrags, nroots)
            Number of basis states in each fragment in each rootspace
        ifrag: integer
            Fragment index for the targeted block
        iroot: integer
            Rootspace index for the targeted block

    Kwargs:
        axis: integer
            The axis of target to which umat is to be applied

    Returns:
        target: same as input target
            After application of unitary transformation'''
    nprods = np.prod (lroots, axis=0)
    offs = [0,] + list (np.cumsum (nprods))
    i, j = offs[iroot], offs[iroot+1]
    newaxes = [axis,] + list (range (axis)) + list (range (axis+1, target.ndim))
    oldaxes = list (np.argsort (newaxes))
    target = target.transpose (*newaxes)
    target[i:j] = _umat_dot_1frag (target[i:j], umat, lroots[:,iroot], ifrag)
    target = target.transpose (*oldaxes)
    return target

def _umat_dot_1frag (target, umat, lroots, ifrag):
    # Remember: COLUMN-MAJOR ORDER!!
    iifrag = len (lroots) - ifrag - 1
    old_shape1 = target.shape
    new_shape = lroots[::-1]
    nrow = np.prod (new_shape[:iifrag]).astype (int)
    ncol1 = lroots[ifrag]
    assert (ncol1==umat.shape[0])
    ncol2 = umat.shape[1]
    nstack = (np.prod (new_shape[iifrag:]) * np.prod (old_shape1[1:])).astype (int) // ncol1
    new_shape = (nrow, ncol1, nstack)
    target = target.reshape (*new_shape).transpose (1,0,2)
    target = np.tensordot (umat.T, target, axes=1).transpose (1,0,2)
    old_shape2 = list (old_shape1)
    old_shape2[0] = old_shape2[0] * ncol2 // ncol1
    return target.reshape (*old_shape2)

def get_orth_basis (ci_fr, norb_f, nelec_frs, _get_ovlp=None, smult_fr=None):
    '''Unitary matrix for an orthonormal product-state basis from a set of CI vectors.

    Args:
        ci_fr: list of length nfrags of list of length nroots of ndarrays
            CI vectors of fragment statelets
        norb_f: ndarray of shape (nfrags,) of int
            Number of orbitals in each fragment
        nelec_frs: ndarray of shape (nfrags,nroots,2) of int
            Number of electrons in each fragment in each rootspace

    Kwargs:
        _get_ovlp: callable with kwarg rootidx
            Produce the overlap matrix between model states in a set of rootspaces,
            identified by ndarray or list "rootidx"

    Returns:
        raw2orth: LinearOperator of shape (north, nraw)
            Apply to SI vector to transform from the primitive to the orthonormal basis
    '''
    if _get_ovlp is None:
        from mrh.my_pyscf.lassi.op_o0 import get_ovlp
        _get_ovlp = functools.partial (get_ovlp, ci_fr, norb_f, nelec_frs)
    nfrags, nroots = nelec_frs.shape[:2]
    tabulator = nelec_frs
    if smult_fr is not None:
        tabulator = np.append (tabulator, smult_fr[:,:,None], axis=2)
    unique, uniq_idx, inverse, cnts = np.unique (tabulator, axis=1, return_index=True,
                                                 return_inverse=True, return_counts=True)
    lroots_fr = np.array ([[1 if c.ndim<3 else c.shape[0]
                            for c in ci_r]
                           for ci_r in ci_fr])
    nprods_r = np.prod (lroots_fr, axis=0)
    offs1 = np.cumsum (nprods_r)
    offs0 = offs1 - nprods_r
    nraw = offs1[-1]
    is_complex = any ([any ([np.iscomplexobj (c) for c in ci_r]) for ci_r in ci_fr])
    dtype = np.complex128 if is_complex else np.float64 

    if not np.count_nonzero (cnts>1): 
        _get_ovlp = None
        return sparse_linalg.LinearOperator (shape=(nraw,nraw), dtype=dtype,
                                             matvec=lambda x: x,
                                             rmatvec=lambda x: x)
    uniq_prod_idx = []
    for i in uniq_idx[cnts==1]: uniq_prod_idx.extend (list(range(offs0[i],offs1[i])))
    manifolds_prod_idx = []
    manifolds_xmat = []
    nuniq_prod = north = len (uniq_prod_idx)
    for manifold_idx in np.where (cnts>1)[0]:
        manifold = np.where (inverse==manifold_idx)[0]
        manifold_prod_idx = []
        for i in manifold: manifold_prod_idx.extend (list(range(offs0[i],offs1[i])))
        ovlp = _get_ovlp (rootidx=manifold)
        ovlp[np.diag_indices_from (ovlp)] -= 1.0
        err_from_diag = np.amax (np.abs (ovlp))
        if err_from_diag > 1e-8:
            ovlp[np.diag_indices_from (ovlp)] += 1.0
            manifolds_prod_idx.append (manifold_prod_idx)
            xmat = canonical_orth_(ovlp, thr=LINDEP_THRESH)
            north += xmat.shape[1]
            manifolds_xmat.append (xmat)
        else:
            north += ovlp.shape[0]
            nuniq_prod += ovlp.shape[0]
            uniq_prod_idx.extend (manifold_prod_idx)
        ovlp = None

    _get_ovlp = None

    nraw = offs1[-1]
    def raw2orth (rawarr):
        is_out_complex = is_complex or np.iscomplexobj (rawarr)
        my_dtype = np.complex128 if is_out_complex else np.float64
        col_shape = rawarr.shape[1:]
        orth_shape = [north,] + list (col_shape)
        ortharr = np.zeros (orth_shape, dtype=my_dtype)
        ortharr[:nuniq_prod] = rawarr[uniq_prod_idx]
        i = nuniq_prod
        for prod_idx, xmat in zip (manifolds_prod_idx, manifolds_xmat):
            j = i + xmat.shape[1]
            ortharr[i:j] = np.tensordot (xmat.T, rawarr[prod_idx], axes=1)
            i = j
        return ortharr

    def orth2raw (ortharr):
        is_out_complex = is_complex or np.iscomplexobj (ortharr)
        my_dtype = np.complex128 if is_out_complex else np.float64
        col_shape = ortharr.shape[1:]
        raw_shape = [nraw,] + list (col_shape)
        rawarr = np.zeros (raw_shape, dtype=my_dtype)
        rawarr[uniq_prod_idx] = ortharr[:nuniq_prod]
        i = nuniq_prod
        for prod_idx, xmat in zip (manifolds_prod_idx, manifolds_xmat):
            j = i + xmat.shape[1]
            rawarr[prod_idx] = np.tensordot (xmat.conj (), ortharr[i:j], axes=1)
            i = j
        return rawarr

    return sparse_linalg.LinearOperator (shape=(north,nraw), dtype=dtype,
                                         matvec=raw2orth,
                                         rmatvec=orth2raw)

def get_unique_roots (ci, nelec_r, screen_linequiv=True, screen_thresh=SCREEN_THRESH,
                      discriminator=None):
    '''Identify which groups of CI vectors are equal or equivalent from a list.

    Args:
        ci: list of length nroots of ndarray
            CI vectors
        nelec_r: list of length nroots of tuple of length 2 of int
            Numbers of electrons in each group of CI vectors

    Kwargs:
        screen_linequiv: logical
            If False, two groups of CI vectors are only equivalent if they share
            memory. Otherwise, they are equivalent if they span the same space.
        screen_thresh: float
            epsilon for linear equivalence between CI vector spaces
            meaningless if screen_linequiv==False
        discriminator: sequence of length nroots
            Tags to forcibly discriminate otherwise equivalent rootspaces

    Returns:
        root_unique: ndarray of logical of length nroots
            Whether a given set of CI vectors is unique
        unique_root: ndarray of ints length nroots
            The index of the unique image of each set of CI vectors in the list
        umat_root: dict
            Unitary vectors for transforming a nonunique set of CI vectors into
            its unique image
    '''
    nroots = len (ci)
    if discriminator is None: discriminator = np.zeros (nroots, dtype=int)
    lroots = get_lroots (ci)
    root_unique = np.ones (nroots, dtype=bool)
    unique_root = np.arange (nroots, dtype=int)
    umat_root = {}
    for i, j in combinations (range (nroots), 2):
        if not root_unique[i]: continue
        if not root_unique[j]: continue
        if nelec_r[i] != nelec_r[j]: continue
        if lroots[i] != lroots[j]: continue
        if ci[i].shape != ci[j].shape: continue
        if discriminator[i] != discriminator[j]: continue
        isequal = False
        if ci[i] is ci[j]: isequal = True
        else:
            try:
                isequal = np.shares_memory (ci[i], ci[j], max_work=1000)
            except np.exceptions.TooHardError:
                isequal = False
        if (not isequal) and screen_linequiv:
            if np.all (ci[i]==ci[j]): isequal = True
            elif np.all (np.abs (ci[i]-ci[j]) < screen_thresh): isequal=True
            else:
                ci_i = ci[i].reshape (lroots[i],-1)
                ci_j = ci[j].reshape (lroots[j],-1)
                ovlp = ci_i.conj () @ ci_j.T
                isequal = np.all (np.abs (ovlp - np.eye (lroots[i])) < screen_thresh)
                                  # need extremely high precision on this one
                if not isequal:
                    err1 = abs ((np.trace (ovlp @ ovlp.conj ().T) / lroots[i]) - 1.0)
                    err2 = abs ((np.trace (ovlp.conj ().T @ ovlp) / lroots[i]) - 1.0)
                    isequal = (err1 < screen_thresh) and (err2 < screen_thresh)
                    if isequal:
                        u, svals, vh = linalg.svd (ovlp)
                        assert (len (svals) == lroots[i])
                        umat_root[j] = u @ vh
        if isequal:
            root_unique[j] = False
            unique_root[j] = i
    for i in range (nroots):
        assert (root_unique[unique_root[i]])
    return root_unique, unique_root, umat_root

def _fake_gen_contract_op_si_hdiag (matrix_builder, las, h1, h2, ci_fr, nelec_frs, soc=0,
                                    orbsym=None, wfnsym=None, **kwargs):
    ham, s2, ovlp, _get_ovlp = matrix_builder (las, h1, h2, ci_fr, nelec_frs, soc=soc,
                                               orbsym=orbsym, wfnsym=wfnsym, **kwargs)
    def contract_ham_si (x):
        return ham @ x
    def contract_s2 (x):
        return s2 @ x
    def contract_ovlp (x):
        return ovlp @ x
    hdiag = np.diagonal (ham)
    return contract_ham_si, contract_s2, contract_ovlp, hdiag, _get_ovlp

def hci_dot_sivecs (hci_fr_pabq, si_bra, si_ket, lroots):
    nfrags, nroots = lroots.shape
    for i, hci_r_pabq in enumerate (hci_fr_pabq):
        for j, hci_pabq in enumerate (hci_r_pabq):
            hci_fr_pabq[i][j] = hci_dot_sivecs_ij (hci_pabq, si_bra, si_ket, lroots, i, j)
    return hci_fr_pabq

def hci_dot_sivecs_ij (hci_pabq, si_bra, si_ket, lroots, i, j):
    nprods = np.prod (lroots, axis=0)
    j1 = np.cumsum (nprods)
    j0 = j1 - nprods
    if si_ket is not None:
        hci_pabq = np.tensordot (hci_pabq, si_ket, axes=1)
    if si_bra is not None:
        from mrh.my_pyscf.lassi.op_o1.utilities import transpose_sivec_make_fragments_slow
        is1d = (si_bra.ndim==1)
        c = np.asfortranarray (si_bra[j0[j]:j1[j]])
        c = transpose_sivec_make_fragments_slow (c, lroots[:,j], i)
        hci_pabq = np.tensordot (c.conj (), hci_pabq, axes=1)
        if si_ket is not None:
            if si_ket.ndim == 1:
                assert (hci_pabq.shape[0] == 1)
                hci_pabq = hci_pabq[0]
            else:
                assert (hci_pabq.shape[0] == hci_pabq.shape[4])
                hci_pabq = np.diagonal (hci_pabq, axis1=0, axis2=4)
                hci_pabq = hci_pabq.transpose (3,0,1,2).copy ()
                if is1d: hci_pabq = hci_pabq[0]
    return hci_pabq


