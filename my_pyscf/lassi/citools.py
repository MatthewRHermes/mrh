import numpy as np
import functools
from pyscf.scf.addons import canonical_orth_
from pyscf import __config__

LINDEP_THRESH = getattr (__config__, 'lassi_lindep_thresh', 1.0e-5)

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

def get_orth_basis (ci_fr, norb_f, nelec_frs, _get_ovlp=None):
    if _get_ovlp is None:
        from mrh.my_pyscf.lassi.op_o0 import get_ovlp
        _get_ovlp = functools.partial (get_ovlp, ci_fr, norb_f, nelec_frs)
    nfrags, nroots = nelec_frs.shape[:2]
    unique, uniq_idx, inverse, cnts = np.unique (nelec_frs, axis=1, return_index=True,
                                                 return_inverse=True, return_counts=True)
    if not np.count_nonzero (cnts>1): 
        _get_ovlp = None
        def raw2orth (rawarr):
            return rawarr
        def orth2raw (ortharr):
            return ortharr
        return raw2orth, orth2raw
    lroots_fr = np.array ([[1 if c.ndim<3 else c.shape[0]
                            for c in ci_r]
                           for ci_r in ci_fr])
    nprods_r = np.prod (lroots_fr, axis=0)
    offs1 = np.cumsum (nprods_r)
    offs0 = offs1 - nprods_r
    uniq_prod_idx = []
    for i in uniq_idx[cnts==1]: uniq_prod_idx.extend (list(range(offs0[i],offs1[i])))
    manifolds_prod_idx = []
    manifolds_xmat = []
    nuniq_prod = north = len (uniq_prod_idx)
    for manifold_idx in np.where (cnts>1)[0]:
        manifold = np.where (inverse==manifold_idx)[0]
        manifold_prod_idx = []
        for i in manifold: manifold_prod_idx.extend (list(range(offs0[i],offs1[i])))
        manifolds_prod_idx.append (manifold_prod_idx)
        ovlp = _get_ovlp (rootidx=manifold)
        xmat = canonical_orth_(ovlp, thr=LINDEP_THRESH)
        north += xmat.shape[1]
        manifolds_xmat.append (xmat)

    _get_ovlp = None

    nraw = offs1[-1]
    def raw2orth (rawarr):
        col_shape = rawarr.shape[1:]
        orth_shape = [north,] + list (col_shape)
        ortharr = np.zeros (orth_shape, dtype=rawarr.dtype)
        ortharr[:nuniq_prod] = rawarr[uniq_prod_idx]
        i = nuniq_prod
        for prod_idx, xmat in zip (manifolds_prod_idx, manifolds_xmat):
            j = i + xmat.shape[1]
            ortharr[i:j] = np.tensordot (xmat.T, rawarr[prod_idx], axes=1)
            i = j
        return ortharr

    def orth2raw (ortharr):
        col_shape = ortharr.shape[1:]
        raw_shape = [nraw,] + list (col_shape)
        rawarr = np.zeros (raw_shape, dtype=ortharr.dtype)
        rawarr[uniq_prod_idx] = ortharr[:nuniq_prod]
        i = nuniq_prod
        for prod_idx, xmat in zip (manifolds_prod_idx, manifolds_xmat):
            j = i + xmat.shape[1]
            rawarr[prod_idx] = np.tensordot (xmat.conj (), ortharr[i:j], axes=1)
            i = j
        return rawarr

    return raw2orth, orth2raw

def _fake_gen_contract_op_si_hdiag (matrix_builder, las, h1, h2, ci_fr, nelec_frs, soc=0, orbsym=None, wfnsym=None):
    ham, s2, ovlp, raw2orth, orth2raw = matrix_builder (las, h1, h2, ci_fr, nelec_frs, soc=soc,
                                                        orbsym=orbsym, wfnsym=wfnsym)
    def contract_ham_si (x):
        return ham @ x
    def contract_s2 (x):
        return s2 @ x
    def contract_ovlp (x):
        return ovlp @ x
    hdiag = np.diagonal (ham)
    return contract_ham_si, contract_s2, contract_ovlp, hdiag, raw2orth, orth2raw


