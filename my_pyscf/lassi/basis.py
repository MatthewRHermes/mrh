import sys
import numpy as np
import functools
from scipy.sparse import linalg as sparse_linalg
from scipy import linalg
from pyscf.scf.addons import canonical_orth_
from pyscf import __config__, lib
from mrh.my_pyscf.lassi.citools import get_unique_roots_with_spin

LINDEP_THRESH = getattr (__config__, 'lassi_lindep_thresh', 1.0e-5)

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
    tabulator = nelec_frs.sum (2)
    if smult_fr is not None:
        tabulator = np.append (tabulator, smult_fr, axis=0)
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
        return NullOrthBasis (nraw, dtype, nprods_r)
    north = 0
    manifolds = []
    manifolds_xmat = []
    manifolds_roots = []
    # iterate over smult & nelec strings
    for sn_string_idx, sn_str in enumerate (unique.T):
        n_str = sn_str[:nfrags]
        s_str = sn_str[nfrags:]
        pm_blocks, pm_strs = _get_spin_split_manifolds (ci_fr, norb_f, nelec_frs, smult_fr,
                                                        lroots_fr, inverse==sn_string_idx)
        # iterate over spatial wave functions. I think that the length of this iteration
        # should be 1 if the model space is really spin-adapted; but in this function,
        # I don't want to require that
        for m_blocks, m_strs in zip (pm_blocks, pm_strs):
            num_m_blocks = len (m_blocks)
            # iterate over m strings, but only to sanity check
            nprod = np.asarray ([(offs1[m]-offs0[m]).sum () for m in m_blocks])
            assert (np.all (nprod==nprod[0]))
            nprod = nprod[0]
            ovlp = _get_ovlp (rootidx=m_blocks[0])
            ovlp[np.diag_indices_from (ovlp)] -= 1.0
            err_from_diag = np.amax (np.abs (ovlp))
            if err_from_diag > 1e-8:
                ovlp[np.diag_indices_from (ovlp)] += 1.0
                xmat = canonical_orth_(ovlp, thr=LINDEP_THRESH)
                north += xmat.shape[1] * num_m_blocks
            else:
                north += ovlp.shape[0] * num_m_blocks
                xmat = None
            manifolds.append (RootspaceManifold (n_str, s_str, m_strs, m_blocks, xmat))
            ovlp = None

    _get_ovlp = None

    return OrthBasis ((north,nraw), dtype, nprods_r, manifolds)

class RootspaceManifold:
    def __init__(self, n_str, s_str, m_strs, m_blocks, xmat):
        self.n_str = n_str
        self.s_str = s_str
        self.m_strs = m_strs
        self.m_blocks = np.asarray (m_blocks, dtype=int)
        self.xmat = xmat

def _get_spin_split_manifolds (ci_fr, norb_f, nelec_frs, smult_fr, lroots_fr, idx):
    '''The same as _get_spin_split_manifolds_idx, except that all of the arguments need to be
    indexed down from the full model space into the input manifold via "idx" first. The returned
    submanifold list is likewise indexed back into the full model space.'''
    nelec_frs = nelec_frs[:,idx,:]
    if smult_fr is not None: smult_fr = smult_fr[:,idx]
    lroots_fr = lroots_fr[:,idx]
    idx = np.where (idx)[0]
    ci1_fr = [[ci_r[i] for i in idx] for ci_r in ci_fr]
    manifolds, m_strs = _get_spin_split_manifolds_idx (ci1_fr, norb_f, nelec_frs, smult_fr, lroots_fr)
    for i in range (len (manifolds)):
        manifolds[i] = [idx[j] for j in manifolds[i]]
    return manifolds, m_strs

def _get_spin_split_manifolds_idx (ci_fr, norb_f, nelec_frs, smult_fr, lroots_fr):
    '''Split a manifold of model state rootspaces which have same numbers of electrons and spin
    multiplicities in each fragment into submanifolds according to their spin-projection quantum
    numbers Na-Nb.'''
    # after indexing down to the current spinless manifold
    nfrags = len (norb_f)
    spins_fr = nelec_frs[:,:,0] - nelec_frs[:,:,1]
    tabulator = spins_fr.T
    m_strs, inverse = np.unique (tabulator, axis=0, return_inverse=True)
    num_m_blocks = len (m_strs)
    m_blocks = [np.where (inverse==i)[0] for i in range (num_m_blocks)]
    if smult_fr is None or num_m_blocks<2:
        return [m_block[None,:] for m_block in m_blocks], [m_str[None,:] for m_str in m_strs]
    fprint = np.stack ([get_unique_roots_with_spin (
        ci_fr[ifrag], norb_f[ifrag], [tuple (n) for n in nelec_frs[ifrag]], smult_fr[ifrag]
    ) for ifrag in range (nfrags)], axis=1)
    fprint = [fprint[m_block] for m_block in m_blocks]
    uniq, inverse = np.unique (fprint, axis=0, return_inverse=True)
    manifolds = [np.stack ([m_blocks[i] for i in np.where (inverse==j)[0]],
                           axis=0)
                 for j in range (len (uniq))]
    m_strs = [np.stack ([m_strs[i] for i in np.where (inverse==j)[0]],
                        axis=0)
              for j in range (len (uniq))]
    return manifolds, m_strs

class OrthBasisBase (sparse_linalg.LinearOperator):
    def get_nbytes (self):
        def _get (x):
            if isinstance (x, np.ndarray):
                return int (x.nbytes)
            elif lib.issequence (x):
                return sum ([_get (xi) for xi in x])
            else:
                return int (sys.getsizeof (x))
        nbytes = sum ([_get (x) for x in self.__dict__.values ()])
        return nbytes

    def same_block (self, i, j):
        return self.roots2blks (i) == self.roots2blks (j)

class NullOrthBasis (OrthBasisBase):
    def __init__(self, nraw, dtype, nprods_r):
        self.shape = (nraw,nraw)
        self.dtype = dtype
        self.nprods_raw = nprods_r
        offs1 = np.cumsum (nprods_r)
        offs0 = offs1 - nprods_r
        self.offs_raw = np.stack ([offs0,offs1], axis=1)

    def _matvec (self, x): return x

    def _rmatvec (self, x): return x

    def get_xmat_rows (self, iroot, _col=None):
        xmat = np.eye (self.nprods_raw[iroot])
        if _col is not None:
            xmat = xmat[:,_col]
        return xmat

    def rootspaces_covering_addrs (self, addrs):
        prods = np.atleast_1d (addrs)
        return np.searchsorted (self.offs_raw[:,0], addrs, side='right')-1

    def split_addrs_by_blocks (self, addrs):
        blks = np.searchsorted (self.offs_raw[:,0], addrs, side='right')-1
        cols = np.asarray (addrs) - self.offs_raw[blks,0]
        assert (np.all (cols>=0))
        return blks, cols

    def roots2blks (self, roots):
        return roots


class OrthBasis (OrthBasisBase):
    def __init__(self, shape, dtype, nprods_r, manifolds):
        self.shape = shape
        self.dtype = dtype
        self.manifolds_roots = [manifold.m_blocks for manifold in manifolds]
        self.manifolds_xmat = [manifold.xmat for manifold in manifolds]
        self.nprods_raw = nprods_r
        offs1 = np.cumsum (nprods_r)
        offs0 = offs1 - nprods_r
        self.offs_raw = np.stack ([offs0,offs1], axis=1)
        manifolds_prod_idx = []
        for mi in self.manifolds_roots:
            pi = []
            for mij in mi:
                pij = []
                for mijk in mij:
                    pij.extend (list (range (offs0[mijk], offs1[mijk])))
                pi.append (pij)
            manifolds_prod_idx.append (pi)
        self.manifolds_prod_idx = [np.asarray (x, dtype=int) for x in manifolds_prod_idx]
        # lookup for diagonal blocking
        # rows are rootspaces
        # [blk_idx, rootspace_idx]
        self.root_block_addr = -np.ones ((len (nprods_r),2), dtype=int)
        # lookup for diagonal blocking
        # rows are blocks
        # [snp_idx, m_idx]
        self.block_manifold_addr = []
        nman = 0
        self.manifolds_nprods_raw = []
        self.manifolds_offs_raw = []
        self.nprods_orth = []
        for i, mi in enumerate (self.manifolds_roots):
            # common xmat: a "manifold"
            my_nprods_raw = nprods_r[mi[0]]
            self.manifolds_nprods_raw.append (my_nprods_raw)
            offs1 = np.cumsum (my_nprods_raw)
            offs0 = offs1 - my_nprods_raw
            self.manifolds_offs_raw.append (np.stack ([offs0, offs1], axis=1))
            xmat = self.manifolds_xmat[i]
            if xmat is None:
                xmat_shape_1 = offs1[-1]
            else:
                xmat_shape_1 = xmat.shape[1]
                assert (offs1[-1] == xmat.shape[0])
            for j, mij in enumerate (mi):
                # common m string: a "block"
                for k, mijk in enumerate (mij):
                    # an individual root
                    self.root_block_addr[mijk,:] = [nman,k]
                self.nprods_orth.append (xmat_shape_1)
                self.block_manifold_addr.append ([i,j])
                nman += 1
        assert (np.all (self.root_block_addr>-1))
        self.block_manifold_addr = np.stack (self.block_manifold_addr, axis=0)
        self.nprods_orth = np.asarray (self.nprods_orth)
        offs1 = np.cumsum (self.nprods_orth)
        offs0 = offs1 - self.nprods_orth
        self.offs_orth = np.stack ([offs0, offs1], axis=1)
        assert (self.offs_orth[-1,-1] == self.shape[0])

    def rootspaces_covering_addrs (self, addrs):
        blocks = np.searchsorted (self.offs_orth[:,0], addrs, side='right')-1
        manaddrs = self.block_manifold_addr[blocks]
        return np.concatenate ([self.manifolds_roots[i][j] for i,j in manaddrs])

    def roots2blks (self, roots):
        return self.root_block_addr[:,0][roots]

    def split_addrs_by_blocks (self, addrs):
        blks = np.searchsorted (self.offs_orth[:,0], addrs, side='right')-1
        cols = np.asarray (addrs) - self.offs_orth[blks,0]
        assert (np.all (cols>=0))
        return blks, cols

    def get_xmat_rows (self, iroot, _col=None):
        x, j = self.root_block_addr[iroot]
        i = self.block_manifold_addr[x,0]
        if self.manifolds_xmat[i] is None:
            xmat = np.eye (self.manifolds_nprods_raw[i].sum ())
        else:
            xmat = self.manifolds_xmat[i]
        p, q = self.manifolds_offs_raw[i][j]
        xmat = xmat[p:q,:]
        nraw = self.manifolds_nprods_raw[i][j]
        north = self.nprods_orth[x]
        assert (xmat.shape == (nraw, north))
        if _col is not None:
            xmat = xmat[:,_col]
        return xmat

    def _matvec (self, rawarr):
        is_out_complex = (self.dtype==np.complex128) or np.iscomplexobj (rawarr)
        my_dtype = np.complex128 if is_out_complex else np.float64
        col_shape = rawarr.shape[1:]
        orth_shape = [self.shape[0],] + list (col_shape)
        ortharr = np.zeros (orth_shape, dtype=my_dtype)
        p = 0
        for prod_idx, xmat in zip (self.manifolds_prod_idx, self.manifolds_xmat):
            if xmat is None:
                for mirror in prod_idx:
                    i, j = self.offs_orth[p]
                    p += 1
                    ortharr[i:j] = rawarr[mirror]
            else:
                for mirror in prod_idx:
                    i, j = self.offs_orth[p]
                    p += 1
                    ortharr[i:j] = np.tensordot (xmat.T, rawarr[mirror], axes=1)
        return ortharr

    def _rmatvec (self, ortharr):
        is_out_complex = (self.dtype==np.complex128) or np.iscomplexobj (ortharr)
        my_dtype = np.complex128 if is_out_complex else np.float64
        col_shape = ortharr.shape[1:]
        raw_shape = [self.shape[1],] + list (col_shape)
        rawarr = np.zeros (raw_shape, dtype=my_dtype)
        p = 0
        for prod_idx, xmat in zip (self.manifolds_prod_idx, self.manifolds_xmat):
            if xmat is None:
                for mirror in prod_idx:
                    i, j = self.offs_orth[p]
                    p += 1
                    rawarr[mirror] = ortharr[i:j]
            else:
                for mirror in prod_idx:
                    i, j = self.offs_orth[p]
                    p += 1
                    rawarr[mirror] = np.tensordot (xmat.conj (), ortharr[i:j], axes=1)
        return rawarr




