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
    uniq_roots = list (uniq_idx[cnts==1])
    north = (offs1[uniq_roots] - offs0[uniq_roots]).sum ()
    manifolds_xmat = []
    manifolds_roots = []
    for manifold_idx in np.where (cnts>1)[0]:
        manifolds = _get_spin_split_manifolds (ci_fr, norb_f, nelec_frs, smult_fr, lroots_fr,
                                               inverse==manifold_idx)
        for manifold in manifolds:
            nmirror = len (manifold)
            nprod = np.asarray ([(offs1[spin_mirror]-offs0[spin_mirror]).sum () for spin_mirror in manifold])
            assert (np.all (nprod==nprod[0]))
            nprod = nprod[0]
            ovlp = _get_ovlp (rootidx=manifold[0])
            ovlp[np.diag_indices_from (ovlp)] -= 1.0
            err_from_diag = np.amax (np.abs (ovlp))
            if err_from_diag > 1e-8:
                ovlp[np.diag_indices_from (ovlp)] += 1.0
                xmat = canonical_orth_(ovlp, thr=LINDEP_THRESH)
                north += xmat.shape[1] * nmirror
                manifolds_xmat.append (xmat)
                manifolds_roots.append (manifold)
            else:
                north += ovlp.shape[0] * nmirror
                for spin_mirror in manifold:
                    uniq_roots.extend (spin_mirror)
            ovlp = None

    _get_ovlp = None

    return OrthBasis ((north,nraw), dtype, nprods_r, uniq_roots, manifolds_roots, manifolds_xmat)

def _get_spin_split_manifolds (ci_fr, norb_f, nelec_frs, smult_fr, lroots_fr, idx):
    '''The same as _get_spin_split_manifolds_idx, except that all of the arguments need to be
    indexed down from the full model space into the input manifold via "idx" first. The returned
    submanifold list is likewise indexed back into the full model space.'''
    nelec_frs = nelec_frs[:,idx,:]
    if smult_fr is not None: smult_fr = smult_fr[:,idx]
    lroots_fr = lroots_fr[:,idx]
    idx = np.where (idx)[0]
    ci1_fr = [[ci_r[i] for i in idx] for ci_r in ci_fr]
    manifolds = _get_spin_split_manifolds_idx (ci1_fr, norb_f, nelec_frs, smult_fr, lroots_fr)
    for i in range (len (manifolds)):
        manifolds[i] = [idx[j] for j in manifolds[i]]
    return manifolds

def _get_spin_split_manifolds_idx (ci_fr, norb_f, nelec_frs, smult_fr, lroots_fr):
    '''Split a manifold of model state rootspaces which have same numbers of electrons and spin
    multiplicities in each fragment into submanifolds according to their spin-projection quantum
    numbers Na-Nb.'''
    # after indexing down to the current spinless manifold
    nfrags = len (norb_f)
    spins_fr = nelec_frs[:,:,0] - nelec_frs[:,:,1]
    tabulator = spins_fr.T
    uniq, inverse = np.unique (tabulator, axis=0, return_inverse=True)
    nmanifolds = len (uniq)
    manifolds = [np.where (inverse==i)[0] for i in range (nmanifolds)]
    if smult_fr is None or nmanifolds<2:
        return [manifold[None,:] for manifold in manifolds]
    fprint = np.stack ([get_unique_roots_with_spin (
        ci_fr[ifrag], norb_f[ifrag], [tuple (n) for n in nelec_frs[ifrag]], smult_fr[ifrag]
    ) for ifrag in range (nfrags)], axis=1)
    fprint = [fprint[manifold] for manifold in manifolds]
    uniq, inverse = np.unique (fprint, axis=0, return_inverse=True)
    manifolds = [np.stack ([manifolds[i] for i in np.where (inverse==j)[0]],
                           axis=0)
                 for j in range (len (uniq))]
    return manifolds

class OrthBasis (sparse_linalg.LinearOperator):
    def __init__(self, shape, dtype, nprods_r, uniq_roots, manifolds_roots, manifolds_xmat):
        self.shape = shape
        self.dtype = dtype
        self.uniq_roots = uniq_roots
        nuniq_roots = len (uniq_roots)
        self.manifolds_roots = [np.asarray (x, dtype=int) for x in manifolds_roots]
        self.manifolds_xmat = manifolds_xmat
        self.nprods_raw = nprods_r
        offs1 = np.cumsum (nprods_r)
        offs0 = offs1 - nprods_r
        self.offs_raw = np.stack ([offs0,offs1], axis=1)
        uniq_prod_idx = []
        for i in uniq_roots: uniq_prod_idx.extend (list(range(offs0[i],offs1[i])))
        self.uniq_prod_idx = np.asarray (uniq_prod_idx, dtype=int)
        manifolds_prod_idx = []
        for mi in manifolds_roots:
            pi = []
            for mij in mi:
                pij = []
                for mijk in mij:
                    pij.extend (list (range (offs0[mijk], offs1[mijk])))
                pi.append (pij)
            manifolds_prod_idx.append (pi)
        self.manifolds_prod_idx = [np.asarray (x, dtype=int) for x in manifolds_prod_idx]
        # lookup for diagonal blocking
        self.root_manifold_addr = -2*np.ones ((len (nprods_r),3), dtype=int)
        self.root_manifold_addr[uniq_roots,0] = np.arange (len (uniq_roots), dtype=int)
        self.root_manifold_addr[uniq_roots,1] = -(self.root_manifold_addr[uniq_roots,0]+1)
        self.root_manifold_addr[uniq_roots,2] = -1
        nman = len (uniq_roots)
        self.manifolds_nprods_raw = []
        self.manifolds_offs_raw = []
        self.snm_blocks = [np.asarray ([u,]) for u in uniq_roots]
        manifolds_nprods_orth_flat = []
        for i, mi in enumerate (manifolds_roots):
            my_nprods_raw = nprods_r[mi[0]]
            self.manifolds_nprods_raw.append (my_nprods_raw)
            offs1 = np.cumsum (my_nprods_raw)
            offs0 = offs1 - my_nprods_raw
            self.manifolds_offs_raw.append (np.stack ([offs0, offs1], axis=1))
            xmat = manifolds_xmat[i]
            assert (offs1[-1] == xmat.shape[0])
            for j, mij in enumerate (mi):
                for k, mijk in enumerate (mij):
                    self.root_manifold_addr[mijk,:] = [nman,i,k]
                manifolds_nprods_orth_flat.append (xmat.shape[1])
                self.snm_blocks.append (np.asarray (mij))
                nman += 1
        assert (np.all (self.root_manifold_addr[:,2]>-2))
        self.nprods_orth = np.empty (nman, dtype=int)
        self.nprods_orth[:len(uniq_roots)] = nprods_r[uniq_roots]
        self.nprods_orth[len(uniq_roots):] = manifolds_nprods_orth_flat
        offs1 = np.cumsum (self.nprods_orth)
        offs0 = offs1 - self.nprods_orth
        self.offs_orth = np.stack ([offs0, offs1], axis=1)
        assert (self.offs_orth[-1,-1] == self.shape[0])

    def roots_in_same_block (self, i, j):
        return self.root_manifold_addr[i,0]==self.root_manifold_addr[j,0]

    def get_orth_prod_range (self, iroot):
        p = self.root_manifold_addr[iroot,0]
        return tuple (self.offs_orth[p])

    def prods_2_roots (self, prods):
        blocks = np.searchsorted (self.offs_orth[:,0], prods, side='right')-1
        return [self.snm_blocks[b] for b in blocks]

    def roots_2_snm (self, roots):
        return self.root_manifold_addr[:,0][roots]

    def map_prod_subspace (self, prods):
        prods = np.asarray (prods)
        rootlist = self.prods_2_roots (prods)
        rootmap = {}
        for i in range (len (prods)):
            roots = tuple (rootlist[i])
            val = rootmap.get (roots, [])
            val.append (prods[i])
            rootmap[roots] = val
        return rootmap

    def interpret_address (self, prods):
        blocks = np.searchsorted (self.offs_orth[:,0], prods, side='right')-1
        psi = np.asarray (prods) - self.offs_orth[blocks,0]
        assert (np.all (psi>=0))
        return blocks, psi

    def get_xmat_rows (self, iroot, _col=None):
        x, i, j = self.root_manifold_addr[iroot]
        if j == -1:
            xmat = np.eye (self.nprods_raw[iroot])
            if _col is not None:
                xmat = xmat[:,_col]
            return xmat
        assert (i >= 0)
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
        nuniq_prod = len (self.uniq_prod_idx)
        is_out_complex = (self.dtype==np.complex128) or np.iscomplexobj (rawarr)
        my_dtype = np.complex128 if is_out_complex else np.float64
        col_shape = rawarr.shape[1:]
        orth_shape = [self.shape[0],] + list (col_shape)
        ortharr = np.zeros (orth_shape, dtype=my_dtype)
        ortharr[:nuniq_prod] = rawarr[self.uniq_prod_idx]
        i = nuniq_prod
        for prod_idx, xmat in zip (self.manifolds_prod_idx, self.manifolds_xmat):
            for mirror in prod_idx:
                j = i + xmat.shape[1]
                ortharr[i:j] = np.tensordot (xmat.T, rawarr[mirror], axes=1)
                i = j
        return ortharr

    def _rmatvec (self, ortharr):
        nuniq_prod = len (self.uniq_prod_idx)
        is_out_complex = (self.dtype==np.complex128) or np.iscomplexobj (ortharr)
        my_dtype = np.complex128 if is_out_complex else np.float64
        col_shape = ortharr.shape[1:]
        raw_shape = [self.shape[1],] + list (col_shape)
        rawarr = np.zeros (raw_shape, dtype=my_dtype)
        rawarr[self.uniq_prod_idx] = ortharr[:nuniq_prod]
        i = nuniq_prod
        for prod_idx, xmat in zip (self.manifolds_prod_idx, self.manifolds_xmat):
            for mirror in prod_idx:
                j = i + xmat.shape[1]
                rawarr[mirror] = np.tensordot (xmat.conj (), ortharr[i:j], axes=1)
                i = j
        return rawarr

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

    def gen_mixed_state_vectors (self, _yield_roots=False):
        x0 = np.zeros (self.shape[1])
        for i in range (len (self.manifolds_xmat)):
            prod_idx = self.manifolds_prod_idx[i]
            roots = self.manifolds_roots[i]
            xmat = self.manifolds_xmat[i]
            for spincase in range (len (prod_idx)):
                for row in xmat.T:
                    x0[prod_idx[spincase]] = row
                    if _yield_roots:
                        yield x0, roots[spincase]
                    else:
                        yield x0
                    x0[prod_idx[spincase]] = 0

class NullOrthBasis (sparse_linalg.LinearOperator):
    def __init__(self, nraw, dtype, nprods_r):
        self.shape = (nraw,nraw)
        self.dtype = dtype
        self.nprods_raw = nprods_r
        offs1 = np.cumsum (nprods_r)
        offs0 = offs1 - nprods_r
        self.offs_raw = np.stack ([offs0,offs1], axis=1)

    def _matvec (self, x): return x

    def _rmatvec (self, x): return x

    get_nbytes = OrthBasis.get_nbytes

    def get_xmat_rows (self, iroot, _col=None):
        xmat = np.eye (self.nprods_raw[iroot])
        if _col is not None:
            xmat = xmat[:,_col]
        return xmat

    @property
    def uniq_prod_idx (self): return np.arange (self.shape[0], dtype=int)

    def gen_mixed_state_vectors (self, _yield_roots=False):
        # zero-length generator
        return
        yield

    def prods_2_roots (self, prods):
        prods = np.atleast_1d (prods)
        roots = np.searchsorted (self.offs_raw[:,0], prods, side='right')-1
        return [tuple ((r,)) for r in np.atleast_1d (roots)]

    map_prod_subspace = OrthBasis.map_prod_subspace

    def interpret_address (self, prods):
        roots = np.searchsorted (self.offs_raw[:,0], prods, side='right')-1
        psi = np.asarray (prods) - self.offs_raw[roots,0]
        assert (np.all (psi>=0))
        return roots, psi

    def roots_2_snm (self, roots):
        return roots

