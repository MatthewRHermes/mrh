import sys
import numpy as np
import functools
import itertools
from scipy.sparse import linalg as sparse_linalg
from scipy import linalg
from pyscf.scf.addons import canonical_orth_
from pyscf import __config__, lib
from mrh.my_pyscf.lassi.citools import get_unique_roots_with_spin
from mrh.my_pyscf.lassi.op_o1.utilities import fermion_spin_shuffle

LINDEP_THRESH = getattr (__config__, 'lassi_lindep_thresh', 1.0e-5)

def get_orth_basis (ci_fr, norb_f, nelec_frs, _get_ovlp=None, smult_fr=None, smult_si=None):
    '''Unitary matrix for an orthonormal product-state basis from a set of CI vectors.

    Args:
        ci_fr: list of length nfrags of list of length nroots of ndarrays
            CI vectors of fragment statelets
        norb_f: ndarray of shape (nfrags,) of int
            Number of orbitals in each fragment
        nelec_frs: ndarray of shape (nfrags,nroots,2) of int
            Number of electrons in each fragment in each rootspace

    Kwargs:
        smult_fr: ndarray of shape (nfrags,nroots) of int
            Spin multiplicity in each fragment in each rootspace
        smult_si: integer
            Target spin multiplicity. If included, smult_fr is also required
        _get_ovlp: callable with kwarg rootidx
            Produce the overlap matrix between model states in a set of rootspaces,
            identified by ndarray or list "rootidx"

    Returns:
        raw2orth: LinearOperator of shape (north, nraw)
            Apply to SI vector to transform from the primitive to the orthonormal basis.
            The orthonormal basis is also spin-coupled if smult_si is provided.
    '''
    if _get_ovlp is None:
        from mrh.my_pyscf.lassi.op_o0 import get_ovlp
        _get_ovlp = functools.partial (get_ovlp, ci_fr, norb_f, nelec_frs)
    nfrags, nroots = nelec_frs.shape[:2]
    tabulator = nelec_frs.sum (2)
    if smult_fr is not None:
        tabulator = np.append (tabulator, smult_fr, axis=0)
    else:
        assert (smult_si is None)
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
    # iterate over smult & nelec strings
    found1st = False
    for sn_string_idx, sn_str in enumerate (unique.T):
        contains1st = (inverse[0]==sn_string_idx)
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
            else:
                xmat = None
            new_manifold = get_rootspace_manifold (norb_f, lroots_fr, nprods_r, n_str, s_str,
                                                   m_strs, m_blocks, xmat, smult_si=smult_si)
            if contains1st and _is_first (nelec_frs, smult_fr, n_str, s_str, m_strs):
                assert (not found1st)
                manifolds = [new_manifold,] + manifolds
                found1st = True
            else:
                manifolds.append (new_manifold)
            north += np.prod (new_manifold.orth_shape)
            ovlp = None

    assert (found1st)
    _get_ovlp = None

    if smult_si is None:
        return OrthBasis ((north,nraw), dtype, nprods_r, manifolds)
    else:
        return SpinCoupledOrthBasis ((north,nraw), dtype, nprods_r, manifolds)

def _is_first (nelec_frs, smult_fr, n_str, s_str, m_strs):
    if np.any (n_str != nelec_frs[:,0,:].sum (1)):
        return False
    if smult_fr is None:
        n0_fs = nelec_frs[:,0,:][None,:,:]
        n1_rfs = np.asarray ([[n_str + m_str, n_str - m_str]
                              for m_str in m_strs]).transpose (0,2,1) // 2
        return (n0_fs==n1_rfs).all ((1,2)).any ()
    else:
        return np.all (smult_fr[:,0]==s_str)

def get_nbytes (obj):
    def _get (x):
        if isinstance (x, np.ndarray):
            return int (x.nbytes)
        elif callable (getattr (x, 'get_nbytes', None)):
            return x.get_nbytes ()
        elif lib.issequence (x):
            return sum ([_get (xi) for xi in x])
        else:
            return int (sys.getsizeof (x))
    nbytes = sum ([_get (x) for x in obj.__dict__.values ()])
    return nbytes

def get_rootspace_manifold (norb_f, lroots_fr, nprods_r, n_str, s_str, m_strs, m_blocks, xmat,
                            smult_si=None):
    if smult_si is None:
        return RootspaceManifold (norb_f, lroots_fr, nprods_r, n_str, s_str, m_strs, m_blocks,
                                  xmat)
    else:
        return SpinCoupledRootspaceManifold (norb_f, lroots_fr, nprods_r, n_str, s_str, m_strs,
                                             m_blocks, xmat, smult_si)

class RootspaceManifold:
    get_nbytes = get_nbytes
    def __init__(self, norb_f, lroots_fr, nprods_r, n_str, s_str, m_strs, m_blocks, xmat):
        self.norb_f = norb_f
        self.n_str = n_str
        self.s_str = s_str
        self.m_strs = m_strs
        self.m_blocks = np.asarray (m_blocks, dtype=int)
        self.xmat = xmat
        self.umat = np.eye (self.m_blocks.shape[0])

        offs1 = np.cumsum (nprods_r)
        offs0 = offs1 - nprods_r
        offs_raw = np.stack ([offs0,offs1], axis=1)
        self.prod_idx = []
        for m_block in self.m_blocks:
            pij = []
            for iroot in m_block:
                pij.extend (list (range (offs0[iroot], offs1[iroot])))
            self.prod_idx.append (pij)
        self.prod_idx = np.asarray (self.prod_idx, dtype=int)
        self.lroots_rf = lroots_fr.T[self.m_blocks[0]]
        self.nprods_raw = nprods_r[self.m_blocks[0]]
        offs1 = np.cumsum (self.nprods_raw)
        offs0 = offs1 - self.nprods_raw
        self.offs_raw = np.stack ([offs0, offs1], axis=1)
        if xmat is None:
            xmat_shape = (offs1[-1], offs1[-1])
        else:
            xmat_shape = xmat.shape
            assert (offs1[-1] == xmat.shape[0])
        self.raw_shape = (self.m_blocks.shape[0], xmat_shape[0])
        self.orth_shape = (self.raw_shape[0], xmat_shape[1])

    def idx_m_str (self, m_str, inv):
        m_str = np.asarray (m_str)[None,:]
        m_strs = self.m_strs[:,inv]
        return np.where (np.all (m_str==m_strs, axis=1))[0]

    def get_t_strs (self): return self.m_strs

    def sprintf_address_book_sn (self, i):
        return 'manifold {}: smult = {}; nelec = {}'.format (i, self.s_str, self.n_str)

    def sprintf_address_book_spin_raw (self):
        out = 'm_strs:\n'
        for i, m_str in enumerate (self.m_strs):
            out += ' {} {}\n'.format (i, m_str)
        return out[:-1]   

    sprintf_address_book_spin_orth = sprintf_address_book_spin_raw

    def sprintf_address_book_spat_raw (self):
        out = 'rootspace address and size:\n'
        for i, (ir, lr) in enumerate (zip (self.m_blocks[0], self.lroots_rf)):
            out += ' {} {} {}\n'.format (i, ir, lr)
        return out[:-1]

    def sprintf_address_book_spat_orth (self):
        return '{} generic spatial states'.format (self.orth_shape[1])

    def sprintf_address_book_raw (self, i):
        return '\n'.join ([self.sprintf_address_book_sn (i),
                           self.sprintf_address_book_spin_raw (),
                           self.sprintf_address_book_spat_raw ()])

    def sprintf_address_book_orth (self, i):
        return '\n'.join ([self.sprintf_address_book_sn (i),
                           self.sprintf_address_book_spin_orth (),
                           self.sprintf_address_book_spat_orth ()])


class SpinCoupledRootspaceManifold (RootspaceManifold):
    def __init__(self, norb_f, lroots_fr, nprods_r, n_str, s_str, m_strs, m_blocks, xmat,
                 smult_si):
        super().__init__(norb_f, lroots_fr, nprods_r, n_str, s_str, m_strs, m_blocks, xmat)
        spin_si = np.sum (self.m_strs[0])
        spins_table, smult_table = get_spincoup_bases (self.s_str, spin_lsf=spin_si,
                                                       smult_lsf=smult_si)
        self.t_strs = smult_table
        spins_table = [tuple (row) for row in spins_table]
        idx = np.asarray ([spins_table.index (tuple (row)) for row in self.m_strs])
        self.umat = get_spincoup_umat (self.s_str, spin_si, smult_si)[idx,:]
        for i, m_str in enumerate (self.m_strs):
            na = (self.n_str + m_str) // 2
            nb = (self.n_str - m_str) // 2
            self.umat[i,:] *= fermion_spin_shuffle (na, nb)
        self.orth_shape = (self.umat.shape[1], self.orth_shape[1])

    def get_t_strs (self): return self.t_strs

    def sprintf_address_book_spin_orth (self):
        out = 't_strs:\n'
        for i, t_str in enumerate (self.t_strs):
            out += ' {} {}\n'.format (i, t_str)
        return out[:-1]   

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
    for iblk in range (len (m_blocks)):
        for ifrag in range (nfrags):
            idx = np.argsort (fprint[iblk][:,ifrag], kind='mergesort')
            fprint[iblk] = fprint[iblk][idx]
            m_blocks[iblk] = m_blocks[iblk][idx]
    uniq, inverse = np.unique (fprint, axis=0, return_inverse=True)
    manifolds = [np.stack ([m_blocks[i] for i in np.where (inverse==j)[0]],
                           axis=0)
                 for j in range (len (uniq))]
    m_strs = [np.stack ([m_strs[i] for i in np.where (inverse==j)[0]],
                        axis=0)
              for j in range (len (uniq))]
    for iblk in range (len (uniq)):
        for ifrag in range (nfrags):
            idx = np.argsort (m_strs[iblk][:,ifrag], kind='mergesort')
            m_strs[iblk] = m_strs[iblk][idx]
            manifolds[iblk] = manifolds[iblk][idx]
    return manifolds, m_strs

class OrthBasisBase (sparse_linalg.LinearOperator):
    get_nbytes = get_nbytes
    def roots_coupled_in_hdiag (self, i, j):
        return self.roots2blks (i) == self.roots2blks (j)

    def split_oblocks_by_manifolds (self, blocks):
        blocks_shape = np.asarray (blocks).shape
        return blocks, np.zeros (blocks_shape, dtype=int)

    split_rblocks_by_manifolds=split_oblocks_by_manifolds

    def idx2addrs_orth (self, idx):
        blks, addrs_p = self.split_addrs_by_oblocks (idx)
        addrs_sn, addrs_t = self.split_oblocks_by_manifolds (blks)
        return addrs_sn, addrs_t, addrs_p

    def idx2addrs_raw (self, idx):
        blks, addrs_p = self.split_addrs_by_rblocks (idx)
        addrs_sn, addrs_m = self.split_rblocks_by_manifolds (blks)
        return addrs_sn, addrs_m, addrs_p

    idx2addrs = idx2addrs_orth

    def are_tstrs_coupled (self, bra_sn, ket_sn, bra_t, ket_t, inv):
        coup = (bra_t==ket_t)
        if isinstance (coup, np.ndarray):
            coup[:] = True
        else:
            coup = True
        return coup

class NullOrthBasis (OrthBasisBase):
    def __init__(self, nraw, dtype, nprods_r):
        self.shape = (nraw,nraw)
        self.dtype = dtype
        self.nprods_raw = nprods_r
        offs1 = np.cumsum (nprods_r)
        offs0 = offs1 - nprods_r
        self.offs_raw = np.stack ([offs0,offs1], axis=1)

    def get_ref_man_size (self): return self.nprods_raw[0]

    def _matvec (self, x): return x

    def _rmatvec (self, x): return x

    def get_xmat_rows (self, iroot, _col=None):
        xmat = np.eye (self.nprods_raw[iroot])
        if _col is not None:
            xmat = xmat[:,_col]
        return xmat

    def rootspaces_covering_addrs (self, addrs):
        prods = np.atleast_1d (addrs)
        roots = np.searchsorted (self.offs_raw[:,0], addrs, side='right')-1
        return np.atleast_1d (roots).astype (int)

    def split_addrs_by_blocks (self, addrs):
        blks = np.searchsorted (self.offs_raw[:,0], addrs, side='right')-1
        cols = np.asarray (addrs) - self.offs_raw[blks,0]
        assert (np.all (cols>=0))
        return blks, cols

    split_addrs_by_oblocks = split_addrs_by_blocks
    split_addrs_by_rblocks = split_addrs_by_blocks

    def roots2blks (self, roots):
        return roots

    def roots2mans (self, roots):
        return roots

    def get_manifold_orth_shape (self, iroot):
        return (1, self.nprods_raw[iroot])

    def hdiag_spincoup_loop (self, iroot, mstr_bra, mstr_ket):
        yield 1, self.offs_raw[iroot]
        return

    def spincase_mstrs (self, roots, inv):
        return tuple (roots)

    def find_spin_nonvanishing_overlaps (self, bra_sn, ket_sn, m_exc, inv):
        return np.ones (len (m_exc), dtype=bool)

    def pspace_ham_spincoup_dm (self, bra_sn, ket_sn, mtidx_bra, mtidx_ket, sgnvec, inv):
        assert (len (sgnvec) == 1)
        return np.atleast_2d (sgnvec)

    def log_debug1_hdiag_raw (self, log, hdiag, idx=None):
        return

    def log_debug_hdiag_orth (self, log, hdiag, idx=None):
        return

class OrthBasis (OrthBasisBase):
    def __init__(self, shape, dtype, nprods_r, manifolds):
        self.shape = shape
        self.dtype = dtype
        self.manifolds = manifolds
        self.nprods_raw = nprods_r
        offs1 = np.cumsum (nprods_r)
        offs0 = offs1 - nprods_r
        self.offs_raw = np.stack ([offs0,offs1], axis=1)
        # lookup for diagonal blocking
        # rows are rootspaces
        # [blk_idx, rootspace_idx]
        self.root_block_addr = -np.ones ((len (nprods_r),2), dtype=int)
        self.root_block_prod_offs = -np.ones (len (nprods_r), dtype=int)
        # lookup for diagonal blocking
        # rows are blocks
        # [snp_idx, m_idx]
        self.rblock_manifold_addr = []
        self.oblock_manifold_addr = []
        nman = 0
        self.nprods_orth = []
        blk_off = 0
        for i, manifold in enumerate (manifolds):
            for j in range (manifold.orth_shape[0]):
                self.oblock_manifold_addr.append ([i,j])
                self.nprods_orth.append (manifold.orth_shape[1])
            for j in range (manifold.raw_shape[0]):
                self.rblock_manifold_addr.append ([i,j])
            for j, m_block in enumerate (manifold.m_blocks):
                # common m string: a "block"
                offs = np.cumsum (self.nprods_raw[m_block])
                offs -= self.nprods_raw[m_block]
                for k, iroot in enumerate (m_block):
                    # an individual root
                    self.root_block_addr[iroot,:] = [nman,k]
                    self.root_block_prod_offs[iroot] = offs[k]
                nman += 1
        assert (np.all (self.root_block_addr>-1))
        self.rblock_manifold_addr = np.stack (self.rblock_manifold_addr, axis=0)
        self.oblock_manifold_addr = np.stack (self.oblock_manifold_addr, axis=0)
        self.nprods_orth = np.asarray (self.nprods_orth)
        offs1 = np.cumsum (self.nprods_orth)
        offs0 = offs1 - self.nprods_orth
        self.offs_orth = np.stack ([offs0, offs1], axis=1)
        assert (self.offs_orth[-1,-1] == self.shape[0])

    def get_ref_man_size (self): return np.prod (self.manifolds[0].orth_shape)

    def rootspaces_covering_addrs (self, addrs):
        blocks = np.searchsorted (self.offs_orth[:,0], addrs, side='right')-1
        manaddrs = self.rblock_manifold_addr[blocks]
        return np.concatenate ([self.manifolds[i].m_blocks[j] for i,j in manaddrs]).astype (int)

    def roots2blks (self, roots):
        return self.root_block_addr[:,0][roots]

    def roots2mans (self, roots):
        return self.rblock_manifold_addr[:,0][self.roots2blks (roots)]

    def get_manifold_orth_shape (self, iman):
        return self.manifolds[iman].orth_shape

    def hdiag_spincoup_loop (self, iman, mstr_bra, mstr_ket, inv):
        assert (mstr_bra == mstr_ket)
        offs0 = 0
        for man in self.manifolds[:iman]:
            offs0 += np.prod (man.orth_shape)
        man = self.manifolds[iman]
        iblks = man.idx_m_str (mstr_ket, inv)
        ncols = man.orth_shape[1]
        for iblk in iblks:
            p = offs0 + iblk*ncols
            q = p + ncols
            yield 1, (p,q)
        return

    def spincase_mstrs (self, roots, inv):
        mstrs = []
        for iroot in roots:
            iblk = self.root_block_addr[iroot,0]
            iman, imstr = self.rblock_manifold_addr[iblk]
            mstrs.append (tuple (np.asarray (self.manifolds[iman].m_strs)[:,inv][imstr]))
        return tuple (mstrs)

    def split_addrs_by_blocks (self, addrs):
        # This usage implies "orth" blocks
        blks = np.searchsorted (self.offs_orth[:,0], addrs, side='right')-1
        cols = np.asarray (addrs) - self.offs_orth[blks,0]
        assert (np.all (cols>=0))
        return blks, cols

    split_addrs_by_oblocks = split_addrs_by_blocks

    def split_addrs_by_rblocks (self, addrs):
        roots = np.searchsorted (self.offs_raw[:,0], addrs, side='right')-1
        cols = np.asarray (addrs) - self.offs_raw[roots,0]
        cols += self.root_block_prod_offs[roots]
        blks = self.root_block_addr[roots,0]
        assert (np.all (cols>=0))
        return blks, cols

    def split_oblocks_by_manifolds (self, blocks):
        blocks = np.asarray (blocks)
        blocks_shape = blocks.shape
        blocks = np.ravel (blocks)
        mans, sps = list (self.oblock_manifold_addr[blocks].T)
        return mans.reshape (blocks_shape), sps.reshape (blocks_shape)

    def split_rblocks_by_manifolds (self, blocks):
        blocks = np.asarray (blocks)
        blocks_shape = blocks.shape
        blocks = np.ravel (blocks)
        mans, sps = list (self.rblock_manifold_addr[blocks].T)
        return mans.reshape (blocks_shape), sps.reshape (blocks_shape)

    def get_xmat_rows (self, iroot, _col=None):
        x, j = self.root_block_addr[iroot]
        i = self.rblock_manifold_addr[x,0]
        if self.manifolds[i].xmat is None:
            xmat = np.eye (self.manifolds[i].nprods_raw.sum ())
        else:
            xmat = self.manifolds[i].xmat
        p, q = self.manifolds[i].offs_raw[j]
        xmat = xmat[p:q,:]
        nraw = self.manifolds[i].nprods_raw[j]
        if _col is not None:
            xmat = xmat[:,_col]
        return xmat

    def get_mstr_env (self, addr_sn, addr_m, inv):
        m_strs = self.manifolds[addr_sn].m_strs
        nm, nfrags = m_strs.shape
        spec = np.ones (nfrags, dtype=bool)
        spec[np.asarray (inv)] = False
        addr_m = np.atleast_1d (addr_m)
        m_strs = m_strs[np.ix_(addr_m,spec)]
        assert (m_strs.ndim==2), '{} {}'.format (addr_m, spec)
        return m_strs 

    def find_spin_nonvanishing_overlaps (self, bra_sn, ket_sn, m_exc, inv):
        mbra = self.get_mstr_env (bra_sn, m_exc[:,0], inv)
        mket = self.get_mstr_env (ket_sn, m_exc[:,1], inv)
        idx = [np.all (self.get_mstr_env (bra_sn, m_exc[i,0], inv)
                       == self.get_mstr_env (ket_sn, m_exc[i,1], inv),
                       axis=1)[0]
               for i in range (len (m_exc))]
        return np.asarray (idx)

    def pspace_ham_spincoup_dm (self, bra_sn, ket_sn, mtidx_bra, mtidx_ket, sgnvec, inv):
        mstr_bra = self.get_mstr_env (bra_sn, mtidx_bra[0].ravel (), inv)
        mstr_ket = self.get_mstr_env (ket_sn, mtidx_ket[0].ravel (), inv)
        umat_bra = self.manifolds[bra_sn].umat[mtidx_bra]
        umat_ket = self.manifolds[ket_sn].umat[mtidx_ket]
        assert (umat_bra.shape[0] == mstr_bra.shape[0])
        assert (umat_ket.shape[0] == mstr_ket.shape[0])
        sgnvec[np.any (mstr_bra!=mstr_ket, axis=1)] = 0
        dm = (umat_bra.conj ().T @ (sgnvec[:,None] * umat_ket))
        return dm

    def _matvec (self, rawarr):
        is_out_complex = (self.dtype==np.complex128) or np.iscomplexobj (rawarr)
        my_dtype = np.complex128 if is_out_complex else np.float64
        col_shape = rawarr.shape[1:]
        orth_shape = [self.shape[0],] + list (col_shape)
        ortharr = np.zeros (orth_shape, dtype=my_dtype)
        p = 0
        for manifold in self.manifolds:
            prod_idx = manifold.prod_idx
            xmat = manifold.xmat
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
        for manifold in self.manifolds:
            prod_idx = manifold.prod_idx
            xmat = manifold.xmat
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

    def are_tstrs_coupled (self, bra_sn, ket_sn, bra_t, ket_t, inv):
        brastr = self.manifolds[bra_sn].get_t_strs ()
        ketstr = self.manifolds[ket_sn].get_t_strs ()
        spec = np.ones (brastr.shape[-1], dtype=bool)
        spec[np.asarray (inv)] = False
        brastr = brastr.T[spec].T
        ketstr = ketstr.T[spec].T
        brastr = brastr[bra_t]
        ketstr = ketstr[ket_t]
        return np.all (brastr==ketstr, axis=-1)

    def log_debug1_hdiag_raw (self, log, hdiag, idx=None):
        if idx is None:
            idx = np.arange (len (hdiag), dtype=int)
        idx = idx[np.argsort (hdiag[idx])]
        addrs_sn, addrs_m, addrs_p = self.idx2addrs_raw (idx)
        log.debug1 ("Address book for individual raw states:")
        for i in np.sort (np.unique (addrs_sn)):
            man = self.manifolds[i]
            log.debug1 ("%s", man.sprintf_address_book_raw (i))
        log.debug1 ("Raw basis hdiag:")
        log.debug1 ("ix e addr(manifold) addr(spin) addr(spat)")
        for i, ix in enumerate (idx):
            log.debug1 ("%d %15.10e %d %d %d", idx[i], hdiag[ix], addrs_sn[i], addrs_m[i],
                       addrs_p[i])

    def log_debug_hdiag_orth (self, log, hdiag, idx=None):
        if idx is None:
            idx = np.arange (len (hdiag), dtype=int)
        idx = idx[np.argsort (hdiag[idx])]
        addrs_sn, addrs_t, addrs_p = self.idx2addrs_orth (idx)
        log.debug ("Address book for individual orth states:")
        for i in np.sort (np.unique (addrs_sn)):
            man = self.manifolds[i]
            log.debug ("%s", man.sprintf_address_book_orth (i))
        log.debug ("Orth basis hdiag:")
        log.debug ("ix e addr(manifold) addr(spin) addr(spat)")
        for i, ix in enumerate (idx):
            log.debug ("%d %15.10e %d %d %d", idx[i], hdiag[ix], addrs_sn[i], addrs_t[i],
                       addrs_p[i])

class SpinCoupledOrthBasis (OrthBasis):
    def roots_coupled_in_hdiag (self, i, j):
        return self.roots2mans (i) == self.roots2mans (j)

    def rootspaces_covering_addrs (self, addrs):
        blocks = np.searchsorted (self.offs_orth[:,0], addrs, side='right')-1
        manaddrs = self.oblock_manifold_addr[blocks]
        roots = []
        for i in manaddrs[:,0]:
            for m_block in self.manifolds[i].m_blocks:
                roots.extend (m_block)
        return np.atleast_1d (roots).astype (int)

    def hdiag_spincoup_loop (self, iman, mstr_bra, mstr_ket, inv):
        offs0 = 0
        for man in self.manifolds[:iman]:
            offs0 += np.prod (man.orth_shape)
        man = self.manifolds[iman]
        # This indexing should be properly sorted for the dot product below
        # I explicitly sorted the mstrs above
        iblks_bra = man.idx_m_str (mstr_bra, inv)
        iblks_ket = man.idx_m_str (mstr_ket, inv)
        nlsf = man.umat.shape[1]
        ubra = man.umat[iblks_bra,:]
        uket = man.umat[iblks_ket,:]
        ncols = man.orth_shape[1]
        for ilsf in range (nlsf):
            p = offs0 + ilsf*ncols
            q = p + ncols
            spin_fac = np.dot (ubra[:,ilsf], uket[:,ilsf])
            yield spin_fac, (p,q)
        return

    def _matvec (self, rawarr):
        is_out_complex = (self.dtype==np.complex128) or np.iscomplexobj (rawarr)
        my_dtype = np.complex128 if is_out_complex else np.float64
        col_shape = rawarr.shape[1:]
        orth_shape = [self.shape[0],] + list (col_shape)
        ortharr = np.zeros (orth_shape, dtype=my_dtype)
        i = 0
        for manifold in self.manifolds:
            prod_idx = manifold.prod_idx
            xmat = manifold.xmat
            umat = manifold.umat
            uxarr = np.stack ([rawarr[mirror] for mirror in prod_idx], axis=0)
            if xmat is not None:
                uxarr = np.tensordot (xmat, uxarr, axes=((0),(1)))
                uxarr = np.moveaxis (uxarr, 0, 1)
            uxarr = np.tensordot (umat.T, uxarr, axes=1)
            ux_rows = np.prod (manifold.orth_shape)
            uxarr = uxarr.reshape ([ux_rows,] + list (col_shape))
            j = i + ux_rows
            ortharr[i:j] = uxarr[:]
            i = j
        return ortharr

    def _rmatvec (self, ortharr):
        is_out_complex = (self.dtype==np.complex128) or np.iscomplexobj (ortharr)
        my_dtype = np.complex128 if is_out_complex else np.float64
        col_shape = ortharr.shape[1:]
        raw_shape = [self.shape[1],] + list (col_shape)
        rawarr = np.zeros (raw_shape, dtype=my_dtype)
        i = 0
        for manifold in self.manifolds:
            prod_idx = manifold.prod_idx
            xmat = manifold.xmat
            umat = manifold.umat
            ux_rows = np.prod (manifold.orth_shape)
            j = i + ux_rows
            uxarr = ortharr[i:j].reshape (list (manifold.orth_shape) + list (col_shape))
            i = j
            uxarr = np.tensordot (umat, uxarr, axes=1)
            if xmat is not None:
                uxarr = np.tensordot (xmat.conj (), uxarr, axes=((1),(1)))
                uxarr = np.moveaxis (uxarr, 0, 1)
            for mirror, xarr in zip (prod_idx, uxarr):
                rawarr[mirror] = xarr
        return rawarr

    def are_tstrs_coupled (self, bra_sn, ket_sn, bra_t, ket_t, inv):
        brastr = np.cumsum (self.manifolds[bra_sn].get_t_strs (), axis=1)
        ketstr = np.cumsum (self.manifolds[ket_sn].get_t_strs (), axis=1)
        inv = np.arange (brastr.shape[-1], dtype=int)[np.asarray (inv)]
        if len (inv) == 1:
            return bra_t==ket_t
        spec = np.ones (brastr.shape[-1], dtype=bool)
        spec[inv[0]:inv[-1]] = False
        brastr = brastr.T[spec].T[bra_t]
        ketstr = ketstr.T[spec].T[ket_t]
        return np.all (brastr==ketstr, axis=-1)

def get_spincoup_bases (smults_f, spin_lsf=None, smult_lsf=None):
    from mrh.my_pyscf.lassi.spaces import SingleLASRootspace
    norb_f = np.zeros_like (smults_f)
    nelec_f = np.zeros_like (smults_f)
    spins_f = smults_f-1
    if (spin_lsf is None) and (smult_lsf is None):
        smult_lsf = spins_f.sum () + 1
        spin_lsf = smult_lsf - 1
    elif spin_lsf is None:
        spin_lsf = smult_lsf - 1
    elif smult_lsf is None:
        smult_lsf = spin_lsf + 1
    for ifrag in range (len (smults_f)):
        dspin = min (spins_f.sum()-spin_lsf, 2*(smults_f[ifrag]-1))
        assert (dspin >= 0)
        dspin = (dspin // 2) * 2
        spins_f[ifrag] -= dspin
    assert (spins_f.sum () == spin_lsf)
    assert (np.all (np.abs (spins_f)<smults_f))
    assert (np.all (np.divmod (spins_f, 2)[1] == np.divmod (smults_f-1, 2)[1]))
    space = SingleLASRootspace (None, spins_f, smults_f, np.zeros_like (smults_f), None,
                                nlas=norb_f, nelelas=nelec_f, verbose=0, stdout=0, fragsym=0)
    spins_table = space.make_spin_shuffle_table ()
    smult_table = space.make_smult_shuffle_table (smult_lsf)
    return spins_table, smult_table

def make_s2mat (smults_f, spin_lsf):
    spins_table = get_spincoup_bases (smults_f, spin_lsf=spin_lsf)[0]
    nbas, nfrags = spins_table.shape
    s = (smults_f - 1) / 2
    s2mat = (s * (s + 1)).sum () * np.eye (nbas, dtype=float)
    for i, m2 in enumerate (spins_table):
        s2mat[i,i] += 0.25 * ((m2[None,:] * m2[:,None]).sum () - np.dot (m2, m2))
    m2 = spins_table
    m = m2 * .5
    s = s[None,:]
    cg = np.sqrt ((s-m+1)*(s+m))
    for i, j in itertools.combinations (range (nbas), 2):
        dm2 = m2[i] - m2[j]
        if np.count_nonzero (dm2) > 2: continue
        if dm2.sum () != 0: continue
        if np.abs (dm2).sum () != 4: continue
        ifrag = np.where (dm2==2)[0][0]
        jfrag = np.where (dm2==-2)[0][0]
        s2mat[i,j] += cg[i,ifrag] * cg[j,jfrag]
        s2mat[j,i] += cg[i,ifrag] * cg[j,jfrag]
    return s2mat

def get_spincoup_umat (smults_f, spin_lsf, smult_lsf):
    spins_table, smult_table = get_spincoup_bases (smults_f, spin_lsf=spin_lsf, smult_lsf=smult_lsf)
    from sympy import S
    from sympy.physics.quantum.cg import CG
    gencoup_table = np.cumsum (smult_table, axis=1)
    spinsum_table = np.cumsum (spins_table, axis=1)
    nfrags = len (smults_f)
    nunc = spins_table.shape[0]
    nlsf = smult_table.shape[0]
    umat = np.ones ((nunc, nlsf), dtype=float)
    for i in range (1,nfrags):
        si = S(int(smults_f[i])-1)/2
        for j in range (nunc):
            m0 = S(int(spinsum_table[j,i-1]))/2
            m1 = S(int(spinsum_table[j,i]))/2
            mi = S(int(spins_table[j,i]))/2
            assert (abs (mi) <= si)
            for k in range (nlsf):
                s0 = S(int(gencoup_table[k,i-1]))/2
                s1 = S(int(gencoup_table[k,i]))/2
                assert ((s0 + si) >= s1)
                assert ((m0 + mi) == m1)
                umat[j,k] *= float (CG (s0,m0,si,mi,s1,m1).doit ())
    return umat

