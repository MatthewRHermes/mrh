import numpy as np
from mrh.util import bigdim
from mrh.my_pyscf.lassi.citools import umat_dot_1frag_
from mrh.my_pyscf.lassi.op_o0 import civec_spinless_repr

def fermion_spin_shuffle (na_list, nb_list):
    ''' Compute the sign factor corresponding to the convention
        difference between

        ... a2' a1' a0' ... b2' b1' b0' |vac>

        and

        ... a2' b2' a1' b1' a0' b0' |vac>

        where subspaces 0, 1, 2, etc. have arbitrary numbers of spin-up
        and spin-down electrons 

        Args:
            na: list of up-spin electrons for each subspace
            nb: list of down-spin electrons for each subspace

        Returns:
            sgn: +-1
    '''
    assert (len (na_list) == len (nb_list))
    nperms = 0
    for ix, nb in enumerate (nb_list[1:]):
        na = sum(na_list[:ix+1])
        nperms += na * nb
    return (1,-1)[nperms%2]

def fermion_frag_shuffle (nelec_f, frag_list):
    ''' Compute the sign factor associated with the isolation of
        particular fragments in a product of fermion field operators;
        i.e., the difference between

        ... c2' ... c1' ... c0' ... |vac>

        and

        ... c2' c1' c0' ... |vac>  

        Args:
            nelec_f: list of electron numbers per fragment for the
                whole state
            frag_list: list of fragments to coalesce

        Returns:
            sgn: +- 1
    '''

    frag_list = list (set (frag_list))
    nperms = 0
    nbtwn = 0
    for ix, frag in enumerate (frag_list[1:]):
        lfrag = frag_list[ix]
        if (frag - lfrag) > 1:
            nbtwn += sum ([nelec_f[jx] for jx in range (lfrag+1,frag)])
        if nbtwn:
            nperms += nelec_f[frag] * nbtwn
    return (1,-1)[nperms%2]

def fermion_des_shuffle (nelec_f, frag_list, i):
    ''' Compute the sign factor associated with anticommuting a destruction
        operator past creation operators of unrelated fragments, i.e.,    
        
        ci ... cj' ci' ch' .. |vac> -> ... cj' ci ci' ch' ... |vac>

        Args:
            nelec_f: list of electron numbers per fragment for the whole state
            frag_list: list of fragment numbers actually involved in a given
                transfer; i.e., the argument 'frag_list' of a recent call to
                fermion_frag_shuffle
            i: fragment of the destruction operator to commute foward

        Returns:
            sgn: +- 1
        
    '''
    assert (i in frag_list)
    # Assuming that low orbital indices touch the vacuum first,
    # the destruction operator commutes past the high-index field
    # operators first -> reverse the order of frag_list
    frag_list = list (set (frag_list))[::-1]
    nelec_f = [nelec_f[ix] for ix in frag_list]
    i = frag_list.index (i)
    nperms = sum (nelec_f[:i]) if i else 0
    return (1,-1)[nperms%2]

def lst_hopping_index (nelec_frs):
    ''' Build the LAS state transition hopping index

        Args:
            nelec_frs : ndarray of shape (nfrags,nroots,2)
                Number of electrons of each spin in each rootspace in each
                fragment

        Returns:
            hopping_index: ndarray of ints of shape (nfrags, 2, nroots, nroots)
                element [i,j,k,l] reports the change of number of electrons of
                spin j in fragment i between LAS rootspaces k and l
            zerop_index: ndarray of bools of shape (nroots, nroots)
                element [i,j] is true where the ith and jth LAS spaces are
                connected by a null excitation; i.e., no electron, pair,
                or spin hopping or pair splitting/coalescence. This implies
                nonzero 1- and 2-body transition density matrices within
                all fragments.
            onep_index: ndarray of bools of shape (nroots, nroots)
                element [i,j] is true where the ith and jth LAS spaces
                are connected by exactly one electron hop from i to j or vice
                versa, implying nonzero 1-body transition density matrices
                within spectator fragments and phh/pph modes within
                source/dest fragments.
    '''
    nelec_fsr = nelec_frs.transpose (0,2,1)
    hopping_index = np.array ([[np.subtract.outer (spin, spin)
        for spin in frag] for frag in nelec_fsr])
    symm_index = np.all (hopping_index.sum (0) == 0, axis=0)
    zerop_index = symm_index & (np.count_nonzero (hopping_index, axis=(0,1)) == 0)
    onep_index = symm_index & (np.abs (hopping_index).sum ((0,1)) == 2)
    return hopping_index, zerop_index, onep_index

def get_contig_blks (mask):
    '''Get contiguous chunks from a mask index into an array'''
    mask = np.ravel (mask)
    prevm = np.append ([False], mask[:-1])
    destblkstart = np.asarray (np.where (mask & (~prevm))[0])
    nextm = np.append (mask[1:], [0])
    destblkend = np.asarray (np.where (mask & (~nextm))[0])
    blklen = destblkend - destblkstart + 1
    srcblkstart = np.cumsum (np.append ([0], blklen[:-1]))
    return srcblkstart, destblkstart, blklen

def split_contig_array (arrlen, nthreads):
    '''Divide a contiguous array into chunks to be handled by each thread'''
    nthreads = min (arrlen, nthreads)
    blklen, rem = divmod (arrlen, nthreads);
    blklen = np.array ([blklen,]*nthreads)
    blklen[:rem] += 1
    blkstart = np.cumsum (blklen)
    assert (blkstart[-1] == arrlen), '{}'.format (blklen)
    blkstart -= blklen
    return blkstart, blklen

def ci_map2spinless (ci0_fr, norb_f, nelec_frs):
    ''' Map CI vectors to the spinless representation, preserving references to save memory '''
    nfrags, nroots = nelec_frs.shape[:2]

    # Only transform unique CI vectors
    ci_ptrs = np.asarray ([[c.__array_interface__['data'][0] for c in ci_r] for ci_r in ci0_fr])
    _, idx, inv = np.unique (ci_ptrs, return_index=True, return_inverse=True)
    inv = inv.reshape (ci_ptrs.shape)
    ci1 = []
    for ix in idx:
        i, j = divmod (ix, nroots)
        if isinstance (ci0_fr[i][j], (list,tuple)) or ci0_fr[i][j].ndim>2:
            ci0 = ci0_fr[i][j]
        else:
            ci0 = [ci0_fr[i][j],]
        nelec = [tuple (nelec_frs[i][j]),]*len(ci0)
        ci1.append (civec_spinless_repr (ci0, norb_f[i], nelec))

    return [[ci1[inv[i,j]] for j in range (nroots)] for i in range (nfrags)]

def spin_shuffle_idx (norb_f):
    '''Obtain the index vector to take an orbital-basis array from fragment-major to spin-major
    order

    Args:
        norb_f : sequence of int
            Number of spatial orbitals in each fragment

    Returns:
        idx : list of int
            Indices of spinorbitals in fragment-major order rearranged to spin-major order
    '''
    norb_f = np.asarray (norb_f)
    off = np.cumsum (norb_f)
    idxa = []
    idxb = []
    for i,j in zip (off, norb_f):
        idxa.extend (list (range (2*(i-j),2*i-j)))
        idxb.extend (list (range (2*i-j,2*i)))
    return idxa + idxb

def transpose_sivec_make_fragments_slow (vec, lroots, *inv):
    '''A single-rootspace slice of the SI vectors, transposed so that involved fragments
    are slower-moving

    Args:
        vec: col-major ndarray of shape (np.prod (lroots), nroots_si)
            Single-rootspace sivec
        lroots: ndarray of shape (nfrags)
            Number of fragment states
        *inv: integers 
            Indices of nonspectator fragments, which are placed to the right in column-major order.

    Returns:
        vec: ndarray of shape (nroots_si, nrows, ncols)
            SI vectors with the faster dimension iterating over states of fragments not in
            inv and the slower dimension iterating over states of fragments in inv 
    '''
    nfrags = len (lroots)
    nprods = np.prod (lroots)
    nroots_si = vec.size // nprods
    assert ((vec.size % nprods) == 0), 'lroots does not match vec size'
    nrows = np.prod (lroots[list (inv)])
    ncols = nprods // nrows
    axes = [i for i in range (nfrags) if i not in inv] + list (inv) + [nfrags,]
    shape = list (lroots) + [nroots_si,]
    vec = bigdim.transpose (vec, shape=shape, axes=axes, order='F')
    return vec.reshape ((ncols, nrows, nroots_si), order='F').T

def transpose_sivec_with_slow_fragments (vec, lroots, *inv):
    '''The inverse operation of transpose_sivec_make_fragments_slow.

    Args:
        vec: col-major ndarray of shape (ncols, nrows, nroots_si)
            Single-rootspace sivec, where nrows are the dimensions of the fragments inv
        lroots: ndarray of shape (nfrags)
            Number of fragment states
        *inv: integers 
            Indices of slow-moving fragments, assuming they have been placed in column-major order.

    Returns:
        vec: ndarray of shape (nroots_si, np.prod (lroots))
            SI vectors in canonical fragment order
    '''
    nfrags = len (lroots)
    nprods = np.prod (lroots)
    axes = [i for i in range (nfrags) if i not in inv] + list (inv) + [nfrags,]
    rdim = nroots_si = vec.size // nprods
    assert ((vec.size % nprods) == 0), 'lroots does not match vec size'
    shape = list (lroots[axes[:-1]]) + [nroots_si,]
    axes = np.argsort (axes)
    vec = bigdim.transpose (vec, shape=shape, axes=axes, order='F')
    return vec.reshape ((nprods, nroots_si), order='F').T



