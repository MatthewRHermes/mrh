import numpy as np
from scipy import linalg
from pyscf import lib

# Let's finally implement, in the more pure LASSCF rewrite, the ERI-
# related cost savings that I made such a big deal about in JCTC 2020,
# 16, 4923

def make_schmidt_spaces (h_op):
    ''' Build the spaces which active orbitals will explore in this
    macrocycle, based on gradient and Hessian SVDs and Schmidt
    decompositions. Must be called after __init__ is complete

    Args:
        h_op: LASSCF Hessian operator instance

    Returns:
        uschmidt : list of ndarrays of shape (nmo, *)
            The number of subspaces built is equal to the
            product of the number of irreps in the molecule
            and the number of fragments, minus the number
            of null spaces.

    '''
    las = h_op.las
    ugg = h_op.ugg
    mo_coeff = h_op.mo_coeff
    nao, nmo = mo_coeff.shape
    ncore, ncas = las.ncore, las.ncas
    nocc = ncore + ncas
    dm1 = h_op.dm1s.sum (0)
    g_vec = h_op.get_grad ()
    prec_op = h_op.get_prec ()
    hx_vec = prec_op (-g_vec)
    gorb1 = ugg.unpack (g_vec)[0]
    gorb2 = gorb1 + ugg.unpack (hx_vec)[0]

    def _svd (metric, q, p):
        m, n = q.shape[1], p.shape[1]
        k = min (m, n)
        qh = q.conj ().T
        u, svals, vh = linalg.svd (qh @ metric @ p)
        idx_sort = np.argsort (-np.abs (svals))
        svals = svals[idx_sort]
        u[:,:k] = u[:,idx_sort]
        return q @ u, svals

    def _check (tag, umat_p, umat_q):
        np, nq = umat_p.shape[1], umat_q.shape[1]
        k = min (np, nq)
        lib.logger.debug (h_op, '%s size of pspace = %d, qspace = %d', tag, np, nq)
        return k

    def _grad_svd (tag, geff, umat_p, umat_q, ncoup=0):
        umat_q = _svd (geff, umat_q, umat_p)[0]
        umat_p = np.append (umat_p, umat_q[:,:ncoup], axis=1)
        umat_q = umat_q[:,ncoup:]
        k = _check ('after {} SVD'.format (tag), umat_p, umat_q)
        return umat_p, umat_q, k

    def _schmidt (tag, umat_p, umat_q, thresh=1e-8):
        umat_q, svals = _svd (dm1, umat_q, umat_p)
        ncoup = np.count_nonzero (np.abs (svals) > thresh)
        umat_p = np.append (umat_p, umat_q[:,:ncoup], axis=1)
        umat_q = umat_q[:,ncoup:]
        dm_pp = umat_p.conj ().T @ dm1 @ umat_p
        lib.logger.debug (h_op, 'number of electrons in p-space after %s Schmidt = %e', tag, np.trace (dm_pp))
        k = _check ('after {} Schmidt'.format (tag), umat_p, umat_q)
        return umat_p, umat_q, k

    def _make_single_space (mask):
        # Active orbitals other than the current should be in
        # neither the p-space nor the q-space
        # This is because I am still assuming we have all 
        # eris of type aaaa
        nlas = np.count_nonzero (mask)
        umat_p = np.diag (mask.astype (mo_coeff.dtype))[:,mask]
        umat_q = np.eye (nmo)
        umat_q = np.append (umat_q[:,:ncore], umat_q[:,nocc:])
        # At any of these steps we might run out of orbitals...
        k = _check ('initial', umat_p, umat_q)
        if k == 0: return umat_p
        umat_p, umat_q, k = _grad_svd ('g', gorb1, umat_p, umat_q, ncoup=k)
        if k == 0: return umat_p
        umat_p, umat_q, k = _schmidt ('first', umat_p, umat_q)
        if k == 0: return umat_p
        umat_p, umat_q, k = _grad_svd ('g+hx', gorb2, umat_p, umat_q, ncoup=min(k,3*nlas))
        if k == 0: return umat_p
        umat_p, umat_q, k = _schmidt ('second', umat_p, umat_q)
        return umat_p

    orbsym = getattr (mo_coeff, 'orbsym', np.zeros (nmo))
    uschmidt = []
    for ilas in range (len (las.ncas_sub)):
        i = sum (las.ncas_sub[:ix]) + ncore
        j = i + las.ncas_sub[ix]
        irreps, idx_irrep = np.unique (orbsym[i:j], return_inverse=True)
        for ix in range (len (irreps)):
            idx = np.squeeze (idx_irrep==ix) + i
            idx_mask = np.zeros (nmo, dtype=np.bool_)
            idx_mask[idx] = True
            uschmidt.append (_make_single_space (idx_mask))

    return uschmidt


