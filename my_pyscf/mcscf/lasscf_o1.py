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
    gorb = ugg.unpack (g_vec)[0]
    hxorb = ugg.unpack (hx_vec)[0]

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

    def _make_single_space (mask):
        # Active orbitals other than the current should be in
        # neither the p-space nor the q-space
        # This is because I am still assuming we have all 
        # eris of type aaaa
        nlas = np.count_nonzero (mask)
        umat_p = np.diag (mask.astype (mo_coeff.dtype))[:,mask]
        umat_q = np.eye (nmo)
        umat_q = np.append (umat_q[:,:ncore], umat_q[:,nocc:])
        k = _check ('initial', umat_p, umat_q)
        if k == 0: return umat_p
        # Initial gradient SVD
        umat_q = _svd (gorb, umat_q, umat_p)[0]
        umat_p = np.append (umat_p, umat_q[:,:k], axis=1)
        umat_q = umat_q[:,k:]
        k = _check ('after gradient SVD', umat_p, umat_q)
        if k == 0: return umat_p
        # First Schmidt decomposition
        umat_q, svals = _svd (dm1, umat_q, umat_p)
        ncoup = np.count_nonzero (np.abs (svals)>1e-8)
        umat_p = np.append (umat_p, umat_q[:,:ncoup], axis=1)
        umat_q = umat_q[:,ncoup:]
        dm_pp = umat_p.conj ().T @ dm1 @ umat_p
        lib.logger.debug (h_op, 'number of electrons in p-space after first Schmidt = %e', np.trace (dm_pp))
        k = _check ('after first Schmidt', umat_p, umat_q)
        if k == 0: return umat_p
        # (g + H.x)_qp SVD 
        umat_q = _svd (gorb, umat_q, umat_p)[0]
        ncoup = min (k, 3*nlas)
        umat_p = np.append (umat_p, umat_q[:,:ncoup], axis=1)
        umat_q = umat_q[:,ncoup:]
        k = _check ('after g+H.x SVD', umat_p, umat_q)
        if k == 0: return umat_p
        # Second Schmidt decomposition
        umat_q, svals = _svd (dm1, umat_q, umat_p)
        ncoup = np.count_nonzero (np.abs (svals)>1e-8)
        umat_p = np.append (umat_p, umat_q[:,:ncoup], axis=1)
        umat_q = umat_q[:,ncoup:]
        dm_pp = umat_p.conj ().T @ dm1 @ umat_p
        lib.logger.debug (h_op, 'number of electrons in p-space after second Schmidt = %e', np.trace (dm_pp))
        _check ('after second Schmidt', umat_p, umat_q)
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


