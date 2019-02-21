import numpy as np
import scipy
from mrh.util import params

# A collection of simple manipulations of matrices that I somehow can't find in numpy

def is_matrix_zero (test_matrix, rtol=params.num_zero_rtol, atol=params.num_zero_atol):
    test_zero = np.zeros (test_matrix.shape, dtype=test_matrix.dtype)
    return np.allclose (test_matrix, test_zero, rtol=rtol, atol=atol)

def is_matrix_eye (test_matrix, matdim=None, rtol=params.num_zero_rtol, atol=params.num_zero_atol):
    if (test_matrix.shape[0] != test_matrix.shape[1]):
        return False
    test_eye = np.eye (test_matrix.shape[0], dtype=test_matrix.dtype)
    return np.allclose (test_matrix, test_eye, atol=atol, rtol=rtol)

def is_matrix_idempotent (test_matrix, rtol=params.num_zero_rtol, atol=params.num_zero_atol):
    if (test_matrix.shape[0] != test_matrix.shape[1]):
        return False
    test_m2 = np.dot (test_matrix, test_matrix)
    return np.allclose (test_matrix, test_m2, atol=atol, rtol=rtol)

def is_matrix_diagonal (test_matrix, rtol=params.num_zero_rtol, atol=params.num_zero_atol):
    test_diagonal = np.diag (np.diag (test_matrix))
    return np.allclose (test_matrix, test_diagonal, atol=atol, rtol=rtol)

def is_matrix_hermitian (test_matrix, rtol=params.num_zero_rtol, atol=params.num_zero_atol):
    test_adjoint = np.transpose (np.conjugate (test_matrix))
    return np.allclose (test_matrix, test_adjoint, atol=atol, rtol=rtol)

def assert_matrix_square (test_matrix, matdim=None):
    if (matdim == None):
        matdim = test_matrix.shape[0]
    assert ((test_matrix.ndim == 2) and (test_matrix.shape[0] == matdim) and (test_matrix.shape[1] == matdim)), "Matrix shape is {0}; should be ({1},{1})".format (test_matrix.shape, matdim)
    return matdim

def matrix_svd_control_options (the_matrix, full_matrices=False, sort_vecs=-1, only_nonzero_vals=False,
    num_zero_rtol=params.num_zero_rtol, num_zero_atol=params.num_zero_atol):
    if 0 in the_matrix.shape:
        M = the_matrix.shape[0]
        N = the_matrix.shape[1]
        K = min (M,N)
        if full_matrices: return np.zeros ((M,M)), np.zeros ((K)), np.zeros ((N,N))
        return np.zeros ((M,K)), np.zeros ((K)), np.zeros ((N,K))
    pMq = np.asarray (the_matrix)
    lvecs_pl, svals_lr, rvecs_rq = scipy.linalg.svd (np.asarray (the_matrix), full_matrices=full_matrices)
    p2l = lvecs_pl
    r2q = rvecs_rq
    q2r = r2q.conjugate ().T
    if sort_vecs:
        idx_sval = (np.abs (svals_lr)).argsort ()[::sort_vecs]
        idx_q2r = np.append (idx_sval, np.arange (len (idx_sval), q2r.shape[1], dtype=idx_sval.dtype))
        idx_p2l = np.append (idx_sval, np.arange (len (idx_sval), p2l.shape[1], dtype=idx_sval.dtype))
        svals_lr = svals_lr[idx_sval]
        q2r = q2r[:,idx_q2r]
        p2l = p2l[:,idx_p2l]
    if only_nonzero_vals:
        idx = np.where (np.abs (svals_lr) > num_zero_atol)[0]
        svals_lr = svals_lr[idx]
        q2r = q2r[:,idx]
        p2l = p2l[:,idx]

    lvecs, svals_lr, rvecs = (np.asarray (output) for output in (p2l, svals_lr, q2r))
    return lvecs, svals_lr, rvecs

def matrix_eigen_control_options (the_matrix, b_matrix=None, symmetry=None, strong_symm=False, 
    subspace=None, subs_symm=None, sort_vecs=-1, only_nonzero_vals=False, round_zero_vals=False, 
    num_zero_atol=params.num_zero_atol, num_zero_rtol=params.num_zero_rtol):
    ''' Diagonalize a matrix using scipy's driver and also a whole lot of pre-/post-processing,
        most significantly sorting, throwing away numerical null spaces, 
        subspace projections, and symmetry alignments.
        
        Args:
            the_matrix: square np.ndarray with M rows

        Kwargs:
            b_matrix: square np.ndarray with M rows
                The second matrix for the generalized eigenvalue problem
            symmetry: list of block labels of length M
            or list of non-square matrices of shape (M,P), sum_P = M
            or None
                Formal symmetry information. Neither the matrix nor the
                subspace need to be symmetry adapted, and unless strong_symm=True,
                symmetry is used only to define a gauge convention within
                degenerate manifolds. Orthonormal linear combinations of degenerate
                eigenvectors with the highest possible projections onto any symmetry
                block are sequentially generated using repeated singular value decompositions
                in successively smaller subspaces. Eigenvectors within degenerate
                manifolds are therefore sorted from least to most symmetry-breaking;
                ordering within quantitatively symmetry-adapted eigenvectors
                (i.e. sum_{symm_u} <symm_u|u> == 1) is currently arbitrary.
            strong_symm: logical
                If true, the actual diagonalization is carried out symmetry-blockwise.
                Requires symmetry. Eigenvectors will be symmetry-adapted but this does not
                check that the whole matrix is actually diagonalized by them so user beware!
                Extra risky if a subspace is used because the vectors of the subspace
                are assigned to symmetry blocks in the same way as degenerate eigenvectors,
                and the symmetry labels of the final eigenvectors are inherited from
                the corresponding subspace symmetry block without double-checking.
                (Subspace always outranks symmetry.)
            subspace: index list for accessing Mprime elements array of shape (M,)
            or ndarray of shape (M,Mprime)
            or None
                Defines a subspace in which the matrix is diagonalized. Note
                that symmetry is applied to the matrix, not the subspace states.
                Subspace always outranks symmetry, meaning that the eigenvectors are
                guaranteed within round-off error to be contained within the
                subspace, but using a subspace may decrease the reliability
                of symmetry assignments, even if strong_symm==True.
            subs_symm: list of block labels of length Mprime
            or list of non-square matrices of shape (Mprime,P), sum_P = Mprime
            or None
                To be implemented: explicit symmetry information for the subspace
            sort_vecs: integer
                Defines overall sorting of non-degenerate eigenvalues. -1 means from largest
                to smallest; +1 means from smallest to largest. Play with other values at
                your peril
            only_nonzero_vals: logical
                If true, only the K <= M nonzero eigenvalues and corresponding eigenvectors
                are returned
            round_zero_vals: logical
                If true, sets all eigenvalues of magnitude less than num_zero_atol to identically zero
            num_zero_atol: float
                Absolute tolerance for numpy's "isclose" function and its relatives.
                Used in determining what counts as a degenerate manifold.
            num_zero_rtol: float
                Relative tolerance for numpy's "isclose" function and its relatives.
                Used in determining what counts as a degenerate manifold.

        Returns:
            evals: ndarray of shape (K,); K<=M (see only_nonzero_vals kwarg)
            evecs: ndarray of shape (M,K); K<=M (see only_nonzero_vals kwarg)
                If a subspace is specified, the eigenvectors are transformed back into the original
                basis before returning
            labels: list of length K; K<=M
                Only returned if symmetry is not None. Identifies symmetry block
                with the highest possible projection onto each eigenvector unless
                strong_symm==True, in which case the labels are derived
                from labels of subspace vectors computed in this way.. Does not
                guarantee that the eigenvector is symmetry-adapted.
                
            

    '''

    if subs_symm is not None:
        raise NotImplementedError ("Explicit subspace symmetry information.")

    if the_matrix.shape == tuple((0,0)):
        return np.zeros ((0)), np.zeros ((0,0))
    subspace = None if subspace is None else np.asarray (subspace)
    subspace_isvectorblock = False if subspace is None else subspace.ndim == 2

    # Prevent wasting time. Matt, you should gradually remove these lines to test edge-case tolerance
    if symmetry is not None and len (symmetry) == 1: symmetry = False
    if (symmetry is None) and strong_symm: raise RuntimeError ("Needs symmetry information or strong_symm == False!")
    if (symmetry is False): strong_symm = False
    if subspace is not None and subspace.shape[int (subspace_isvectorblock)] == the_matrix.shape[1]: subspace = None
    symm_isvectorblock = False if symmetry is False or symmetry is None else isinstance (symmetry[0], np.ndarray)

    # If symmetry information is provided as a vector block, transform into a symmetry-adapted basis and recurse
    if symm_isvectorblock:
        symm_umat = np.concatenate (symmetry, axis=1)
        assert (symm_umat.shape == the_matrix.shape), "I can't guess how to map symmetry blocks to different bases! Matt fix this bug!"
        symm_lbls = np.concatenate ([idx * np.ones (blk.shape[1], dtype=int) for idx, blk in enumerate (symmetry)])
        symm_matr = symm_umat.conjugate ().T @ the_matrix @ symm_umat
        symm_bmat = symm_umat.conjugate ().T @ b_matrix @ symm_umat if b_matrix is not None else None
        if subspace is not None:
            # Since symm_isidx == False, I have to turn the subspace into a vector block too! Dang!
            if subspace_isvectorblock: symm_subs = symm_umat.conjugate ().T @ subspace
            else: symm_subs = symm_umat.conjugate ().T [:,subspace]
        # Be 200% sure that this recursion can't trigger this conditional block again!
        assert (not (isinstance (symm_lbls[0], np.ndarray))), 'Infinite recursion detected! Fix this bug!'
        evals, symm_evecs, labels = matrix_eigen_control_options (symm_matr, symmetry=symm_lbls, strong_symm=strong_symm,
            subspace=symm_subs, sort_vecs=sort_vecs, only_nonzero_vals=only_nonzero_vals, round_zero_vals=round_zero_vals,
            b_matrix=symm_bmat, num_zero_atol=num_zero_atol, num_zero_rtol=num_zero_rtol)
        evecs = symm_umat @ symm_evecs
        return evals, evecs, labels

    # Recurse from strong symmetry enforcement to diagonalization over individual symmetry blocks
    if strong_symm:

        # If a subspace is being diagonalized and it's provided as a vector block, align the subspace vectors with the symmetry
        # Include an option to raise an error if the subspace isn't symmetry-adapted to within a tolerance: see align_vecs
        subs_symm_lbls = subspace
        if subspace is not None and subspace_isvectorblock:
            subspace, subs_symm_lbls = align_vecs (subspace, symmetry, rtol=num_zero_rtol, atol=num_zero_atol)

        # Recurse into diagonalization of individual symmetry blocks
        uniq_lbls = np.unique (symmetry)
        evals = []
        evecs = []
        labels = []
        for lbl in uniq_lbls:
            if subspace is not None and np.count_nonzero (subs_symm_lbls == lbl) == 0:
                continue
            idx_blk = np.ix_(symmetry == lbl, symmetry == lbl)
            mat_blk = the_matrix[idx_blk]
            b_blk = b_matrix[idx_blk] if b_matrix is not None else None
            subs_blk = None
            if sub_symm_lbls is not None:
                subs_blk = (subs_symm_lbls==lbl)
                if subspace_isvectorblock: subs_blk = subspace[:,subs_blk]
            # Be 200% sure that this recursion can't trigger this conditional block again!
            evals_blk, _evecs_blk = matrix_eigen_control_options (mat_blk, symmetry=None, strong_symm=False,
                sort_vecs=sort_vecs, only_nonzero_vals=only_nonzero_vals, round_zero_vals=round_zero_vals,
                b_matrix=b_blk, num_zero_atol=num_zero_atol, subspace=subs_blk)
            evecs_blk = np.zeros ((the_matrix.shape[0], _evecs_blk.shape[1]), dtype=_evecs_blk.dtype)
            evecs_blk[symmetry == lbl,:] = _evecs_blk
            evals.append (evals_blk)
            evecs.append (evecs_blk)
            labels.extend ([lbl for ix in range (len (evals_blk))])
        evals = np.concatenate (evals)
        evecs = np.concatenate (evecs, axis=1)
        labels = np.asarray (labels)
        if sort_vecs:
            idx = evals.argsort ()[::sort_vecs]
            evals = evals[idx]
            evecs = evecs[:,idx]
            labels = labels[idx]
        return evals, evecs, labels

    # Wrap in subspace projection. This should be triggered if and only if strong_symm==False!
    if subspace is not None:
        assert (not strong_symm)
        if subspace_isvectorblock:
            the_matrix = subspace.conjugate ().T @ the_matrix @ subspace
            b_matrix   = subspace.conjugate ().T @ b_matrix @ subspace if b_matrix is not None else None
        else:
            idx = np.ix_(subspace,subspace)
            ndim_full = the_matrix.shape[0]
            the_matrix = the_matrix[idx]
            b_matrix   = b_matrix[idx] if b_matrix is not None else None

    # Now for the actual damn kernel            
    # Subtract a diagonal average from the matrix to fight rounding error
    diag_avg = np.eye (the_matrix.shape[0]) * np.mean (np.diag (the_matrix))
    pMq = np.asmatrix (the_matrix - diag_avg)
    qSr = None if b_matrix is None else np.asmatrix (b_matrix)
    # Use hermitian diagonalizer if possible and don't do anything if the matrix is already diagonal
    evals = np.diagonal (pMq)
    evecs = np.asmatrix (np.eye (len (evals), dtype=evals.dtype))
    if not is_matrix_diagonal (pMq):
        evals, evecs = scipy.linalg.eigh (pMq, qSr) if is_matrix_hermitian (pMq) else scipy.linalg.eig (pMq, qSr)
    # Add the diagonal average to the eigenvalues when returning!
    evals = evals + np.diag (diag_avg)
    if only_nonzero_vals:
        idx = np.where (np.abs (evals) > num_zero_atol)[0]
        evals = evals[idx]
        evecs = evecs[:,idx]
    if sort_vecs:
        idx = evals.argsort ()[::sort_vecs]
        evals = evals[idx]
        evecs = evecs[:,idx]
    if round_zero_vals:
        idx = np.where (np.abs (evals) < num_zero_atol)[0]
        evals[idx] = 0
    evals, evecs = (np.asarray (output) for output in (evals, evecs))

    # Wrap out subspace projection. This should be triggered if and only if strong_symm==False!
    if subspace is not None:
        assert (not strong_symm)
        if subspace_isvectorblock:
            evecs = subspace @ evecs
        else:
            subs_evecs = evecs.copy ()
            evecs = np.zeros ((ndim_full, subs_evecs.shape[1]), dtype=subs_evecs.dtype)
            evecs[subspace,:] = subs_evecs

    # ~Weak~ symmetry sorting part
    if symmetry is not False and symmetry is not None:
        assert (not isinstance (symmetry[0], np.ndarray)), "Shouldn't have been able to get here with a vector-block format of symmetry data! Matt fix this bug!"
        uniq_labels = np.unique (symmetry)
        idx_unchk = np.ones (len (evals), dtype=np.bool_)
        labels = np.empty (len (evals), dtype=uniq_labels.dtype)
        while np.count_nonzero (idx_unchk > 0):
            chk_1st_eval = evals[idx_unchk][0]
            idx_degen = np.isclose (evals, chk_1st_eval, rtol=num_zero_rtol, atol=num_zero_atol)
            if np.count_nonzero (idx_degen) > 1:
                evecs[:,idx_degen], labels[idx_degen] = align_vecs (evecs[:,idx_degen], symmetry, atol=num_zero_atol, rtol=num_zero_rtol)
            else:
                symmweight = [np.square (evecs[symmetry==lbl,idx_degen]).sum () for lbl in uniq_labels]
                labels[idx_degen] = uniq_labels[np.argmax (symmweight)]
            idx_unchk[idx_degen] = False
        return evals, evecs, labels
    if symmetry is None:
        return evals, evecs
    return evals, evecs, None

def align_vecs (vecs, row_labels, rtol=params.num_zero_rtol, atol=params.num_zero_atol, assert_tol=0):
    col_labels = np.empty (vecs.shape[1], dtype=row_labels.dtype)
    uniq_labels = np.unique (row_labels)
    i = 0
    while i < vecs.shape[1]:
        svdout = [matrix_svd_control_options (vecs[row_labels==lbl,i:], sort_vecs=-1,
            only_nonzero_vals=False, full_matrices=True) for lbl in uniq_labels]
        # This argmax identifies the single best irrep assignment possible for all of vecs[:,i:]
        symm_label_idx = np.argmax ([sval[0] for lvec, sval, rvec in svdout])
        # lvecs and svals are now useless but I write this out just to make sure that I
        # am remembering the return signature of svd correctly
        lvecs, svals, rvecs = svdout[symm_label_idx]
        j = i + np.count_nonzero (np.isclose (svals, svals[0], atol=atol, rtol=rtol))
        col_labels[i:j] = uniq_labels[symm_label_idx]
        if assert_tol and not np.all (np.isclose (svals, 0, atol=assert_tol, rtol=rtol) | np.isclose (svals, 1, atol=assert_tol, rtol=rtol)):
            raise RuntimeError ('Vectors not block-adapted in space {}; svals = {}'.format (col_labels[i], svals))
        # This puts the best-aligned vector at position i and causes all of vecs[:,i+1:] to be orthogonal to it
        vecs[:,i:] = vecs[:,i:] @ rvecs
        assert (j > i)
        i = j
        # This is a trick to grab a whole bunch of degenerate vectors at once (i.e., numerically symmetry-adapted vectors with sval = 1)
        # It may improve numerical stability
    return vecs, col_labels


def assign_blocks_weakly (the_states, the_blocks):
    projectors = [blk @ blk.conjugate ().T for blk in the_blocks]
    vals = np.stack ([((proj @ the_states) * the_states).sum (0) for proj in projectors], axis=-1)
    return np.argmax (vals, axis=1)





