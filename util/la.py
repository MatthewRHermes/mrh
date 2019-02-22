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

def matrix_svd_control_options (the_matrix, full_matrices=False, only_nonzero_vals=False, sort_vecs=-1,
    lspace=None, rspace=None, lsymm=None, rsymm=None, symmetry=None,
    num_zero_rtol=params.num_zero_rtol, num_zero_atol=params.num_zero_atol):
    ''' Perform SVD of a matrix using scipy's linalg driver with a lot of pre-/post-processing

        Args:
            the_matrix: ndarray of shape (M,N)

        Kwargs:
            full_matrices: logical
                If true, lvecs and rvecs include the null space and have shapes (M,M) and (N,N).
                Otherwise, lvecs and rvecs omit the null space and have shapes (M,K) and (N,K).
            only_nonzero_vals: logical
                If true, the formaal [max (M,N) - K] and numerical (sval == 0)
                null spaces are both omitted from the returned singular values
                and possibly (depending on the value of full_matrices)
                the left- and right-singular vectors: K = count_nonzero (svals).
                Otherwise: only the formal null space is omitted: K = min (M,N).
            sort_vecs: integer
                Defines overall sorting of non-degenerate eigenvalues. -1 means from largest
                to smallest; +1 means from smallest to largest. Play with other values at
                your peril
            lspace: index list for accessing Mprime elements array of shape (M,)
            or ndarray of shape (M,Mprime)
            or None
                Defines a subspace for the rows in which SVD is performed.
            rspace: index list for accessing Nprime elements array of shape (N,)
            or ndarray of shape (N,Nprime)
            or None
                Defines a subspace for the columns in which SVD is performed.
            num_zero_atol: float
                Absolute tolerance for numpy's "isclose" function and its relatives.
            num_zero_rtol: float
                Relative tolerance for numpy's "isclose" function and its relatives.

        Returns:
            lvecs: ndarray of shape (M,M) or (M,K), K <= M (see full_matrices and only_nonzero_vals kwargs)
                If a subspace is specified, the eigenvectors are transformed back into the original
                basis before returning (Mprime -> M)
            svals: ndarray of shape (K,), K <= min (M,N) (see only_nonzero_vals kwarg)
            rvecs: ndarray of shape (N,N) or (N,K), K <= N (see full_matrices and only_nonzero_vals kwargs)
                If a subspace is specified, the eigenvectors are transformed back into the original
                basis before returning (Nprime -> N)


    '''

    # Interpret symmetry information
    if lsymm is None and symmetry is not None: lsymm = symmetry
    if rsymm is None and symmetry is not None: rsymm = symmetry
    return_llabels = not (lsymm is None)
    return_rlabels = not (rsymm is None)
    lsymm_isvectorblock = False if lsymm is None else isinstance (lsymm[0], np.ndarray)
    if lsymm is not None and lsymm_isvectorblock and len (lsymm) == 1: symmetry = None
    if lsymm is not None and not lsymm_isvectorblock and len (np.unique (lsymm)) == 1: symmetry = None
    if lsymm is None: lsymm_isvectorblock = False
    rsymm_isvectorblock = False if rsymm is None else isinstance (rsymm[0], np.ndarray)
    if rsymm is not None and rsymm_isvectorblock and len (rsymm) == 1: symmetry = None
    if rsymm is not None and not rsymm_isvectorblock and len (np.unique (rsymm)) == 1: symmetry = None
    if rsymm is None: rsymm_isvectorblock = False

    # Interpret subspace information
    lspace = None if lspace is None else np.asarray (lspace)
    lspace_isvectorblock = False if lspace is None else lspace.ndim == 2
    rspace = None if rspace is None else np.asarray (rspace)
    rspace_isvectorblock = False if rspace is None else rspace.ndim == 2

    # Zero matrix escape
    M = the_matrix.shape[0] if lspace is None else lspace.shape[-1]
    Mbasis = the_matrix.shape[0]
    N = the_matrix.shape[1] if rspace is None else rspace.shape[-1]
    Nbasis = the_matrix.shape[1]
    if 0 in (M, N):
        if full_matrices: return np.zeros ((Mbasis,M)), np.zeros ((0)), np.zeros ((Nbasis,N))
        return np.zeros ((Mbasis,0)), np.zeros ((K)), np.zeros ((Nbasis,0))

    # If symmetry information is provided as a vector block, transform into a symmetry-adapted basis and recurse
    if lsymm_isvectorblock:
        lsymm_umat = np.concatenate (lsymm, axis=1)
        assert (lsymm_umat.shape == tuple([the_matrix.shape[1],the_matrix.shape[1]])), "I can't guess how to map symmetry blocks to different bases! Matt fix this bug!"
        lsymm_lbls = np.concatenate ([idx * np.ones (blk.shape[1], dtype=int) for idx, blk in enumerate (lsymm)])
        lsymm_matr = lsymm_umat.conjugate ().T @ the_matrix 
        if lspace is not None:
            # Since symm_isidx == False, I have to turn the subspace into a vector block too! Dang!
            if lspace_isvectorblock: lsymm_subs = lsymm_umat.conjugate ().T @ lspace
            else: lsymm_subs = lsymm_umat.conjugate ().T [:,subspace]
        # Be 200% sure that this recursion can't trigger this conditional block again!
        assert (not (isinstance (lsymm_lbls[0], np.ndarray))), 'Infinite recursion detected! Fix this bug!'
        lvecs, svals, rvecs = matrix_svd_control_options (lsymm_matr, only_nonzero_vals=only_nonzero_vals,
            full_matrices=full_matrices, sort_vecs=sort_vecs,
            lspace=lsymm_subs, rspace=rspace,
            lsymm=lsymm_lbls, rsymm=rsymm, symmetry=None,
            num_zero_atol=num_zero_atol, num_zero_rtol=num_zero_rtol)
        lvecs = lsymm_umat @ lvecs
        return lvecs, svals, rvecs
    if rsymm_isvectorblock:
        rsymm_umat = np.concatenate (rsymm, axis=1)
        assert (rsymm_umat.shape == tuple([the_matrix.shape[1],the_matrix.shape[1]])), "I can't guess how to map symmetry blocks to different bases! Matt fix this bug!"
        rsymm_lbls = np.concatenate ([idx * np.ones (blk.shape[1], dtype=int) for idx, blk in enumerate (rsymm)])
        rsymm_matr = the_matrix @ rsymm_umat
        if rspace is not None:
            # Since symm_isidx == False, I have to turn the subspace into a vector block too! Dang!
            if rspace_isvectorblock: rsymm_subs = rsymm_umat.conjugate ().T @ lspace
            else: rsymm_subs = rsymm_umat.conjugate ().T [:,subspace]
        # Be 200% sure that this recursion can't trigger this conditional block again!
        assert (not (isinstance (rsymm_lbls[0], np.ndarray))), 'Infinite recursion detected! Fix this bug!'
        lvecs, svals, rvecs = matrix_svd_control_options (rsymm_matr, only_nonzero_vals=only_nonzero_vals,
            full_matrices=full_matrices, sort_vecs=sort_vecs,
            lspace=lspace, rspace=rsymm_subs,
            lsymm=lsymm, rsymm=rsymm_lbls, symmetry=None,
            num_zero_atol=num_zero_atol, num_zero_rtol=num_zero_rtol)
        lvecs = rsymm_umat @ lvecs
        return lvecs, svals, rvecs

    # Wrap in subspaces
    if lspace_isvectorblock:
        the_matrix = lspace.conjugate ().T @ the_matrix
    elif lspace is not None:
        the_matrix = the_matrix[lspace,:]
    if rspace_isvectorblock:
        the_matrix = the_matrix @ rspace
    elif rspace is not None:
        the_matrix = the_matrix[:,rspace]

    # Kernel
    lvecs, svals, r2q = scipy.linalg.svd (the_matrix, full_matrices=full_matrices)
    rvecs = r2q.conjugate ().T
    nsvals = len (svals)
    if only_nonzero_vals:
        idx = np.isclose (svals, 0, atol=num_zero_atol, rtol=num_zero_rtol)
        svals = svals[~idx]
        if full_matrices:
            lvecs[:,:nsvals] = np.append (lvecs[:,:nsvals][:,~idx], lvecs[:,:nsvals][:,idx], axis=1)
            rvecs[:,:nsvals] = np.append (rvecs[:,:nsvals][:,~idx], rvecs[:,:nsvals][:,idx], axis=1)
        else:
            lvecs = lvecs[:,~idx]
            rvecs = rvecs[:,~idx]
        nsvals = len (svals)
    if sort_vecs:
        idx = (np.abs (svals)).argsort ()[::sort_vecs]
        svals = svals[idx]
        rvecs[:,:nsvals] = rvecs[:,:nsvals][:,idx]
        lvecs[:,:nsvals] = lvecs[:,:nsvals][:,idx]

    # Wrap out subspaces
    if lspace_isvectorblock:
        lvecs = lspace @ lvecs
    elif lspace is not None:
        subs_lvecs = lvecs.copy ()
        lvecs = np.zeros (Mbasis, lvecs.shape[1], dtype=lvecs.dtype)
        lvecs[lspace,:] = subs_lvecs
    if rspace_isvectorblock:
        rvecs = rspace @ rvecs
    elif rspace is not None:
        subs_rvecs = rvecs.copy ()
        rvecs = np.zeros (Nbasis, rvecs.shape[1], dtype=rvecs.dtype)
        rvecs[rspace,:] = subs_rvecs

    return lvecs, svals, rvecs

def matrix_eigen_control_options (the_matrix, b_matrix=None, symmetry=None, strong_symm=False, 
    subspace=None, subs_symm=None, sort_vecs=-1, only_nonzero_vals=False, round_zero_vals=False, 
    num_zero_atol=params.num_zero_atol, num_zero_rtol=params.num_zero_rtol):
    ''' Diagonalize a matrix using scipy's driver and also a whole lot of pre-/post-processing,
        most significantly sorting, throwing away numerical null spaces, 
        subspace projections, and symmetry alignments.
        
        Args:
            the_matrix: square ndarray with M rows

        Kwargs:
            b_matrix: square ndarray with M rows
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
                quantitatively symmetry-adapted eigenvectors are grouped by block 
                with the blocks in a currently arbitrary order.
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
            evals: ndarray of shape (K,); K <= M (see only_nonzero_vals kwarg)
            evecs: ndarray of shape (M,K); K <= M (see only_nonzero_vals kwarg)
                If a subspace is specified, the eigenvectors are transformed back into the original
                basis before returning
            labels: list of length K; K<=M
                Only returned if symmetry is not None. Identifies symmetry block
                with the highest possible projection onto each eigenvector unless
                strong_symm==True, in which case the labels are derived
                from labels of subspace vectors computed in this way.. Does not
                guarantee that the eigenvector is symmetry-adapted.
                
            

    '''

    # Interpret symmetry information
    return_labels = not (symmetry is None)
    symm_isvectorblock = False if symmetry is None else isinstance (symmetry[0], np.ndarray)
    if symmetry is not None and symm_isvectorblock and len (symmetry) == 1: symmetry = None
    if symmetry is not None and not symm_isvectorblock and len (np.unique (symmetry)) == 1: symmetry = None
    if symmetry is None: symm_isvectorblock = False

    # Interpret subspace information
    subspace = None if subspace is None else np.asarray (subspace)
    subspace_isvectorblock = False if subspace is None else subspace.ndim == 2

    # Interpret subspace symmetry information
    if subs_symm is not None:
        raise NotImplementedError ("Explicit subspace symmetry information.")

    # Prevent wasting time. Matt, you should gradually remove these lines to test edge-case tolerance
    nouter = the_matrix.shape[-1] if subspace is None else subspace.shape[-1]
    ninner = the_matrix.shape[-1]
    if not nouter:
        if return_labels: return np.zeros ((0)), np.zeros ((ninner,0)), np.ones ((0))
        return np.zeros ((0)), np.zeros ((ninner,0))
    if (symmetry is None): strong_symm = False

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
    if symmetry is not None:
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
    else:
        labels = np.zeros (len (evals))    

    if return_labels: return evals, evecs, labels
    return evals, evecs

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

def align_coupled_vecs (lvecs, coupl, rvecs, lrow_labels, rrow_labels, rtol=params.num_zero_rtol, atol=params.num_zero_atol, assert_tol=0):
    assert (lvecs.shape[1] == rvecs.shape[1])
    npairs = lvecs.shape[1]
    assert (coupl.shape == (npairs, npairs))
    col_labels = np.empty (npairs, dtype=rrow_labels.dtype)
    uniq_labels = np.unique (np.concatenate (lrow_labels, rrow_labels))
    i = 0
    metric = lvecs @ coupl @ rvecs.conjugate ().T
    while i < npairs:
        svdout = [matrix_svd_control_options (metric, sort_vecs=-1, only_nonzero_vals=False, full_matrices=True,
            lspace=lrow_labels==lbl, rspace=rrow_labels==lbl) for lbl in uniq_labels]
        symm_label_idx = np.argmax ([sval[0] for lvec, sval, rvec in svdout])
        

def assign_blocks_weakly (the_states, the_blocks):
    projectors = [blk @ blk.conjugate ().T for blk in the_blocks]
    vals = np.stack ([((proj @ the_states) * the_states).sum (0) for proj in projectors], axis=-1)
    return np.argmax (vals, axis=1)





