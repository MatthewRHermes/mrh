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
    M = lspace.shape[-1] if lspace is not None else the_matrix.shape[0] 
    Mbasis = lspace.shape[0] if lspace_isvectorblock else the_matrix.shape[0]
    N = rspace.shape[-1] if rspace is not None else the_matrix.shape[1] 
    Nbasis = rspace.shape[0] if rspace_isvectorblock else the_matrix.shape[1]
    if 0 in (M, N):
        if full_matrices: return np.zeros ((Mbasis,M)), np.zeros ((0)), np.zeros ((Nbasis,N))
        return np.zeros ((Mbasis,0)), np.zeros ((K)), np.zeros ((Nbasis,0))

    # If symmetry information is provided as a vector block, transform into a symmetry-adapted basis and recurse
    if lsymm_isvectorblock:
        lsymm_umat = np.concatenate (lsymm, axis=1)
        assert (lsymm_umat.shape == tuple((Mbasis, Mbasis))), "I can't guess how to map symmetry blocks to different bases! Matt fix this bug!"
        lsymm_lbls = np.concatenate ([idx * np.ones (blk.shape[1], dtype=int) for idx, blk in enumerate (lsymm)])
        lsymm_matr = lsymm_umat.conjugate ().T @ the_matrix if isinstance (the_matrix, np.ndarray) else lsymm_umat.conjugate ().T * the_matrix
        if lspace is not None:
            # Since symm_isidx == False, I have to turn the subspace into a vector block too! Dang!
            if lspace_isvectorblock: lsymm_subs = lsymm_umat.conjugate ().T @ lspace
            else: lsymm_subs = lsymm_umat.conjugate ().T [:,subspace]
        # Be 200% sure that this recursion can't trigger this conditional block again!
        assert (not (isinstance (lsymm_lbls[0], np.ndarray))), 'Infinite recursion detected! Fix this bug!'
        rets = matrix_svd_control_options (lsymm_matr, only_nonzero_vals=only_nonzero_vals,
            full_matrices=full_matrices, sort_vecs=sort_vecs,
            lspace=lsymm_subs, rspace=rspace,
            lsymm=lsymm_lbls, rsymm=rsymm, symmetry=None,
            num_zero_atol=num_zero_atol, num_zero_rtol=num_zero_rtol)
        rets[0] = lsymm_umat @ rets[0]
        return rets
    if rsymm_isvectorblock:
        rsymm_umat = np.concatenate (rsymm, axis=1)
        assert (rsymm_umat.shape == tuple((Nbasis, Nbasis))), "I can't guess how to map symmetry blocks to different bases! Matt fix this bug!"
        rsymm_lbls = np.concatenate ([idx * np.ones (blk.shape[1], dtype=int) for idx, blk in enumerate (rsymm)])
        rsymm_matr = the_matrix @ rsymm_umat if isinstance (the_matrix, np.ndarray) else the_matrix * rsymm_umat
        if rspace is not None:
            # Since symm_isidx == False, I have to turn the subspace into a vector block too! Dang!
            if rspace_isvectorblock: rsymm_subs = rsymm_umat.conjugate ().T @ lspace
            else: rsymm_subs = rsymm_umat.conjugate ().T [:,subspace]
        # Be 200% sure that this recursion can't trigger this conditional block again!
        assert (not (isinstance (rsymm_lbls[0], np.ndarray))), 'Infinite recursion detected! Fix this bug!'
        rets = matrix_svd_control_options (rsymm_matr, only_nonzero_vals=only_nonzero_vals,
            full_matrices=full_matrices, sort_vecs=sort_vecs,
            lspace=lspace, rspace=rsymm_subs,
            lsymm=lsymm, rsymm=rsymm_lbls, symmetry=None,
            num_zero_atol=num_zero_atol, num_zero_rtol=num_zero_rtol)
        rets[2] = rsymm_umat @ rets[2]
        return rets

    # Wrap in subspaces
    if lspace_isvectorblock:
        if isinstance (the_matrix, np.ndarray):
            the_matrix = lspace.conjugate ().T @ the_matrix 
        else:
            the_matrix = lspace.conjugate ().T * the_matrix
    if rspace_isvectorblock:
        if isinstance (the_matrix, np.ndarray):
            the_matrix = the_matrix @ rspace
        else:
            the_matrix = the_matrix * rspace
    elif rspace is not None:
        the_matrix = the_matrix[:,rspace]
    if not lspace_isvectorblock and lspace is not None:
        the_matrix = the_matrix[lspace,:]

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
        lvecs = np.zeros ((Mbasis, lvecs.shape[1]), dtype=lvecs.dtype)
        lvecs[lspace,:] = subs_lvecs
    if rspace_isvectorblock:
        rvecs = rspace @ rvecs
    elif rspace is not None:
        subs_rvecs = rvecs.copy ()
        rvecs = np.zeros ((Nbasis, rvecs.shape[1]), dtype=rvecs.dtype)
        rvecs[rspace,:] = subs_rvecs

    lvecs, rvecs, llabels, rlabels = align_degenerate_coupled_vecs (lvecs, svals, rvecs, lsymm, rsymm, rtol=num_zero_rtol, atol=num_zero_atol)

    if return_llabels and return_rlabels:
        return lvecs, svals, rvecs, llabels, rlabels
    elif return_llabels:
        return lvecs, svals, rvecs, llabels
    elif return_rlabels:
        return lvecs, svals, rvecs, rlabels
    return lvecs, svals, rvecs

def matrix_eigen_control_options (the_matrix, b_matrix=None, symmetry=None, strong_symm=False, 
    subspace=None, subspace_symmetry=None, sort_vecs=-1, only_nonzero_vals=False, round_zero_vals=False, 
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
                Requires symmetry or subspace_symmetry.
                Eigenvectors will be symmetry-adapted but this does not
                check that the whole matrix is actually diagonalized by them so user beware!
                Extra risky if a subspace is used without subspace_symmetry
                because the vectors of the subspace
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
            subspace_symmetry: list of block labels of length Mprime
            or list of non-square matrices of shape (Mprime,P), sum_P = Mprime
            or None
                Formal symmetry information of the subspace. If included,
                "symmetry" is ignored.
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

    # Interpret subspace information
    subspace = None if subspace is None else np.asarray (subspace)
    subspace_isvectorblock = False if subspace is None else subspace.ndim == 2

    # Interpret subspace symmetry information
    return_labels = not (symmetry is None) or not (subspace_symmetry is None)
    subs_symm_isvectorblock = False if subspace_symmetry is None else isinstance (subspace_symmetry[0], np.ndarray)
    if subspace_symmetry is not None:
        symmetry = None
        if subs_symm_isvectorblock:
            if len (subspace_symmetry) == 1: subspace_symmetry = None
        elif len (np.unique (subspace_symmetry)) == 1: subspace_symmetry = None
    if subspace_symmetry is None: subs_symm_isvectorblock = False

    # Interpret (full space) symmetry information (discarded if subspace symmetry is provided!)
    symm_isvectorblock = False if symmetry is None else isinstance (symmetry[0], np.ndarray)
    if symmetry is not None and symm_isvectorblock and len (symmetry) == 1: symmetry = None
    if symmetry is not None and not symm_isvectorblock and len (np.unique (symmetry)) == 1: symmetry = None
    if symmetry is None: symm_isvectorblock = False
    if (symmetry is None) and (subspace_symmetry is None): strong_symm = False

    # Zero matrix escape
    M = subspace.shape[-1] if subspace is not None else the_matrix.shape[0]
    Mbasis = subspace.shape[0] if subspace_isvectorblock else the_matrix.shape[0]
    if not M:
        if return_labels: return np.zeros ((0)), np.zeros ((Mbasis,0)), np.ones ((0))
        return np.zeros ((0)), np.zeros ((Mbasis,0))

    # If subspace symmetry is provided as a vector block, transform subspace into a symmetry-adapted form
    # No recursion necessary because the eigenvectors are meant to be provided in the full basis :)
    if subs_symm_isvectorblock:
        subspace = subspace @ np.concatenate (subspace_symmetry, axis=1)
        subspace_symmetry = np.concatenate ([idx * np.ones (blk.shape[1], dtype=int) for idx, blk in enumerate (subspace_symmetry)])

    # If symmetry information is provided as a vector block, transform into a symmetry-adapted basis and recurse
    if symm_isvectorblock:
        symm_umat = np.concatenate (symmetry, axis=1)
        symm_lbls = np.concatenate ([idx * np.ones (blk.shape[1], dtype=int) for idx, blk in enumerate (symmetry)])
        if isinstance (the_matrix, np.ndarray):
            symm_matr = symm_umat.conjugate ().T @ the_matrix @ symm_umat
        else:
            symm_matr = (symm_umat.conjugate ().T * the_matrix) @ symm_umat
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

        # If a subspace is being diagonalized, recurse into symmetry blocks via the subspace
        if subspace is None:
            symm_lbls = symmetry
            subspace = np.ones (the_matrix.shape[0], dtype=np.bool_)
        elif subspace_symmetry is not None:
            symm_lbls = subspace_symmetry
        elif subspace_isvectorblock:
            subspace, symm_lbls = align_vecs (subspace, symmetry, rtol=num_zero_rtol, atol=num_zero_atol)
        else:
            symm_lbls = symmetry[subspace]
            
        # Recurse into diagonalization of individual symmetry blocks via the "subspace" option!
        uniq_lbls = np.unique (symm_lbls)
        evals = []
        evecs = []
        labels = []
        for lbl in uniq_lbls:
            subs_blk = subspace[...,symm_lbls==lbl]
            evals_blk, evecs_blk = matrix_eigen_control_options (the_matrix, symmetry=None, strong_symm=False,
                sort_vecs=sort_vecs, only_nonzero_vals=only_nonzero_vals, round_zero_vals=round_zero_vals,
                b_matrix=b_blk, num_zero_atol=num_zero_atol, subspace=subs_blk)
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
            if isinstance (the_matrix, np.ndarray):
                the_matrix = subspace.conjugate ().T @ the_matrix @ subspace
            else:
                the_matrix = (subspace.conjugate ().T * the_matrix) @ subspace
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

    evecs, labels = align_degenerate_vecs (evals, evecs, subspace_symmetry, rtol=num_zero_rtol, atol=num_zero_atol)
    # Wrap out subspace projection. This should be triggered if and only if strong_symm==False!
    if subspace is not None:
        assert (not strong_symm)
        if subspace_isvectorblock:
            evecs = subspace @ evecs
        else:
            subs_evecs = evecs.copy ()
            evecs = np.zeros ((ndim_full, subs_evecs.shape[1]), dtype=subs_evecs.dtype)
            evecs[subspace,:] = subs_evecs
    if labels is None: evecs, labels = align_degenerate_vecs (evals, evecs, symmetry, rtol=num_zero_rtol, atol=num_zero_atol)
    if labels is None: labels = np.zeros (len (evals))

    if return_labels: return evals, evecs, labels
    return evals, evecs

def align_degenerate_vecs (vals, vecs, symm, rtol=params.num_zero_rtol, atol=params.num_zero_atol, assert_tol=0):
    if symm is None:
        return vecs, None
    uniq_labels = np.unique (symm)
    idx_unchk = np.ones (len (vals), dtype=np.bool_)
    labels = np.empty (len (vals), dtype=uniq_labels.dtype)
    while np.count_nonzero (idx_unchk):
        chk_1st_eval = vals[idx_unchk][0]
        idx_degen = np.isclose (vals, chk_1st_eval, rtol=rtol, atol=atol)
        if np.count_nonzero (idx_degen) > 1:
            vecs[:,idx_degen], labels[idx_degen] = align_vecs (vecs[:,idx_degen], symm, atol=atol, rtol=rtol)
        else:
            symmweight = [np.square (vecs[symm==lbl,idx_degen]).sum () for lbl in uniq_labels]
            labels[idx_degen] = uniq_labels[np.argmax (symmweight)]
        idx_unchk[idx_degen] = False
    return vecs, labels

def align_vecs (vecs, row_labels, rtol=params.num_zero_rtol, atol=params.num_zero_atol, assert_tol=0):
    col_labels = np.empty (vecs.shape[1], dtype=row_labels.dtype)
    uniq_labels = np.unique (row_labels)
    i = 0
    while i < vecs.shape[1]:
        svdout = [matrix_svd_control_options (vecs[:,i:], sort_vecs=-1, only_nonzero_vals=False,
            full_matrices=True, lspace=row_labels==lbl) for lbl in uniq_labels]
        # This argmax identifies the single best irrep assignment possible for all of vecs[:,i:]
        symm_label_idx = np.argmax ([svals[0] for lvecs, svals, rvecs in svdout])
        lvecs, svals, rvecs = svdout[symm_label_idx]
        j = i + np.count_nonzero (np.isclose (svals, svals[0], atol=atol, rtol=rtol))
        col_labels[i:j] = uniq_labels[symm_label_idx]
        if assert_tol and not np.all (np.isclose (svals, 0, atol=assert_tol, rtol=rtol) | np.isclose (svals, 1, atol=assert_tol, rtol=rtol)):
            raise RuntimeError ('Vectors not block-adapted in space {}; svals = {}'.format (col_labels[i], svals))
        # This puts the best-aligned vector at position i and causes all of vecs[:,j:] to be orthogonal to it
        vecs[:,i:] = vecs[:,i:] @ rvecs
        assert (j > i)
        i = j
        # This is a trick to grab a whole bunch of degenerate vectors at once (i.e., numerically symmetry-adapted vectors with sval = 1)
        # It may improve numerical stability
    return vecs, col_labels

def align_degenerate_coupled_vecs (lvecs, svals, rvecs, lsymm, rsymm, rtol=params.num_zero_rtol, atol=params.num_zero_atol, assert_tol=0):
    nvals = len (svals)
    if lsymm is None and rsymm is None:
        return lvecs, rvecs, None, None
    elif lsymm is None:
        rvals = np.append (svals, np.zeros (rvecs.shape[1] - nvals))
        rvecs, rlabels = align_degenerate_vecs (rvals, rvecs, rsymm, rtol=rtol, atol=atol, assert_tol=assert_tol)
        return lvecs, rvecs, None, rlabels
    elif rsymm is None:
        lvals = np.append (svals, np.zeros (lvecs.shape[1] - nvals))
        lvecs, llabels = align_degenerate_vecs (lvals, lvecs, lsymm, rtol=rtol, atol=atol, assert_tol=assert_tol)
        return lvecs, rvecs, llabels, None
    if lsymm is None: lsymm = []
    if rsymm is None: rsymm = []
    uniq_labels = np.unique (np.append (lsymm, rsymm))
    idx_unchk = np.ones (len (svals), dtype=np.bool_)
    llabels = np.empty (lvecs.shape[1], dtype=uniq_labels.dtype)
    rlabels = np.empty (rvecs.shape[1], dtype=uniq_labels.dtype)
    lv = lvecs[:,:nvals]
    rv = rvecs[:,:nvals]
    ll = llabels[:nvals]
    while np.count_nonzero (idx_unchk):
        chk_1st_sval = svals[idx_unchk][0]
        idx = np.isclose (svals, chk_1st_sval, rtol=rtol, atol=atol)
        if np.count_nonzero (idx) > 1:
            lv[:,idx], rv[:,idx], ll[idx] = align_coupled_vecs (
                lv[:,idx], rv[:,idx], lsymm, rsymm, rtol=rtol, atol=atol)
        else:
            proj = lv[:,idx] @ rv[:,idx].conjugate ().T
            symmweight = []
            for lbl in uniq_labels:
                if len (lsymm) > 0: proj = proj[lsymm==lbl,:]
                if len (rsymm) > 0: proj = proj[:,rsymm==lbl]
                symmweight.append (linalg.norm (proj))
            ll[idx] = uniq_labels[np.argmax (symmweight)]
        idx_unchk[idx_degen] = False
    rlabels = llabels
    if lvecs.shape[1] > nvals and len (lsymm) > 0:
        lvecs[:,nvals:], llabels[nvals:] = align_vecs (lvecs[:,nvals:], lsymm, rtol=rtol, atol=atol)
    if rvecs.shape[1] > nvals and len (rsymm) > 0:
        rvecs[:,nvals:], rlabels[nvals:] = align_vecs (rvecs[:,nvals:], rsymm, rtol=rtol, atol=atol)
    if len (lsymm) == 0: llabels = 0
    if len (rsymm) == 0: rlabels = 0
    return lvecs, rvecs, llabels, rlabels

def align_coupled_vecs (lvecs, rvecs, lrow_labels, rrow_labels, rtol=params.num_zero_rtol, atol=params.num_zero_atol, assert_tol=0):
    assert (lvecs.shape[1] == rvecs.shape[1])
    npairs = lvecs.shape[1]
    assert (coupl.shape == (npairs, npairs))
    col_labels = np.empty (npairs, dtype=rrow_labels.dtype)
    if lrow_labels is None: lrow_labels = []
    if rrow_labels is None: rrow_labels = []
    uniq_labels = np.unique (np.concatenate (lrow_labels, rrow_labels))
    coupl = lvecs @ rvecs.conjugate ().T
    i = 0
    while i < npairs:
        svdout = []
        for lbl in uniq_labels:
            metric = coupl.copy ()
            if len (lrow_labels) > 0: metric[lrow_labels!=lbl,:] = 0
            if len (rrow_labels) > 0: metric[:,rrow_labels!=lbl] = 0
            svdout.append (matrix_svd_control_options (metric, sort_vecs=-1, only_nonzero_vals=False, full_matrices=True,
                lspace=lvecs[:,i:], rspace=rvecs[:,i:]))
        symm_label_idx = np.argmax ([svals[0] for lu, svals, ru in svdout])
        lu, svals, ru = svdout[symm_label_idx]
        j = i + np.count_nonzero (np.isclose, svals, svals[0], atol=atol, rtol=rtol)
        col_labels[i:j] = uniq_labels[symm_label_idx]
        if assert_tol and not np.all (np.isclose (svals, 0, atol=assert_tol, rtol=rtol) | np.isclose (svals, 1, atol=assert_tol, rtol=rtol)):
            raise RuntimeError ('Vectors not block-adapted in space {}; svals = {}'.format (col_labels[i], svals))
        lvecs[:,i:] = lu
        rvecs[:,i:] = ru
        assert (j > i)
        i = j
    return lvecs, rvecs, col_labels

def assign_blocks_weakly (the_states, the_blocks):
    projectors = [blk @ blk.conjugate ().T for blk in the_blocks]
    vals = np.stack ([((proj @ the_states) * the_states).sum (0) for proj in projectors], axis=-1)
    return np.argmax (vals, axis=1)





