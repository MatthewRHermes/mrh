# A collection of simple manipulations of matrices that I somehow can't find in numpy

def is_matrix_zero (test_matrix):
	test_zero = np.zeros (test_matrix.shape, dtype=test_matrix.dtype)
	return np.allclose (test_matrix, test_zero)

def is_matrix_eye (test_matrix, matdim=None):
	test_eye = np.eye (test_matrix.shape[0], dtype=test_matrix.dtype)
	return np.allclose (test_matrix, test_eye)

def is_matrix_idempotent (test_matrix):
	test_m2 = np.dot (test_matrix, test_matrix)
	return np.allclose (test_matrix, test_m2)

def is_matrix_diagonal (test_matrix):
	test_diagonal = np.diag (np.diag (test_matrix))
	return np.allclose (test_matrix, test_diagonal)

def is_matrix_hermitian (test_matrix):
	test_adjoint = np.transpose (np.conjugate (test_matrix))
	return np.allclose (test_matrix, test_adjoint)

def assert_matrix_square (test_matrix, matdim=None):
	if (matdim == None):
		matdim = test_matrix.shape[0]
	assert ((test_matrix.ndim == 2) and (test_matrix.shape[0] == matdim) and (test_matrix.shape[1] == matdim)), "Matrix shape is {0}; should be ({1},{1})".format (test_matrix.shape, matdim)
	return matdim

def matrix_svd_control_options (the_matrix, full_matrices=False, sort_vecs=True, only_nonzero_vals=False, num_zero_atol=1.0e-8):
	pMq = np.asmatrix (the_matrix)
	lvecs_pl, svals_lr, rvecs_rq = np.linalg.svd (pMq, full_matrices=full_matrices)
	p2l = np.asmatrix (lvecs_pl)
	r2q = np.asmatrix (rvecs_rq)
	q2r = r2q.H
	if sort_vecs:
		idx_sval = (np.abs (svals_lr)).argsort ()[::-1]
		idx_q2r = np.append (idx_sval, np.arange (idx_sval, q2r.shape[1], dtype=idx_sval.dtype))
		idx_p2l = np.append (idx_sval, np.arange (idx_sval, p2l.shape[1], dtype=idx_sval.dtype))
		svals_lr = svals_lr[idx_sval]
		q2r = q2r[idx_q2r]
		p2l = p2l[idx_p2l]
	if only_nonzero_vals:
		idx = np.where (np.abs (svals_lr) > num_zero_atol)[0]
		svals_lr = svals_lr[idx]
		q2r = q2r[idx]
		p2l = p2l[idx]

	# I'll return them in an order evocative of my favorite way of writing an svd, pMq * q2r = p2l * svals_lr
	return q2r, p2l, svals_lr

def matrix_eigen_control_options (the_matrix, sort_vecs=True, only_nonzero_vals=False, num_zero_atol=1.0e-8):
	# Subtract a diagonal average from the matrix to fight rounding error
	diag_avg = np.eye (the_matrix.shape[0]) * np.mean (np.diag (the_matrix))
	pMq = np.asmatrix (the_matrix - diag_avg)
	# Use hermitian diagonalizer if possible and don't do anything if the matrix is already diagonal
	evals = np.diagonal (pMq)
	evecs = np.asmatrix (np.eye (len (evals), dtype=evals.dtype))
	if not is_matrix_diagonal (pMq):
		evals, evecs = np.linalg.eigh (pMq) if is_matrix_hermitian (pMq) else np.linalg.eig (pMq)
	# Add the diagonal average to the eigenvalues when returning!
	evals = evals + np.diag (diag_avg)
	if only_nonzero_vals:
		idx = np.where (np.abs (evals) > num_zero_atol)
		evals = evals[idx]
		evecs = evecs[:,idx]
	if sort_vecs:
		idx = np.abs (evals[idx]).argsort ()[::-1]
		evals = evals[idx]
		evecs = evecs[:,idx]
	return evals, evecs



