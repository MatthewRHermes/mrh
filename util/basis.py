# A collection of useful manipulations of basis sets (i.e., rectangular matrices) and operators (square matrices)

import numpy as np
from mrh.util.la import is_matrix_zero, is_matrix_eye, is_matrix_idempotent, matrix_eigen_control_options, matrix_svd_control_options

################	basic queries and assertions for basis-set-related objects	################



def assert_vector_statelist (test_vector, max_element=None, max_length=None, min_length=0):
	if (max_length == None):
		max_length = test_vector.shape[0] + 1
	if (max_element == None):
		max_element = np.amax (test_vector) + 1
	err_str = "vector not 1d array of unique nonnegative integers with {0} <= length <= {1} and maximum element < {2}\n({3})".format (min_length, max_length, max_element, test_vector)
	cond_isvec = (test_vector.ndim == 1)
	cond_min = (np.amin (test_vector) >= 0)
	cond_max = (np.amax (test_vector) <= max_element)
	cond_length = (test_vector.shape[0] <= max_length) and (test_vector.shape[0] >= min_length)
	cond_int = np.all (np.mod (test_vector, 1.0) == 0.0)
	u, counts = np.unique (test_vector, return_counts=True)
	cond_uniq = np.all (counts == 1)
	assert (cond_isvec and cond_min and cond_max and cond_length and cond_int and cond_uniq), err_str
	return test_vector.shape[0]

def assert_vector_stateis (test_vector, vecdim=None):
	err_str = "vector not 1d boolean array"
	cond_isvec = (test_vector.ndim == 1)
	cond_length = (test_vector.shape[0] == vecdim) or (not vecdim)
	if not cond_length:
		err_str = err_str + " (vector has length {0}, should be {1})".format (test_vector.shape[0], vecdim)
	cond_bool = np.all (np.logical_or (test_vector == 1, test_vector == 0))
	assert (cond_isvec and cond_len and cond_bool), err_str

def is_basis_orthonormal (the_basis):
	c2b = np.asmatrix (the_basis)
	b2c = c2b.H
	return is_matrix_eye (b2c * c2b)

def is_basis_orthonormal_and_complete (the_basis):
	return (is_basis_orthonormal (the_basis) and (the_basis.shape[1] == the_basis.shape[0]))

def are_bases_orthogonal (bra_basis, ket_basis):
	return is_matrix_zero (basis_olap (bra_basis, ket_basis))



################	simple manipulations and common calculations		################



def basis_olap (bra_basis, ket_basis):
	c2p = np.asmatrix (bra_basis)
	c2q = np.asmatrix (ket_basis)
	p2c = c2p.H
	return (p2c * c2q)

def represent_operator_in_subspace (braOket, ket_basis, bra_basis = None):
	# Note that this CHANGES the basis that braOket is stored in
	lOr = np.asmatrix (braOket)
	if not bra_basis:
		bra_basis = ket_basis

	l2p = np.asmatrix (bra_basis)
	r2q = np.asmatrix (ket_basis)

	p2l = l2p.H
	q2r = r2q.H
	return (p2l * lOr * r2q)

def project_operator_into_subspace (braOket, ket_basis, bra_basis = None):
	# Note that this DOESN'T change the basis that braOket is stored in
	lOr = np.asmatrix (braOket)
	if not bra_basis:
		bra_basis = ket_basis

	l2p = np.asmatrix (bra_basis)
	p2l = l2p.H
	r2q = np.asmatrix (ket_basis)
	q2r = r2q.H

	lPl = l2p * p2l
	rPr = r2q * q2r
	return (lPl * lOr * rPr)

def compute_operator_trace_in_subset (the_operator, the_subset_basis):
	return np.trace (represent_operator_in_subspace (the_operator, the_subset_basis))

compute_nelec_in_subspace = compute_operator_trace_in_subset



################	More complicated basis manipulation functions		################



def get_overlapping_states (bra_basis, ket_basis, across_operator = None, nrvecs=0, nlvecs=0, num_zero_atol=1.0e-8):
	c2p = np.asmatrix (bra_basis)
	c2q = np.asmatrix (ket_basis)
	cOc = np.asmatrix (across_operator) if np.any (across_operator) else 1
	assert (c2p.shape[0] == c2q.shape[0]), "you need to give the two spaces in the same basis"
	assert (c2p.shape[1] <= c2p.shape[0]), "you need to give the first state in a complete basis (c2p). Did you accidentally transpose it?"
	assert (c2q.shape[1] <= c2q.shape[0]), "you need to give the second state in a complete basis (c2q). Did you accidentally transpose it?"
	assert (nlvecs <= c2p.shape[1]), "you can't ask for more left states than are in your left space"
	assert (nrvecs <= c2q.shape[1]), "you can't ask for more right states than are in your right space"
	if np.any (across_operator):
		assert (c2p.shape[0] == cOc.shape[0] and c2p.shape[0] == cOc.shape[1]), "when specifying an across_operator, it's dimensions need to be the same as the external basis"

	p2c = c2p.H
	pOq = p2c * cOc * c2q

	''' I think the eigendecomposition is actually more stable than the explicit svd so I'm going to do it that way (it's equivalent)
	q2r, p2l, svals = matrix_svd_control_options (pOq, svd_full_matrices = (not omit_id_zero_svals), sort_vecs=True)
	c2l = c2p * p2l
	c2r = c2q * q2r
	'''

	pQp = pOq * pOq.H
	qPq = pOq.H * pOq
	pevals, p2l = matrix_eigen_control_options (pQp, sort_vecs=True, only_nonzero_vals=True)
	qevals, q2r = matrix_eigen_control_options (qPq, sort_vecs=True, only_nonzero_vals=True)
	assert (np.allclose (pevals, qevals)), str (pevals) + "\n" + str (qevals)
	svals = np.sqrt (np.mean ([pevals, qevals], axis=0))

	# Get the left- and right-vectors back in the external basis
	c2l = c2p * p2l
	c2r = c2q * q2r

	# get_top_nvecs = 0 means get all nvecs (or nonzero nvecs if omit_id_zero_svals, which I took care of above)
	def getbasis_from_olap_trunc_ (c2b, c2b_len):
		if c2b_len < 1:
			c2b_len = len (svals)

		# Print a note if you asked for more states than are left at this point
		if (c2b_len > len (svals)):
			head_str = "get_overlapping_states :: note : "
			note_1 = "{0} states projected into overlap space requested, but only {1} such pairs found ; ".format (c2b_len, len (svals))
			note_2 = "returning only {0} states to caller".format (len (svals))
			print (head_str + note_1 + note_2)
			c2b_len = len (svals)
		return c2b[:,:c2b_len]

	c2r = getbasis_from_olap_trunc_ (c2r, nrvecs)
	c2l = getbasis_from_olap_trunc_ (c2l, nlvecs)

	return c2l, c2r, svals
	
def measure_basis_olap (bra_basis, ket_basis):
	svals = get_overlapping_states (bra_basis, ket_basis)[2]
	olap_ndf = len (svals)
	olap_mag = np.norm (svals)
	return olap_mag, olap_ndf

def orthonormalize_a_basis (overlapping_basis, num_zero_atol=1.0e-8):
	if (is_basis_orthonormal (overlapping_basis)):
		return overlapping_basis
	c2b = np.asmatrix (incomplete_basis)
	b2c = c2b.H
	bOb = b2c * c2b
	assert (not is_matrix_zero (bOb)), "overlap matrix is zero! problem with basis?"
	assert (np.allclose (bOb, bOb.H)), "overlap matrix not hermitian! problem with basis?"
	assert (np.abs (np.trace (bOb)) > num_zero_atol), "overlap matrix zero or negative trace! problem with basis?"
	 
	evals, p2x = matrix_eigen_control_options (bOb, sort_vecs=True, only_nonzero_vals=True)
	c2x = c2b * p2x 
	assert (not np.any (evals < 0)), "overlap matrix has negative eigenvalues! problem with basis?"

	# I want c2n = c2x * x2n
	# x2n defined such that n2c * c2n = I
	# n2x * x2c * c2x * x2n = n2x * evals_xx * x2n = I
	# therefore
	# x2n = evals_xx^{-1/2}
	x2n = np.asmatrix (np.diag (np.reciprocal (np.sqrt (evals))))
	c2n = c2x * x2n
	assert (is_basis_orthonormal (c2n)), "failed to orthonormalize basis somehow\n" + str (c2n)

	return c2n

def get_basis_from_projector (the_projector, num_zero_atol=1.0e-8):
	proj_cc = np.asmatrix (the_projector)
	assert (np.allclose (proj_cc, proj_cc.H)), "projector must be hermitian\n" + str (proj_cc - proj_cc.H)
	assert (is_matrix_idempotent (proj_cc)), "projector must be idempotent\n" + str ((proj_cc * proj_cc) - proj_cc)
	evals, p2x = matrix_eigen_control_options (proj_cc, sort_vecs=True, only_nonzero_vals=True)
	return p2x

def basis_complement (incomplete_basis):
	orthonormal_basis = orthonormalize_a_basis (incomplete_basis)
	if is_basis_orthonormal_and_complete (orthonormal_basis):
		print ("warning: tried to construct a complement for a basis that was already complete")
		return None

	c2b = np.asmatrix (orthonormal_basis)
	nstates_b = c2b.shape[1]
	nstates_c = c2b.shape[0]

	b2c = c2b.H
	Projb_cc = c2b * b2c
	Projq_cc = np.eye (nstates_c, dtype=Projb_cc.dtype) - Projb_cc
	
	return get_basis_from_projector (Projq_cc)

