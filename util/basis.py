# A collection of useful manipulations of basis sets (i.e., rectangular matrices) and operators (square matrices)

import numpy as np
from mrh.util.la import is_matrix_zero, is_matrix_eye, is_matrix_idempotent, matrix_eigen_control_options, matrix_svd_control_options
from mrh.util import params

################    basic queries and assertions for basis-set-related objects    ################



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



################    simple manipulations and common calculations        ################



def enforce_maxel_positive (the_basis):
    '''Multiply coefficients for states with negative largest coefficient by -1'''
    idx0 = np.asarray (np.abs (the_basis)).argmax (axis=0)
    idx1 = list(range(the_basis.shape[1]))
    cols = np.where (np.asarray (the_basis)[idx0,idx1]<0)[0]
    the_basis[:,cols] *= -1
    return the_basis

def sort_states_by_diag_maxabs (the_basis):
    '''Sort states so that the coefficients in each with the maximum absolute value are on the diagonal'''
    cols = np.asarray (np.abs (the_basis)).argmax (axis=0).argsort ()
    the_basis = the_basis[:,cols]
    return the_basis

def basis_olap (bra_basis, ket_basis):
    c2p = np.asmatrix (bra_basis)
    c2q = np.asmatrix (ket_basis)
    p2c = c2p.H
    return np.asarray (p2c * c2q)

def represent_operator_in_basis (braOket, bra1_basis = None, ket1_basis = None, bra2_basis = None, ket2_basis = None):
    # This CHANGES the basis that braOket is stored in
    allbases = [i for i in [bra1_basis, ket1_basis, bra2_basis, ket2_basis] if i is not None]
    if len (allbases) == 0:
        raise RuntimeError ("needs at least one basis")
    bra1_basis = allbases[0]
    ket1_basis = bra1_basis if ket1_basis is None else ket1_basis
    bra2_basis = bra1_basis if bra2_basis is None else bra2_basis
    ket2_basis = bra2_basis if ket2_basis is None else ket2_basis
    the_bases = [bra1_basis, ket1_basis, bra2_basis, ket2_basis]
    if any ([i.shape[0] == 0 for i in the_bases]):
        newshape = tuple ([i.shape[1] for i in the_bases])
        return np.zeros (newshape, dtype=braOket.dtype)
    if all ([is_matrix_eye (i) for i in the_bases]):
        return braOket
    if len (braOket.shape) == 2:
        return represent_operator_in_basis_1body (braOket, bra1_basis, ket1_basis)
    elif len (braOket.shape) == 4:
        return represent_operator_in_basis_2body (braOket, bra1_basis, ket1_basis, bra2_basis, ket2_basis)
    else:
        raise ValueError ("Only one- and two-body operators (two- and four-index arrays) supported")

def represent_operator_in_basis_1body (braOket, bra_basis, ket_basis):
    lOr = np.asmatrix (braOket)
    l2p = np.asmatrix (bra_basis)
    r2q = np.asmatrix (ket_basis)
    p2l = l2p.H
    return np.asarray (p2l * lOr * r2q)

def represent_operator_in_basis_2body (braOket, bra1_basis, ket1_basis, bra2_basis, ket2_basis):
    abcd = braOket
    az = np.conj (bra1_basis)
    bz = ket1_basis
    cz = np.conj (bra2_basis)
    dz = ket2_basis

    abcd = np.einsum ('abcd,az->zbcd',abcd,az)
    abcd = np.einsum ('abcd,bz->azcd',abcd,bz)
    abcd = np.einsum ('abcd,cz->abzd',abcd,cz)
    abcd = np.einsum ('abcd,dz->abcz',abcd,dz)
    return abcd

def project_operator_into_subspace (braOket, ket1_basis = None, bra1_basis = None, ket2_basis = None, bra2_basis = None):
    # This DOESN'T CHANGE the basis that braOket is stored in
    allbases = [i for i in [bra1_basis, ket1_basis, bra2_basis, ket2_basis] if i is not None]
    if len (allbases) == 0:
        raise RuntimeError ("needs at least one basis")
    bra1_basis = allbases[0]
    ket1_basis = bra1_basis if ket1_basis is None else ket1_basis
    bra2_basis = bra1_basis if bra2_basis is None else bra2_basis
    ket2_basis = bra2_basis if ket2_basis is None else ket2_basis
    the_bases = [bra1_basis, ket1_basis, bra2_basis, ket2_basis]
    if any ([basis.shape[1] == 0 for basis in the_bases]):
        return np.zeros_like (braOket)
    if all ([is_matrix_eye (basis) for basis in the_bases]):
        return braOket
    if len (braOket.shape) == 2:
        return project_operator_into_subspace_1body (braOket, bra1_basis, ket1_basis)
    elif len (braOket.shape) == 4:
        return project_operator_into_subspace_2body (braOket, bra1_basis, ket1_basis, bra2_basis, ket2_basis)
    else:
        raise ValueError ("Only one- and two-body operators (two- and four-index arrays) supported")

def project_operator_into_subspace_1body (braOket, bra_basis, ket_basis):
    lOr = np.asmatrix (braOket)

    l2p = np.asmatrix (bra_basis)
    p2l = l2p.H
    r2q = np.asmatrix (ket_basis)
    q2r = r2q.H

    lPl = l2p * p2l
    rPr = r2q * q2r
    return np.asarray (lPl * lOr * rPr)

def project_operator_into_subspace_2body (braOket, bra1_basis, ket1_basis, bra2_basis, ket2_basis):
    abcd = braOket

    l2a = np.asmatrix (bra1_basis)
    l2b = np.asmatrix (ket1_basis)
    l2c = np.asmatrix (bra2_basis)
    l2d = np.asmatrix (ket2_basis)

    abcd = np.einsum ('abcd,az->zbcd', abcd, np.asarray (l2a * l2a.H))
    abcd = np.einsum ('abcd,bz->azcd', abcd, np.asarray (l2b * l2b.H))
    abcd = np.einsum ('abcd,cz->abzd', abcd, np.asarray (l2c * l2c.H))
    abcd = np.einsum ('abcd,dz->abcz', abcd, np.asarray (l2d * l2d.H))
    return abcd



def compute_operator_trace_in_subset (the_operator, the_subset_basis):
    return np.trace (represent_operator_in_basis (the_operator, the_subset_basis))

compute_nelec_in_subspace = compute_operator_trace_in_subset



################    More complicated basis manipulation functions        ################



def get_overlapping_states (bra_basis, ket_basis, across_operator=None, max_nrvecs=0, max_nlvecs=0, num_zero_atol=params.num_zero_atol, only_nonzero_vals=True):
    c2p = np.asmatrix (bra_basis)
    c2q = np.asmatrix (ket_basis)
    cOc = 1 if across_operator is None else np.asmatrix (across_operator)
    assert (c2p.shape[0] == c2q.shape[0]), "you need to give the two spaces in the same basis"
    assert (c2p.shape[1] <= c2p.shape[0]), "you need to give the first state in a complete basis (c2p). Did you accidentally transpose it?"
    assert (c2q.shape[1] <= c2q.shape[0]), "you need to give the second state in a complete basis (c2q). Did you accidentally transpose it?"
    assert (max_nlvecs <= c2p.shape[1]), "you can't ask for more left states than are in your left space"
    assert (max_nrvecs <= c2q.shape[1]), "you can't ask for more right states than are in your right space"
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
    pevals, p2l = matrix_eigen_control_options (pQp, sort_vecs=True, only_nonzero_vals=only_nonzero_vals, round_zero_vals=True, num_zero_atol=num_zero_atol)
    qevals, q2r = matrix_eigen_control_options (qPq, sort_vecs=True, only_nonzero_vals=only_nonzero_vals, round_zero_vals=True, num_zero_atol=num_zero_atol)
    nsvals = min (c2p.shape[1], c2q.shape[1])
    pevals = pevals[:nsvals] # Has no effect if only_nonzero_vals==True, because len (pevals) couldn't possibly be > nsvals in that case
    qevals = qevals[:nsvals] # ditto
    try:
        svals = np.sqrt (np.mean ([pevals, qevals], axis=0))
    except ValueError: # If only_nonzero_vals==True, numerical noise might strip an eigenvalue on one side but not the other
        p2l, svals, q2l = matrix_svd_control_options (pOq, sort_vecs=True, only_nonzero_vals=only_nonzero_vals, full_matrices=False)

    # Get the left- and right-vectors back in the external basis
    c2l = c2p * p2l
    c2r = c2q * q2r

    # Truncate the basis if requested
    max_nlvecs = max_nlvecs or len (svals)
    max_nrvecs = max_nrvecs or len (svals)

    # But you can't truncate it smaller than it already is
    notes = (side for side in [(max_nlvecs, 'left'), (max_nrvecs, 'right')] if side[0] > len (svals))
    for requested, side in notes:
        head_str = "get_overlapping_states :: note : "
        note_1 = "{0} states projected into overlap space requested on the {1} side, but only {2} such pairs found ; ".format (requested, side, len (svals))
        note_2 = "returning only {0} states to caller".format (len (svals))
        print (head_str + note_1 + note_2)
    max_nlvecs = min (max_nrvecs, len (svals))
    max_nrvecs = min (max_nlvecs, len (svals))
    c2r = c2r[:,:max_nrvecs]
    c2l = c2l[:,:max_nlvecs]

    c2l, c2r, svals = (np.asarray (output) for output in (c2l, c2r, svals))
    return c2l, c2r, svals
    
def measure_basis_olap (bra_basis, ket_basis):
    if bra_basis.shape[1] == 0 or ket_basis.shape[1] == 0:
        return 0, 0
    svals = get_overlapping_states (bra_basis, ket_basis)[2]
    olap_ndf = len (svals)
    olap_mag = np.sum (svals * svals)
    return olap_mag, svals

def orthonormalize_a_basis (overlapping_basis, num_zero_atol=params.num_zero_atol):
    if (is_basis_orthonormal (overlapping_basis)):
        return overlapping_basis
    c2b = np.asmatrix (overlapping_basis)
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

    return np.asarray (c2n)

def get_states_from_projector (the_projector, num_zero_atol=params.num_zero_atol):
    proj_cc = np.asmatrix (the_projector)
    assert (np.allclose (proj_cc, proj_cc.H)), "projector must be hermitian\n" + str (proj_cc - proj_cc.H)
    assert (is_matrix_idempotent (proj_cc)), "projector must be idempotent\n" + str ((proj_cc * proj_cc) - proj_cc)
    evals, p2x = matrix_eigen_control_options (proj_cc, sort_vecs=True, only_nonzero_vals=True)
    return np.asarray (p2x)

def get_complementary_states (incomplete_basis, in_subspace = None, already_complete_warning=True):
    if incomplete_basis.shape[1] == 0:
        return np.eye (incomplete_basis.shape[0])
    orthonormal_basis = orthonormalize_a_basis (incomplete_basis)
    if is_basis_orthonormal_and_complete (orthonormal_basis) and already_complete_warning:
        print ("warning: tried to construct a complement for a basis that was already complete")
        return None

    c2b = np.asmatrix (orthonormal_basis)
    nstates_b = c2b.shape[1]
    nstates_c = c2b.shape[0]

    c2s = np.asmatrix (np.eye (c2b.shape[0], dtype=c2b.dtype)) if in_subspace is None else np.asmatrix (in_subspace)
    s2c = c2s.H
    cSc = c2s * s2c

    c2b = cSc * c2b
    b2c = c2b.H
    Projb_cc = c2b * b2c
    Projq_cc = np.eye (nstates_c, dtype=Projb_cc.dtype) - Projb_cc
    
    return get_states_from_projector (Projq_cc)

def get_complete_basis (incomplete_basis):
    complementary_states = get_complementary_states (incomplete_basis, already_complete_warning = False)
    if np.any (complementary_states):
        return np.append (incomplete_basis, complementary_states, axis=1)
    else:
        return incomplete_basis

def get_projector_from_states (the_states):
    l2p = np.asmatrix (the_states)
    p2l = l2p.H
    return np.asarray (l2p * p2l)

