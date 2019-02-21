# A collection of useful manipulations of basis sets (i.e., rectangular matrices) and operators (square matrices)

import sys
import numpy as np
from scipy import linalg
from mrh.util.io import prettyprint_ndarray
from mrh.util.la import is_matrix_zero, is_matrix_eye, is_matrix_idempotent, matrix_eigen_control_options, matrix_svd_control_options, align_vecs
from mrh.util import params
from itertools import combinations
from math import sqrt
import copy

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

def is_basis_orthonormal (the_basis, ovlp=1, rtol=params.num_zero_rtol, atol=params.num_zero_atol):
    cOc = np.atleast_2d (ovlp) 
    c2b = np.asarray (the_basis)
    b2c = c2b.conjugate ().T
    try:
        test_matrix = b2c @ cOc @ c2b
    except ValueError:
        test_matrix = (b2c * cOc[0,0]) @ c2b
    rtol *= test_matrix.shape[0]
    atol *= test_matrix.shape[0]
    return is_matrix_eye (test_matrix, rtol=rtol, atol=atol)

def is_basis_orthonormal_and_complete (the_basis, rtol=params.num_zero_rtol, atol=params.num_zero_atol):
    return (is_basis_orthonormal (the_basis, rtol=rtol, atol=atol) and (the_basis.shape[1] == the_basis.shape[0]))

def are_bases_orthogonal (bra_basis, ket_basis, ovlp=1, rtol=params.num_zero_rtol, atol=params.num_zero_atol):
    test_matrix = basis_olap (bra_basis, ket_basis, ovlp)
    rtol *= sqrt (bra_basis.shape[1] * ket_basis.shape[1])
    atol *= sqrt (bra_basis.shape[1] * ket_basis.shape[1])
    return is_matrix_zero (test_matrix, rtol=rtol, atol=atol), test_matrix

def are_bases_equivalent (bra_basis, ket_basis, ovlp=1, rtol=params.num_zero_rtol, atol=params.num_zero_atol):
    bra_basis = orthonormalize_a_basis (bra_basis)
    ket_basis = orthonormalize_a_basis (ket_basis)
    if bra_basis.shape[1] != ket_basis.shape[1]: return False
    svals = get_overlapping_states (bra_basis, ket_basis, only_nonzero_vals=True)[2]
    rtol *= sqrt (bra_basis.shape[1] * ket_basis.shape[1])
    atol *= sqrt (bra_basis.shape[1] * ket_basis.shape[1])
    return np.allclose (svals, 1, rtol=rtol, atol=atol)
    


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

def basis_olap (bra_basis, ket_basis, ovlp=1):
    c2p = np.asmatrix (bra_basis)
    c2q = np.asmatrix (ket_basis)
    p2c = c2p.H
    cOc = np.asmatrix (ovlp)
    try:
        return np.asarray (p2c * cOc * c2q)
    except ValueError:
        return np.asarray (p2c * cOc[0,0] * c2q)

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
    try:
        return np.asarray (p2l * lOr * r2q)
    except ValueError as err:
        print (p2l.shape)
        print (lOr.shape)
        print (r2q.shape)
        raise (err)


def represent_operator_in_basis_2body (braOket, bra1_basis, ket1_basis, bra2_basis, ket2_basis):
    abcd = braOket
    az = np.conj (bra1_basis)
    bz = ket1_basis
    cz = np.conj (bra2_basis)
    dz = ket2_basis


    #abcd = np.einsum ('abcd,az->zbcd',abcd,az)
    #abcd = np.einsum ('abcd,bz->azcd',abcd,bz)
    #abcd = np.einsum ('abcd,cz->abzd',abcd,cz)
    #abcd = np.einsum ('abcd,dz->abcz',abcd,dz)
    # Order matters when doing this with tensordot! It puts the remaining
    # axes in the order that the tensors are supplied as arguments.
    abcd = np.tensordot (bz, abcd, axes=(0,1)) # xacd 
    abcd = np.tensordot (az, abcd, axes=(0,1)) # wxcd
    abcd = np.tensordot (abcd, cz, axes=(2,0)) # wxdy
    abcd = np.tensordot (abcd, dz, axes=(2,0)) # wxyz
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

    #abcd = np.einsum ('abcd,az->zbcd', abcd, np.asarray (l2a * l2a.H))
    #abcd = np.einsum ('abcd,bz->azcd', abcd, np.asarray (l2b * l2b.H))
    #abcd = np.einsum ('abcd,cz->abzd', abcd, np.asarray (l2c * l2c.H))
    #abcd = np.einsum ('abcd,dz->abcz', abcd, np.asarray (l2d * l2d.H))
    # Order matters when doing this with tensordot! It puts the remaining
    # axes in the order that the tensors are supplied as arguments.
    abcd = np.tensordot (np.asarray (l2b * l2b.H), abcd, axes=(0,1)) # xacd
    abcd = np.tensordot (np.asarray (l2a * l2a.H), abcd, axes=(0,1)) # wxcd
    abcd = np.tensordot (abcd, np.asarray (l2c * l2c.H), axes=(2,0)) # wxdy
    abcd = np.tensordot (abcd, np.asarray (l2d * l2d.H), axes=(2,0)) # wxyz
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

    try:
        p2l, svals, q2r = matrix_svd_control_options (pOq, sort_vecs=-1, only_nonzero_vals=only_nonzero_vals, num_zero_atol=num_zero_atol)
    except ValueError as e:
        if 0 in pOq.shape:
            print ("get_overlapping_states: one of bra or ket basis is zero size; returning bra and ket basis unchanged")
            return bra_basis, ket_basis, np.zeros (0)
        else:
            print (pOq.shape)
            raise (e)
            

    '''
    pQp = pOq * pOq.H
    qPq = pOq.H * pOq
    pevals, p2l = matrix_eigen_control_options (pQp, sort_vecs=-1, only_nonzero_vals=only_nonzero_vals, round_zero_vals=True, num_zero_atol=num_zero_atol)
    qevals, q2r = matrix_eigen_control_options (qPq, sort_vecs=-1, only_nonzero_vals=only_nonzero_vals, round_zero_vals=True, num_zero_atol=num_zero_atol)
    nsvals = min (c2p.shape[1], c2q.shape[1])
    pevals = pevals[:nsvals] # Has no effect if only_nonzero_vals==True, because len (pevals) couldn't possibly be > nsvals in that case
    qevals = qevals[:nsvals] # ditto
    try:
        svals = np.sqrt (np.mean ([pevals, qevals], axis=0))
    except ValueError: # If only_nonzero_vals==True, numerical noise might strip an eigenvalue on one side but not the other
        p2l, svals, q2l = matrix_svd_control_options (pOq, sort_vecs=-1, only_nonzero_vals=only_nonzero_vals, full_matrices=False)
    '''

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

def orthonormalize_a_basis (overlapping_basis, symm_blocks=None, ovlp=1, num_zero_atol=params.num_zero_atol):
    if (is_basis_orthonormal (overlapping_basis)):
        return overlapping_basis

    c2b = np.asmatrix (overlapping_basis)
    b2c = c2b.H
    cOc = ovlp
    if isinstance (ovlp, np.ndarray):
        cOc = np.asmatrix (ovlp)
    bOb = b2c * cOc * c2b
    assert (not is_matrix_zero (bOb)), "overlap matrix is zero! problem with basis?"
    assert (np.allclose (bOb, bOb.H)), "overlap matrix not hermitian! problem with basis?"
    assert (np.abs (np.trace (bOb)) > num_zero_atol), "overlap matrix zero or negative trace! problem with basis?"
     
    evals, evecs = matrix_eigen_control_options (bOb, sort_vecs=-1, only_nonzero_vals=True)
    if len (evals) == 0:
        return np.zeros ((c2b.shape[0], 0), dtype=c2b.dtype)
    p2x = np.asmatrix (evecs)
    c2x = c2b * p2x 
    assert (not np.any (evals < 0)), "overlap matrix has negative eigenvalues! problem with basis?"

    # I want c2n = c2x * x2n
    # x2n defined such that n2c * c2n = I
    # n2x * x2c * c2x * x2n = n2x * evals_xx * x2n = I
    # therefore
    # x2n = evals_xx^{-1/2}
    x2n = np.asmatrix (np.diag (np.reciprocal (np.sqrt (evals))))
    c2n = c2x * x2n
    n2c = c2n.H
    nOn = n2c * cOc * c2n
    if not is_basis_orthonormal (c2n):
        # Assuming numerical problem due to massive degeneracy; remove constant from diagonal to improve solver?
        assert (np.all (np.isclose (np.diag (nOn), 1))), np.diag (nOn) - 1
        nOn[np.diag_indices_from (nOn)] -= 1
        evals, evecs = matrix_eigen_control_options (nOn, sort_vecs=-1, only_nonzero_vals=False)
        n2x = np.asmatrix (evecs)
        c2x = c2n * n2x
        x2n = np.asmatrix (np.diag (np.reciprocal (np.sqrt (evals + 1))))
        c2n = c2x * x2n
        n2c = c2n.H
        nOn = n2c * cOc * c2n
        assert (is_basis_orthonormal (c2n)), "failed to orthonormalize basis even after two tries somehow\n" + str (
            prettyprint_ndarray (nOn)) + "\n" + str (np.linalg.norm (nOn - np.eye (c2n.shape[1]))) + "\n" + str (evals)

    return np.asarray (c2n)

def get_states_from_projector (the_projector, num_zero_atol=params.num_zero_atol):
    proj_cc = np.asmatrix (the_projector)
    assert (np.allclose (proj_cc, proj_cc.H)), "projector must be hermitian\n" + str (np.linalg.norm (proj_cc - proj_cc.H))
    assert (is_matrix_idempotent (proj_cc)), "projector must be idempotent\n" + str (np.linalg.norm ((proj_cc * proj_cc) - proj_cc))
    evals, evecs = matrix_eigen_control_options (proj_cc, sort_vecs=-1, only_nonzero_vals=True)
    idx = np.isclose (evals, 1)
    return evecs[:,idx]

def get_complementary_states (incomplete_basis, symm_blocks=None, already_complete_warning=True, atol=params.num_zero_atol):
    if incomplete_basis.shape[1] == 0:
        if symm_blocks is None:
            return np.eye (incomplete_basis.shape[0])
        else:
            return symm_blocks
    orthonormal_basis = orthonormalize_a_basis (incomplete_basis)

    # Symmetry wrapper
    if symm_blocks is not None:
        if not isinstance (symm_blocks[0], np.ndarray):
            raise RuntimeError ("You need to pass the actual symmetry basis, I can't just guess how many states are supposed to be in each irrep!")
        c2p = align_states (orthonormal_basis, symm_blocks)
        labels = assign_blocks_weakly (c2p, symm_blocks)
        c2q = []
        for idx, c2s in enumerate (symm_blocks):
            if np.count_nonzero (labels==idx) == 0:
                c2q.append (c2s)
                continue
            s2p = c2s.conjugate ().T @ c2p[:,labels==idx]
            s2q = get_complementary_states (s2p, already_complete_warning=False)
            c2q.append (c2s @ s2q)
        # Yadda yadda linear algebra breaks orthogonality
        c2q = np.concatenate (c2q, axis=1)
        ovlp_PQ = c2p.conjugate ().T @ c2q
        assert (are_states_block_adapted (c2q, symm_blocks))
        assert (is_basis_orthonormal (c2q))
        if not are_bases_orthogonal (c2p, c2q):
            assert (linalg.norm (ovlp_PQ) / sum (ovlp_PQ.shape) < 1e-8)
            proj_PP = c2p @ c2p.conjugate ().T
            c2q -= proj_PP @ c2q
            c2q = align_states (c2q, symm_blocks)
            assert (are_bases_orthogonal (c2p, c2q))
        return c2q

    # Kernel
    nbas = orthonormal_basis.shape[1]
    if is_basis_orthonormal_and_complete (orthonormal_basis):
        if already_complete_warning:
            print ("warning: tried to construct a complement for a basis that was already complete")
        return np.zeros ((incomplete_basis.shape[0], 0))
    Q, R = linalg.qr (orthonormal_basis)
    assert (are_bases_equivalent (Q[:,:nbas], orthonormal_basis))
    assert (are_bases_orthogonal (Q[:,nbas:], orthonormal_basis))
    '''
    err = linalg.norm (ovlp[:nbas,:].T @ ovlp[:nbas,:]) - np.eye (nbas)) / nbas
    assert (abs (err) < 1e-8), err
    err = linalg.norm (ovlp[nbas:,:]) / nbas
    assert (abs (err) < 1e-8), err
    '''
    return orthonormalize_a_basis (Q[:,nbas:])


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




################    symmetry block manipulations   ################



# Should work with overlapping states!
def is_operator_block_adapted (the_operator, the_blocks, tol=params.num_zero_atol):
    tol *= the_operator.shape[0]
    if isinstance (the_blocks[0], np.ndarray):
        umat = np.concatenate (the_blocks, axis=1)
        assert (is_basis_orthonormal_and_complete (umat)), 'Symmetry blocks must be orthonormal and complete, {}'.format (len (the_blocks))
        operator_block = represent_operator_in_basis (the_operator, umat)
        labels = np.concatenate ([[idx,] * blk.shape[1] for idx, blk in enumerate (the_blocks)])
        return is_operator_block_adapted (operator_block, labels)
    iterable = the_blocks if isinstance (the_blocks[0], np.ndarray) else np.unique (the_blocks)
    offblk_operator = the_operator.copy ()
    for blk in np.unique (the_blocks):
        offblk_operator[np.ix_(the_blocks==blk,the_blocks==blk)] = 0
    return is_matrix_zero (offblk_operator, atol=tol)

# Should work with overlapping states!
def is_subspace_block_adapted (the_basis, the_blocks, tol=params.num_zero_atol):
    return is_operator_block_adapted (the_basis @ the_basis.conjugate ().T, the_blocks, tol=tol)

# Should work with overlapping states!
def are_states_block_adapted (the_basis, the_blocks, atol=params.num_zero_atol, rtol=params.num_zero_rtol):
    if not is_subspace_block_adapted (the_basis, the_blocks, tol=atol): return False
    atol *= the_basis.shape[0]
    rtol *= the_basis.shape[0]
    for blk in the_blocks:
        projector = blk @ blk.conjugate ().T
        is_symm = ((projector @ the_basis) * the_basis).sum (0)
        if not (np.all (np.logical_or (np.isclose (is_symm, 0, atol=atol, rtol=rtol),
                                       np.isclose (is_symm, 1, atol=atol, rtol=rtol)))): return False
    return True

# Should work with overlapping states!
def assign_blocks (the_basis, the_blocks, atol=params.num_zero_atol, rtol=params.num_zero_rtol):
    assert (is_subspace_block_adapted (the_basis, the_blocks, tol=atol)), 'Basis space must be block-adapted before assigning states'
    atol *= the_basis.shape[0]
    rtol *= the_basis.shape[0]
    labels = -np.ones (the_basis.shape[1], dtype=int)
    for idx, blk in enumerate (the_blocks):
        projector = blk @ blk.conjugate ().T
        is_symm = ((projector @ the_basis) * the_basis).sum (0)
        check = np.all (np.logical_or (np.isclose (is_symm, 0, atol=atol, rtol=rtol), np.isclose (is_symm, 1, atol=atol, rtol=rtol)))
        assert (check), 'Basis states must be individually block-adapted before being assigned (is_symm = {} for label {})'.format (is_symm, idx)
        labels[np.isclose(is_symm, 1, atol=atol, rtol=rtol)] = idx
    assert (np.all (labels>=0)), 'Failed to assign states {}'.format (np.where (labels<0)[0])
    return labels.astype (int)
    
def symmetrize_basis (the_basis, the_blocks, sorting_metric=None, sort_vecs=1, do_eigh_metric=True, check_metric_block_adapted=True, atol=params.num_zero_atol, rtol=params.num_zero_rtol):
    atol_scl = atol * the_basis.shape[0]
    rtol_scl = rtol * the_basis.shape[0]
    if the_blocks is None: the_blocks=[np.eye (the_basis.shape[0])]
    assert (is_subspace_block_adapted (the_basis, the_blocks, tol=atol)), 'Basis space must be block-adapted before blockifying states'
    if are_states_block_adapted (the_basis, the_blocks, atol=atol, rtol=rtol):
        symmetrized_basis = the_basis
        labels = assign_blocks (the_basis, the_blocks)
    else:
        orthonormal_basis = orthonormalize_a_basis (the_basis)
        labels = []
        symmetrized_basis = []
        for idx, blk in enumerate (the_blocks):
            c2s_blk, c2s_p, svals = get_overlapping_states (blk, orthonormal_basis, only_nonzero_vals=True, num_zero_atol=1e-3)
            assert (np.all (np.isclose (svals, 1, atol=atol_scl, rtol=rtol_scl))), 'Failed to find block-adapted states in block {}; svals = {}'.format (idx, svals)
            labels.append (idx * svals)
            symmetrized_basis.append (c2s_blk)
        labels = np.around (np.concatenate (labels)).astype (int)
        symmetrized_basis = np.concatenate (symmetrized_basis, axis=1)
        assert (is_basis_orthonormal (symmetrized_basis, atol=atol, rtol=rtol)), "? labels = {}".format (labels)

    if sorting_metric is None:
        return symmetrized_basis, labels
    else:
        if sorting_metric.shape[0] == the_basis.shape[0]:
            metric_symm = represent_operator_in_basis (sorting_metric, symmetrized_basis)
        else:
            assert (sorting_metric.shape[0] == the_basis.shape[1]), 'The sorting metric must be in either the row or column basis of the orbital matrix that is being symmetrized'
            metric_symm = represent_operator_in_basis (sorting_metric, the_basis.conjugate ().T @ symmetrized_basis)
        if check_metric_block_adapted: assert (is_operator_block_adapted (metric_symm, labels, tol=atol))
        metric_evals, evecs, labels = matrix_eigen_control_options (metric_symm, symm_blocks=labels, sort_vecs=sort_vecs, only_nonzero_vals=False, num_zero_atol=atol)
        symmetrized_basis = symmetrized_basis @ evecs
        return symmetrized_basis, labels, metric_evals

def align_states (unaligned_states, the_blocks, sorting_metric=None, sort_vecs=1, atol=params.num_zero_atol, rtol=params.num_zero_rtol):
    ''' Symmbreak-tolerant '''
    unaligned_states = orthonormalize_a_basis (unaligned_states)
    if the_blocks is None: the_blocks=[np.eye (unaligned_states.shape[0])]
    block_umat = np.concatenate (the_blocks, axis=1)
    assert (is_basis_orthonormal_and_complete (block_umat)), 'Symmetry blocks must be orthonormal and complete, {}'.format (len (the_blocks))

    if sorting_metric is None: sorting_metric=np.diag (np.arange (unaligned_states.shape[1]))
    if sorting_metric.shape[0] == unaligned_states.shape[1]: sorting_metric=represent_operator_in_basis (sorting_metric, unaligned_states.conjugate ().T)
    block_idx = np.concatenate ([[idx,] * blk.shape[1] for idx, blk in enumerate (the_blocks)])

    c2u = unaligned_states
    c2s = block_umat
    s2c = c2s.conjugate ().T
    s2u = s2c @ c2u
    s2a = align_vecs (s2u, block_idx)[0]
    c2a = c2s @ s2a
    aligned_states = c2a

    sortval = ((sorting_metric @ aligned_states) * aligned_states).sum (0)
    aligned_states = aligned_states[:,np.argsort (sortval)[::sort_vecs]]
    assert (are_bases_equivalent (unaligned_states, aligned_states)), linalg.norm (ovlp - np.eye (ovlp.shape[0]))
    return aligned_states

def eigen_weaksymm (the_matrix, the_blocks, subspace=None, sort_vecs=1, only_nonzero_vals=False, atol=params.num_zero_atol, rtol=params.num_zero_rtol):
    if the_blocks is None: the_blocks=[np.eye (the_matrix.shape[0])]
    if subspace is None: subspace = np.eye (the_matrix.shape[0])
    subspace_matrix = represent_operator_in_basis (the_matrix, subspace)
    evals, evecs = matrix_eigen_control_options (subspace_matrix, symm_blocks=None, sort_vecs=sort_vecs, only_nonzero_vals=only_nonzero_vals,
        num_zero_atol=atol)
    evecs = subspace @ evecs
    idx_unchk = np.ones (len (evals), dtype=np.bool_)
    while np.count_nonzero (idx_unchk > 0):
        chk_1st_eval = evals[idx_unchk][0]
        idx_degen = np.isclose (evals, chk_1st_eval, rtol=rtol, atol=atol)
        if np.count_nonzero (idx_degen) > 1:
            evecs[:,idx_degen] = align_states (evecs[:,idx_degen], the_blocks, atol=atol, rtol=rtol)
        idx_unchk[idx_degen] = False
    return evals, evecs, assign_blocks_weakly (evecs, the_blocks)

def assign_blocks_weakly (the_states, the_blocks):
    projectors = [blk @ blk.conjugate ().T for blk in the_blocks]
    vals = np.stack ([((proj @ the_states) * the_states).sum (0) for proj in projectors], axis=-1)
    return np.argmax (vals, axis=1)

def cleanup_operator_symmetry (the_operator, the_blocks):
    if the_blocks is None or len (the_blocks) == 1:
        return the_operator
    if not isinstance (the_blocks[0], np.ndarray):
        dummy_blocks = [np.eye (the_operator.shape[0])[:,the_blocks==lbl] for lbl in np.unique (the_blocks)]
        return cleanup_operator_symmetry (the_operator, dummy_blocks)
    trashbin = np.zeros_like (the_operator)
    for blk1, blk2 in combinations (the_blocks, 2):
        trashbin += project_operator_into_subspace (the_operator, blk1, blk2)
        trashbin += project_operator_into_subspace (the_operator, blk2, blk1)
    print ("Norm of matrix elements thrown in the trash: {}".format (linalg.norm (trashbin)))
    the_operator -= trashbin
    assert (is_operator_block_adapted (the_operator, the_blocks))
    return the_operator

def analyze_operator_blockbreaking (the_operator, the_blocks, block_labels=None):
    if block_labels is None: block_labels = np.arange (len (the_blocks))
    if isinstance (the_blocks[0], np.ndarray):
        c2s = np.concatenate (the_blocks, axis=1)
        assert (is_basis_orthonormal_and_complete (c2s)), "Symmetry block problem? Not a complete, orthonormal basis."
        blocked_operator = represent_operator_in_basis (the_operator, c2s)
        blocked_idx = np.concatenate ([[idx,] * blk.shape[1] for idx, blk in enumerate (the_blocks)])
        c2l, op_svals, c2r = analyze_operator_blockbreaking (blocked_operator, blocked_idx, block_labels=block_labels)
        c2l = [c2s @ s2l for s2l in c2l]
        c2r = [c2s @ s2r for s2r in c2r]
        return c2l, op_svals, c2r
    elif np.asarray (the_blocks).dtype == np.asarray (block_labels).dtype:
        the_indices = np.empty (len (the_blocks), dtype=int)
        for idx, lbl in enumerate (block_labels):
            idx_indices = (the_blocks == lbl)
            the_indices[idx_indices] = idx
        the_blocks = the_indices
    c2l = []
    c2r = []
    op_svals = []
    norbs = the_operator.shape[0]
    my_range = [idx for idx, bl in enumerate (block_labels) if idx in the_blocks]
    for idx1, idx2 in combinations (my_range, 2):
        blk1 = block_labels[idx1]
        blk2 = block_labels[idx2]
        idx12 = np.ix_(the_blocks==idx1, the_blocks==idx2)
        lvecs = np.eye (norbs, dtype=the_operator.dtype)[:,the_blocks==idx1]
        rvecs = np.eye (norbs, dtype=the_operator.dtype)[:,the_blocks==idx2]
        mat12 = the_operator[idx12]
        if is_matrix_zero (mat12):
            c2l.append (np.zeros ((norbs,0), dtype=the_operator.dtype))
            c2r.append (np.zeros ((norbs,0), dtype=the_operator.dtype))
            op_svals.append (np.zeros ((0), dtype=the_operator.dtype))
            continue
        try:
            vecs1, svals, vecs2 = matrix_svd_control_options (mat12, sort_vecs=-1, only_nonzero_vals=False)
            lvecs = lvecs @ vecs1
            rvecs = rvecs @ vecs2
        except ValueError as e:
            if the_operator[idx12].size > 0: raise (e)
            c2l.append (np.zeros ((norbs,0), dtype=the_operator.dtype))
            c2r.append (np.zeros ((norbs,0), dtype=the_operator.dtype))
            op_svals.append (np.zeros ((0), dtype=the_operator.dtype))
            continue
        #print ("Coupling between {} and {}: {} svals, norm = {}".format (idx1, idx2, len (svals), linalg.norm (svals)))
        c2l.append (lvecs)
        c2r.append (rvecs)
        op_svals.append (svals)
    return c2l, op_svals, c2r

def measure_operator_blockbreaking (the_operator, the_blocks, block_labels=None):
    op_svals = np.concatenate (analyze_operator_blockbreaking (the_operator, the_blocks, block_labels=block_labels)[1])
    if len (op_svals) == 0: return 0,0
    return np.amax (np.abs (op_svals)), linalg.norm (op_svals)

def analyze_subspace_blockbreaking (the_basis, the_blocks, block_labels=None):
    projector = the_basis @ the_basis.conjugate ().T
    return analyze_operator_blockbreaking (projector, the_blocks, block_labels=block_labels)

def measure_subspace_blockbreaking (the_basis, the_blocks, block_labels=None):
    projector = the_basis @ the_basis.conjugate ().T
    return measure_operator_blockbreaking (projector, the_blocks, block_labels=block_labels)


