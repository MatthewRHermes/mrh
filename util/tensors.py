import numpy as np
import itertools
from .la import matrix_eigen_control_options, assert_matrix_square
from .basis import represent_operator_in_basis, get_complementary_states, get_overlapping_states, get_complete_basis, is_basis_orthonormal, is_basis_orthonormal_and_complete, compute_nelec_in_subspace


def symmetrize_tensor_conj (tensor):
    # tensors are by default in Mulliken/chemist's order
    # The even indices are bra and the odd indices are ket
    # So the permutation is simply [1, 0, 3, 2, 5, 4, ...]
    perm = tuple(sum ([[x+1, x] for x in range (0, len(tensor.shape), 2)],[]))
    tensor = 0.5 * (tensor + np.conj (np.transpose (tensor, perm)))
    return tensor

def symmetrize_tensor_elec (tensor):
    # tensors are by default in Mulliken/chemists order
    nelec = len (tensor.shape) // 2
    if nelec == 1:
        return tensor
    norbs = tensor.shape[0]
    ngems = norbs * norbs
    orbshape = tuple(norbs for x in range (nelec*2))
    gemshape = tuple(ngems for x in range (nelec))
    tensor = tensor.reshape(gemshape)
    ###
    start_perm = range (nelec)
    allperms = tuple (itertools.permutations (range (nelec)))
    tensor *= (1.0 / len (allperms))
    for perm in allperms[1:]:
        tensor += np.transpose (tensor, perm)
    ###
    tensor = tensor.reshape (orbshape) 
    return tensor

def symmetrize_tensor (tensor):
    return symmetrize_tensor_elec (symmetrize_tensor_conj (tensor))

