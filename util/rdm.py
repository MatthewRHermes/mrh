import numpy as np
from .la import matrix_eigen_control_options
from .basis import represent_operator_in_basis


def get_1RDM_from_OEI (one_electron_hamiltonian, nocc):
    evals, evecs = matrix_eigen_control_options (one_electron_hamiltonian, sort_vecs=True)
    l2p = np.asmatrix (evecs[:,::-1])[:,:nocc]
    p2l = l2p.H
    return np.asarray (l2p * p2l)

def get_1RDM_from_OEI_in_subspace (one_electron_hamiltonian, subspace_basis, nocc_subspace, num_zero_atol):
    l2w = np.asmatrix (subspace_basis)
    w2l = l2w.H
    OEI_wrk = represent_operator_in_basis (one_electron_hamiltonian, l2w)
    oneRDM_wrk = get_1RDM_from_OEI (OEI_wrk, nocc_subspace)
    oneRDM_loc = represent_operator_in_basis (oneRDM_wrk, w2l)
    return oneRDM_loc
    

