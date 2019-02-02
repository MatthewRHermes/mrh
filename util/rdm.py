import numpy as np
import itertools
from scipy import linalg
from math import factorial
from mrh.util.la import *
from mrh.util.basis import *
from mrh.util.my_math import is_close_to_integer
from mrh.util import params
from mrh.util.io import warnings

def get_1RDM_from_OEI (one_electron_hamiltonian, nocc):
    evals, evecs = matrix_eigen_control_options (one_electron_hamiltonian, sort_vecs=1, only_nonzero_vals=False)
    l2p = np.asmatrix (evecs[:,:nocc])
    p2l = l2p.H
    return np.asarray (l2p * p2l)

def get_1RDM_from_OEI_in_subspace (one_electron_hamiltonian, subspace_basis, nocc_subspace, num_zero_atol):
    l2w = np.asmatrix (subspace_basis)
    w2l = l2w.H
    OEI_wrk = represent_operator_in_basis (one_electron_hamiltonian, l2w)
    oneRDM_wrk = get_1RDM_from_OEI (OEI_wrk, nocc_subspace)
    oneRDM_loc = represent_operator_in_basis (oneRDM_wrk, w2l)
    return oneRDM_loc
    
def Schmidt_decompose_1RDM (the_1RDM, loc2frag, norbs_bath_max, bath_tol=1e-5, num_zero_atol=params.num_zero_atol):
    norbs_tot = assert_matrix_square (the_1RDM)
    norbs_frag = loc2frag.shape[1]
    assert (norbs_tot >= norbs_frag and loc2frag.shape[0] == norbs_tot)
    assert (is_basis_orthonormal (loc2frag)), linalg.norm (np.dot (loc2frag.T, loc2frag) - np.eye (loc2frag.shape[1]))
    norbs_env = norbs_tot - norbs_frag
    nelec_tot = np.trace (the_1RDM)

    # We need to SVD the environment-fragment block of the 1RDM
    # The bath states are from the left-singular vectors corresponding to nonzero singular value
    # The fragment semi-natural orbitals are from the right-singular vectors of ~any~ singular value
    # Note that only ~entangled~ fragment orbitals are returned so don't overwrite loc2frag!
    loc2env = get_complementary_states (loc2frag)
    loc2bath, loc2efrag, svals = get_overlapping_states (loc2env, loc2frag, across_operator=the_1RDM, only_nonzero_vals=False)

    # If I specified that I want less than the maxiumum possible number of bath orbitals, I need to implement that here
    norbs_bath = min (np.count_nonzero (np.abs (svals)>bath_tol), norbs_bath_max)
    dropped_svals_norm = 0 if len (svals) == norbs_bath else linalg.norm (svals[norbs_bath:])
    svals = svals[:norbs_bath]
    loc2bath = loc2bath[:,:norbs_bath]

    # Check that I haven't messed up the entangled orbitals so far
    loc2ent = np.append (loc2efrag, loc2bath, axis=1)
    assert (is_basis_orthonormal (loc2ent))

    # Get unentangled natural orbitals, separated into fragment and core. Unfortunately, it will be necessary to do ~3~ diagonalizations here, because of degeneracy leading to
    # undetermined rotations.
    Pfrag_loc = np.dot (loc2frag, loc2frag.conjugate ().T)
    loc2une = get_complementary_states (loc2ent)
    pfrag, pfrag_evecs = matrix_eigen_control_options (represent_operator_in_basis (Pfrag_loc, loc2une))
    loc2une = np.dot (loc2une, pfrag_evecs)
    idx_frag = np.isclose (pfrag, 1)
    idx_core = np.isclose (pfrag, 0)
    # Check that math works
    assert (np.all (np.logical_or (idx_frag, idx_core))), pfrag
    loc2ufrag = loc2une[:,idx_frag]
    loc2core  = loc2une[:,idx_core]
    no_occs_frag, no_evecs_frag = matrix_eigen_control_options (represent_operator_in_basis (the_1RDM, loc2ufrag))
    no_occs_core, no_evecs_core = matrix_eigen_control_options (represent_operator_in_basis (the_1RDM, loc2core))
    loc2ufrag = np.dot (loc2ufrag, no_evecs_frag)
    loc2core  = np.dot (loc2core, no_evecs_core)
    print ("Found {} unentangled fragment orbitals and {} core orbitals".format (loc2ufrag.shape[1], loc2core.shape[1]))

    # Build embedding basis: frag (efrag then ufrag, check that this is complete!), bath, core. Check for zero frag-core entanglement
    loc2frag = np.append (loc2efrag, loc2ufrag, axis=1)
    assert (is_matrix_eye (represent_operator_in_basis (Pfrag_loc, loc2frag)))
    errmat = represent_operator_in_basis (the_1RDM, loc2frag, loc2core)
    assert (loc2core.shape[1] == 0 or is_matrix_zero (errmat))
    loc2imp = np.append (loc2frag, loc2bath, axis=1)
    assert (is_basis_orthonormal (loc2imp))
    loc2emb = np.append (loc2imp, loc2core, axis=1)
    assert (is_basis_orthonormal_and_complete (loc2emb))

    # Calculate the number of electrons in the would-be impurity model
    nelec_imp = compute_nelec_in_subspace (the_1RDM, loc2imp)

    # Check the fidelity of the diagonalizations
    test = represent_operator_in_basis (the_1RDM, loc2emb)
    test[np.diag_indices_from (test)] = 0
    idx_rectdiag = np.diag_indices (norbs_bath)
    test_fe = test[:norbs_frag,norbs_frag:]
    svals_test = np.copy (test_fe[idx_rectdiag])
    test_fe[idx_rectdiag] = 0
    test_ef = test[norbs_frag:,:norbs_frag]
    test_ef[idx_rectdiag] = 0
    print ("Schmidt decomposition total diagonal error: {}".format (linalg.norm (test)))
    sec = ('frag', 'bath', 'core')
    lim = (0, norbs_frag, norbs_frag+norbs_bath, norbs_tot)
    for i, j in itertools.product (range (3), repeat=2):
        test_view = test[lim[i]:lim[i+1],lim[j]:lim[j+1]]
        #print ("Schmidt decomposition {}-{} block diagonal error: {}".format (sec[i],sec[j],linalg.norm(test_view)))
    print ("Schmidt decomposition svals error: {}".format (linalg.norm (svals - svals_test)))
    print ("Schmidt decomposition smallest sval: {}".format (np.amin (np.insert (np.abs (svals), 0, 0))))
    
    return loc2emb, norbs_bath, nelec_imp

def electronic_energy_orbital_decomposition (norbs_tot, OEI=None, oneRDM=None, TEI=None, twoRDM=None):
    E_bas = np.zeros (norbs_tot)
    if (OEI is not None) and (oneRDM is not None):
        # Let's make sure that matrix-multiplication doesn't mess me up
        OEI     = np.asarray (OEI)
        oneRDM  = np.asarray (oneRDM)
        prod    = OEI * oneRDM
        E_bas  += 0.5 * np.einsum ('ij->i', prod)[:norbs_tot]
        E_bas  += 0.5 * np.einsum ('ij->j', prod)[:norbs_tot]
    if (TEI is not None) and (twoRDM is not None):
        # Let's make sure that matrix-multiplication doesn't mess me up
        TEI    = np.asarray (TEI)
        twoRDM = np.asarray (twoRDM)
        prod = TEI * twoRDM
        E_bas += (0.125 * np.einsum ('ijkl->i', prod))[:norbs_tot]
        E_bas += (0.125 * np.einsum ('ijkl->j', prod))[:norbs_tot]
        E_bas += (0.125 * np.einsum ('ijkl->k', prod))[:norbs_tot]
        E_bas += (0.125 * np.einsum ('ijkl->l', prod))[:norbs_tot]
    return E_bas

def get_E_from_RDMs (EIs, RDMs):
    energy = 0.0
    for EI, RDM in zip (EIs, RDMs):
        pref    = 1.0 / factorial (len (EI.shape))
        EI      = np.ravel (np.asarray (EI))
        RDM     = np.ravel (np.asarray (RDM))
        energy += pref * np.dot (EI, RDM)
    return energy

def idempotize_1RDM (oneRDM, thresh):
    evals, evecs = linalg.eigh (oneRDM)
    diff_evals = (2.0 * np.around (evals / 2.0)) - evals
    # Only allow evals to change by at most +-thresh
    idx_floor = np.where (diff_evals < -abs (thresh))[0]
    idx_ceil  = np.where (diff_evals >  abs (thresh))[0]
    diff_evals[idx_floor] = -abs(thresh)
    diff_evals[idx_ceil]  =  abs(thresh)
    nelec_diff = np.sum (diff_evals)
    new_evals = evals + diff_evals
    new_oneRDM = represent_operator_in_basis (np.diag (new_evals), evecs.T)
    return new_oneRDM, nelec_diff

def Schmidt_decomposition_idempotent_wrapper (working_1RDM, loc2wfrag, norbs_bath_max, bath_tol=1e-5, idempotize_thresh=0, num_zero_atol=0):
    norbs_tot = loc2wfrag.shape[0]
    norbs_wfrag = loc2wfrag.shape[1]
    loc2wemb, norbs_wbath, nelec_wimp = Schmidt_decompose_1RDM (working_1RDM, loc2wfrag, norbs_bath_max, bath_tol=bath_tol)
    norbs_wimp  = norbs_wfrag + norbs_wbath
    norbs_wcore = norbs_tot - norbs_wimp
    loc2wimp  = loc2wemb[:,:norbs_wimp]
    loc2wcore = loc2wemb[:,norbs_wimp:]
    print ("Schmidt decomposition found {0} bath orbitals for this fragment, of an allowed total of {1}".format (norbs_wbath, norbs_bath_max))
    print ("Schmidt decomposition found {0} electrons in this impurity".format (nelec_wimp))
    working_1RDM_core = np.zeros(working_1RDM.shape)
    if norbs_wcore > 0:
        working_1RDM_core = project_operator_into_subspace (working_1RDM, loc2wcore)
        if abs (idempotize_thresh) > num_zero_atol:
            working_1RDM_core, nelec_wcore_diff = idempotize_1RDM (working_1RDM_core, idempotize_thresh)
            nelec_wimp -= nelec_wcore_diff
            print ("After attempting to idempotize the core (part of the putatively idempotent guide) 1RDM with a threshold of "
            + "{0}, {1} electrons were found in the impurity".format (idempotize_thresh, nelec_wimp))
    if not is_close_to_integer (nelec_wimp / 2, num_zero_atol):
        raise RuntimeError ("Can't solve impurity problems without even-integer number of electrons! nelec_wimp={0}".format (nelec_wimp))
    return loc2wemb, norbs_wbath, int (round (nelec_wimp)), working_1RDM_core

def get_2CDM_from_2RDM (twoRDM, oneRDMs):
    oneRDMs = np.asarray (oneRDMs)
    if len (oneRDMs.shape) < 3:
        #warnings.warn ("requires spin-separated 1-RDM - approximating as [1/2 dm, 1/2 dm]", RuntimeWarning)
        oneRDM = oneRDMs.copy ()
        oneRDMs = oneRDM / 2
        oneRDMs = np.stack ((oneRDMs, oneRDMs), axis=0)
    else:
        oneRDM = oneRDMs[0] + oneRDMs[1]
    #twoCDM  = twoRDM - np.einsum ('pq,rs->pqrs', oneRDM, oneRDM)
    #twoCDM +=    0.5 * np.einsum ('ps,rq->pqrs', oneRDM, oneRDM)
    twoCDM  = twoRDM.copy ()
    twoCDM -= np.multiply.outer (oneRDM, oneRDM)
    twoCDM += np.multiply.outer (oneRDMs[0], oneRDMs[0]).transpose (0, 3, 2, 1)
    twoCDM += np.multiply.outer (oneRDMs[1], oneRDMs[1]).transpose (0, 3, 2, 1)
    return twoCDM

def get_2CDMs_from_2RDMs (twoRDM, oneRDMs):
    ''' PySCF stores spin-separated twoRDMs as (aa, ab, bb) '''
    oneRDMs = np.asarray (oneRDMs)
    if len (oneRDMs.shape) < 3:
        #warnings.warn ("requires spin-separated 1-RDM - approximating as [1/2 dm, 1/2 dm]", RuntimeWarning)
        oneRDM = oneRDMs.copy ()
        oneRDMs = oneRDM / 2
        oneRDMs = np.stack ((oneRDMs, oneRDMs), axis=0)
    #twoCDM  = twoRDM - np.einsum ('pq,rs->pqrs', oneRDM, oneRDM)
    #twoCDM +=    0.5 * np.einsum ('ps,rq->pqrs', oneRDM, oneRDM)
    twoCDM = [i.copy () for i in twoRDM]
    twoCDM[0] -= np.multiply.outer (oneRDMs[0], oneRDMs[0])
    twoCDM[1] -= np.multiply.outer (oneRDMs[0], oneRDMs[1]) 
    twoCDM[2] -= np.multiply.outer (oneRDMs[1], oneRDMs[1])
    twoCDM[0] += np.multiply.outer (oneRDMs[0], oneRDMs[0]).transpose (0, 3, 2, 1)
    twoCDM[2] += np.multiply.outer (oneRDMs[1], oneRDMs[1]).transpose (0, 3, 2, 1)
    return tuple (twoCDM)

def get_2RDM_from_2CDM (twoCDM, oneRDMs):
    oneRDMs = np.asarray (oneRDMs)
    if len (oneRDMs.shape) < 3:
        #warnings.warn ("requires spin-separated 1-RDM - approximating as [1/2 dm, 1/2 dm]", RuntimeWarning)
        oneRDM = oneRDMs.copy ()
        oneRDMs = oneRDM / 2
        oneRDMs = np.stack ((oneRDMs, oneRDMs), axis=0)
    else:
        oneRDM = oneRDMs[0] + oneRDMs[1]
    #twoRDM  = twoCDM + np.einsum ('pq,rs->pqrs', oneRDM, oneRDM)
    #twoRDM -=    0.5 * np.einsum ('ps,rq->pqrs', oneRDM, oneRDM)
    twoRDM  = twoCDM.copy ()
    twoRDM += np.multiply.outer (oneRDM, oneRDM)
    twoRDM -= np.multiply.outer (oneRDMs[0], oneRDMs[0]).transpose (0, 3, 2, 1)
    twoRDM -= np.multiply.outer (oneRDMs[1], oneRDMs[1]).transpose (0, 3, 2, 1)
    return twoRDM

def get_2RDMs_from_2CDMs (twoCDM, oneRDMs):
    ''' PySCF stores spin-separated twoRDMs as (aa, ab, bb) '''
    oneRDMs = np.asarray (oneRDMs)
    if len (oneRDMs.shape) < 3:
        #warnings.warn ("requires spin-separated 1-RDM - approximating as [1/2 dm, 1/2 dm]", RuntimeWarning)
        oneRDM = oneRDMs.copy ()
        oneRDMs = oneRDM / 2
        oneRDMs = np.stack ((oneRDMs, oneRDMs), axis=0)
    #twoCDM  = twoRDM - np.einsum ('pq,rs->pqrs', oneRDM, oneRDM)
    #twoCDM +=    0.5 * np.einsum ('ps,rq->pqrs', oneRDM, oneRDM)
    twoRDM = [i.copy () for i in twoCDM]
    twoRDM[0] += np.multiply.outer (oneRDMs[0], oneRDMs[0])
    twoRDM[1] += np.multiply.outer (oneRDMs[0], oneRDMs[1])
    twoRDM[2] += np.multiply.outer (oneRDMs[1], oneRDMs[1])
    twoRDM[0] -= np.multiply.outer (oneRDMs[0], oneRDMs[0]).transpose (0, 3, 2, 1)
    twoRDM[2] -= np.multiply.outer (oneRDMs[1], oneRDMs[1]).transpose (0, 3, 2, 1)
    return tuple (twoRDM)



def S2_exptval (oneDM, twoDM, Nelec=None, cumulant=False):
    ''' Evaluate S^2 expectation value from spin-summed one- and two-body density matrices.
        <S^2> = 1/4 (3N - sum_pq [2P_pqqp + P_ppqq])
              = Tr[D-(D**2)/2] - 1/2 sum_pq L_pqqp

        Args:

        oneDM: ndarray of shape = (M,M)
            spin-summed one-body density matrix

        twoDM: ndarray of shape = (M,M,M,M)
            spin-summed two-body density matrix if cumulant == False or density cumulant if cumulant == True

        Kwargs:

        Nelec: int, default = None
            if not supplied, is computed as trace of oneDM

        cumulant: bool, default = False
            whether the cumulant expansion is used for oneDM and twoDM
    '''

    if not cumulant:
        twoDM = get_2CDM_from_2RDM (twoDM, oneDM)

    return np.sum (np.diag (oneDM) - np.einsum ('pq,qp->p', oneDM, oneDM)/2 - np.einsum ('pqqp->p', twoDM)/2)



