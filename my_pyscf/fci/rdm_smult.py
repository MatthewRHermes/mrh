'''This file was generated from the execution of
/home/herme068/anaconda3/envs/pyscf/bin/python /home/herme068/gits/mrh/my_sympy/spin/lassi_tdms_spins/main.py
using mrh
GIT ORIG_HEAD 0decd1229805eaaf53f54d64b7c02e2723bcf083
GIT HEAD (branch sympy_docstrings) 0decd1229805eaaf53f54d64b7c02e2723bcf083
installed at
/home/herme068/anaconda3/envs/pyscf/lib/python3.12/site-packages/mrh
on 2025-09-17 17:32:06.347333'''
import numpy as np
from mrh.my_pyscf.fci import spin_op

def _get_highm_civecs (cibra, ciket, norb, nelec, dnelec, smult_bra, smult_ket):
    if smult_bra is None or smult_ket is None:
        return cibra, ciket, nelec
    nelec_ket = nelec
    nelec_bra = (nelec[0]+dnelec[0], nelec[1]+dnelec[1])
    cibra = spin_op.mup (cibra, norb, nelec_bra, smult_bra)
    ciket = spin_op.mup (ciket, norb, nelec_ket, smult_ket)
    nelec_bra = sum (nelec_bra)
    nelec_ket = sum (nelec_ket)
    dspin_op = dnelec[0]-dnelec[1]
    spin_ket = min (smult_ket-1, smult_bra-1-dspin_op)
    spin_bra = spin_ket + dspin_op
    nelec_bra = ((nelec_bra + spin_bra)//2, (nelec_bra-spin_bra)//2)
    nelec_ket = ((nelec_ket + spin_ket)//2, (nelec_ket-spin_ket)//2)
    cibra = spin_op.mdown (cibra, norb, nelec_bra, smult_bra)
    ciket = spin_op.mdown (ciket, norb, nelec_ket, smult_ket)
    return cibra, ciket, nelec_ket

def get_highm_civecs_h (cibra, ciket, norb, nelec, spin_op, smult_bra=None, smult_ket=None):
    '''Maximize the spin polarization quantum number m = 1/2 (na - nb) for a pair of CI vectors
    corresponding to the bra and ket vectors of a transition density matrix of the type

    <cp>, <cp' cq cr>

    and return the corresponding CI vectors. To say the same thing a different way: rotate the
    laboratory Z-axis parallel to either the bra or ket CI vectors, depending on which one leaves
    the spin vector of the operator string above unaltered. If either spin multiplicity quantum
    is omitted, it defaults to returning the input vectors unaltered.

    Args:
        cibra: ndarray or list of ndarrays
            CI vectors of the bra
        ciket: ndarray or list of ndarrays
            CI vectors of the ket
        norb: integer
            Number of orbitals in the CI vectors
        nelec: list of length 2 of integers
            Number of electrons of each spin in the ket CI vectors
        spin_op: 0 or 1
            identify spin sector of operator: alpha (0) or beta (1)

    Kwargs:
        smult_bra: integer
            spin multiplicity of the bra
        smult_ket: integer
            spin multiplicity of the ket

    Returns:
        cibra: ndarray or list of ndarrays
            CI vectors of the rotated bra
        ciket: ndarray or list of ndarrays
            CI vectors of the rotated ket
        nelec: list of length 2 of integers
            Number of electrons of each spin in the rotated ket CI vectors'''
    dneleca = (-1, 0)[spin_op]
    dnelecb = (0, -1)[spin_op]
    return _get_highm_civecs (cibra, ciket, norb, nelec, (dneleca,dnelecb), smult_bra, smult_ket)

_scale_h = [
    [lambda s,m: ((1/2)*np.sqrt(2)*np.sqrt(-m + s)*np.sqrt(m + s)*np.sqrt(m + s - 1)/(np.sqrt(s - 1)*np.sqrt(2*s - 1))),
     lambda s,m: ((1/2)*np.sqrt(-m + s)*np.sqrt(m + s)*np.sqrt(-m + s - 1)/np.sqrt(s - 1))],
    [lambda s,m: ((1/2)*np.sqrt(2*m + 2*s)/np.sqrt(s)),
     lambda s,m: (np.sqrt(-m + s))],
    [lambda s,m: (np.sqrt(-m + s + 1)),
     lambda s,m: (np.sqrt(m + s + 1)/np.sqrt(2*s + 1))],
    [lambda s,m: ((1/2)*np.sqrt(2)*np.sqrt(-m + s + 1)*np.sqrt(-m + s + 2)*np.sqrt(m + s + 1)/np.sqrt(2*s + 1)),
     lambda s,m: ((1/2)*np.sqrt(2)*np.sqrt(-m + s + 1)*np.sqrt(m + s + 1)*np.sqrt(m + s + 2)/np.sqrt(2*s**2 + 3*s + 1))]
    ]

def scale_h (smult_bra, spin_op, smult_ket, spin_ket):
    '''Compute the scale factor A(s',s,m) for the transition density matrices

    <s',m+m"|cp, cp' cq cr|s,m> = A(s',s,m) <s',m'+m"|cp, cp' cq cr|s,m'>

    where m' = min (s,s'-m") and m" is the spin sector of the operator
    not accounting for any transposition of spin sectors among the operators if present.

    Args:
        smult_bra: integer
            spin multiplicity of the bra
        spin_op: 0 or 1
            identify spin sector of operator: alpha (0) or beta (1)
        smult_ket: integer
            spin multiplicity of the ket
        spin_ket: integer
            2*spin polarization (= na - nb) in the ket

    Returns:
        A: float
            scale factor'''
    d2s_idx = (smult_bra - smult_ket + 3)//2
    if (d2s_idx < 0) or (d2s_idx >= 4): return 0
    s = (smult_ket-1)/2
    m = spin_ket/2
    return _scale_h[d2s_idx][spin_op] (s, m)

def get_highm_civecs_hh (cibra, ciket, norb, nelec, spin_op, smult_bra=None, smult_ket=None):
    '''Maximize the spin polarization quantum number m = 1/2 (na - nb) for a pair of CI vectors
    corresponding to the bra and ket vectors of a transition density matrix of the type

    <cp cq>

    and return the corresponding CI vectors. To say the same thing a different way: rotate the
    laboratory Z-axis parallel to either the bra or ket CI vectors, depending on which one leaves
    the spin vector of the operator string above unaltered. If either spin multiplicity quantum
    is omitted, it defaults to returning the input vectors unaltered.

    Args:
        cibra: ndarray or list of ndarrays
            CI vectors of the bra
        ciket: ndarray or list of ndarrays
            CI vectors of the ket
        norb: integer
            Number of orbitals in the CI vectors
        nelec: list of length 2 of integers
            Number of electrons of each spin in the ket CI vectors
        spin_op: 0, 1, or 2
            identify spin sector of operator: aa (0), ba (1), or bb (2)

    Kwargs:
        smult_bra: integer
            spin multiplicity of the bra
        smult_ket: integer
            spin multiplicity of the ket

    Returns:
        cibra: ndarray or list of ndarrays
            CI vectors of the rotated bra
        ciket: ndarray or list of ndarrays
            CI vectors of the rotated ket
        nelec: list of length 2 of integers
            Number of electrons of each spin in the rotated ket CI vectors'''
    dneleca = (-2, -1, 0)[spin_op]
    dnelecb = (0, -1, -2)[spin_op]
    return _get_highm_civecs (cibra, ciket, norb, nelec, (dneleca,dnelecb), smult_bra, smult_ket)

_scale_hh = [
    [lambda s,m: ((1/2)*np.sqrt(2)*np.sqrt((m + s - 1)/(2*s - 1))*np.sqrt(m + s)/np.sqrt(s)),
     lambda s,m: (np.sqrt((-m + s)/(2*s - 1))*np.sqrt(m + s)/(np.sqrt(2*s - 1)*np.sqrt(1/(2*s - 1)))),
     lambda s,m: ((1/2)*np.sqrt(2)*np.sqrt((-m + s - 1)/(2*s - 1))*np.sqrt(-m + s)/np.sqrt(1/(2*s - 1)))],
    [lambda s,m: ((1/2)*np.sqrt(2)*np.sqrt(m + s)*np.sqrt(-m + s + 1)/np.sqrt(s)),
     lambda s,m: (1),
     lambda s,m: ((1/2)*np.sqrt(2)*np.sqrt(-m + s)*np.sqrt(m + s + 1)/np.sqrt(s))],
    [lambda s,m: ((1/2)*np.sqrt(2)*np.sqrt(-m + s + 1)*np.sqrt(-m + s + 2)),
     lambda s,m: (np.sqrt(-m + s + 1)*np.sqrt(m + s + 1)*np.sqrt(2*s**2 + 3*s + 1)/np.sqrt(4*s**3 + 8*s**2 + 5*s + 1)),
     lambda s,m: ((1/2)*np.sqrt(2)*np.sqrt(m + s + 1)*np.sqrt(m + s + 2)/np.sqrt(2*s**2 + 3*s + 1))]
    ]

def scale_hh (smult_bra, spin_op, smult_ket, spin_ket):
    '''Compute the scale factor A(s',s,m) for the transition density matrices

    <s',m+m"|cp cq|s,m> = A(s',s,m) <s',m'+m"|cp cq|s,m'>

    where m' = min (s,s'-m") and m" is the spin sector of the operator
    not accounting for any transposition of spin sectors among the operators if present.

    Args:
        smult_bra: integer
            spin multiplicity of the bra
        spin_op: 0, 1, or 2
            identify spin sector of operator: aa (0), ba (1), or bb (2)
        smult_ket: integer
            spin multiplicity of the ket
        spin_ket: integer
            2*spin polarization (= na - nb) in the ket

    Returns:
        A: float
            scale factor'''
    d2s_idx = (smult_bra - smult_ket + 2)//2
    if (d2s_idx < 0) or (d2s_idx >= 3): return 0
    s = (smult_ket-1)/2
    m = spin_ket/2
    return _scale_hh[d2s_idx][spin_op] (s, m)

def get_highm_civecs_sm (cibra, ciket, norb, nelec, smult_bra=None, smult_ket=None):
    '''Maximize the spin polarization quantum number m = 1/2 (na - nb) for a pair of CI vectors
    corresponding to the bra and ket vectors of a transition density matrix of the type

    <bp' aq>

    and return the corresponding CI vectors. To say the same thing a different way: rotate the
    laboratory Z-axis parallel to either the bra or ket CI vectors, depending on which one leaves
    the spin vector of the operator string above unaltered. If either spin multiplicity quantum
    is omitted, it defaults to returning the input vectors unaltered.

    Args:
        cibra: ndarray or list of ndarrays
            CI vectors of the bra
        ciket: ndarray or list of ndarrays
            CI vectors of the ket
        norb: integer
            Number of orbitals in the CI vectors
        nelec: list of length 2 of integers
            Number of electrons of each spin in the ket CI vectors

    Kwargs:
        smult_bra: integer
            spin multiplicity of the bra
        smult_ket: integer
            spin multiplicity of the ket

    Returns:
        cibra: ndarray or list of ndarrays
            CI vectors of the rotated bra
        ciket: ndarray or list of ndarrays
            CI vectors of the rotated ket
        nelec: list of length 2 of integers
            Number of electrons of each spin in the rotated ket CI vectors'''
    dneleca = -1
    dnelecb = 1
    return _get_highm_civecs (cibra, ciket, norb, nelec, (dneleca,dnelecb), smult_bra, smult_ket)

_scale_sm = [
    lambda s,m: ((1/2)*np.sqrt(2)*np.sqrt(m + s)*np.sqrt(m + s - 1)/(np.sqrt(s)*np.sqrt(2*s - 1))),
    lambda s,m: ((1/2)*np.sqrt(2)*np.sqrt(m + s)*np.sqrt(-m + s + 1)/np.sqrt(s)),
    lambda s,m: ((1/2)*np.sqrt(2)*np.sqrt(-m + s + 1)*np.sqrt(-m + s + 2))
    ]

def scale_sm (smult_bra, smult_ket, spin_ket):
    '''Compute the scale factor A(s',s,m) for the transition density matrices

    <s',m-2|bp' aq|s,m> = A(s',s,m) <s',m'-2|bp' aq|s,m'>

    where m' = min (s,s'+2)
    not accounting for any transposition of spin sectors among the operators if present.

    Args:
        smult_bra: integer
            spin multiplicity of the bra
        smult_ket: integer
            spin multiplicity of the ket
        spin_ket: integer
            2*spin polarization (= na - nb) in the ket

    Returns:
        A: float
            scale factor'''
    d2s_idx = (smult_bra - smult_ket + 2)//2
    if (d2s_idx < 0) or (d2s_idx >= 3): return 0
    s = (smult_ket-1)/2
    m = spin_ket/2
    return _scale_sm[d2s_idx] (s, m)

def get_highm_civecs_dm (cibra, ciket, norb, nelec, smult_bra=None, smult_ket=None):
    '''Maximize the spin polarization quantum number m = 1/2 (na - nb) for a pair of CI vectors
    corresponding to the bra and ket vectors of a transition density matrix of the type

    <cp' cq>, <cp' cq' cr cs>

    and return the corresponding CI vectors. To say the same thing a different way: rotate the
    laboratory Z-axis parallel to either the bra or ket CI vectors, depending on which one leaves
    the spin vector of the operator string above unaltered. If either spin multiplicity quantum
    is omitted, it defaults to returning the input vectors unaltered.

    Args:
        cibra: ndarray or list of ndarrays
            CI vectors of the bra
        ciket: ndarray or list of ndarrays
            CI vectors of the ket
        norb: integer
            Number of orbitals in the CI vectors
        nelec: list of length 2 of integers
            Number of electrons of each spin in the ket CI vectors

    Kwargs:
        smult_bra: integer
            spin multiplicity of the bra
        smult_ket: integer
            spin multiplicity of the ket

    Returns:
        cibra: ndarray or list of ndarrays
            CI vectors of the rotated bra
        ciket: ndarray or list of ndarrays
            CI vectors of the rotated ket
        nelec: list of length 2 of integers
            Number of electrons of each spin in the rotated ket CI vectors'''
    dneleca = 0
    dnelecb = 0
    return _get_highm_civecs (cibra, ciket, norb, nelec, (dneleca,dnelecb), smult_bra, smult_ket)

_scale_dm = [
    lambda s,m: ((1/2)*np.sqrt(-m + s)*np.sqrt(m + s)*np.sqrt(-m + s - 1)*np.sqrt(m + s - 1)/np.sqrt(2*s**2 - 5*s + 3)),
    lambda s,m: (np.sqrt(-m + s)*np.sqrt(m + s)/np.sqrt(2*s - 1)),
    lambda s,m: (1)
    ]

def scale_dm (smult_bra, smult_ket, spin_ket):
    '''Compute the scale factor A(s',s,m) for the transition density matrices

    <s',m|cp' cq, cp' cq' cr cs|s,m> = A(s',s,m) <s',m'|cp' cq, cp' cq' cr cs|s,m'>

    where m' = min (s,s')
    not accounting for any transposition of spin sectors among the operators if present.

    Args:
        smult_bra: integer
            spin multiplicity of the bra
        smult_ket: integer
            spin multiplicity of the ket
        spin_ket: integer
            2*spin polarization (= na - nb) in the ket

    Returns:
        A: float
            scale factor'''
    if smult_bra > smult_ket:
        return scale_dm (smult_ket, smult_bra, spin_ket)
    d2s_idx = (smult_bra - smult_ket + 4)//2
    if (d2s_idx < 0) or (d2s_idx >= 3): return 0
    s = (smult_ket-1)/2
    m = spin_ket/2
    return _scale_dm[d2s_idx] (s, m)

_transpose_mdown_h = {}

def mdown_h (dm_0, smult_bra, spin_op, smult_ket, spin_ket):
    '''Obtain the transition density matrix

    <s',m+m"|cp|s,m>

    from its "high-m" representation in which at least one of the bra or ket spin vectors is
    parallel to the laboratory Z-axis:

    <s',m'+m"|cp|s,m'>

    where m' = min (s,s'-m") and m" is the spin sector of the operator.
    Note that this operation is not guaranteed to be invertible; for instance, it is impossible
    for a triplet to obtain the m=+1 case of the <a'a-b'b> matrix from the m=0 case, since the
    latter is zero by construction.

    Args:
        dm_0: ndarray of shape (*,nmo)
            Transition density matrix or matrices in the "high-m" representation
        smult_bra: integer
            spin multiplicity of the bra
        spin_op: 0 or 1
            identify spin sector of operator: alpha (0) or beta (1)
        smult_ket: integer
            spin multiplicity of the ket
        spin_ket: integer
            2*spin polarization (= na - nb) in the ket

    Returns:
        dm_1: ndarray of shape (*,nmo)
            Transition density matrix or matrices with 2m = spin_ket'''
    dm_1 = dm_0 * scale_h (smult_bra, spin_op, smult_ket, spin_ket)
    old_shape = dm_1.shape
    temp_shape = (-1,) + old_shape[-1:]
    dm_1 = dm_1.reshape (*temp_shape)
    d2s_idx = smult_bra-smult_ket 
    key = (d2s_idx, spin_op)
    transpose = _transpose_mdown_h.get (key, lambda x, s, m: x)
    s = (smult_ket - 1) / 2
    m = spin_ket / 2
    return transpose (dm_1, s, m).reshape (*old_shape)

def _transpose_mdown_phh_0(dm_0, s, m):
    dm_1 = np.empty_like (dm_0)
    if round (m,1) == round (s,1):
        dm_1[:,0] = ((1) * dm_0[:,0])
        dm_1[:,1] = ((1) * dm_0[:,1])
    else:
        dm_1[:,0] = (((m + s - 1)/(2*s - 1)) * dm_0[:,0]
                     + ((-m + s)/(2*s - 1)) * dm_0[:,1]
                     + ((m - s)/(2*s - 1)) * dm_0[:,1].transpose (0,3,2,1))
        dm_1[:,1] = (((-m + s)/(2*s - 1)) * dm_0[:,0]
                     + ((m + s - 1)/(2*s - 1)) * dm_0[:,1]
                     + ((-m + s)/(2*s - 1)) * dm_0[:,1].transpose (0,3,2,1))
    return dm_1
def _transpose_mdown_phh_1(dm_0, s, m):
    dm_1 = np.empty_like (dm_0)
    if round (m,1) == round (s - 1,1):
        dm_1[:,0] = ((1) * dm_0[:,0])
        dm_1[:,1] = ((1) * dm_0[:,1])
    else:
        dm_1[:,0] = (((1/2)*(-m + s - 1)/(2*s - 1)) * dm_0[:,1]
                     + ((1/2)*(-m + s - 1)/(2*s - 1)) * dm_0[:,0].transpose (0,3,2,1)
                     + ((1/2)*(m + 3*s - 1)/(2*s - 1)) * dm_0[:,0])
        dm_1[:,1] = (((1/2)*(m + 3*s - 1)/(2*s - 1)) * dm_0[:,1]
                     + ((1/2)*(m - s + 1)/(2*s - 1)) * dm_0[:,0].transpose (0,3,2,1)
                     + ((1/2)*(-m + s - 1)/(2*s - 1)) * dm_0[:,0])
    return dm_1
def _transpose_mdown_phh_2(dm_0, s, m):
    dm_1 = np.empty_like (dm_0)
    if round (m,1) == round (s,1):
        dm_1[:,0] = ((1) * dm_0[:,0])
        dm_1[:,1] = ((1) * dm_0[:,1])
    else:
        dm_1[:,0] = (((1/4)*(m - s)/s) * dm_0[:,1].transpose (0,3,2,1)
                     + ((1/4)*(m + 3*s)/s) * dm_0[:,0]
                     + ((1/4)*(-m + s)/s) * dm_0[:,1])
        dm_1[:,1] = (((1/4)*(-m + s)/s) * dm_0[:,1].transpose (0,3,2,1)
                     + ((1/4)*(-m + s)/s) * dm_0[:,0]
                     + ((1/4)*(m + 3*s)/s) * dm_0[:,1])
    return dm_1
def _transpose_mdown_phh_3(dm_0, s, m):
    dm_1 = np.empty_like (dm_0)
    if round (m,1) == round (s,1):
        dm_1[:,0] = ((1) * dm_0[:,0])
        dm_1[:,1] = ((1) * dm_0[:,1])
    else:
        dm_1[:,0] = (((1/2)*(-m + s)/s) * dm_0[:,0].transpose (0,3,2,1)
                     + ((1/2)*(m + s)/s) * dm_0[:,0]
                     + ((1/2)*(-m + s)/s) * dm_0[:,1])
        dm_1[:,1] = (((1/2)*(m - s)/s) * dm_0[:,0].transpose (0,3,2,1)
                     + ((1/2)*(-m + s)/s) * dm_0[:,0]
                     + ((1/2)*(m + s)/s) * dm_0[:,1])
    return dm_1
_transpose_mdown_phh = {(-1, 0): _transpose_mdown_phh_0,
                        (-1, 1): _transpose_mdown_phh_1,
                        (1, 0): _transpose_mdown_phh_2,
                        (1, 1): _transpose_mdown_phh_3}

def mdown_phh (dm_0, smult_bra, spin_op, smult_ket, spin_ket):
    '''Obtain the transition density matrix

    <s',m+m"|cp' cq cr|s,m>

    from its "high-m" representation in which at least one of the bra or ket spin vectors is
    parallel to the laboratory Z-axis:

    <s',m'+m"|cp' cq cr|s,m'>

    where m' = min (s,s'-m") and m" is the spin sector of the operator.
    Note that this operation is not guaranteed to be invertible; for instance, it is impossible
    for a triplet to obtain the m=+1 case of the <a'a-b'b> matrix from the m=0 case, since the
    latter is zero by construction.

    Args:
        dm_0: ndarray of shape (*,2,nmo,nmo,nmo)
            Transition density matrix or matrices in the "high-m" representation
        smult_bra: integer
            spin multiplicity of the bra
        spin_op: 0 or 1
            identify spin sector of operator: alpha (0) or beta (1)
        smult_ket: integer
            spin multiplicity of the ket
        spin_ket: integer
            2*spin polarization (= na - nb) in the ket

    Returns:
        dm_1: ndarray of shape (*,2,nmo,nmo,nmo)
            Transition density matrix or matrices with 2m = spin_ket'''
    dm_1 = dm_0 * scale_h (smult_bra, spin_op, smult_ket, spin_ket)
    old_shape = dm_1.shape
    temp_shape = (-1,) + old_shape[-4:]
    dm_1 = dm_1.reshape (*temp_shape)
    d2s_idx = smult_bra-smult_ket 
    key = (d2s_idx, spin_op)
    transpose = _transpose_mdown_phh.get (key, lambda x, s, m: x)
    s = (smult_ket - 1) / 2
    m = spin_ket / 2
    return transpose (dm_1, s, m).reshape (*old_shape)

def _transpose_mdown_hh_0(dm_0, s, m):
    dm_1 = np.empty_like (dm_0)
    if round (m,1) == round (s,1):
        dm_1[:] = ((1) * dm_0[:])
    else:
        dm_1[:] = (((1/2)*(m + s)/s) * dm_0[:]
                   + ((1/2)*(-m + s)/s) * dm_0[:].transpose (0,2,1))
    return dm_1
_transpose_mdown_hh = {(0, 1): _transpose_mdown_hh_0}

def mdown_hh (dm_0, smult_bra, spin_op, smult_ket, spin_ket):
    '''Obtain the transition density matrix

    <s',m+m"|cp cq|s,m>

    from its "high-m" representation in which at least one of the bra or ket spin vectors is
    parallel to the laboratory Z-axis:

    <s',m'+m"|cp cq|s,m'>

    where m' = min (s,s'-m") and m" is the spin sector of the operator.
    Note that this operation is not guaranteed to be invertible; for instance, it is impossible
    for a triplet to obtain the m=+1 case of the <a'a-b'b> matrix from the m=0 case, since the
    latter is zero by construction.

    Args:
        dm_0: ndarray of shape (*,nmo,nmo)
            Transition density matrix or matrices in the "high-m" representation
        smult_bra: integer
            spin multiplicity of the bra
        spin_op: 0, 1, or 2
            identify spin sector of operator: aa (0), ba (1), or bb (2)
        smult_ket: integer
            spin multiplicity of the ket
        spin_ket: integer
            2*spin polarization (= na - nb) in the ket

    Returns:
        dm_1: ndarray of shape (*,nmo,nmo)
            Transition density matrix or matrices with 2m = spin_ket'''
    dm_1 = dm_0 * scale_hh (smult_bra, spin_op, smult_ket, spin_ket)
    old_shape = dm_1.shape
    temp_shape = (-1,) + old_shape[-2:]
    dm_1 = dm_1.reshape (*temp_shape)
    d2s_idx = smult_bra-smult_ket 
    key = (d2s_idx, spin_op)
    transpose = _transpose_mdown_hh.get (key, lambda x, s, m: x)
    s = (smult_ket - 1) / 2
    m = spin_ket / 2
    return transpose (dm_1, s, m).reshape (*old_shape)

_transpose_mdown_sm = {}

def mdown_sm (dm_0, smult_bra, smult_ket, spin_ket):
    '''Obtain the transition density matrix

    <s',m-2|bp' aq|s,m>

    from its "high-m" representation in which at least one of the bra or ket spin vectors is
    parallel to the laboratory Z-axis:

    <s',m'-2|bp' aq|s,m'>

    where m' = min (s,s'+2).
    Note that this operation is not guaranteed to be invertible; for instance, it is impossible
    for a triplet to obtain the m=+1 case of the <a'a-b'b> matrix from the m=0 case, since the
    latter is zero by construction.

    Args:
        dm_0: ndarray of shape (*,nmo,nmo)
            Transition density matrix or matrices in the "high-m" representation
        smult_bra: integer
            spin multiplicity of the bra
        smult_ket: integer
            spin multiplicity of the ket
        spin_ket: integer
            2*spin polarization (= na - nb) in the ket

    Returns:
        dm_1: ndarray of shape (*,nmo,nmo)
            Transition density matrix or matrices with 2m = spin_ket'''
    dm_1 = dm_0 * scale_sm (smult_bra, smult_ket, spin_ket)
    old_shape = dm_1.shape
    temp_shape = (-1,) + old_shape[-2:]
    dm_1 = dm_1.reshape (*temp_shape)
    d2s_idx = smult_bra-smult_ket 
    key = (d2s_idx, 0)
    transpose = _transpose_mdown_sm.get (key, lambda x, s, m: x)
    s = (smult_ket - 1) / 2
    m = spin_ket / 2
    return transpose (dm_1, s, m).reshape (*old_shape)

def _transpose_mdown_dm1_0(dm_0, s, m):
    dm_1 = np.empty_like (dm_0)
    if round (m,1) == round (s,1):
        dm_1[:,0] = ((1) * dm_0[:,0])
        dm_1[:,1] = ((1) * dm_0[:,1])
    elif round (m,1) == round (-s,1):
        dm_1[:,0] = ((1) * dm_0[:,1])
        dm_1[:,1] = ((1) * dm_0[:,0])
    else:
        dm_1[:,0] = (((1/2)*(m + s)/s) * dm_0[:,0]
                     + ((1/2)*(-m + s)/s) * dm_0[:,1])
        dm_1[:,1] = (((1/2)*(-m + s)/s) * dm_0[:,0]
                     + ((1/2)*(m + s)/s) * dm_0[:,1])
    return dm_1
_transpose_mdown_dm1 = {(0, 0): _transpose_mdown_dm1_0}

def mdown_dm1 (dm_0, smult_bra, smult_ket, spin_ket):
    '''Obtain the transition density matrix

    <s',m|cp' cq|s,m>

    from its "high-m" representation in which at least one of the bra or ket spin vectors is
    parallel to the laboratory Z-axis:

    <s',m'|cp' cq|s,m'>

    where m' = min (s,s').
    Note that this operation is not guaranteed to be invertible; for instance, it is impossible
    for a triplet to obtain the m=+1 case of the <a'a-b'b> matrix from the m=0 case, since the
    latter is zero by construction.

    Args:
        dm_0: ndarray of shape (*,2,nmo,nmo)
            Transition density matrix or matrices in the "high-m" representation
        smult_bra: integer
            spin multiplicity of the bra
        smult_ket: integer
            spin multiplicity of the ket
        spin_ket: integer
            2*spin polarization (= na - nb) in the ket

    Returns:
        dm_1: ndarray of shape (*,2,nmo,nmo)
            Transition density matrix or matrices with 2m = spin_ket'''
    dm_1 = dm_0 * scale_dm (smult_bra, smult_ket, spin_ket)
    old_shape = dm_1.shape
    temp_shape = (-1,) + old_shape[-3:]
    dm_1 = dm_1.reshape (*temp_shape)
    d2s_idx = -abs(smult_bra-smult_ket) 
    key = (d2s_idx, 0)
    transpose = _transpose_mdown_dm1.get (key, lambda x, s, m: x)
    s = (max (smult_bra, smult_ket) - 1) / 2
    m = spin_ket / 2
    return transpose (dm_1, s, m).reshape (*old_shape)

def _transpose_mdown_dm2_0(dm_0, s, m):
    dm_1 = np.empty_like (dm_0)
    if round (m,1) == round (s - 1,1):
        dm_1[:,0] = ((1) * dm_0[:,0])
        dm_1[:,1] = ((1) * dm_0[:,1])
        dm_1[:,2] = ((1) * dm_0[:,2])
        dm_1[:,3] = ((1) * dm_0[:,3])
    elif round (m,1) == round (1 - s,1):
        dm_1[:,0] = ((-1) * dm_0[:,3])
        dm_1[:,1] = ((1) * dm_0[:,3]
                     + (1) * dm_0[:,0]
                     + (1) * dm_0[:,1])
        dm_1[:,2] = ((-1) * dm_0[:,3].transpose (0,1,4,3,2)
                     + (-1) * dm_0[:,0].transpose (0,1,4,3,2)
                     + (1) * dm_0[:,2])
        dm_1[:,3] = ((-1) * dm_0[:,0])
    else:
        dm_1[:,0] = (((1/2)*(m - s + 1)/(s - 1)) * dm_0[:,3]
                     + ((1/2)*(m + s - 1)/(s - 1)) * dm_0[:,0])
        dm_1[:,1] = (((1/2)*(-m + s - 1)/(s - 1)) * dm_0[:,3]
                     + ((1/2)*(-m + s - 1)/(s - 1)) * dm_0[:,0]
                     + (1) * dm_0[:,1])
        dm_1[:,2] = ((-1/2*(-m + s - 1)/(s - 1)) * dm_0[:,3].transpose (0,1,4,3,2)
                     + (-1/2*(-m + s - 1)/(s - 1)) * dm_0[:,0].transpose (0,1,4,3,2)
                     + (1) * dm_0[:,2])
        dm_1[:,3] = (((1/2)*(m + s - 1)/(s - 1)) * dm_0[:,3]
                     + ((1/2)*(m - s + 1)/(s - 1)) * dm_0[:,0])
    return dm_1
def _transpose_mdown_dm2_1(dm_0, s, m):
    dm_1 = np.empty_like (dm_0)
    if round (m,1) == round (s,1):
        dm_1[:,0] = ((1) * dm_0[:,0])
        dm_1[:,1] = ((1) * dm_0[:,1])
        dm_1[:,2] = ((1) * dm_0[:,2])
        dm_1[:,3] = ((1) * dm_0[:,3])
    elif round (m,1) == round (-s,1):
        dm_1[:,0] = ((1) * dm_0[:,3])
        dm_1[:,1] = ((1) * dm_0[:,2])
        dm_1[:,2] = ((1) * dm_0[:,1])
        dm_1[:,3] = ((1) * dm_0[:,0])
    else:
        dm_1[:,0] = (((1/2)*(-m**2 + s**2)/(s*(2*s - 1))) * dm_0[:,1]
                     + ((1/2)*(m**2 - s**2)/(s*(2*s - 1))) * dm_0[:,2].transpose (0,1,4,3,2)
                     + ((1/2)*(m**2 + 2*m*s - m + s**2 - s)/(s*(2*s - 1))) * dm_0[:,0]
                     + ((1/2)*(-m**2 + s**2)/(s*(2*s - 1))) * dm_0[:,2]
                     + ((1/2)*(m**2 - 2*m*s + m + s**2 - s)/(s*(2*s - 1))) * dm_0[:,3]
                     + ((1/2)*(m**2 - s**2)/(s*(2*s - 1))) * dm_0[:,1].transpose (0,1,4,3,2))
        dm_1[:,1] = (((1/2)*(m**2 + 2*m*s - m + s**2 - s)/(s*(2*s - 1))) * dm_0[:,1]
                     + ((1/2)*(-m**2 + s**2)/(s*(2*s - 1))) * dm_0[:,2].transpose (0,1,4,3,2)
                     + ((1/2)*(-m**2 + s**2)/(s*(2*s - 1))) * dm_0[:,0]
                     + ((1/2)*(m**2 - 2*m*s + m + s**2 - s)/(s*(2*s - 1))) * dm_0[:,2]
                     + ((1/2)*(-m**2 + s**2)/(s*(2*s - 1))) * dm_0[:,3]
                     + ((1/2)*(-m**2 + s**2)/(s*(2*s - 1))) * dm_0[:,1].transpose (0,1,4,3,2))
        dm_1[:,2] = (((1/2)*(m**2 + 2*m*s - m + s**2 - s)/(s*(2*s - 1))) * dm_0[:,2]
                     + ((1/2)*(-m**2 + s**2)/(s*(2*s - 1))) * dm_0[:,1].transpose (0,1,4,3,2)
                     + (-1/2*(-m**2 + s**2)/(s*(2*s - 1))) * dm_0[:,0].transpose (0,1,4,3,2)
                     + ((1/2)*(m**2 - 2*m*s + m + s**2 - s)/(s*(2*s - 1))) * dm_0[:,1]
                     + (-1/2*(-m**2 + s**2)/(s*(2*s - 1))) * dm_0[:,3].transpose (0,1,4,3,2)
                     + ((1/2)*(-m**2 + s**2)/(s*(2*s - 1))) * dm_0[:,2].transpose (0,1,4,3,2))
        dm_1[:,3] = (((1/2)*(-m**2 + s**2)/(s*(2*s - 1))) * dm_0[:,1]
                     + ((1/2)*(m**2 - s**2)/(s*(2*s - 1))) * dm_0[:,2].transpose (0,1,4,3,2)
                     + ((1/2)*(m**2 - 2*m*s + m + s**2 - s)/(s*(2*s - 1))) * dm_0[:,0]
                     + ((1/2)*(-m**2 + s**2)/(s*(2*s - 1))) * dm_0[:,2]
                     + ((1/2)*(m**2 + 2*m*s - m + s**2 - s)/(s*(2*s - 1))) * dm_0[:,3]
                     + ((1/2)*(m**2 - s**2)/(s*(2*s - 1))) * dm_0[:,1].transpose (0,1,4,3,2))
    return dm_1
_transpose_mdown_dm2 = {(-2, 0): _transpose_mdown_dm2_0,
                        (0, 0): _transpose_mdown_dm2_1}

def mdown_dm2 (dm_0, smult_bra, smult_ket, spin_ket):
    '''Obtain the transition density matrix

    <s',m|cp' cq' cr cs|s,m>

    from its "high-m" representation in which at least one of the bra or ket spin vectors is
    parallel to the laboratory Z-axis:

    <s',m'|cp' cq' cr cs|s,m'>

    where m' = min (s,s').
    Note that this operation is not guaranteed to be invertible; for instance, it is impossible
    for a triplet to obtain the m=+1 case of the <a'a-b'b> matrix from the m=0 case, since the
    latter is zero by construction.

    Args:
        dm_0: ndarray of shape (*,4,nmo,nmo,nmo,nmo)
            Transition density matrix or matrices in the "high-m" representation
        smult_bra: integer
            spin multiplicity of the bra
        smult_ket: integer
            spin multiplicity of the ket
        spin_ket: integer
            2*spin polarization (= na - nb) in the ket

    Returns:
        dm_1: ndarray of shape (*,4,nmo,nmo,nmo,nmo)
            Transition density matrix or matrices with 2m = spin_ket'''
    dm_1 = dm_0 * scale_dm (smult_bra, smult_ket, spin_ket)
    old_shape = dm_1.shape
    temp_shape = (-1,) + old_shape[-5:]
    dm_1 = dm_1.reshape (*temp_shape)
    d2s_idx = -abs(smult_bra-smult_ket) 
    key = (d2s_idx, 0)
    transpose = _transpose_mdown_dm2.get (key, lambda x, s, m: x)
    s = (max (smult_bra, smult_ket) - 1) / 2
    m = spin_ket / 2
    return transpose (dm_1, s, m).reshape (*old_shape)

