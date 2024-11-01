import numpy as np

def yamaguchi (e_roots, s2, highsmult=None):
    ''' Evaluate the Yamaguchi formula 

    J = (ELS - EHS) / (<S2>HS - <S2>LS)

    in cm-1

    Args:
        e_roots: sequence
            State energies
        s2: sequence
            Spin expectation values of states

    Kwargs:
        highsmult: integer
            High-spin spin multiplicity. Parity of the wave function
            (i.e. whether low-spin is S=0 or S=1/2) is derived from this.
            Defaults to maximum value of s2

    Returns:
        J: float
            Magnetic coupling parameter in cm-1
    '''
    e_roots, s2 = np.asarray (e_roots), np.asarray (s2)
    if highsmult is not None:
        highs = (highsmult - 1) / 2
        highs2 = np.around (highs * (highs+1), 2)
    else:
        highs2 = np.around (np.amax (s2), 2)
    lows2 = highs2 - np.floor (highs2)
    idx = np.argsort (e_roots)
    e_roots = e_roots[idx]
    s2 = s2[idx]
    idx_hs = (np.around (s2, 2) == highs2)
    assert (np.count_nonzero (idx_hs)), 'high-spin ground state not found {} {}'.format (np.around (s2,2), highs2)
    idx_hs = np.where (idx_hs)[0][0]
    e_hs = e_roots[idx_hs]
    idx_ls = (np.around (s2, 2) == lows2)
    assert (np.count_nonzero (idx_ls)), 'low-spin ground state not found {} {}'.format (np.around (s2,2), lows2)
    idx_ls = np.where (idx_ls)[0][0]
    e_ls = e_roots[idx_ls]
    j = (e_ls - e_hs) / 24
    from pyscf.data import nist
    au2cm = nist.HARTREE2J / nist.PLANCK / nist.LIGHT_SPEED_SI * 1e-2
    return j*au2cm


