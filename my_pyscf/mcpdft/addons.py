import numpy as np

def dm2_cumulant (dm2, dm1s):
    '''
    Evaluate the spin-summed two-body cumulant reduced density
    matrix:

    cm2[p,q,r,s] = (dm2[p,q,r,s] - dm1[p,q]*dm1[r,s]
                       + dm1s[0][p,s]*dm1s[0][r,q]
                       + dm1s[1][p,s]*dm1s[1][r,q])

    Args:
        dm2 : ndarray of shape [norb,]*4
            Contains spin-summed 2-RDMs
        dm1s : ndarray (or compatible) of overall shape [2,norb,norb]
            Contains spin-separated 1-RDMs

    Returns:
        cm2 : ndarray of shape [norb,]*4
    '''

    dm1s = np.asarray (dm1s)
    if len (dm1s.shape) < 3:
        dm1 = dm1s.copy ()
        dm1s = dm1 / 2
        dm1s = np.stack ((dm1s, dm1s), axis=0)
    else:
        dm1 = dm1s[0] + dm1s[1]
    #cm2  = dm2 - np.einsum ('pq,rs->pqrs', dm1, dm1)
    #cm2 +=    0.5 * np.einsum ('ps,rq->pqrs', dm1, dm1)
    cm2  = dm2.copy ()
    cm2 -= np.multiply.outer (dm1, dm1)
    cm2 += np.multiply.outer (dm1s[0], dm1s[0]).transpose (0, 3, 2, 1)
    cm2 += np.multiply.outer (dm1s[1], dm1s[1]).transpose (0, 3, 2, 1)
    return cm2

def dm2s_cumulant (dm2s, dm1s):
    '''
    Evaluate the spin-summed two-body cumulant reduced density
    matrix:

    cm2s[0][p,q,r,s] = (dm2s[0][p,q,r,s] - dm1s[0][p,q]*dm1s[0][r,s]
                       + dm1s[0][p,s]*dm1s[0][r,q])
    cm2s[1][p,q,r,s] = (dm2s[1][p,q,r,s] - dm1s[0][p,q]*dm1s[1][r,s])
    cm2s[2][p,q,r,s] = (dm2s[2][p,q,r,s] - dm1s[1][p,q]*dm1s[1][r,s]
                       + dm1s[1][p,s]*dm1s[1][r,q])

    Args:
        dm2s : ndarray of shape [norb,]*4
            Contains spin-separated 2-RDMs
        dm1s : ndarray (or compatible) of overall shape [2,norb,norb]
            Contains spin-separated 1-RDMs

    Returns:
        cm2s : (cm2s[0], cms2[1], cm2s[2])
            ndarrays of shape [norb,]*4; contain spin components
            aa, ab, bb respectively
    '''
    dm1s = np.asarray (dm1s)
    if len (dm1s.shape) < 3:
        dm1 = dm1s.copy ()
        dm1s = dm1 / 2
        dm1s = np.stack ((dm1s, dm1s), axis=0)
    #cm2  = dm2 - np.einsum ('pq,rs->pqrs', dm1, dm1)
    #cm2 +=    0.5 * np.einsum ('ps,rq->pqrs', dm1, dm1)
    cm2s = [i.copy () for i in dm2s]
    cm2s[0] -= np.multiply.outer (dm1s[0], dm1s[0])
    cm2s[1] -= np.multiply.outer (dm1s[0], dm1s[1]) 
    cm2s[2] -= np.multiply.outer (dm1s[1], dm1s[1])
    cm2s[0] += np.multiply.outer (dm1s[0], dm1s[0]).transpose (0, 3, 2, 1)
    cm2s[2] += np.multiply.outer (dm1s[1], dm1s[1]).transpose (0, 3, 2, 1)
    return tuple (cm2s)

