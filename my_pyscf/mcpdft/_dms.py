# Common density-matrix manipulations

import numpy as np
from pyscf import lib
from scipy import linalg

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

def casdm1s_to_dm1s (mc, casdm1s, mo_coeff=None, ncore=None, ncas=None):
    '''
    Generate AO-basis spin-separated 1-RDM from active space part. This
    is necessary because the StateAverageMCSCFSolver class doesn't have
    API for getting the AO-basis density matrix of a single state.

    Args:
        mc : object of CASCI or CASSCF class
        casdm1s : ndarray or compatible of shape (2,ncas,ncas)
            Active-space spin-separated 1-RDM

    Kwargs:
        ncore : integer
            Number of occupied inactive orbitals
        ncas : integer
            Number of active orbitals

    Returns:
        dm1s : ndarray of shape (2,nao,nao)
    '''
    if mo_coeff is None: mo_coeff=mc.mo_coeff
    if ncore is None: ncore=mc.ncore
    if ncas is None: ncas=mc.ncas
    mo_core = mo_coeff[:,:ncore]
    mo_cas = mo_coeff[:,ncore:][:,:ncas]
    moH_core = mo_core.conj ().T
    moH_cas = mo_cas.conj ().T

    casdm1s = np.asarray (casdm1s)
    dm1s_cas = np.dot (casdm1s, moH_cas)
    dm1s_cas = np.dot (mo_cas, dm1s_cas).transpose (1,0,2)
    dm1s_core = np.dot (mo_core, moH_core)
    dm1s = dm1s_cas + dm1s_core[None,:,:]

    # Tags for speeding up rho generators and DF fns
    no_coeff = mo_coeff[:,:ncore+ncas]
    no_coeff = np.stack ([no_coeff, no_coeff], axis=0)
    no_occ = np.zeros ((2,ncore+ncas))
    no_occ[:,:ncore] = 1.0
    no_cas = no_coeff[:,:,ncore:]
    for i in range (2):
        no_occ[i,ncore:], umat = linalg.eigh (-casdm1s[i])
        no_cas[i,:,:] = np.dot (no_cas[i,:,:], umat)
    no_occ[:,ncore:] *= -1
    dm1s = lib.tag_array (dm1s, mo_coeff=no_coeff, mo_occ=no_occ)

    return dm1s

