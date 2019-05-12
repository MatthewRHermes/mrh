import numpy as np
from scipy import linalg
from functools import reduce

def get_translational_coordinates (carts, masses):
    ''' Construct mass-weighted translational coordinate vectors '''
    natm = carts.shape[0]
    u = np.stack ([np.eye (3) for iatm in range (natm)], axis=0)
    u *= np.sqrt (masses)[:,None,None]
    return u

def get_rotational_coordinates (carts, masses):
    ''' Construct mass-weighted rotational coordinate vectors '''
    natm = carts.shape[0]
    # Translate to center of mass
    carts = carts - (np.einsum ('i,ij->j', masses, carts) / np.sum (masses))[None,:]
    # Generate and diagonalize moment-of-inertia vector
    rad2 = (carts * carts).sum (1)
    I = rad2[:,None,None] * np.stack ([np.eye (3) for iatm in range (natm)], axis=0)
    I -= np.stack ([np.outer (icart, icart) for icart in carts], axis=0)
    I = np.einsum ('m,mij->ij', masses, I)
    X, mI = linalg.eigh (I)
    # Generate rotational coordinates: cross-product of X axes with radial displacement from X axes
    u = np.zeros ((natm, 3, 3))
    RXt = np.dot (carts, X.T)
    for iatm in range (natm):
        u[iatm] = np.stack ([np.cross (RXt[iatm], xyz) for xyz in X], axis=0)
    u = u.transpose (0,2,1)
    u /= np.sqrt (masses)[:,None,None]
    # Remove norm = 0 modes (linear molecules)
    norm_u = (u * u).sum ((0,1))
    idx = norm_u > 1e-8
    u = u[:,:,idx]
    mI = mI[idx]
    return mI, u

class InternalCoords (object):
    def __init__(self, mol):
        self.mol = mol
        self.masses = mol.get_atom_masses ()
        self.carts = mol.get_atom_coords ()
    def get_coords (self, carts=None, include_inertia=False):
        if carts is None: carts = self.carts
        utrans = get_translational_coordinates (carts, self.masses)
        mI, urot = get_rotational_coordinates (carts, self.masses)
        nextr = utrans.shape[-1] + urot.shape[-1]
        nintr = carts.size - nextr
        uall = linalg.qr (np.append (utrans, urot, axis=-1).reshape (3*self.mol.natm,nextr))
        uvib = uall[:,nextr:].reshape (self.mol.natm, 3, nintr)
        if include_inertia: return utrans, urot, uvib, mI
        return utrans, urot, uvib
    def transform_1body (vec, carts=None):
        utrans, urot, uvib = self.get_coords (carts=carts)
        vec /= np.sqrt (self.masses)[:,None]
        vec = vec.ravel ()
        vec_t = np.dot (vec, utrans.reshape (3*mol.natm, -1))
        vec_r = np.dot (vec, urot.reshape (3*mol.natm, -1))
        vec_v = np.dot (vec, uvib.reshape (3*mol.natm, -1))
        return vec_t, vec_r, vec_v
    def _project_1body (vec, carts=None, idx=None):
        uvib = self.get_coords (carts=carts)[idx].reshape (3*mol.natm, -1)
        vec /= np.sqrt (self.masses)[:,None]
        vec = vec.ravel ()
        vec = np.dot (vec, uvib)
        vec = np.dot (uvib.conjugate (), vec).reshape (3, mol.natm)
        vec *= np.sqrt (self.masses)[:,None]
        return vec
    def project_1body_trans (vec, carts=None):
        return self._project_1body (vec, carts=carts, idx=0)
    def project_1body_rot (vec, carts=None):
        return self._project_1body (vec, carts=carts, idx=1)
    def project_1body_vib (vec, carts=None):
        return self._project_1body (vec, carts=carts, idx=2)




