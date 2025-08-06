'''
Doughlas Kroll Hess Hamiltonian:
Scalar Relativisitic Corrections Only upto 4th order
Source code received from Prof. Marcus Reiher <markus.reiher@phys.chem.ethz.ch>
REFERENCES:                                                      
1.  M. Reiher, A. Wolf, J. Chem. Phys. 121 (2004) 10944-10956
2.  A. Wolf, M. Reiher, B. A. Hess, J. Chem. Phys. 117 (2002) 9215-9226
'''

import sys
from functools import reduce
import numpy as np
from pyscf.scf import hf, lib, gto
from pyscf.lib import logger
from pyscf.data import nist
from pyscf import __config__
import scipy
import ctypes
import mrh

libdkh = ctypes.CDLL(mrh.__file__.rstrip('__init__.py')+'lib/libdkh.so')
DKH = libdkh.__dkh_main_MOD_dkh
DKH.restype=None
xuncontract = getattr(__config__, 'x2c_X2C_xuncontract', True)

def _uncontract_mol(mol, xuncontract=None):
    '''mol._basis + uncontracted steep functions'''
    pmol, contr_coeff = mol.decontract_basis(atoms=xuncontract)
    contr_coeff = scipy.linalg.block_diag(*contr_coeff)
    return pmol, contr_coeff


def get_hcore(mol, dkhord=2, c=None):

     xmol, contr_coeff = _uncontract_mol(mol, xuncontract)

     t = xmol.intor_symmetric('int1e_kin')
     v = xmol.intor_symmetric('int1e_nuc')
     s = xmol.intor_symmetric('int1e_ovlp')
     pVp = xmol.intor_symmetric('int1e_pnucp')

     if c is None: c  = lib.param.LIGHT_SPEED

     t   = np.asarray(t,dtype=np.float64, order='C')
     v   = np.asarray(v,dtype=np.float64, order='C')
     s   = np.asarray(s,dtype=np.float64, order='C')
     pVp = np.asarray(pVp, dtype=np.float64, order='C')
     nao = ctypes.byref(ctypes.c_int(t.shape[0]))
     dkhord = ctypes.byref(ctypes.c_int(dkhord))
     c = np.asarray(c, dtype=np.float64, order='C')

     double_ndptr = np.ctypeslib.ndpointer(dtype=np.float64, flags='CONTIGUOUS')
     int_ndptr    = ctypes.POINTER(ctypes.c_int)

     DKH.argtypes = [double_ndptr, double_ndptr,double_ndptr,double_ndptr,int_ndptr, int_ndptr, double_ndptr]
     DKH(s, v, t, pVp, nao, dkhord, c)
     h1 = t + v

     h1 = reduce(np.dot, (contr_coeff.T, h1, contr_coeff))

     return h1

def dkhscalar(mol, dkhord=2, c=None):
    '''
    This function will update the hcore (T+V)
    terms of mean-field hamiltonian.
    '''
    assert isinstance(mol, gto.Mole), \
            "Requires the mol object"
    assert 2 <= dkhord <= 4, \
            "Only 2nd-4th order Scalar DKH is defined"
    logger.info(mol, f'Scalar Relativisitc Corrections are added using DKH-{dkhord} Hamiltonian')
    logger.info(mol, f'Speed of light {c}')
    return get_hcore(mol, dkhord=dkhord, c=c)

if __name__ == '__main__':
    import numpy as np
    from pyscf import gto, scf

    mol = gto.M(atom='''Zn 0 0 0''',basis='ano@5s4p2d1f',verbose=0)

    mfdkh = scf.RHF(mol) 
    mfdkh.get_hcore = lambda *args: dkhscalar(mol, dkhord=4) # Orca's Speed of Light
    mfdkh.kernel()

    mfx2c = scf.RHF(mol).sfx2c1e().run()
    mfnr = scf.RHF(mol).run()
    
    print('\nE(Non-Relativisitic) : ', mfnr.e_tot)
    print('E(DKH) 4th Order     : ', mfdkh.e_tot)
    print('E(SFX2C1e)           : ', mfx2c.e_tot)

