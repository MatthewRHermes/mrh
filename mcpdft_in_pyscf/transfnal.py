import numpy as np
from pyscf.dft.numint import eval_rho, eval_ao 
from pyscf.dft.libxc import eval_xc

deriv_dict = {'LDA': 0,
              'GGA': 1,
              'MGGA': 2}

def get_E_ot_translated_rho (ks, rho, Pi, translator):
    r''' E_ot[rho, Pi] = V_xc[rho_translated] 

        Args:
            mc : an instance of a pyscf MC-SCF class
            rho : ndarray containing density and derivatives
                see the documentation of pyscf.dft.numint.eval_rho and
                pyscf.dft.libxc.eval_xc for its shape 
            Pi : 1d ndarray with shape=(grid),
                containing on-top pair density
            translator : callable with args (rho, Pi)
                which returns the translated spin density,
                an object with the same shape as rho

        Returns : float
            The on-top exchange-correlation energy, for an on-top xc functional
            which uses a translated density with an otherwise standard xc functional
    '''
    ngrids = rho.shape[-1]
    rho_t = translator (rho, Pi)
    dexc_ddens = ks._numint.eval_xc (ks.xc, (rho_t[0], rho_t[1]), spin=1, relativity=0, deriv=0, verbose=ks.verbose)
    dens = rho_t[0].flat[:ngrids] + rho_t[1].flat[:ngrids]
    return np.einsum ('i,i,i->', dexc_ddens, dens, ks.grids.weights)
                
def get_E_ot_txc (ks, rho, Pi):
    return get_E_ot_translated_rho (ks, rho, Pi, get_rho_translated)

def get_E_ot_ftxc (ks, rho, Pi):
    return get_E_ot_translated_rho (ks, rho, Pi, get_rho_fully_translated)

def get_rho_translated (Pi, rho):
    raise (NotImplementedError)

def get_rho_fully_translated (Pi, rho):
    raise (NotImplementedError)


