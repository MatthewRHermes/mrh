import numpy as np
from pyscf.dft.numint import eval_rho, eval_ao 
from pyscf.dft.libxc import eval_xc
from mrh.mcpdft_in_pyscf.otpd import get_ontop_pair_density

deriv_dict = {'LDA': 0,
              'GGA': 1,
              'MGGA': 2}

def get_mcpdft_elec_energy (mc, ks, get_E_ot, oneCSDM_ao=None, rho=None, Pi=None):
    ''' E_MCPDFT = h^p_q l^p_q + 1/2 v^pr_qs l^p_q l^r_s + E_ot[rho,Pi] 
        or, in other terms, 
        E_MCPDFT = T_KS[rho] + V_ext[rho] + V_coul[rho] + E_ot[rho, Pi]
                 = E_KSDFT[1rdm] - V_xc[rho] + E_ot[rho, Pi] 
        Args:
            mc : an instance of a pyscf MC-SCF class
            ks : an instance of a pyscf UKS class
            get_E_ot : callable with args (ks, rho, Pi) 
                which returns the on-top pair density exchange-correlation energy

        Kwargs:
            oneCSDM_ao : length 2 tuple of 2d ndarrays with shape=(nao, nao)
                containing spin-separated one-body density matrices
            rho : ndarray containing density and derivatives
                see the documentation of pyscf.dft.numint.eval_rho and
                pyscf.dft.libxc.eval_xc for its shape
            Pi : 1d ndarray with shape=(ngrid),
                containing on-top pair density

        Returns : float
            The MC-PDFT electronic energy (not including the internuclear constant) 

    '''
    xctype = ks._numint._xc_type

    if oneCSDM_ao is None:
        oneCSDM_ao = mc.make_rdm1s ()
    if (rho is None) or (Pi is None):
        ao = eval_ao (ks.mol, ks.grids.coords, deriv=deriv_dict[xctype])
    if rho is None:
        rho = np.asarray ([eval_rho (ks.mol, ao, oneCDM, xctype=xctype) for oneCDM in oneCSDM_ao])
    if Pi is None:
        Pi = get_ontop_pair_density (mc, ks, rho=rho, ao=ao)

    E_DFT = ks.energy_elec (dm=oneCSDM_ao)
    E_xc = ks.get_veff (dm=oneCDSM_ao).exc
    E_ot = get_E_ot (ks, rho, Pi)
    return E_DFT - E_xc + E_ot
    

