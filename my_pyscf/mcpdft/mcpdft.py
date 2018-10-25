import numpy as np
from pyscf import dft
from mrh.my_pyscf.mcpdft.otpd import get_ontop_pair_density
from mrh.util.rdm import get_2CDM_from_2RDM

def get_mcpdft_elec_energy (ot, oneCDMs, twoCDM_amo, ao2amo, deriv=0, max_memory=20000, hermi=0):
    ''' E_MCPDFT = h_pq l_pq + 1/2 v_pqrs l_pq l_rs + E_ot[rho,Pi] 
        or, in other terms, 
        E_MCPDFT = T_KS[rho] + E_ext[rho] + E_coul[rho] + E_ot[rho, Pi]
                 = E_DFT[1rdm] - E_xc[rho] + E_ot[rho, Pi] 
        Args:
            ot : an instance of otfnal class
            oneCDMs : ndarray of shape (2, nao, nao)
                containing spin-separated one-body density matrices
            twoCDM_amo : ndarray of shape (ncas, ncas, ncas, ncas)
                containing spin-summed two-body cumulant density matrix in an active space
            ao2amo : ndarray of shape (nao, ncas)
                containing molecular orbital coefficients for active-space orbitals

        Kwargs:
            deriv : int
                order of spin density and on-top pair density derivatives required by otfnal
            max_memory : int or float
                maximum cache size in MB
                default is 20000
            hermi : int
                1 if 1CDMs are assumed hermitian, 0 otherwise
                default is 0

        Returns : float
            The MC-PDFT electronic energy (not including the internuclear constant) 

    '''
    ks = ot.ks
    xctype = ks._numint._xc_type (ks.xc)
    deriv = ot.xc_deriv
    norbs_ao = ao2amo.shape[0]

    E_DFT = ks.energy_elec (dm=oneCDMs)[0]
    E_xc = ks.get_veff (dm=oneCDMs).exc
    E_ot = 0.0

    make_rho = tuple (ks._numint._gen_rho_evaluator (ot.mol, oneCDMs[i,:,:], hermi) for i in range(2))
    for ao, mask, weight, coords in ks._numint.block_loop (ot.mol, ks.grids, norbs_ao, deriv, max_memory):
        rho = np.asarray ([m[0] (0, ao, mask, xctype) for m in make_rho])
        Pi = get_ontop_pair_density (rho, ao, twoCDM_amo, ao2amo, deriv)        
        E_ot += ot.get_E_ot (rho, Pi, weight)

    E_MCPDFT = E_DFT - E_xc + E_ot
    return E_MCPDFT
    
def mcpdft_kernel (mc, ot):
    dm1 = np.asarray (mc.make_rdm1s ())
    amo = mc.mo_coeff[:,mc.ncore:mc.ncore+mc.ncas]
    # make_rdm12s returns (a, b), (aa, ab, bb)
    adm1s = np.stack (mc.fcisolver.make_rdm12s (mc.ci, mc.ncas, mc.nelecas)[0], axis=0)
    adm2 = get_2CDM_from_2RDM (mc.fcisolver.make_rdm12 (mc.ci, mc.ncas, mc.nelecas)[1], adm1s)
    E_ot = get_mcpdft_elec_energy (ot, dm1, adm2, amo) + mc._scf.energy_nuc ()
    return E_ot

