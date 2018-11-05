import numpy as np
import time
from pyscf import dft
from pyscf.lib import logger
from mrh.my_pyscf.mcpdft.otpd import get_ontop_pair_density
from mrh.util.rdm import get_2CDM_from_2RDM

def kernel (mc, ot):
    ''' Calculate MC-PDFT total energy

        Args:
            mc : an instance of CASSCF or CASCI class
                Note: this function does not currently run the CASSCF or CASCI calculation itself
                prior to calculating the MC-PDFT energy. Call mc.kernel () before passing to this function!
            ot : an instance of on-top density functional class - see otfnal.py

        Returns:
            Total MC-PDFT energy including nuclear repulsion energy.
    '''
    t0 = (time.clock (), time.time ())
    dm1s = np.asarray (mc.make_rdm1s ())
    dm1 = mc.make_rdm1 ()
    amo = mc.mo_coeff[:,mc.ncore:mc.ncore+mc.ncas]
    # make_rdm12s returns (a, b), (aa, ab, bb)
    adm1s = np.stack (mc.fcisolver.make_rdm1s (mc.ci, mc.ncas, mc.nelecas), axis=0)
    adm2 = get_2CDM_from_2RDM (mc.fcisolver.make_rdm12 (mc.ci, mc.ncas, mc.nelecas)[1], adm1s)
    t0 = logger.timer (ot, 'rdms', *t0)

    h = mc._scf.get_hcore ()
    j = mc._scf.get_j (dm=dm1) / 2
    E_TVJ = mc._scf.energy_nuc () + np.tensordot (h + j, dm1)
    logger.debug (ot, 'T + Vext + Vcoul = %s', E_TVJ)
    t0 = logger.timer (ot, 'T + Vext + Vcoul', *t0)

    Exc_wfn = mc.e_tot - E_TVJ
    if ot.verbose >= logger.DEBUG:
        Exc_dm1 = ot.ks.get_veff (dm=dm1s).exc
        logger.debug (ot, 'Exc[wfn] = %s; Exc[dm1] = %s', Exc_wfn, Exc_dm1)
    E_ot = get_E_ot (ot, dm1s, adm2, amo)
    t0 = logger.timer (ot, 'Eot', *t0)
    logger.info (ot, 'MC-PDFT E = %s, Eot(%s) = %s', E_TVJ + E_ot, ot.otxc, E_ot)

    return E_TVJ + E_ot

def get_E_ot (ot, oneCDMs, twoCDM_amo, ao2amo, deriv=0, max_memory=20000, hermi=0):
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
            The MC-PDFT on-top exchange-correlation energy

    '''
    ks = ot.ks
    xctype = ks._numint._xc_type (ks.xc)
    deriv = ot.xc_deriv
    norbs_ao = ao2amo.shape[0]

    E_ot = 0.0

    t0 = (time.clock (), time.time ())
    make_rho = tuple (ks._numint._gen_rho_evaluator (ot.mol, oneCDMs[i,:,:], hermi) for i in range(2))
    for ao, mask, weight, coords in ks._numint.block_loop (ot.mol, ks.grids, norbs_ao, deriv, max_memory):
        rho = np.asarray ([m[0] (0, ao, mask, xctype) for m in make_rho])
        t0 = logger.timer (ot, 'untransformed density', *t0)
        Pi = get_ontop_pair_density (ot, rho, ao, twoCDM_amo, ao2amo, deriv) 
        t0 = logger.timer (ot, 'on-top pair density calculation', *t0) 
        E_ot += ot.get_E_ot (rho, Pi, weight)

    return E_ot
    

