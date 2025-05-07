import numpy as np
from pyscf.mcpdft import _dms
from pyscf.lib import logger
from pyscf.mcpdft.otpd import get_ontop_pair_density
from DMET.my_pyscf.dmet._pdfthelper import get_mc_for_dmet_pdft

def energy_ot (ot, casdm1s, casdm2, mo_coeff, ncore, max_memory=2000, hermi=1):
    '''
    See the docstring of pyscf/mcpdft/otfnal.energy_ot for more information.
    '''
    E_ot = 0.0
    ni = ot._numint
    xctype =  ot.xctype

    if xctype=='HF': 
        return E_ot
    
    dens_deriv = ot.dens_deriv

    nao = mo_coeff.shape[0]
    ncas = casdm2.shape[0]
    cascm2 = _dms.dm2_cumulant (casdm2, casdm1s)
    
    dm1s = _dms.casdm1s_to_dm1s (ot, casdm1s, mo_coeff=mo_coeff, ncore=ncore,
                                ncas=ncas)
    # Just because of this.
    # mo_cas = mo_coeff[:,ncore:][:,:ncas]
    if not hasattr(mo_coeff, 'ao2lo'):
        raise ValueError('The provided mo_coeff must be tagged with ao2lo')
    ao2eo = mo_coeff.ao2eo
    embmo = mo_coeff.emb_mo_coeff
    u_temp = embmo[:, ncore:ncore+ncas] # neo * ncas
    mo_cas = ao2eo @ u_temp

    t0 = (logger.process_clock (), logger.perf_counter ())
    make_rho = tuple (ni._gen_rho_evaluator (ot.mol, dm1s[i,:,:], hermi) for
        i in range(2))
    
    for ao_k1, ao_k2, mask, weight, _ \
        in ni.block_loop(ot.mol, ot.grids, nao, deriv=dens_deriv, kpt=None, max_memory=max_memory):
        '''
        ao_k1 and ao_k2 are the block of AO integrals for the given k-point. They
        are the same for supercell(1x1x1) calculations.
        '''

        rho = np.asarray ([m[0] (0, ao_k1, mask, xctype) for m in make_rho])
        t0 = logger.timer (ot, 'untransformed density', *t0)
        Pi = get_ontop_pair_density (ot, rho, ao_k1, cascm2, mo_cas,
            dens_deriv, mask)
        t0 = logger.timer (ot, 'on-top pair density calculation', *t0)
        if rho.ndim == 2:
            rho = np.expand_dims (rho, 1)
            Pi = np.expand_dims (Pi, 0)
        E_ot += ot.eval_ot (rho, Pi, dderiv=0, weights=weight)[0].dot (weight)
        t0 = logger.timer (ot, 'on-top energy calculation', *t0)

    return E_ot

def get_mc_for_pdmet_pdft(mc, trans_coeff, mf):
    newmc = get_mc_for_dmet_pdft(mc, trans_coeff, mf)
    # mcpdft.otfnal.energy_ot = energy_ot
    # mcpdft.otfnal.otfnal.energy_ot = energy_ot
    from mrh.my_pyscf.mcpdft.otfnalperiodic import otfnalperiodic
    otfnalperiodic.energy_ot = energy_ot
    assert hasattr(mf, "with_df"), "Incorrect mf passed"
    newmc.with_df = mf.with_df
    return newmc
