import numpy as np
import scipy, time
from pyscf import lib, mcscf
from mrh.my_pyscf.mcpdft import mcpdft

def get_heff_cas (mc, mo_coeff, ci):
    ncore, ncas = mc.ncore, mc.ncas
    nocc = ncore + ncas
    mo_core = mo_coeff[:,:ncore]
    mo_cas = mo_coeff[:,ncore:nocc]
    
    veff1, veff2 = mc.get_pdft_veff (mo=mo_coeff, ci=ci, incl_coul=False, paaa_only=True)
    dm_core = 2 * mo_core @ mo_core.conj ().T
    vj_core = mc._scf.get_j (dm=dm_core)
    h1 = mo_cas.conj ().T @ (mc.get_hcore () + veff1 + vj_core) @ mo_cas
    h1 += veff2.vhf_c[ncore:nocc,ncore:nocc]
    h2 = np.zeros ([ncas,]*4)
    for i in range (ncore, nocc): 
        h2[i-ncore] = veff2.ppaa[i][ncore:nocc].copy ()
    return 0.0, h1, h2

def casci(mc, mo_coeff, ci0=None, eris=None, verbose=None, envs=None):
    ''' Minimize the PDFT energy expression for a single state by
    varying the vector using a fixed-point FCI algorithm '''
    if ci0 is None: ci0 = mc.ci
    t0 = (time.clock (), time.time ())
    ncas, nelecas = mc.ncas, mc.nelecas
    linkstrl = mc.fcisolver.gen_linkstr(ncas, nelecas, True)
    linkstr  = mc.fcisolver.gen_linkstr(ncas, nelecas, False)
    log = lib.logger.new_logger (mc, mc.verbose)
    max_memory = max(400, mc.max_memory-lib.current_memory()[0])
   
    if ci0 is None: 
        # Use real Hamiltonian? Or use HF?
        h0, h1 = mcscf.casci.h1e_for_cas (mc, mo_coeff, mc.ncas, mc.ncore)
        h2 = mcscf.casci.CASCI.ao2mo (mc, mo_coeff)
        hdiag = mc.fcisolver.make_hdiag (h1, h2, mc.ncas, mc.nelecas)
        ci0 = mc.fcisolver.get_init_guess (ncas, nelecas, 1, hdiag)[0]

    ci1 = ci0.copy ()
    for it in range (mc.max_cycle_fp):
        mc.ci = ci1 # TODO: remove; fix calls
        h0, h1, h2 = get_heff_cas (mc, mo_coeff, ci1)
        h2eff = mc.fcisolver.absorb_h1e (h1, h2, ncas, nelecas, 0.5)
        hc = mc.fcisolver.contract_2e (h2eff, ci1, ncas, nelecas, link_index=linkstrl).ravel ()
        chc = ci1.conj ().ravel ().dot (hc)
        ci_grad = hc - (chc * ci1.ravel ())
        ci_grad_norm = ci_grad.dot (ci_grad)
        lib.logger.info (mc, 'MC-PDFT CI fp iter |grad| = %e, <c.K.c> = %e', ci_grad_norm, chc)
        if ci_grad_norm < mc.conv_tol_ci_fp: break
       
        e_cas, ci1 = mc.fcisolver.kernel (h1, h2, ncas, nelecas,
                                          ci0=ci1, verbose=log,
                                          max_memory=max_memory,
                                          ecore=0)

    lib.logger.timer (mc, 'MC-PDFT CI fp iteration', *t0)

    with lib.temporary_env (mc, ci=ci1):
        e_tot, e_ot = mcpdft.kernel (mc, mc.otfnal, verbose=0)

    return e_tot, e_cas, ci1

