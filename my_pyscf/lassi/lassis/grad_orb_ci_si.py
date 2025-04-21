from mrh.my_pyscf.lassi import grad_orb_ci_si
from mrh.my_pyscf.lassi.lassis import coords

def get_grad_orb (lsi, mo_coeff=None, ci=None, ci_ref=None, ci_sf=None, ci_ch=None, si=None,
                  state=None, weights=None, h2eff_sub=None, veff=None, dm1s=None, opt=None,
                  hermi=-1):
    '''Return energy gradient for orbital rotation.

    Args:
        lsi : instance of :class:`LASSIS`

    Kwargs:
        mo_coeff : ndarray of shape (nao,nmo)
            Contains molecular orbitals
        ci : list (length=nfrags) of list (length=nroots) of ndarray
            Contains CI vectors. Built from ci_ref, ci_sf, ci_ch if omitted
        ci_ref : list (length=nfrags) of ndarray
            Contains CI vectors for reference statelets
        ci_sf : nested list of shape (nfrags,2) of ndarray
            Contains CI vectors for spin-flip statelets
        ci_ch : nested list of shape (nfrag,nfrags,4,2) of ndarrays
            Contains CI vectors for charge-hop statelets
        si : ndarray of shape (nprods,nroots_si)
            Contains si vectors
        state : integer
            Index of column of si to use
        weights : ndarray of shape (nroots_si)
            Weights for the columns of si vectors
        h2eff_sub : ndarray of shape (nmo,ncas**2*(ncas+1)/2)
            Contains ERIs (p1a1|a2a3), lower-triangular in the a2a3 indices
        veff : ndarray of shape (2,nao,nao)
            Spin-separated, state-averaged 1-electron mean-field potential in the AO basis
        dm1s : ndarray of shape (2,nao,nao)
            Spin-separated, state-averaged 1-RDM in the AO basis
        opt : integer
            Optimization level
        hermi : integer
            Control (anti-)symmetrization. 0 means to return the effective Fock matrix,
            F1 = h.D + g.d. -1 means to return the true orbital-rotation gradient, which is skew-
            symmetric: gorb = F1 - F1.T. +1 means to return the symmetrized effective Fock matrix,
            (F1 + F1.T) / 2. The factor of 2 difference between hermi=-1 and the other two options
            is intentional and necessary.

    Returns:
        gorb : ndarray of shape (nmo,nmo)
            Orbital rotation gradients as a square antihermitian array
    '''
    if ci is None: ci = lsi.prepare_model_states (ci_ref, ci_sf, ci_ch)[0].ci
    return grad_orb_ci_si.get_grad_orb (lsi, mo_coeff=mo_coeff, ci=ci, si=si, state=state,
                                        weights=weights, h2eff_sub=h2eff_sub, veff=veff, dm1s=dm1s,
                                        opt=opt, hermi=hermi)

def get_grad_ci (lsi, mo_coeff=None, ci=None, ci_ref=None, ci_sf=None, ci_ch=None, si=None,
                 state=None, weights=None, opt=None):
    '''Return energy gradient for CI rotation.

    Args:
        lsi : instance of :class:`LASSIS`

    Kwargs:
        mo_coeff : ndarray of shape (nao,nmo)
            Contains molecular orbitals
        ci : list (length=nfrags) of list (length=nroots) of ndarray
            Contains CI vectors. Built from ci_ref, ci_sf, ci_ch if omitted
        ci_ref : list (length=nfrags) of ndarray
            Contains CI vectors for reference statelets
        ci_sf : nested list of shape (nfrags,2) of ndarray
            Contains CI vectors for spin-flip statelets
        ci_ch : nested list of shape (nfrag,nfrags,4,2) of ndarrays
            Contains CI vectors for charge-hop statelets
        si : ndarray of shape (nprods,nroots_si)
            Contains si vectors
        state : integer
            Index of column of si to use
        weights : ndarray of shape (nroots_si)
            Weights for the columns of si vectors
        opt : integer
            Optimization level

    Returns:
        gci_ref : list of length nfrags of ndarrays
            CI rotation gradients for reference wfn
        gci_sf : nested list of shape (nfrags,2) of ndarrays
            CI rotation gradients for spin-flip statelets
        gci_ch : nested list of shape (nfrag,nfrags,4,2) of ndarrays
            CI rotation gradients for charge-hop statelets
    '''
    if ci is None: ci = lsi.prepare_model_states (ci_ref, ci_sf, ci_ch)[0].ci
    hci = grad_orb_ci_si.get_grad_ci (
        lsi, mo_coeff=mo_coeff, ci=ci, si=si, state=state, weights=weights, opt=opt
    )
    return coords.sum_hci (lsi, hci)

def get_grad_si (lsi, mo_coeff=None, ci=None, ci_ref=None, ci_sf=None, ci_ch=None, si=None,
                 opt=None):
    '''Return energy gradient for SI rotation.

    Args:
        lsi : instance of :class:`LASSIS`

    Kwargs:
        mo_coeff : ndarray of shape (nao,nmo)
            Contains molecular orbitals
        ci : list (length=nfrags) of list (length=nroots) of ndarray
            Contains CI vectors. Built from ci_ref, ci_sf, ci_ch if omitted
        ci_ref : list (length=nfrags) of ndarray
            Contains CI vectors for reference statelets
        ci_sf : nested list of shape (nfrags,2) of ndarray
            Contains CI vectors for spin-flip statelets
        ci_ch : nested list of shape (nfrag,nfrags,4,2) of ndarrays
            Contains CI vectors for charge-hop statelets
        si : ndarray of shape (nprods,nroots_si)
            Contains si vectors
        opt : integer
            Optimization level

    Returns:
        gsi : ndarray of shape (nprods,nroots_si)
            SI rotation gradients, ignoring weights and state-averaging
    '''
    if ci is None: ci = lsi.prepare_model_states (ci_ref, ci_sf, ci_ch)[0].ci
    return grad_orb_ci_si.get_grad_si (
        lsi, mo_coeff=mo_coeff, ci=ci, si=si, opt=opt
    )

def get_grad (lsi, mo_coeff=None, ci_ref=None, ci_sf=None, ci_ch=None, si=None, state=None,
              weights=None, opt=None, pack=False):
    '''Return energy gradient all coordinates

    Args:
        lsi : instance of :class:`LASSIS`

    Kwargs:
        mo_coeff : ndarray of shape (nao,nmo)
            Contains molecular orbitals
        ci_ref : list (length=nfrags) of ndarray
            Contains CI vectors for reference statelets
        ci_sf : nested list of shape (nfrags,2) of ndarray
            Contains CI vectors for spin-flip statelets
        ci_ch : nested list of shape (nfrag,nfrags,4,2) of ndarrays
            Contains CI vectors for charge-hop statelets
        si : ndarray of shape (nprods,nroots_si)
            Contains si vectors
        state : integer
            Index of column of si to use
        weights : ndarray of shape (nroots_si)
            Weights for the columns of si vectors
        opt : integer
            Optimization level
        pack : logical
            If True, returns gradients packed into a single vector using UnitaryGroupGenerators    

    Returns:
        gorb : ndarray of shape (nmo,nmo)
            Orbital rotation gradients as a square antihermitian array
        gci_ref : list of length nfrags of ndarrays
            CI rotation gradients for reference wfn
        gci_sf : nested list of shape (nfrags,2) of ndarrays
            CI rotation gradients for spin-flip statelets
        gci_ch : nested list of shape (nfrag,nfrags,4,2) of ndarrays
            CI rotation gradients for charge-hop statelets
        gsi : ndarray of shape (nprods,nroots_si)
            SI rotation gradients, ignoring weights and state-averaging
    '''
    if mo_coeff is None: mo_coeff = lsi.mo_coeff
    if ci_ref is None: ci_ref = lsi.get_ci_ref ()
    if ci_sf is None: ci_sf = lsi.ci_spin_flips
    if ci_ch is None: ci_ch = lsi.ci_charge_hops
    if si is None: si = lsi.si
    ci = lsi.prepare_model_states (ci_ref, ci_sf, ci_ch)[0].ci
    gorb = get_grad_orb (lsi, mo_coeff=mo_coeff, ci=ci, si=si, state=state, weights=weights,
                         opt=opt)
    gci_ref, gci_sf, gci_ch = get_grad_ci (lsi, mo_coeff=mo_coeff, ci=ci, si=si, state=state,
                                           weights=weights, opt=opt)
    gsi = get_grad_si (lsi, mo_coeff=mo_coeff, ci=ci, si=si, opt=opt)
    if state is not None: gsi = gsi[:,state]
    if state is not None: si = si[:,state] 
    if pack:
        ugg = coords.UnitaryGroupGenerators (lsi, mo_coeff, ci_ref, ci_sf, ci_ch, si)
        return ugg.pack (gorb, gci_ref, gci_sf, gci_ch, gsi)
    else:
        return gorb, gci_ref, gci_sf, gci_ch, gsi

