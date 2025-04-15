import numpy as np
from pyscf import lib
from mrh.my_pyscf.lassi import op_o0, op_o1

op = (op_o0, op_o1)

def get_grad_orb (lsi, mo_coeff=None, ci=None, si=None, state=None, weights=None, h2eff_sub=None,
                  veff=None, dm1s=None, opt=None, hermi=-1):
    '''Return energy gradient for orbital rotation.

    Args:
        lsi : instance of :class:`LASSI`

    Kwargs:
        mo_coeff : ndarray of shape (nao,nmo)
            Contains molecular orbitals
        ci : list (length=nfrags) of list (length=nroots) of ndarray
            Contains CI vectors
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
    if mo_coeff is None: mo_coeff = lsi.mo_coeff
    if ci is None: ci = lsi.ci
    if si is None: si = lsi.si
    if si.ndim==1:
        assert ((state is None) and (weights is None))
        si = si[:,None]
        state = 0
    if dm1s is None: dm1s = lsi.make_rdm1s (mo_coeff=mo_coeff, ci=ci, si=si, state=state,
                                            weights=weights, opt=opt)
    if h2eff_sub is None: h2eff_sub = lsi._las.get_h2eff (mo_coeff)
    if veff is None:
        veff = lsi._las.get_veff (dm=dm1s.sum (0))
        veff = lsi._las.split_veff (veff, h2eff_sub, mo_coeff=mo_coeff, ci=ci)
    nao, nmo = mo_coeff.shape
    ncore = lsi.ncore
    ncas = lsi.ncas
    nocc = lsi.ncore + lsi.ncas
    smo_cas = lsi._las._scf.get_ovlp () @ mo_coeff[:,ncore:nocc]
    smoH_cas = smo_cas.conj ().T

    # The orbrot part
    h1s = lsi._las.get_hcore ()[None,:,:] + veff
    f1 = h1s[0] @ dm1s[0] + h1s[1] @ dm1s[1]
    f1 = mo_coeff.conjugate ().T @ f1 @ lsi._las._scf.get_ovlp () @ mo_coeff
    # ^ I need the ovlp there to get dm1s back into its correct basis
    casdm2 = lsi.make_casdm2 (ci=ci, si=si, state=state, weights=weights, opt=opt)
    casdm1s = np.stack ([smoH_cas @ d @ smo_cas for d in dm1s], axis=0)
    casdm1 = casdm1s.sum (0)
    casdm2 -= np.multiply.outer (casdm1, casdm1)
    casdm2 += np.multiply.outer (casdm1s[0], casdm1s[0]).transpose (0,3,2,1)
    casdm2 += np.multiply.outer (casdm1s[1], casdm1s[1]).transpose (0,3,2,1)
    eri = h2eff_sub.reshape (nmo*ncas, ncas*(ncas+1)//2)
    eri = lib.numpy_helper.unpack_tril (eri).reshape (nmo, ncas, ncas, ncas)
    f1[:,ncore:nocc] += np.tensordot (eri, casdm2, axes=((1,2,3),(1,2,3)))

    if hermi == -1:
        return f1 - f1.T
    elif hermi == 1:
        return .5*(f1+f1.T)
    elif hermi == 0:
        return f1
    else:
        raise ValueError ("kwarg 'hermi' must = -1, 0, or +1")

def get_grad_ci (lsi, mo_coeff=None, ci=None, si=None, state=None, weights=None, opt=None):
    '''Return energy gradient for CI rotation.

    Args:
        lsi : instance of :class:`LASSI`

    Kwargs:
        mo_coeff : ndarray of shape (nao,nmo)
            Contains molecular orbitals
        ci : list (length=nfrags) of list (length=nroots) of ndarray
            Contains CI vectors
        si : ndarray of shape (nprods,nroots_si)
            Contains si vectors
        state : integer
            Index of column of si to use
        weights : ndarray of shape (nroots_si)
            Weights for the columns of si vectors
        opt : integer
            Optimization level

    Returns:
        gci_fr_pabq : nested list of shape (nfrags, nroots) containing ndarrays
            CI rotation gradients
    '''
    if ci is None: ci = lsi.ci
    if si is None: si = lsi.si
    if opt is None: opt = lsi.opt
    nelec_frs = lsi.get_nelec_frs ()
    h0, h1, h2 = lsi.ham_2q (mo_coeff)
    lroots = lsi.get_lroots (ci)
    sivec = si
    if state is not None:
        sivec = sivec[:,state]
        n = 1
    elif sivec.ndim==2 and sivec.shape[1]>1:
        n = sivec.shape[1]
        assert (len (weights) == n)
    else:
        n = 1
        assert (weights is None)
    hci = op[opt].contract_ham_ci (lsi, h1, h2, ci, nelec_frs, ci, nelec_frs, sivec, sivec, h0=h0)
    for f in range (lsi.nfrags):
        for r in range (lsi.nroots):
            c = ci[f][r].reshape (lroots[f][r],-1)
            hc = np.diagonal (hci[f][r].reshape (n,lroots[f][r],-1,n), axis1=0, axis2=3)
            hc = hc.transpose (2,0,1).copy ()
            if weights is not None:
                hc = lib.einsum ('r,rli->li', weights, hc)
            else:
                hc = hc[0]
            hc = hc.reshape (ci[f][r].shape)
            hci[f][r] = hc + hc.conj () # + h.c.
    return hci

def get_grad_si (lsi, mo_coeff=None, ci=None, si=None, opt=None):
    '''Return energy gradient for SI rotation.

    Args:
        lsi : instance of :class:`LASSI`

    Kwargs:
        mo_coeff : ndarray of shape (nao,nmo)
            Contains molecular orbitals
        ci : list (length=nfrags) of list (length=nroots) of ndarray
            Contains CI vectors
        si : ndarray of shape (nprods,nroots_si)
            Contains si vectors
        opt : integer
            Optimization level

    Returns:
        gsi : ndarray of shape (nprods,nroots_si)
            SI rotation gradients, ignoring weights and state-averaging
    '''
    if ci is None: ci = lsi.ci
    if si is None: si = lsi.si
    if opt is None: opt = lsi.opt
    is1d = si.ndim==1
    if is1d: si=si[:,None]
    nelec_frs = lsi.get_nelec_frs ()
    h0, h1, h2 = lsi.ham_2q (mo_coeff)
    hop = op[opt].gen_contract_op_si_hdiag (lsi, h1, h2, ci, nelec_frs)[0]
    hsi = hop (si) + (h0*si)
    hsi -= si @ (si.conj ().T @ hsi)
    hsi += hsi.conj () # + h.c.
    if is1d: hsi=hsi[:,0]
    return hsi

def get_grad (lsi, mo_coeff=None, ci=None, si=None, state=None, weights=None, opt=None):
    '''Return energy gradient all coordinates

    Args:
        lsi : instance of :class:`LASSI`

    Kwargs:
        mo_coeff : ndarray of shape (nao,nmo)
            Contains molecular orbitals
        ci : list (length=nfrags) of list (length=nroots) of ndarray
            Contains CI vectors
        si : ndarray of shape (nprods,nroots_si)
            Contains si vectors
        state : integer
            Index of column of si to use
        weights : ndarray of shape (nroots_si)
            Weights for the columns of si vectors
        opt : integer
            Optimization level
    
    Returns:
        gorb : ndarray of shape (nmo,nmo)
            Orbital rotation gradients as a square antihermitian array
        gci : nested list of shape (nfrags, nroots) containing ndarrays
            CI rotation gradients
        gsi : ndarray of shape (nprods,nroots_si)
            SI rotation gradients, ignoring weights and state-averaging
    '''
    gorb = get_grad_orb (lsi, mo_coeff=mo_coeff, ci=ci, si=si, state=state, weights=weights,
                         opt=opt)
    gci = get_grad_ci (lsi, mo_coeff=mo_coeff, ci=ci, si=si, state=state, weights=weights,
                       opt=opt)
    gsi = get_grad_si (lsi, mo_coeff=mo_coeff, ci=ci, si=si, opt=opt)
    return gorb, gci, gsi

