import numpy as np
from pyscf import lib
from mrh.my_pyscf.lassi.grad_orb_ci_si import get_grad_orb

def get_hdiag_orb (lsi, mo_coeff=None, ci=None, si=None, state=None, weights=None, eris=None,
                   veff=None, dm1s=None, opt=None, hermi=1):
    if mo_coeff is None: mo_coeff=lsi.mo_coeff
    if ci is None: ci = lsi.ci
    if si is None: si = lsi.si
    if si.ndim==1:
        assert ((state is None) and (weights is None))
        si = si[:,None]
        state = 0
    if dm1s is None: dm1s = lsi.make_rdm1s (mo_coeff=mo_coeff, ci=ci, si=si, state=state,
                                            weights=weights, opt=opt)
    dm1 = dm1s.sum (0)
    if veff is None: veff = lsi._las.get_veff (dm=dm1)
    if eris is None: h2eff_sub = lsi.get_casscf_eris (mo_coeff)
    nao, nmo = mo_coeff.shape
    ncore = lsi.ncore
    ncas = lsi.ncas
    nocc = ncore + ncas
    casdm1, casdm2 = lsi.make_casdm12 (ci=ci, si=si, state=state, weights=weights, opt=opt)
    h1 = lsi._las.get_hcore () + veff
    h1 = mo_coeff.conj ().T @ h1 @ mo_coeff

    # F^pp_ii
    f2d = np.zeros_like (h1)
    f2d[:,:ncore] = 2*np.diag(h1)[:,None] + 6*eris.k_pc - 2*eris.j_pc

    # F^pp_aa
    f2d[:,ncore:nocc] = np.multiply.outer (np.diag (h1), np.diag (casdm1))
    h2j = np.zeros ((nmo,ncas,ncas), dtype=h1.dtype)
    h2k = np.zeros ((nmo,ncas,ncas), dtype=h1.dtype)
    for p in range (nmo):
        h2j[p] = eris.ppaa[p][p]
        h2k[p] = eris.papa[p][:,p]
    f2d[:,ncore:nocc] += np.tensordot (h2k, np.diagonal (casdm2, axis1=0, axis2=2), axes=2)
    f2d[:,ncore:nocc] += np.tensordot (h2k, np.diagonal (casdm2, axis1=0, axis2=3), axes=2)
    f2d[:,ncore:nocc] += np.tensordot (h2j, np.diagonal (casdm2, axis1=0, axis2=1), axes=2)

    # F^pq_qp
    # TODO: eris -> h2eff_sub
    f1d = np.diag (get_grad_orb (lsi, mo_coeff=mo_coeff, ci=ci, si=si, state=state, weights=weights,
                                 veff=veff, dm1s=dm1s, opt=opt, hermi=1))
    f2d -= f1d[:,None]
    f2d -= f1d[None,:]

    if hermi==1:
        f2d += f2d.T
    return f2d

def get_hdiag_ci (lsi, mo_coeff=None, ci=None, si=None, state=None, weights=None, opt=None,
                  sum_bra=False):
    pass

def get_hdiag_si (lsi, mo_coeff=None, ci=None, si=None, opt=None):
    log = lib.logger.new_logger (lsi, lsi.verbose)
    t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
    if ci is None: ci = lsi.ci
    if si is None: si = lsi.si
    if opt is None: opt = lsi.opt
    is1d = si.ndim==1
    if is1d: si=si[:,None]
    nelec_frs = lsi.get_nelec_frs ()
    h0, h1, h2 = lsi.ham_2q (mo_coeff)
    hop, _, _, hdiag = op[opt].gen_contract_op_si_hdiag (lsi, h1, h2, ci, nelec_frs)[:4]
    hsi = hop (si) + (h0*si)
    ei = si.conj ().T @ hsi
    hsi -= si @ ei
    hdiag = hdiag[:,None] - ei[None,:]
    hdiag -= si.conj () * hsi
    hdiag -= si * hsi.conj ()
    hdiag += hdiag.conj ()
    if is1d: hdiag=hdiag[:,0]
    return hdiag



