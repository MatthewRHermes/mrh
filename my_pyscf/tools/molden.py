from pyscf.lib import temporary_env
from pyscf.mcscf import StateAverageFCISolver, StateAverageMixFCISolver
from pyscf.tools.molden import from_mo, IGNORE_H
from pyscf.tools.molden import from_mcscf as pyscf_from_mcscf
import numpy as np

def from_sa_mcscf (mc, fname, state=None, cas_natorb=False, cas_mo_energy=False, **kwargs):
    if state is None: 
        if cas_mo_energy: return pyscf_from_mcscf (mc, fname, cas_natorb=cas_natorb, **kwargs)
        else: return from_mcscf (mc, fname, cas_natorb=cas_natorb, **kwargs)
    ci, nelecas = mc.ci[state], mc.nelecas
    fcisolver = mc.fcisolver
    if isinstance (fcisolver, StateAverageMixFCISolver):
        p0 = 0
        for s in fcisolver.fcisolvers:
            p1 = p0 + s.nroots
            if p0 <= state and state < p1:
                nelecas = fcisolver._get_nelec (s, nelecas)
                fcisolver = s
                break
            p0 = p1
    elif isinstance (fcisolver, StateAverageFCISolver):
        fcisolver = fcisolver._base_class (mc.mol)
    casdm1 = fcisolver.make_rdm1 (ci, mc.ncas, nelecas)
    mo_coeff, mo_ci, mo_energy = mc.canonicalize (ci=mc.ci, cas_natorb=cas_natorb, casdm1=casdm1)
    if not cas_mo_energy:
        mo_energy[mc.ncore:][:mc.ncas] = 0.0
    # TODO: cleaner interface. Probably invent "state_make_?dm*" functions ("state" singular)
    # and apply them also to the StateAverageMCSCFSolver instance
    mo_occ = np.zeros_like (mo_energy)
    mo_occ[:mc.ncore] = 2.0
    ci = [c.copy () for c in mc.ci]
    ci[state] = mo_ci[state]
    mo_occ[mc.ncore:][:mc.ncas] = fcisolver.make_rdm1 (mo_ci[state], mc.ncas, nelecas).diagonal ()
    return from_mo (mc.mol, fname, mo_coeff, occ=mo_occ, ene=mo_energy, **kwargs)

def from_si_mcscf (mc, fname, state=None, si=None, cas_natorb=False, cas_mo_energy=False, **kwargs):
    if si is None: si = mc.si
    ci = list (np.tensordot (si.T, np.stack (mc.ci, axis=0), axes=1))
    with temporary_env (mc, ci=ci):
        return from_sa_mcscf (mc, fname, state=state, cas_natorb=cas_natorb,
                              cas_mo_energy=cas_mo_energy, **kwargs)

def from_lasscf (las, fname, state=None, natorb_casdm1=None, only_as=False, **kwargs):
    if state is not None: natorb_casdm1 = las.states_make_casdm1s ()[state].sum (0)
    # Rotating CI vectors is stupid slow for large active spaces
    # and completely unnecessary here
    las_ci_is_None = (las.ci is None)
    las1 = las.copy ()
    # Temporary lroots safety; see corresponding block at tope of canonicalize function
    if las1.ci is not None:
        for i, ci_i in enumerate (las1.ci):
            for j in range (len (ci_i)):
                if las1.ci[i][j].ndim>2:
                    las1.ci[i][j] = las1.ci[i][j][0]
    casdm1fs = las1.make_casdm1s_sub ()
    las1.ci = None
    mo_coeff, mo_ene, mo_occ = las1.canonicalize (casdm1fs=casdm1fs,
                                                  natorb_casdm1=natorb_casdm1)[:3]
    assert (las_ci_is_None == (las.ci is None))
    if only_as:
        mo_coeff = mo_coeff[:, las.ncore:las.ncore+las.ncas]
        mo_ene = mo_ene[las.ncore:las.ncore+las.ncas]
        mo_occ = mo_occ[las.ncore:las.ncore+las.ncas]
    return from_mo (las.mol, fname, mo_coeff, occ=mo_occ, ene=mo_ene, **kwargs)

def from_lassi (lsi, fname, state=0, si=None, opt=1, **kwargs):
    if si is None: si = getattr (lsi, 'si', None)
    from mrh.my_pyscf.lassi.lassi import root_make_rdm12s
    from mrh.my_pyscf.lassi import LASSI
    natorb_casdm1 = root_make_rdm12s (lsi, lsi.ci, si, state=state, opt=opt)[0].sum (0)
    if isinstance (lsi, LASSI):
        mo_coeff, mo_ene, mo_occ = lsi._las.canonicalize (natorb_casdm1=natorb_casdm1)[:3]
    else:
        mo_coeff, mo_ene, mo_occ = lsi.canonicalize (natorb_casdm1=natorb_casdm1)[:3]
    return from_mo (lsi.mol, fname, mo_coeff, occ=mo_occ, ene=mo_ene, **kwargs)

def from_mcscf (mc, filename, ignore_h=IGNORE_H, cas_natorb=False):
    ncore, ncas = mc.ncore, mc.ncas
    nocc = ncore + ncas
    mc_can = mc.canonicalize
    def no_active_ene (sort=True, cas_natorb=cas_natorb):
        mo_coeff, ci, mo_energy = mc_can (sort=sort, cas_natorb=cas_natorb)
        mo_energy[ncore:nocc] = 0.0
        return mo_coeff, ci, mo_energy
    mo_energy = mc.mo_energy.copy ()
    mo_energy[ncore:nocc] = 0.0
    with temporary_env (mc, canonicalize=no_active_ene, mo_energy=mo_energy):
        ret = pyscf_from_mcscf (mc, filename, ignore_h=IGNORE_H, cas_natorb=cas_natorb)
    return ret
