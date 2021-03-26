from pyscf.lib import temporary_env
from pyscf.tools.molden import from_mo, IGNORE_H
from pyscf.tools.molden import from_mcscf as pyscf_from_mcscf
import numpy as np

def from_sa_mcscf (mc, fname, state=None, cas_natorb=False, cas_mo_energy=False, **kwargs):
    if state is None: return pyscf_from_mcscf (mc, fname, cas_natorb=cas_natorb, **kwargs)
    casdm1 = mc.fcisolver.states_make_rdm1 (mc.ci, mc.ncas, mc.nelecas)[state]
    mo_coeff, mo_ci, mo_energy = mc.canonicalize (ci=mc.ci[state], cas_natorb=cas_natorb, casdm1=casdm1)
    if not cas_mo_energy:
        mo_energy[mc.ncore:][:mc.ncas] = 0.0
    # TODO: cleaner interface. Probably invent "state_make_?dm*" functions ("state" singular)
    # and apply them also to the StateAverageMCSCFSolver instance
    mo_occ = np.zeros_like (mo_energy)
    mo_occ[:mc.ncore] = 2.0
    ci = mc.ci
    ci[state] = mo_ci
    mo_occ[mc.ncore:][:mc.ncas] = mc.fcisolver.states_make_rdm1 (ci, mc.ncas, mc.nelecas)[state].diagonal ()
    return from_mo (mc.mol, fname, mo_coeff, occ=mo_occ, ene=mo_energy, **kwargs)


def from_lasscf (las, fname, **kwargs):
    mo_coeff, mo_ene, mo_occ = las.canonicalize ()[:3]
    return from_mo (las.mol, fname, mo_coeff, occ=mo_occ, ene=mo_ene, **kwargs)

def from_mcscf (mc, filename, ignore_h=IGNORE_H, cas_natorb=False):
    ncore, ncas = mc.ncore, mc.ncas
    nocc = ncore + ncas
    mc_can = mc.canonicalize
    def no_active_ene (sort=True, cas_natorb=cas_natorb):
        mo_coeff, ci, mo_energy = mc_can (sort=sort, cas_natorb=cas_natorb)
        mo_energy[ncore:nocc] = 0.0
        return mo_coeff, ci, mo_energy
    with temporary_env (mc, canonicalize=no_active_ene):
        ret = pyscf_from_mcscf (mc, filename, ignore_h=IGNORE_H, cas_natorb=cas_natorb)
    return ret
