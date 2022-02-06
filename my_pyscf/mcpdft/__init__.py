# Lahh dee dah
import copy
from mrh.my_pyscf.mcpdft.mcpdft import get_mcpdft_child_class
from mrh.my_pyscf.mcpdft.otfnal import make_hybrid_fnal as hyb
from pyscf import mcscf
from pyscf.mcscf import mc1step, casci

def _MCPDFT (mc_class, mc_or_mf_or_mol, ot, ncas, nelecas, ncore=None, frozen=None, ci_min='ecas',
             **kwargs):
    if isinstance (mc_or_mf_or_mol, (mc1step.CASSCF, casci.CASCI)):
        mc0 = mc_or_mf_or_mol
        mf_or_mol = mc_or_mf_or_mol._scf
    else:
        mc0 = None
        mf_or_mol = mc_or_mf_or_mol
    if frozen is not None: mc1 = mc_class (mf_or_mol, ncas, nelecas, ncore=ncore, frozen=frozen)
    else: mc1 = mc_class (mf_or_mol, ncas, nelecas, ncore=ncore)
    mc2 = get_mcpdft_child_class (mc1, ot, ci_min=ci_min, **kwargs)
    if mc0 is not None:
        mc2.mo_coeff = mc_or_mf_or_mol.mo_coeff.copy ()    
        mc2.ci = copy.deepcopy (mc_or_mf_or_mol.ci)
        mc2.converged = mc0.converged
    return mc2

def CASSCFPDFT (mc_or_mf_or_mol, ot, ncas, nelecas, ncore=None, frozen=None, ci_min='ecas',
                **kwargs):
    return _MCPDFT (mcscf.CASSCF, mc_or_mf_or_mol, ot, ncas, nelecas, ncore=ncore, frozen=frozen,
                    ci_min='ecas', **kwargs)

def CASCIPDFT (mc_or_mf_or_mol, ot, ncas, nelecas, ncore=None, ci_min='ecas', **kwargs):
    return _MCPDFT (mcscf.CASCI, mc_or_mf_or_mol, ot, ncas, nelecas, ncore=ncore, ci_min='ecas',
                    **kwargs)

CASSCF=CASSCFPDFT
CASCI=CASCIPDFT

