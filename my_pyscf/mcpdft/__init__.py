# Lahh dee dah
from mrh.my_pyscf.mcpdft.mcpdft import get_mcpdft_child_class
from mrh.my_pyscf.mcpdft.otfnal import make_hybrid_fnal as hyb
from pyscf import mcscf

def CASSCFPDFT (mf_or_mol, ot, ncas, nelecas, ncore=None, frozen=None, **kwargs):
    mc = mcscf.CASSCF (mf_or_mol, ncas, nelecas, ncore=ncore, frozen=frozen)
    return get_mcpdft_child_class (mc, ot, **kwargs)

def CASCIPDFT (mf_or_mol, ot, ncas, nelecas, ncore=None, **kwargs):
    mc = mcscf.CASCI (mf_or_mol, ncas, nelecas, ncore=ncore)
    return get_mcpdft_child_class (mc, ot, **kwargs)

CASSCF=CASSCFPDFT
CASCI=CASCIPDFT

