# Lahh dee dah
from mrh.my_pyscf.mcpdft.mcpdft import get_mcpdft_child_class
from pyscf import mcscf

def CASSCFPDFT (mf_or_mol, ot, ncas, nelecas, ncore=None, frozen=None):
    mc = mcscf.CASSCF (mf_or_mol, ncas, nelecas, ncore=ncore, frozen=frozen)
    return get_mcpdft_child_class (mc, ot)

def CASCIPDFT (mf_or_mol, ot, ncas, nelecas, ncore=None, frozen=None):
    mc = mcscf.CASCI (mf_or_mol, ncas, nelecas, ncore=ncore, frozen=frozen)
    return get_mcpdft_child_class (mc, ot)

CASSCF=CASSCFPDFT
CASCI=CASCIPDFT

