
from pyscf.pbc import scf
from mrh.my_pyscf.pbc.mcscf import casci


def CASCI(kmf, ncas, nelecas, ncore=None):
    assert isinstance(kmf, scf.hf.SCF),  "CASCI only works with periodic SCF objects"
    kmc = casci.CASCI(kmf, ncas, nelecas, ncore)
    return kmc
