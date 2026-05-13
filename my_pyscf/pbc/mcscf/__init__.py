
from pyscf.pbc import scf, dft
from mrh.my_pyscf.pbc.mcscf import casci
from mrh.my_pyscf.pbc.mcscf import mc1step

def CASCI(kmf, ncas, nelecas, ncore=None):
    assert isinstance(kmf, scf.hf.SCF),  "CASCI only works with periodic SCF objects"
    # Make sure kdft mean field objects are not passed to kCASCI
    if isinstance(kmf, dft.krks.KRKS) or isinstance(kmf, dft.kuks.KUKS) \
        or isinstance(kmf, dft.rks.RKS) or isinstance(kmf, dft.uks.UKS):
        raise NotImplementedError("CASCI with DFT is not implemented yet.")
    if isinstance(kmf, scf.kuhf.KUHF):
        kmf = scf.addons.convert_to_rhf(kmf)
    kmc = casci.CASCI(kmf, ncas, nelecas, ncore)
    return kmc

def CASSCF(kmf, ncas, nelecas, ncore=None):
    assert isinstance(kmf, scf.hf.SCF),  "CASSCF only works with periodic SCF objects"
    # Make sure kdft mean field objects are not passed to kCASSCF
    if isinstance(kmf, dft.krks.KRKS) or isinstance(kmf, dft.kuks.KUKS) \
        or isinstance(kmf, dft.rks.RKS) or isinstance(kmf, dft.uks.UKS):
        raise NotImplementedError("CASSCF with DFT is not implemented yet.")
    # If the mean-field object is KUHF, convert it to RHF before passing to CASSCF, 
    if isinstance(kmf, scf.kuhf.KUHF):
        kmf = scf.addons.convert_to_rhf(kmf)
    kmc = mc1step.CASSCF(kmf, ncas, nelecas, ncore)
    return kmc

