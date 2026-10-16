
from numbers import Integral

from pyscf.pbc import scf, dft
from mrh.my_pyscf.pbc.mcscf import casci
from mrh.my_pyscf.pbc.mcscf import kcasci
from mrh.my_pyscf.pbc.mcscf import mc1step
from mrh.my_pyscf.pbc.mcscf.productstate import (
    PBCTransSymmImpureProductStateFCISolver,
)
from mrh.my_pyscf.pbc.mcscf.klasci import (
    PBCLASCINoSymm,
    PBCLASCITransSymm,
    kLASCI,
)
from mrh.my_pyscf.pbc.mcscf.klasscf import (
    KLASSCF_UnitaryGroupGenerators,
    KLASSCF_HessianOperator,
    PBCLASSCFNoSymm,
    kLASSCF,
)
from mrh.my_pyscf.pbc.mcscf.real_linear_solvers import (
    SolveScipyCGForCplx,
    SolveScipyMINRESForCplx,
)


def _sanity_check_for_kmf(kmf):
    assert isinstance(kmf, scf.hf.SCF), \
        "PBC MCSCF only works with periodic SCF objects"

    if isinstance(kmf, (dft.krks.KRKS, dft.kuks.KUKS,
                        dft.rks.RKS, dft.uks.UKS)):
        msg = "PBC MCSCF only works with periodic HF objects."
        raise NotImplementedError(msg)

    if isinstance(kmf, scf.kuhf.KUHF):
        kmf = scf.addons.convert_to_rhf(kmf)

    return kmf

def CASCI(kmf, ncas, nelecas, ncore=None):
    kmf = _sanity_check_for_kmf(kmf)
    kmc = casci.CASCI(kmf, ncas, nelecas, ncore)
    return kmc


def KCASCI(kmf, ncas, nelecas, ncore=None, target_k=None, charge=None,
           charged_spin=None):
    kmf = _sanity_check_for_kmf(kmf)
    if charge is None:
        if target_k is None: target_k = 0
        return kcasci.KCASCI(kmf, ncas, nelecas, ncore, target_k=target_k,)
    return kcasci.ChargedKCASCI(
        kmf, ncas, nelecas, ncore, charge=charge,
        target_k=target_k, charged_spin=charged_spin,
    )


def CASSCF(kmf, ncas, nelecas, ncore=None):
    kmf = _sanity_check_for_kmf(kmf)
    kmc = mc1step.CASSCF(kmf, ncas, nelecas, ncore)
    return kmc


KLASCI = kLASCI
KLASSCF = kLASSCF
# The kLASCI call function should be added here instead of defining it in kasci.py
