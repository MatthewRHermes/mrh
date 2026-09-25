#!/bin/bash
import copy

from pyscf import mcscf
from pyscf.lib import logger
from pyscf.pbc import scf

from mrh.my_pyscf.pbc import mcscf as pbc_mcscf
from mrh.my_pyscf.pbc.mcscf import _sanity_check_for_kmf
from mrh.my_pyscf.pbc.mcpdft.mcpdft import get_mcpdft_child_class as get_pbc_mcpdft_child_class_gamma
from mrh.my_pyscf.pbc.mcpdft.kmcpdft import (
    get_charged_kcas_mcpdft_child_class,
    get_kcas_mcpdft_child_class,
    get_mcpdft_child_class,
)

# Author: Bhavnesh Jangid

# Implementing MC-PDFT at gamma point and k-MC-PDFT while reusing the
# periodic MCSCF input validation and as much molecular PDFT code as possible.

"""
Periodic MC-PDFT Structure.

Driver hierarchy::

    PySCF _PDFT
    └── _PeriodicMCPDFT
        ├── _MCPDFT
        │    └── Gamma-point CASCI/CASSCF
        └── _MCPDFTCPLX
            ├── based on cplx CASCI/CASSCF
            └── _kCASPDFT
                ├── neutral momentum-resolved kCASCI
                └── _kChargedCASPDFT
                    └── charged sectors and band energies
"""

def _MCPDFT (mc_class, kmc_or_kmf, ot, ncas, nelecas, ncore=None, frozen=None,
            get_mcpdft_child_class=get_mcpdft_child_class,
            allow_frozen=False, **kwargs):
    # Newton SCF objects also expose ``_scf``.  Identify mean-field objects
    # before using that attribute to distinguish an existing MC calculation.
    kmf0 = getattr(kmc_or_kmf, '_scf', None)
    if isinstance(kmc_or_kmf, scf.hf.SCF) or kmf0 is None:
        kmf0 = _sanity_check_for_kmf(kmc_or_kmf)
        kmc0 = None
    else:
        kmf0 = _sanity_check_for_kmf(kmf0)
        kmc0 = kmc_or_kmf

    if frozen is not None and not allow_frozen:
        raise NotImplementedError("Frozen orbitals are not supported in k-MC-PDFT")
    mc_kwargs = {"ncore": ncore}
    if frozen is not None:
        mc_kwargs["frozen"] = frozen
    kmc = get_mcpdft_child_class(
        mc_class(kmf0, ncas, nelecas, **mc_kwargs), ot, **kwargs,
    )

    if kmc0 is not None:
        from mrh.my_pyscf.pbc.mcscf.casci import PBCCASCI
        if isinstance(kmc0, PBCCASCI):
            kmc.kmesh = kmc0.kmesh
            kmc.kpts = kmc0.kpts
        kmc.verbose = kmc0.verbose
        kmc.stdout = kmc0.stdout
        kmc.mo_coeff = kmc_or_kmf.mo_coeff.copy()
        kmc.ci = copy.deepcopy (kmc_or_kmf.ci)
        kmc.converged = kmc0.converged
    return kmc

# For Gamma-point only.
def CASSCFPDFT(kmc_or_kmf, ot, ncas, nelecas, ncore=None, frozen=None, **kwargs):
    get_mcpdft_child_class = get_pbc_mcpdft_child_class_gamma
    return _MCPDFT(mcscf.CASSCF, kmc_or_kmf, ot, ncas, nelecas, ncore=ncore, frozen=frozen,
                    get_mcpdft_child_class=get_mcpdft_child_class,
                    allow_frozen=True,
                **kwargs)

def CASCIPDFT(kmc_or_kmf, ot, ncas, nelecas, ncore=None, frozen=None, **kwargs):
    get_pbc_mcpdft_child_class = get_pbc_mcpdft_child_class_gamma
    return _MCPDFT(mcscf.CASCI, kmc_or_kmf, ot, ncas, nelecas, ncore=ncore, frozen=frozen,
                    get_mcpdft_child_class=get_pbc_mcpdft_child_class,
                **kwargs)

CASSCF = CASSCFPDFT
CASCI = CASCIPDFT

# For k-points
def kCASSCFPDFT(kmc_or_kmf, ot, ncas, nelecas, ncore=None, frozen=None, **kwargs):
    return _MCPDFT(pbc_mcscf.CASSCF, kmc_or_kmf, ot, ncas, nelecas, ncore=ncore, frozen=frozen,
                **kwargs)

def kCASCIPDFT(kmc_or_kmf, ot, ncas, nelecas, ncore=None, frozen=None,
               target_k=None, charge=None, charged_spin=None, **kwargs):
    """Construct conventional or momentum-resolved k-CASCI-PDFT.

    Existing kCASCI objects select the momentum-resolved route automatically.
    For a mean-field input, ``target_k`` selects a neutral momentum sector and
    a nonzero ``charge`` selects charged momentum sectors.  With neither,
    conventional periodic CASCI is used.  Options that differ from an existing
    kCASCI object reset its calculation after issuing a warning.
    """
    from mrh.my_pyscf.pbc.mcscf.kcasci import PBCKCASCI

    is_kcasci = isinstance(kmc_or_kmf, PBCKCASCI)
    use_kcasci = (is_kcasci or target_k is not None
                  or charge not in (None, 0) or charged_spin is not None)
    if not use_kcasci:
        return _MCPDFT(pbc_mcscf.CASCI, kmc_or_kmf, ot, ncas, nelecas,
                       ncore=ncore, frozen=frozen, **kwargs)
    if frozen is not None:
        raise ValueError("Frozen orbitals are not supported in k-MC-PDFT")

    if not is_kcasci:
        if charged_spin is not None and charge in (None, 0):
            raise ValueError("charged_spin requires charge +1 or -1")
        kcas_kwargs = {"ncore": ncore, "target_k": target_k}
        if charge not in (None, 0):
            kcas_kwargs.update(charge=charge, charged_spin=charged_spin)
        kmc = pbc_mcscf.KCASCI(
            kmc_or_kmf, ncas, nelecas, **kcas_kwargs,
        )
    else:
        kmc = kmc_or_kmf
        existing_charge = getattr(kmc, "charge", 0)
        existing_spin   = getattr(kmc, "charged_spin", None)
        existing_target = getattr(kmc, "target_k", None)

        reset_charge = existing_charge if charge is None else charge
        reset_spin = existing_spin if charged_spin is None else charged_spin
        if charge == 0 and charged_spin is None:
            reset_spin = None
        if reset_charge in (None, 0) and reset_spin is not None:
            raise ValueError("charged_spin requires charge +1 or -1")
        reset_target = existing_target
        if target_k is not None:
            reset_target = int(target_k) % kmc.nkpts

        changes = []
        for name, old, new in (
            ("charge", existing_charge, reset_charge),
            ("charged_spin", existing_spin, reset_spin),
            ("target_k", existing_target, reset_target),
        ):
            if old != new:
                changes.append(f"{name}: {old!r} -> {new!r}")

        # If the existing kCASCI object has different options, reset it to a new one
        # new kCAS object.
        if changes:
            logger.warn(kmc, "Resetting existing PBCKCASCI options (%s)",
                        ", ".join(changes),)
            old_kmc = kmc
            kcas_kwargs = {
                "ncore": getattr(old_kmc, "ncore", ncore),
                "target_k": reset_target,
            }
            if reset_charge not in (None, 0):
                kcas_kwargs.update(
                    charge=reset_charge, charged_spin=reset_spin,
                )
            kmc = pbc_mcscf.KCASCI(
                old_kmc._scf,
                getattr(old_kmc, "ncas", ncas),
                getattr(old_kmc, "nelecas", nelecas),
                **kcas_kwargs,
            )
            for name in (
                "verbose", "stdout", "max_memory", "chkfile", "kmesh",
                "kpts", "mo_coeff", "canonicalization",
            ):
                if hasattr(old_kmc, name):
                    value = getattr(old_kmc, name)
                    if name in ("kmesh", "kpts", "mo_coeff"):
                        value = copy.deepcopy(value)
                    setattr(kmc, name, value)
            if hasattr(old_kmc, "fcisolver"):
                kmc.fcisolver = copy.copy(old_kmc.fcisolver)
                kmc.fcisolver.target_k = (
                    0 if reset_target is None else reset_target
                )

    make_pdft = (get_charged_kcas_mcpdft_child_class
                 if getattr(kmc, "charge", 0)
                 else get_kcas_mcpdft_child_class)
    pdft = make_pdft(kmc, ot, **kwargs)
    if getattr(kmc, "charge", 0) and target_k is not None:
        pdft.target_k = int(target_k) % pdft.nkpts
    return pdft

KCASSCF = kCASSCFPDFT
KCASCI = kCASCIPDFT


def _laspdftEnergy(mc_class, mc_or_mf, ot, ncas_sub, nelecas_sub,
                   ncore=None, spin_sub=None, frozen=None, **kwargs):
    from mrh.my_pyscf.mcscf.lasscf_sync_o0 import LASSCFNoSymm, LASSCFSymm
    from mrh.my_pyscf.mcscf.lasscf_async import (
        LASSCFNoSymm as AsyncLASSCFNoSymm,
        LASSCFSymm as AsyncLASSCFSymm,
    )
    from mrh.my_pyscf.pbc.mcpdft.laspdft import get_mcpdft_child_class

    las_classes = (
        LASSCFNoSymm,
        LASSCFSymm,
        AsyncLASSCFNoSymm,
        AsyncLASSCFSymm,
    )
    if isinstance(mc_or_mf, las_classes):
        las = mc_or_mf
        if frozen is not None:
            las.frozen = frozen
    else:
        las_kwargs = {"ncore": ncore, "spin_sub": spin_sub, "frozen": frozen}
        las = mc_class(mc_or_mf, ncas_sub, nelecas_sub, **las_kwargs)
    return get_mcpdft_child_class(las, ot, **kwargs)


def LASSCFPDFT(mc_or_mf, ot, ncas_sub=None, nelecas_sub=None, ncore=None,
               spin_sub=None, frozen=None, **kwargs):
    """
    Create a gamma-point periodic LAS-PDFT solver.
    """
    if ncas_sub is None:
        ncas_sub = getattr(mc_or_mf, "ncas_sub", None)
    if nelecas_sub is None:
        nelecas_sub = getattr(mc_or_mf, "nelecas_sub", None)
    from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF as MolecularLASSCF
    return _laspdftEnergy(
        MolecularLASSCF,
        mc_or_mf,
        ot,
        ncas_sub,
        nelecas_sub,
        ncore=ncore,
        spin_sub=spin_sub,
        frozen=frozen,
        **kwargs,
    )


LASSCF = LASSCFPDFT
