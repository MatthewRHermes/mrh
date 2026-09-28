# !/bin/bash

import numpy as np
from functools import reduce

from pyscf import gto, lib
from pyscf.lib import logger, param
from pyscf.mcpdft import _dms
from pyscf.mcpdft.otpd import get_ontop_pair_density
from pyscf.mcpdft.otfnal import otfnal
from pyscf.mcpdft.otfnal import get_transfnal, transfnal, ftransfnal
from pyscf import __config__
from pyscf.pbc import gto as pbcgto, dft
from pyscf.mcscf import mc1step, casci
from pyscf.pbc.lib import kpts_helper

from mrh.my_pyscf.pbc.mcscf import mc1step as pbc_mc1step, casci as pbc_casci
from mrh.my_pyscf.pbc.mcscf.k2R import get_mo_coeff_k2R_wokmf
from mrh.my_pyscf.pbc.mcscf.mc1step import _get_casdm2_kpts as _basis_transform_casdm2_kpts
from mrh.my_pyscf.pbc.mcpdft.kotpd import get_ontop_pair_density_kpts
from mrh.my_pyscf.pbc.mcpdft._dms import dm2_cumulant_complex
from mrh.my_pyscf.pbc.mcpdft import _dms as pbc_dms

# Author: Bhavnesh Jangid

def redefine_fnal(original_fnal, new_parent, **kwargs):
    class transfnal(original_fnal.__class__, new_parent):
        pass
    new_fnal = lib.view(original_fnal, transfnal)

    # Hack to pass on the cell and the kpts info to ot object.
    # otherwise I need to refactor the whole code to pass the cell 
    # and kpts info to ot object.
    for key, value in kwargs.items():
        setattr(new_fnal, key, value)
    return new_fnal

redefine_transfnal = redefine_fnal
redefine_ftransfnal = redefine_fnal

def _get_mol_or_cell(kmc_or_kmf_mol_cell):
    '''
    A function to get the mol object from the kmc_or_kmf_mol object
    '''
    if isinstance(kmc_or_kmf_mol_cell, (mc1step.CASSCF, casci.CASCI)):
        return kmc_or_kmf_mol_cell._scf.mol
    elif isinstance(kmc_or_kmf_mol_cell, (pbc_mc1step.CASSCF, pbc_casci.CASCI)):
        return kmc_or_kmf_mol_cell._scf.cell
    elif isinstance(kmc_or_kmf_mol_cell, gto.Mole) or \
        isinstance(kmc_or_kmf_mol_cell, pbcgto.cell.Cell):
        return kmc_or_kmf_mol_cell
    elif getattr(kmc_or_kmf_mol_cell, 'mol', None) is not None:
        return kmc_or_kmf_mol_cell.mol
    elif getattr(kmc_or_kmf_mol_cell, 'cell', None) is not None:
        return kmc_or_kmf_mol_cell.cell
    else:
        raise ValueError ("The input object is not recognized. " \
        "It should be either MC-SCF/SCF or Mole/Cell object.")


class otfnalperiodic_gamma(otfnal):
    '''
    Child class to define the otfnal class for periodic systems. Only at the Gamma point.
    '''
    def energy_ot (ot, casdm1s, casdm2, mo_coeff, ncore, 
                   max_memory=param.MAX_MEMORY, hermi=1):
        '''
        See the docstring of pyscf/mcpdft/otfnal.energy_ot for more information.
        '''

        E_ot = 0.0
        ni = ot._numint
        xctype =  ot.xctype

        if xctype=='HF': 
            return E_ot
        
        dens_deriv = ot.dens_deriv
        Pi_deriv = ot.Pi_deriv
        
        nao = mo_coeff.shape[0]
        ncas = casdm2.shape[0]

        # First construct the cumulant then transform it to block mo-orbitals basis.
        cascm2 = _dms.dm2_cumulant(casdm2, casdm1s)
        
        dm1s = _dms.casdm1s_to_dm1s (ot, casdm1s, mo_coeff=mo_coeff, ncore=ncore,
                                    ncas=ncas)
        mo_cas = mo_coeff[:,ncore:][:,:ncas]
        t0 = (logger.process_clock (), logger.perf_counter ())
        make_rho = tuple (ni._gen_rho_evaluator (ot.mol, dm1s[i,:,:], hermi) for
            i in range(2))
        
        for ao_k1, ao_k2, mask, weight, _ \
            in ni.block_loop(ot.mol, ot.grids, nao, deriv=dens_deriv, 
                             kpt=None, max_memory=max_memory):
            '''
            ao_k1 and ao_k2 are the block of AO integrals for the given k-point. They
            are the same for supercell(1x1x1) calculations.
            '''
            rho = np.asarray ([m[0] (0, ao_k1, mask, xctype) for m in make_rho])
            t0 = logger.timer (ot, 'untransformed density', *t0)
            Pi = get_ontop_pair_density (ot, rho, ao_k1, cascm2, mo_cas,
                Pi_deriv, mask)
            t0 = logger.timer (ot, 'on-top pair density calculation', *t0)
            if rho.ndim == 2:
                rho = np.expand_dims (rho, 1)
                Pi = np.expand_dims (Pi, 0)
            E_ot += ot.eval_ot (rho, Pi, dderiv=0, weights=weight)[0].dot (weight)
            t0 = logger.timer (ot, 'on-top energy calculation', *t0)
        return E_ot

    energy_ot.__doc__ = otfnal.energy_ot.__doc__

    def reset(self, mol=None):
        '''
        Discard cached grid data and optionally update the cell object.
        I am not changing the input parameter so that it is compatible with the current
        MCPDFT code.
        '''
        if mol is not None:
            self.mol = mol
        # A hack to reset the grids for the new cell object.
        self.grids.reset (mol) 

def _energy_ot_from_kpts(ot, casdm1s_kpts, cascm2_kpts, mo_coeff,
                         ncore, kconserv, max_memory=param.MAX_MEMORY,
                         hermi=1):
    """Evaluate an on-top functional from prepared k-space active RDMs."""
    if ot.xctype == 'HF':
        return 0.0

    mo_coeff = np.asarray(mo_coeff)
    casdm1s_kpts = np.asarray(casdm1s_kpts)
    cascm2_kpts = np.asarray(cascm2_kpts)
    kconserv = np.asarray(kconserv)

    if mo_coeff.ndim != 3:
        raise ValueError("mo_coeff must have shape (nkpts, nao, nmo)",)

    nkpts, nao = mo_coeff.shape[:2]

    if casdm1s_kpts.ndim != 4 or casdm1s_kpts.shape[:2] != (2, nkpts):
        raise ValueError("casdm1s_kpts must have shape (2, nkpts, ncas, ncas)",)

    ncas = casdm1s_kpts.shape[2]
    expected_dm1_shape = (2, nkpts, ncas, ncas)
    if casdm1s_kpts.shape != expected_dm1_shape:
        raise ValueError(f"Expected casdm1s_kpts shape {expected_dm1_shape}, "
                         f"got {casdm1s_kpts.shape}",)
    expected_cm2_shape = (
        nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas,
    )
    if cascm2_kpts.shape != expected_cm2_shape:
        raise ValueError(f"Expected cascm2_kpts shape {expected_cm2_shape}, "
                         f"got {cascm2_kpts.shape}",)

    if kconserv.shape != (nkpts, nkpts, nkpts):
        raise ValueError(f"kconserv must have shape (nkpts, nkpts, nkpts)",)

    if ncore < 0 or ncore + ncas > mo_coeff.shape[2]:
        raise ValueError("ncore and ncas are incompatible with mo_coeff")

    dm1s_kpts = pbc_dms.casdm1s_kpts_to_dm1s(
        ot, casdm1s_kpts, mo_coeff, ncore,
    )

    ni = ot._numint


    make_rho_alpha, nset_a, nao_a = ni._gen_rho_evaluator(
        ot.cell, dm1s_kpts[0], hermi, False,
    )
    make_rho_beta, nset_b, nao_b = ni._gen_rho_evaluator(
        ot.cell, dm1s_kpts[1], hermi, False,
    )

    if nset_a != 1 or nset_b != 1:
        raise NotImplementedError("k-MC-PDFT requires one density set")
    if nao_a != nao or nao_b != nao:
        raise ValueError("Density evaluator and MO AO dimensions differ")

    mo_cas = np.ascontiguousarray(
        mo_coeff[:, :, ncore:ncore + ncas],
    )
    make_rho = (make_rho_alpha, make_rho_beta)
    kpts = np.asarray(ot.kpts).reshape(-1, 3)
    if kpts.shape[0] != nkpts:
        raise ValueError("ot.kpts and mo_coeff contain different k-point counts")

    energy_ot = 0.0
    t0 = (logger.process_clock(), logger.perf_counter())
    for ao_k1, ao_k2, mask, weight, _ in ni.block_loop(
            ot.cell, ot.grids, nao, deriv=ot.dens_deriv, kpts=kpts,
            max_memory=max_memory):
        rho = np.asarray([
            make_rho_spin(0, ao_k1, mask, ot.xctype).real
            for make_rho_spin in make_rho
        ])
        t0 = logger.timer(ot, 'untransformed density', *t0)
        Pi = get_ontop_pair_density_kpts(
            ot, rho, ao_k2, cascm2_kpts, mo_cas, kconserv,
            deriv=ot.Pi_deriv, non0tab=mask,
        )
        t0 = logger.timer(ot, 'on-top pair density calculation', *t0)
        if rho.ndim == 2:
            rho = np.expand_dims(rho, 1)
            Pi = np.expand_dims(Pi, 0)
        energy_ot += ot.eval_ot(
            rho, Pi, dderiv=0, weights=weight,
        )[0].dot(weight)
        t0 = logger.timer(ot, 'on-top energy calculation', *t0)
    return energy_ot


def _prepare_wannier_rdms(ot, casdm1s, casdm2, mo_coeff, ncore):
    """Transform Wannier-basis RDMs to momentum blocks."""
    mo_coeff = np.asarray(mo_coeff)
    casdm1s = np.asarray(casdm1s)
    casdm2 = np.asarray(casdm2)
    assert mo_coeff.ndim == 3

    nkpts = mo_coeff.shape[0]
    ncastot = casdm2.shape[0]
    ncas = ncastot // nkpts
    assert getattr(ot, 'kmesh', None) is not None
    assert casdm2.shape == (ncastot,) * 4
    assert casdm1s.shape == (2, ncastot, ncastot)

    mo_phase = get_mo_coeff_k2R_wokmf(
        ot.cell, mo_coeff, ncore, ncas, ot.kpts, kmesh=ot.kmesh,
    )[-1]

    cascm2 = dm2_cumulant_complex(casdm2, casdm1s)
    cascm2_kpts = np.zeros(
        (nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas),
        dtype=mo_coeff.dtype,
    )

    kconserv = kpts_helper.get_kconserv(ot.cell, ot.kpts)
    for k1, k2, k3 in kpts_helper.loop_kkk(nkpts):
        k4 = kconserv[k1, k2, k3]
        cascm2_kpts[k1, k2, k3] = _basis_transform_casdm2_kpts(
            cascm2, mo_phase, (k1, k2, k3, k4),
        )

    casdm1s_kpts = []
    for k in range(nkpts):
        casdm1s_kpts.append([
            reduce(np.dot, (mo_phase[k], dm1, mo_phase[k].conj().T))
            for dm1 in casdm1s
        ])
    casdm1s_kpts = np.asarray(casdm1s_kpts).transpose(1, 0, 2, 3)
    return casdm1s_kpts, cascm2_kpts, kconserv


def _prepare_bloch_rdms(ot, casdm1s, casdm2, mo_coeff, momentum_tol):
    """Extract momentum blocks from flattened Bloch-basis RDMs."""
    mo_coeff = np.asarray(mo_coeff)
    casdm2 = np.asarray(casdm2)
    if mo_coeff.ndim != 3:
        raise ValueError("mo_coeff must have shape (nkpts, nao, nmo)")
    if casdm2.ndim != 4 or len(set(casdm2.shape)) != 1:
        raise ValueError("casdm2 must have shape (ncas * nkpts,) * 4")

    nkpts = mo_coeff.shape[0]
    ncastot = casdm2.shape[0]
    if ncastot % nkpts:
        raise ValueError("The active RDM size must be divisible by nkpts")
    ncas = ncastot // nkpts
    kconserv = getattr(ot, 'kconserv', None)
    if kconserv is None:
        kconserv = kpts_helper.get_kconserv(ot.cell, ot.kpts)
    casdm1s_kpts, cascm2_kpts = pbc_dms.make_kcas_rdms_kpts(
        casdm1s, casdm2, nkpts, ncas, kconserv,
        momentum_tol=momentum_tol,
    )
    return casdm1s_kpts, cascm2_kpts, kconserv


def _prepare_kpts_rdms(ot, casdm1s, casdm2, mo_coeff, ncore,
                       representation, momentum_tol):
    """Prepare active RDMs for the shared k-point evaluator."""
    if representation == 'wannier':
        return _prepare_wannier_rdms(
            ot, casdm1s, casdm2, mo_coeff, ncore,
        )
    if representation == 'bloch':
        return _prepare_bloch_rdms(
            ot, casdm1s, casdm2, mo_coeff, momentum_tol,
        )
    raise ValueError(f"Unknown RDM representation {representation!r}")


class otfnalperiodic_kpts(otfnal):
    """On-top functional for periodic k-point calculations."""

    def energy_ot(ot, casdm1s, casdm2, mo_coeff, ncore,
                  max_memory=param.MAX_MEMORY, hermi=1,
                  rdm_representation='wannier', momentum_tol=1e-8):
        """Evaluate the on-top energy from Wannier- or Bloch-basis RDMs."""
        if ot.xctype == 'HF':
            return 0.0
        casdm1s_kpts, cascm2_kpts, kconserv = _prepare_kpts_rdms(
            ot, casdm1s, casdm2, mo_coeff, ncore,
            rdm_representation, momentum_tol,
        )
        return _energy_ot_from_kpts(
            ot, casdm1s_kpts, cascm2_kpts, mo_coeff, ncore, kconserv,
            max_memory=max_memory, hermi=hermi,
        )

def _get_ks_obj(kmc_or_kmf_or_cell, khf=False, kpts=None):
    '''
    Initialize KS object with appropriate density fitting object GDF, 
    MDF or FFTDF
    args:
        kmc_or_kmf_or_cell : kMC or kMF object with cell object
    returns:
        ks : KS object with app. density fitting object GDF, MDF or FFTDF
    '''
    cell = _get_mol_or_cell (kmc_or_kmf_or_cell)
    if hasattr(kmc_or_kmf_or_cell, 'with_df'):
        dfclass = kmc_or_kmf_or_cell.with_df.__class__.__name__
    
    elif hasattr(kmc_or_kmf_or_cell, '_las'):
        dfclass = kmc_or_kmf_or_cell._las.with_df.__class__.__name__
    else:
        raise ValueError ("The input object does not have with_df attribute. \
                          Start with Mean-field object")
    
    df_method = 'density_fit' if dfclass == 'GDF' \
        else 'mix_density_fit' if dfclass == 'MDF' else None
    if df_method is None:
        raise NotImplementedError("PBD-MCPDFT is yet not implemented for FFTDF")
    
    ks_class = dft.KRKS(cell, kpts=kpts) if khf else dft.RKS(cell)
    ks = getattr(ks_class, df_method)()

    return ks


def _get_pbc_otfnal(kmc_or_kmf_or_cell, otxc, otfnalperiodic_class,
                    cell_kptsinfo=None):
    '''
    This is wrapper function to get the appropriate fnal class 
    for the given cell object
    args:
        kmc_or_kmf_or_cell : kMC or kMF object with cell object
        otxc : str, on-top functional name
    kwargs:
        cell_kptsinfo : dict, optional, default: {}
            Dictionary containing the cell and kpts info to be passed to the 
            otfnalperiodic class. This is a hack to avoid refactoring the whole code 
            to pass the cell and kpts only needed for the kpts calculations.
    '''
    if cell_kptsinfo is None:
        cell_kptsinfo = {}

    cell = _get_mol_or_cell (kmc_or_kmf_or_cell)
    fnal_class = get_transfnal (cell, otxc)
    fnal_class_type = fnal_class.__class__.__name__

    assert isinstance(otxc, str), "The otxc should be a string"
    xc_base = fnal_class.otxc

    # If k-points info is provided in the cell_kptsinfo dict, use it
    if isinstance(cell_kptsinfo, dict) and cell_kptsinfo.get('kpts') is not None:
        ks = _get_ks_obj(kmc_or_kmf_or_cell, khf=True, kpts=cell_kptsinfo['kpts'])
    else:
        ks = _get_ks_obj(kmc_or_kmf_or_cell)

    if fnal_class_type == 'transfnal':
        xc_base = xc_base[1:]
        ks.xc = xc_base
        org_transfnal = transfnal(ks)
        new_func_class = redefine_transfnal (org_transfnal, 
                                             otfnalperiodic_class, **cell_kptsinfo)
        del org_transfnal

    elif fnal_class_type == 'ftransfnal':
        xc_base = xc_base[2:]
        ks.xc = xc_base
        org_ftransfnal = ftransfnal(ks)
        new_func_class = redefine_ftransfnal (org_ftransfnal, 
                                              otfnalperiodic_class, **cell_kptsinfo)
        del org_ftransfnal
    else:
        raise ValueError ("The fnal class is not recognized")

    logger.info(cell, 'Periodic OT-FNAL class is used')
    return new_func_class

def get_pbc_otfnal_gamma(kmc_or_kmf_or_cell, otxc):
    return _get_pbc_otfnal(kmc_or_kmf_or_cell, otxc, otfnalperiodic_gamma)

def get_pbc_otfnal_kpts(kmc_or_kmf_or_cell, otxc):
    cell = _get_mol_or_cell (kmc_or_kmf_or_cell)
    kpts = getattr(kmc_or_kmf_or_cell, 'kpts', None)
    try:
        kmesh = getattr(kmc_or_kmf_or_cell, 'kmesh', None)
    except TypeError:
        # PySCF's SCF.kmesh property changed with the kpts_to_kmesh API.
        kmesh = None

    assert kpts is not None, "kpts is required for kpts-based OT-FNAL"
    if kmesh is None:
        from pyscf.pbc.tools.k2gamma import kpts_to_kmesh
        try:
            kmesh = kpts_to_kmesh(cell, kpts)
        except TypeError:
            # Compatibility with PySCF releases whose helper accepted only
            # the k-point array.
            kmesh = kpts_to_kmesh(kpts)

    cell_kptsinfo = {
        'cell': cell, 
        'kpts': kpts, 
        'kmesh': kmesh}

    return _get_pbc_otfnal(kmc_or_kmf_or_cell, otxc, otfnalperiodic_kpts, 
                           cell_kptsinfo=cell_kptsinfo)


def sanity_check_for_kpts(mc_or_mf_or_cell):
    """Require the single k-point supported by gamma-point MC-PDFT."""
    obj = mc_or_mf_or_cell
    if hasattr(obj, "_las"):
        obj = obj._las
    if hasattr(obj, "_scf"):
        obj = obj._scf

    kpts = getattr(obj, "kpts", None)
    if kpts is None:
        raise NotImplementedError("The input object does not have kpts attribute")
    if len(kpts) > 1:
        raise ValueError("Only supercell calculations can be performed with MC-PDFT")


def periodicpdft(mc_or_mf_or_mol, ot):
    """Return the periodic gamma-point on-top functional when appropriate.

    Molecular inputs are returned unchanged.  This behavior is retained for
    callers of the historical ``mrh.my_pyscf.mcpdft.periodicpdft`` helper.
    """
    assert isinstance(ot, str), "The ot should be a string"
    mol_or_cell = _get_mol_or_cell(mc_or_mf_or_mol)
    if isinstance(mol_or_cell, pbcgto.cell.Cell):
        sanity_check_for_kpts(mc_or_mf_or_mol)
        return get_pbc_otfnal_gamma(mc_or_mf_or_mol, ot)
    return ot


# Historical names kept here so the compatibility module in
# ``mrh.my_pyscf.mcpdft`` can be a thin re-export of the periodic code.
otfnalperiodic = otfnalperiodic_gamma
_get_transfnal = get_pbc_otfnal_gamma
sanity_check_for_df = _get_ks_obj
