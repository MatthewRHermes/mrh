import numpy as np

from pyscf.lib import logger
from pyscf.mcpdft import _dms

from mrh.my_pyscf.pbc.mcpdft.otfnalperiodic import (
    _prepare_kpts_rdms,
    get_pbc_otfnal_kpts,
    otfnalperiodic_kpts,
)
from mrh.my_pyscf.pbc.mcscf.casci import get_h2eff_kpts
from mrh.my_pyscf.pbc.mcpdft.mcpdft import _PeriodicMCPDFT
from mrh.my_pyscf.pbc.mcpdft import _dms as pbc_dms

'''
Author: Bhavnesh Jangid
k-MC-PDFT for periodic systems at the gamma point or k-points.
'''

_get_fcisolver = _dms._get_fcisolver


def _select_charged_kcas_result(mc, target_k=None):
    """Select one stored charged KCASCI momentum-sector result."""
    results = list(getattr(mc, "charged_results", ()))
    target_k = getattr(mc, "target_k", None) if target_k is None else target_k
    if target_k is None:
        if len(results) == 1: return results[0]
        raise ValueError("target_k is required when multiple charged KCASCI sectors "
            "are available",)

    target_k = int(target_k) % mc.nkpts

    result = next((
        item for item in results
        if int(item["target_k"]) % mc.nkpts == target_k
    ), None)

    if result is None:
        raise ValueError(f"No charged KCASCI result is available for target_k={target_k}",)
    return result


def _get_charged_kcas_rdm_context(mc, ci=None, state=0, target_k=None):
    """Resolve RDM arguments after selecting a charged momentum sector."""
    result = _select_charged_kcas_result(mc, target_k=target_k)
    return pbc_dms._get_charged_kcas_rdm_context(
        mc, result, ci=ci, state=state,
    )


def make_one_casdm1s_charged_kcas(mc, ci=None, state=0, target_k=None):
    """Build a charged kCASCI 1-RDM for one selected momentum sector."""
    result = _select_charged_kcas_result(mc, target_k=target_k)
    return pbc_dms.make_one_casdm1s_charged_kcas(
        mc, result, ci=ci, state=state,
    )


def make_one_casdm2_charged_kcas(mc, ci=None, state=0, target_k=None):
    """Build a charged kCASCI 2-RDM for one selected momentum sector."""
    result = _select_charged_kcas_result(mc, target_k=target_k)
    return pbc_dms.make_one_casdm2_charged_kcas(
        mc, result, ci=ci, state=state,
    )


# Need to redefine the casdm1s and casdm2 because of shape mismatch.
def make_one_casdm1s (mc, ci, state=0):
    '''
    Spin-separated 1-RDMs.
    Note: the returned RDM1a, and RDM1b are in the shape of (ncas*nkpts, ncas*nkpts)
    and not (ncas, ncas) and in wannier orbital basis. Transform it before using 
    it k-pts machinary.
    '''
    nkpts = mc.nkpts
    ncastot = mc.ncas *  nkpts
    fcisolver, ci, nelecas = _get_fcisolver (mc, ci, state=state)
    nelecastot = (nelecas[0]*nkpts, nelecas[1]*nkpts)
    return fcisolver.make_rdm1s (ci, ncastot, nelecastot)

def make_one_casdm2 (mc, ci, state=0):
    '''
    Spin-summed 2-RDM
    Note: the returned RDM2 is in the shape of (ncas*nkpts, ncas*nkpts, ncas*nkpts, ncas*nkpts)
    and not (ncas, ncas, ncas, ncas) and in wannier orbital basis. Transform it before using 
    it k-pts machinary.
    '''
    ncas = mc.ncas
    fcisolver, ci, nelecas = _get_fcisolver (mc, ci, state=state)
    ncastot = ncas * mc.nkpts
    nelecastot = (nelecas[0]*mc.nkpts, nelecas[1]*mc.nkpts)
    try:
        casdm2 = fcisolver.make_rdm2 (ci, ncastot, nelecastot)
    except AttributeError:
        _, casdm2 = fcisolver.make_rdm12 (ci, ncastot, nelecastot)
    return casdm2

def _energy_mcwfn_from_kpts(mc, casdm1s_kpts, cascm2_kpts, mo_coeff=None,
                            ot=None, verbose=None):
    """Compute the MC wavefunction energy from k-point active-space RDMs."""
    if ot is None:
        ot = mc.otfnal
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    if verbose is None:
        verbose = mc.verbose

    mo_coeff = np.asarray(mo_coeff)
    nkpts = mc.nkpts
    dm1s_kpts = pbc_dms.casdm1s_kpts_to_dm1s(
        mc, casdm1s_kpts, mo_coeff, mc.ncore,
    )
    dm1_kpts = dm1s_kpts[0] + dm1s_kpts[1]

    log = logger.new_logger(mc, verbose=verbose)
    hyb_x, hyb_c = ot._numint.rsh_and_hybrid_coeff(
        ot.otxc, mc.cell.spin,
    )[2]
    h1e_kpts = np.asarray(mc.get_hcore(kpts=mc.kpts))
    with_exchange = log.verbose >= logger.DEBUG or abs(hyb_x) > 1e-10
    if with_exchange:
        vj_spin, vk_kpts = mc._scf.get_jk(
            mc.cell, dm_kpts=dm1s_kpts, kpts=mc.kpts,
        )
        vj_kpts = vj_spin[0] + vj_spin[1]
    else:
        vj_kpts = mc._scf.get_jk(
            mc.cell, dm_kpts=dm1_kpts, kpts=mc.kpts,
            hermi=1, with_k=False,
        )[0]
        vk_kpts = None

    energy_one = np.einsum("kij,kji->", h1e_kpts, dm1_kpts) / nkpts
    energy_j = 0.5 * np.einsum("kij,kji->", vj_kpts, dm1_kpts,) / nkpts

    # This part is basically copied and kept same as in molecular MC-PDFT code.
    if abs(hyb_x - hyb_c) > 1e-10:
        msg = (
            "exchange and correlation hybridization differ "
            "may lead to unphysical results, see "
            "https://github.com/pyscf/pyscf-forge/issues/128",
        )
        log.warn(msg)

    energy_x = 0.0
    if with_exchange:
        energy_x = -0.5 * (np.einsum("kij,kji->", vk_kpts[0], dm1s_kpts[0])
                           + np.einsum("kij,kji->", vk_kpts[1], dm1s_kpts[1])) / nkpts

    energy_c = 0.0
    if log.verbose >= logger.DEBUG or abs(hyb_c) > 1e-10:
        energy_c = np.einsum(
            "abcuvxy,abcuvxy->",
            get_h2eff_kpts(mc, mo_coeff), cascm2_kpts,
            optimize=True,
        ) / (2 * nkpts)

    energy_nuc = mc.energy_nuc()
    for label, value in (("Vnn", energy_nuc), ("Te + Vne", energy_one),
                         ("E_j", energy_j), ("E_x", energy_x),
                         ("E_c", energy_c)):
        log.debug("%s = %s", label, value)
    return (
        energy_nuc + energy_one + energy_j
        + hyb_x * energy_x + hyb_c * energy_c
    )


def energy_mcwfn(mc, mo_coeff=None, ci=None, ot=None, state=0,
                 casdm1s=None, casdm2=None, verbose=None,
                 rdm_representation=None, momentum_tol=1e-8):
    """Evaluate the periodic MC wavefunction energy."""
    mo_coeff = mc.mo_coeff if mo_coeff is None else mo_coeff
    ci = mc.ci if ci is None else ci
    if casdm1s is None:
        casdm1s = mc.make_one_casdm1s(ci=ci, state=state)
    if casdm2 is None:
        casdm2 = mc.make_one_casdm2(ci=ci, state=state)

    if rdm_representation is None:
        rdm_representation = mc._mcwfn_rdm_representation
    casdm1s_kpts, cascm2_kpts, _ = _prepare_kpts_rdms(
        mc, casdm1s, casdm2, mo_coeff, mc.ncore,
        rdm_representation, momentum_tol,
    )
    return _energy_mcwfn_from_kpts(
        mc, casdm1s_kpts, cascm2_kpts, mo_coeff=mo_coeff,
        ot=ot, verbose=verbose,
    )


def energy_dft_kcas(mc, mo_coeff=None, ci=None, ot=None, state=0,
                    casdm1s=None, casdm2=None, max_memory=None, hermi=1,
                    momentum_tol=1e-8):
    """Evaluate the on-top functional directly from momentum kCAS RDMs."""
    if ot is None:
        ot = mc.otfnal
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    if ci is None:
        ci = mc.ci
    if casdm1s is None:
        casdm1s = mc.make_one_casdm1s(ci, state=state)
    if casdm2 is None:
        casdm2 = mc.make_one_casdm2(ci, state=state)
    if max_memory is None:
        max_memory = mc.max_memory
    return ot.energy_ot(
        casdm1s, casdm2, mo_coeff, mc.ncore,
        max_memory=max_memory, hermi=hermi,
        rdm_representation="bloch",
        momentum_tol=momentum_tol,
    )


def energy_tot_charged_kcas(mc, mo_coeff=None, ci=None, ot=None, state=0,
                            target_k=None, verbose=None):
    """Evaluate MC-PDFT for one charged KCASCI momentum sector and root."""
    result = _select_charged_kcas_result(mc, target_k=target_k)
    target_k = int(result["target_k"]) % int(mc.nkpts)
    if ot is None:
        ot = mc.otfnal
    ot.reset(mol=mc.mol)
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    if ci is None:
        ci = result.get("ci")
    if verbose is None:
        verbose = mc.verbose

    casdm1s = mc.make_one_casdm1s(
        ci=ci, state=state, target_k=target_k,
    )
    casdm2 = mc.make_one_casdm2(
        ci=ci, state=state, target_k=target_k,
    )
    e_mcwfn = mc.energy_mcwfn(
        ot=ot, mo_coeff=mo_coeff, casdm1s=casdm1s,
        casdm2=casdm2, verbose=verbose,
    )
    e_ot = mc.energy_dft(
        ot=ot, mo_coeff=mo_coeff, casdm1s=casdm1s,
        casdm2=casdm2,
    )
    e_tot = (e_mcwfn + e_ot).real
    logger.note(
        mc,
        "MC-PDFT charged target_k %d state %d E = %s, Eot(%s) = %s",
        target_k, state, e_tot, ot.otxc, e_ot,
    )
    return e_tot, e_ot


class _MCPDFTCPLX(_PeriodicMCPDFT):
    """MC-PDFT for conventional complex k-point CASCI/CASSCF."""

    momentum_resolved = False
    _mcwfn_rdm_representation = "wannier"
    _get_pbc_otfnal = staticmethod(get_pbc_otfnal_kpts)

    def multi_state(self, method='Lin'):
        raise NotImplementedError(f"StateAverageMix not available for {method}")

    make_one_casdm1s = make_one_casdm1s
    make_one_casdm2 = make_one_casdm2
    energy_mcwfn = energy_mcwfn

    def energy_tot(self, *args, **kwargs):
        e_tot, e_ot = super().energy_tot(*args, **kwargs)
        return e_tot.real, e_ot

    def dump_chk(self, *args, **kwargs):
        logger.warn(self, "dump_chk is not supported for k-MC-PDFT")
        pass

    def get_energy_decomposition(self, *args, **kwargs):
        raise NotImplementedError("Energy decomposition is not implemented for k-MC-PDFT")

    def update_from_chk(self, chkfile=None, **kwargs):
        raise NotImplementedError("update_from_chk is not implemented for k-MC-PDFT")


class _kCASPDFT(_MCPDFTCPLX):
    """k-MC-PDFT specialization for one total-momentum kCAS sector."""

    momentum_resolved = True
    _mcwfn_rdm_representation = "bloch"

    make_one_casdm1s = pbc_dms.make_one_casdm1s_kcas
    make_one_casdm2 = pbc_dms.make_one_casdm2_kcas
    energy_dft = energy_dft_kcas


class _kChargedCASPDFT(_kCASPDFT):
    """k-MC-PDFT specialization for charged KCASCI momentum sectors."""

    make_one_casdm1s = make_one_casdm1s_charged_kcas
    make_one_casdm2 = make_one_casdm2_charged_kcas
    energy_tot = energy_tot_charged_kcas

    def compute_pdft_energy_(self, mo_coeff=None, ci=None, ot=None,
                             otxc=None, grids_level=None, grids_attr=None,
                             dump_chk=False, verbose=None, target_k=None,
                             **kwargs):
        """Evaluate MC-PDFT independently for each charged k sector."""
        del kwargs
        if dump_chk:
            raise ValueError("dump_chk is not supported for k-MC-PDFT")
        if mo_coeff is not None:
            self.mo_coeff = mo_coeff
        if ot is not None:
            self.otfnal = ot
        if otxc is not None:
            self.otxc = otxc
        if grids_attr is None:
            grids_attr = {}
        if grids_level is not None:
            grids_attr["level"] = grids_level
        if grids_attr:
            self.grids.__dict__.update(grids_attr)
        if verbose is None:
            verbose = self.verbose
        self.verbose = self.otfnal.verbose = verbose

        if target_k is None:
            target_k = self.target_k
        if target_k is None:
            results = list(self.charged_results)
        else:
            results = [_select_charged_kcas_result(
                self, target_k=target_k,
            )]
        if not results:
            raise ValueError("No charged KCASCI results are available")
        if len(results) > 1 and ci is not None and not isinstance(ci, dict):
            raise ValueError(
                "ci for multiple charged sectors must be a dict keyed by "
                "target_k",
            )

        nroots = int(getattr(self.fcisolver, "nroots", 1))
        pdft_results = []
        for result in results:
            sector = int(result["target_k"]) % int(self.nkpts)
            if isinstance(ci, dict):
                sector_ci = ci.get(sector)
            elif ci is None:
                sector_ci = result.get("ci")
            else:
                sector_ci = ci
            epdft = [
                self.energy_tot(
                    mo_coeff=self.mo_coeff, ci=sector_ci, state=state,
                    target_k=sector, verbose=verbose,
                )
                for state in range(nroots)
            ]
            e_tot = [energy for energy, _ in epdft]
            e_ot = [energy for _, energy in epdft]
            if nroots == 1:
                e_tot = e_tot[0]
                e_ot = e_ot[0]
            pdft_results.append({
                "target_k": sector,
                "charge": result.get("charge", self.charge),
                "nkpts": result.get("nkpts", self.nkpts),
                "e_mcscf": result.get("e_tot"),
                "e_tot": e_tot,
                "e_ot": e_ot,
            })

        self.charged_pdft_results = pdft_results
        if len(pdft_results) == 1:
            self.e_tot = pdft_results[0]["e_tot"]
            self.e_ot = pdft_results[0]["e_ot"]
        else:
            self.e_tot = np.asarray([
                result["e_tot"] for result in pdft_results
            ])
            self.e_ot = np.asarray([
                result["e_ot"] for result in pdft_results
            ])
        return self.e_tot, self.e_ot, self.charged_pdft_results

    def band_energies(self, reference_energy, root=None, kpts=None,
                      per_cell=False, reference_target_k=None):
        """Return quasiparticle energies from charged MC-PDFT results."""
        from mrh.my_pyscf.pbc.mcscf import kcasci

        if not self.charged_pdft_results:
            raise ValueError("No charged KCASCI-PDFT results are available")
        if kpts is None:
            kpts = getattr(self._scf, "kpts", None)
        return kcasci.compute_band_energies(
            self.charged_pdft_results,
            reference_energy,
            charge=self.charge,
            root=root,
            kpts=kpts,
            nkpts=self.nkpts,
            per_cell=per_cell,
            reference_target_k=reference_target_k,
            kmom=kcasci._get_kmom_for_kcasci(self),
            cell=self.cell,
            kconserv=getattr(self, "kconserv", None),
        )

    get_band_energy = band_energies
    band_energy = band_energies


def _get_mcpdft_child_class(kmc, ot, pdft_base, **kwargs):
    mc_doc = (kmc.__class__.__doc__ or 'No docstring for MC-SCF parent method')

    class PDFT(pdft_base, kmc.__class__):
        __doc__ = mc_doc + '\n\n' + pdft_base.__doc__
        _mc_class = kmc.__class__

        # MC-PDFT object requires mol object in ot.reset functions
        _mc_class.mol = kmc._scf.cell.to_mol()
        
        def compute_pdft_energy_(self, mo_coeff=None, ci=None, ot=None, otxc=None,
                                 grids_level=None, grids_attr=None, dump_chk=False, **kwargs):
            # Some sanity checks:
            if ot is not None:
                assert isinstance(ot, otfnalperiodic_kpts)
                cell_kpts_info = [getattr(ot, 'kmesh', None), 
                                  getattr(ot, 'kpts', None), 
                                  getattr(ot, 'cell', None)]
                assert all(value is not None for value in cell_kpts_info), \
                    "The kmesh and kpts attributes should be set in the otfnal object"
            assert dump_chk is False, "dump_chk is not supported for k-MC-PDFT"
            return pdft_base.compute_pdft_energy_(self, mo_coeff=mo_coeff, ci=ci, ot=ot, otxc=otxc,
                    grids_level=grids_level, grids_attr=grids_attr, dump_chk=False, **kwargs)
     
    pdft = PDFT(kmc._scf, kmc.ncas, kmc.nelecas, my_ot=ot, **kwargs)
    _keys = pdft._keys.copy()
    pdft.__dict__.update(kmc.__dict__)
    pdft._keys = pdft._keys.union(_keys)
    pdft._keys.add("momentum_resolved")
    return pdft


def get_mcpdft_child_class(kmc, ot, **kwargs):
    """Wrap a conventional periodic CAS object with k-MC-PDFT methods."""
    return _get_mcpdft_child_class(kmc, ot, _MCPDFTCPLX, **kwargs)


def get_kcas_mcpdft_child_class(kmc, ot, **kwargs):
    """Wrap a momentum-resolved kCASCI object with k-MC-PDFT methods."""
    pdft = _get_mcpdft_child_class(kmc, ot, _kCASPDFT, **kwargs)
    if getattr(kmc, "converged", False):
        pdft.e_mcscf = kmc.e_tot
    return pdft


def get_charged_kcas_mcpdft_child_class(kmc, ot, **kwargs):
    """Wrap charged KCASCI results with sector-aware k-MC-PDFT methods."""
    pdft = _get_mcpdft_child_class(kmc, ot, _kChargedCASPDFT, **kwargs)
    pdft._keys.add("charged_pdft_results")
    pdft.charged_pdft_results = []
    if getattr(kmc, "converged", False):
        pdft.e_mcscf = kmc.e_tot
    return pdft
