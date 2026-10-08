"""Periodic MC-PDFT wrappers and energy methods for kLAS wave functions.

This module specializes the existing k-MC-PDFT implementation for density
matrices produced by periodic LASCI and LASSCF.  kLAS RDMs and active-space
integrals use a Wannier basis, so the corresponding kLAS mo_phase is
selected through the shared RDM preparation machinery.
"""

import copy
from mrh.my_pyscf.pbc.mcpdft import klaspdft_helper
from mrh.my_pyscf.pbc.mcpdft.kmcpdft import _kMCPDFT


class _kLASPDFT(_kMCPDFT):
    """k-MC-PDFT specialization for periodic LASCI and LASSCF states."""

    make_one_casdm1s = klaspdft_helper.make_one_casdm1s_klas
    make_one_casdm2 = klaspdft_helper.make_one_casdm2_klas
    _mcwfn_rdm_representation = "klas"


def get_klas_mcpdft_child_class(klas, ot, grids_level=None,
                                grids_attr=None, **kwargs):
    """Wrap a completed kLAS object with fixed-wavefunction PDFT methods.

    Unlike the conventional MC-PDFT child-class factory, this routine does not
    call the parent MC-SCF constructor.  Reconstructing a kLAS object would
    rebuild its fragment solvers and requires mesh-specific arguments.  A new
    dynamic child instead receives a shallow copy of the completed kLAS state,
    after which only the PDFT grids and bookkeeping fields are initialized.

    Parameters
    ----------
    klas : object
        Completed periodic LASCI or LASSCF calculation.
    ot : str or on-top functional
        Translated on-top functional specification.
    grids_level : int, optional
        Periodic integration-grid level.
    grids_attr : dict, optional
        Additional periodic integration-grid attributes.
    **kwargs
        Attribute overrides applied to the PDFT wrapper.

    Returns
    -------
    pdft : object
        A new kLAS-PDFT object.  Calling ``kernel`` evaluates PDFT without
        rerunning the underlying LASCI or LASSCF optimization.
    """
    mc_class = klas.__class__
    mc_doc = mc_class.__doc__ or "No docstring for kLAS parent method"

    class PDFT(_kLASPDFT, mc_class):
        __doc__ = mc_doc + "\n\n" + _kLASPDFT.__doc__
        _mc_class = mc_class

        @property
        def mol(self):
            """Molecular view of the primitive cell required by MC-PDFT."""
            return self._scf.cell.to_mol()

        def optimize_mcscf_(self, mo_coeff=None, ci0=None, **unused):
            """Reuse the fixed kLAS wave function without reoptimizing it."""
            if mo_coeff is not None:
                self.mo_coeff = mo_coeff
            if ci0 is not None:
                self.ci = ci0
            self.e_mcscf = self._klas_reference_energy
            return self

        def compute_pdft_energy_(
                self, mo_coeff=None, ci=None, ot=None, otxc=None,
                grids_level=None, grids_attr=None, dump_chk=False, **kw):
            """Evaluate and store fixed-wavefunction kLAS-PDFT energies."""
            if dump_chk:
                raise ValueError("dump_chk is not supported for kLAS-PDFT")
            return _kLASPDFT.compute_pdft_energy_(
                self,
                mo_coeff=mo_coeff,
                ci=ci,
                ot=ot,
                otxc=otxc,
                grids_level=grids_level,
                grids_attr=grids_attr,
                dump_chk=False,
                **kw,
            )

    pdft = PDFT.__new__(PDFT)
    pdft.__dict__ = copy.copy(klas.__dict__)
    pdft._keys = set(getattr(klas, "_keys", ()))
    pdft._keys.update({
        "e_ot", "e_mcscf", "e_states", "otfnal", "grids",
        "max_cycle_fp", "conv_tol_ci_fp", "mcscf_kernel",
        "_in_mcscf_env", "_klas_reference_energy",
    })
    pdft.mcscf_kernel = mc_class.kernel
    pdft._in_mcscf_env = False
    pdft._klas_reference_energy = copy.copy(getattr(klas, "e_tot", None))
    pdft.e_mcscf = copy.copy(pdft._klas_reference_energy)
    pdft.e_ot = None

    if grids_attr is None:
        grids_attr = {}
    else:
        grids_attr = dict(grids_attr)
    if grids_level is not None:
        grids_attr["level"] = grids_level
    pdft._init_ot_grids(ot, grids_attr=grids_attr)
    for key, value in kwargs.items():
        if not hasattr(pdft, key):
            raise TypeError(f"Unknown kLAS-PDFT option: {key}")
        setattr(pdft, key, value)
    return pdft
