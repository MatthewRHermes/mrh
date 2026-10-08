"""Unpack k-LASSCF into synchronous molecular LAS on the same Hamiltonian.

The molecular AO basis is a Born-von Karman supercell. Its integrals are
Fourier transforms of the original periodic GDF integrals, not recomputed supercell
integrals. Nuclear and electronic energies are extensive: divide molecular
energies by the number of k points to compare with kLAS energies per cell.

GDF reads the original three-center HDF5 data through its ao2mo API.
No DF file is deleted or closed by this utility. The returned four-center 
AO ERIs are materialized in memory; their
storage grows as (nk * nao)**4. Orbitals and CI must admit a
real Wannier gauge because the synchronous LAS solver uses real orbitals.
"""

import copy
import os
from typing import NamedTuple

import h5py
import numpy as np

from pyscf import ao2mo, scf as molecular_scf
from pyscf.pbc.lib.kpts_helper import get_kconserv
from pyscf.pbc.tools import k2gamma

from mrh.my_pyscf.mcscf.lasscf_sync_o0 import LASSCF
from mrh.my_pyscf.mcscf.lasci import get_space_info
from mrh.my_pyscf.pbc.util.wannier import get_wannier_orbs


class MolecularHamiltonian(NamedTuple):
    """Real supercell AO integrals; eri has eightfold packed symmetry."""
    hcore: np.ndarray
    eri: np.ndarray
    overlap: np.ndarray
    energy_nuc: float


class UnpackedKLAS(NamedTuple):
    """Transfer result, also unpackable as mo_coeff, hamiltonian, ci, las."""
    mo_coeff: np.ndarray
    hamiltonian: MolecularHamiltonian
    ci: list
    las: object


def require_real(value, label, tol=1e-8):
    """Reject a complex gauge; report small optimizer drift when allowed."""
    value = np.asarray(value)
    error = np.max(np.abs(value.imag)) if value.size else 0.0
    if error > tol:
        raise ValueError(f"{label} has imaginary components of {error:.3e}; "
                         "the real synchronous solver requires a real gauge.")
    if error > 1e-8:
        print(f'{label}: removing imaginary drift of {error:.3e}; '
              'orthonormality and transferred energy will be checked.', flush=True)
    return np.array(value.real, copy=True)


def unpack_orbitals(klas, real_tol):
    """Order columns as all core, then cell-by-cell active, then virtual."""
    coeff = np.asarray(klas.mo_coeff)
    nk, nao, nmo = coeff.shape
    edges = (0, klas.ncore, klas.ncore + klas.ncas, nmo)
    blocks = []
    for start, stop in zip(edges[:-1], edges[1:]):
        if start == stop:
            continue
        wannier = get_wannier_orbs(
            klas._scf, klas.kmesh, coeff[:, :, start:stop])[0]
        blocks.append(wannier.reshape(nk * nao, nk * (stop - start)))
    return require_real(np.hstack(blocks), "Unpacked orbitals", real_tol)


def molecular_hamiltonian(kmf, kmesh, df_file=None):
    """Fourier transform original GDF integrals; retain their approximations.

    Bloch AO ERIs carry 1/nk normalization. Transform each index with the
    normalized Fourier matrix to obtain extensive supercell AO integrals.
    """
    # PySCF GDF streams three-center blocks from its existing HDF5 store.
    # Normalize an open h5py.File to a filename on a private backend copy:
    # PySCF's _load3c opens that filename internally in read mode.
    
    backend = kmf.with_df
    cderi = backend._cderi if df_file is None else df_file
    if isinstance(cderi, h5py.File):
        cderi = cderi.filename
    if isinstance(cderi, os.PathLike):
        cderi = os.fspath(cderi)
    if df_file is not None or cderi is not backend._cderi:
        backend = copy.copy(backend)
        backend._cderi = cderi
    nk = len(kmf.kpts)
    nao = kmf.cell.nao_nr()
    n = nk * nao
    scell, phase = k2gamma.get_phase(kmf.cell, kmf.kpts, kmesh)
    transform = np.kron(phase, np.eye(nao))
    eri_k = np.zeros((n, n, n, n), dtype=np.complex128)
    identity = np.eye(nao, dtype=np.complex128)
    kconserv = get_kconserv(kmf.cell, kmf.kpts)
    for k1, k2, k3 in np.ndindex(nk, nk, nk):
        k4 = kconserv[k1, k2, k3]
        indices = (k1, k2, k3, k4)
        slices = tuple(slice(k * nao, (k + 1) * nao) for k in indices)
        eri_k[slices] = backend.ao2mo(
            [identity] * 4, kmf.kpts[list(indices)], compact=False
        ).reshape((nao,) * 4) / nk
    eri = np.einsum('ap,bq,cr,ds,pqrs->abcd',
                    transform, transform.conj(), transform,
                    transform.conj(), eri_k, optimize=True)
    eri = require_real(eri, "Supercell ERIs")
    hcore = k2gamma.to_supercell_ao_integrals(
        kmf.cell, kmf.kpts, kmf.get_hcore(), kmesh, force_real=False)
    overlap = k2gamma.to_supercell_ao_integrals(
        kmf.cell, kmf.kpts, kmf.get_ovlp(), kmesh, force_real=False)
    hcore = require_real(hcore, "Supercell hcore")
    overlap = require_real(overlap, "Supercell overlap")
    mol = scell.to_mol()
    # The molecule supplies AO bookkeeping, but every Hamiltonian term is PBC.
    nuclear = nk * kmf.cell.energy_nuc()
    mf = molecular_scf.RHF(mol)
    mf.energy_nuc = lambda *args: nuclear
    mf.get_hcore = lambda *args: hcore
    mf.get_ovlp = lambda *args: overlap
    mf._eri = ao2mo.restore(8, eri, n)
    return mf


def unpack_klas(klas, *, real_tol=1e-4, energy_tol=1e-8, df_file=None):
    """Return molecular orbitals, Hamiltonian, CI and initialized sync LASSCF.

    Parameters
    ----------
    klas
        Optimized k-LASSCF object with mo_coeff, ci, kmesh and a GDF mean field.
    real_tol : float
        Maximum imaginary drift removed from Wannier orbitals and CI. The
        transferred energy and AO-metric orthonormality are checked afterward.
    energy_tol : float
        Absolute tolerance in Ha per primitive cell for energy preservation.
    df_file : str, pathlib.Path or h5py.File, optional
        Existing PySCF periodic GDF three-center file. By default reuse
        klas._scf.with_df._cderi. An open file remains owned by the caller;
        its filename is opened independently for reading by PySCF. Its k-point
        pairs and auxiliary basis must match the original calculation.

    Returns
    -------
    UnpackedKLAS
        mo_coeff ordered as all core, cell-major fragment active, then virtual;
        Hamiltonian with AO hcore, eightfold-packed eri, overlap and energy_nuc;
        copied CI in the same fragment/root order; and a synchronous LAS object
        initialized with those arrays. No LAS optimization is performed here.

    Example
    -------
    >>> mo, ham, ci, las = unpack_klas(klas)
    >>> las.kernel(mo_coeff=mo, ci0=ci)
    >>> energy_per_cell = las.e_tot / len(klas.kpts)
    """
    nk = len(klas._scf.kpts)
    mo = unpack_orbitals(klas, real_tol)
    mf = molecular_hamiltonian(klas._scf, klas.kmesh, df_file=df_file)
    overlap = mf.get_ovlp()
    np.testing.assert_allclose(mo.T @ overlap @ mo, np.eye(mo.shape[1]),
                               atol=1e-8, rtol=0)
    mf.mo_coeff = mo
    mf.mo_occ = np.zeros(mo.shape[1])
    mf.mo_occ[:nk * klas.ncore] = 2
    charges, spins, smults, wfnsyms = get_space_info(klas)
    las = LASSCF(mf, tuple(klas.ncas_sub), tuple(map(tuple, klas.nelecas_sub)),
                 ncore=nk * klas.ncore, spin_sub=smults[0].tolist())
    if klas.nroots > 1 or np.any(charges) or np.any(spins != spins[0]):
        las.state_average_(weights=klas.weights, charges=charges, spins=spins,
                           smults=smults, wfnsyms=wfnsyms)
    las.conv_tol_grad = klas.conv_tol_grad
    las.max_cycle_macro = klas.max_cycle_macro
    ci = [[require_real(root, 'Fragment CI', real_tol) for root in fragment]
          for fragment in klas.ci]
    las.mo_coeff, las.ci = mo, ci
    energy = float(las.energy_elec(mo_coeff=mo, ci=ci) + las.energy_nuc())
    reference = float(np.asarray(klas.e_tot).real)
    if abs(energy / nk - reference) > energy_tol:
        raise ValueError(f'Unpacking changed E/cell by {energy / nk - reference:.3e} Ha')
    # e_tot is extensive here, even though the input kLAS energy is per cell.
    las.e_tot = energy
    ham = MolecularHamiltonian(mf.get_hcore(), mf._eri, overlap, float(mf.energy_nuc()))
    return UnpackedKLAS(mo, ham, ci, las)
