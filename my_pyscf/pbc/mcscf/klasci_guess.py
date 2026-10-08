# !/usr/bin/env python

import numpy as np

from mrh.my_pyscf.pbc.util.orth import meta_lowdin_orbitals
from mrh.my_pyscf.pbc.util.orth import orthogonality_check


# Author: Bhavnesh Jangid

'''
In this file: I have added localize_init_guess function which is used to localize the 
active space orbitals in periodic systems unit cell by unit-cell. This fn assumes that
the active space is defined for the unit-cell and the same active space is repeated in
all unit-cells.

Tried my best to keep the code similar to molecular fns.
'''


def _interpret_fragment_orbitals(cell, frag_atoms, frags_by_AOs=False):
    '''
    Convert one unit-cell fragment specification to AO indices.
    The input frag_atoms can be specified in two ways:
        1. list of atom indices (integers) or AO indices (if frags_by_AOs=True)
           This code will convert the atom indices to AO indices using the cell's 
           offset_ao_by_atom() method.
        2. list of AO-label strings (e.g., ['C 2s', 'H 1s']).
    
    args:
        cell: 
            The unit-cell object.
        frag_atoms: list of atom indices (integers) or 
                    AO-label strings (e.g., ['C 2s', 'H 1s']).
                    The fragment specification.
        frags_by_AOs: boolean, optional, (default: True)
            If True, frag_atoms is interpreted as AO indices.
    returns:
        ao_idx: np.array 
            AO indices corresponding to the specified fragment
    '''

    # Sanity check for frag_atoms
    if frag_atoms is None: return None
    if isinstance(frag_atoms, (str, np.character, int, np.integer)):
        frag_atoms = [frag_atoms]
    elif len(frag_atoms) == 1 and isinstance(
            frag_atoms[0], (list, tuple, np.ndarray)):
        frag_atoms = frag_atoms[0]

    # Check if frag_atoms is a list of integers (atom indices or AO indices)
    is_int = all(isinstance(i, (int, np.integer)) for i in frag_atoms)
    is_str = all(isinstance(i, (str, np.character)) for i in frag_atoms)

    if is_int:
        if frags_by_AOs: 
            ao_idx = np.asarray(frag_atoms, dtype=int)
        else:
            ao_offset = cell.offset_ao_by_atom()
            ao_idx = np.asarray([ao 
                                 for atom in frag_atoms 
                                 for ao in range(ao_offset[atom, 2], 
                                                 ao_offset[atom, 3])], dtype=int)
    elif is_str:
        ao_idx = np.asarray(sorted(set(ao 
                                       for label in frag_atoms 
                                       for ao in cell.search_ao_label(label))), dtype=int)
    else:
        msg = ("Fragment must be specified using only atom/AO indices or only "
               "AO-label strings")
        raise TypeError(msg)

    # Final sanity checks for ao_idx
    if ao_idx.size == 0:
        msg = "The fragment specification does not select any AOs"
        raise ValueError(msg)
    if np.unique(ao_idx).size != ao_idx.size:
        msg = "The fragment specification contains duplicate AOs"
        raise ValueError(msg)
    if np.any(ao_idx < 0) or np.any(ao_idx >= cell.nao_nr()):
        msg = "Fragment AO index is outside the unit-cell AO space"
        raise IndexError(msg)
    
    return ao_idx

_interpret_unit_cell_orbitals = _interpret_fragment_orbitals


def _active_occupations(mo_occ, nkpts, nmo, ncore, ncas):
    if mo_occ is None:
        return np.ones((nkpts, ncas), dtype=int)
    mo_occ = np.asarray(mo_occ)
    if mo_occ.shape == (nmo,):
        mo_occ = np.broadcast_to(mo_occ, (nkpts, nmo))
    if mo_occ.shape != (nkpts, nmo):
        msg = (f"mo_occ must have shape ({nmo},) or ({nkpts}, {nmo}); "
               f"got {mo_occ.shape}")
        raise ValueError(msg)
    return list(mo_occ[:, ncore:ncore+ncas])


def localize_init_guess(klas, frag_atoms=None, mo_coeff=None, spin=None, 
                        lo_coeff=None,fock=None, mo_occ=None, freeze_cas_spaces=True,
                        frags_by_AOs=False, smults_f=None, nelec_f=None, 
                        return_umat=False, return_svals=False, sval_thresh=1e-8,
                        align_phases=True, stabilize_virtuals=None,
                        align_core_phases=None, align_active_phases=None,
                        align_virtual_phases=None):
    '''
    Localize one active space per unit cell.Some args are not used in this function
    but are kept for API compatibility with molecular LAS localization. Those variables
    are ``spin``, ``smults_f`` and ``nelec_f``.  They are not required
    when there is one translationally repeated active space per unit cell.

    Now coming back to this function, it is the periodic, active-space-preserving 
    analogue of molecular ``localize_init_guess``.  At every k-point, an overlap
    SVD selects the combinations of the complete active-band manifold with the largest
    projection onto the same unit-cell orbital space.
    After Fock canonicalization, each active band is phase-aligned to the same
    fragment local orbital at every k-point. This removes arbitrary eigenvector
    phases before constructing Wannier orbitals.
    
    By default ``align_phases`` enables core phase alignment, active phase
    alignment, virtual phase alignment, and virtual stabilization together.
    Each operation can be overridden individually. The core subspace and orbital ordering are
    preserved; only individual core phases change. Core and virtual phase alignment
    and virtual stabilization preserve the fixed-CI LAS density
    and energy.
    Active phase alignment is an initialization convention, not generally
    an energy-neutral change of an existing LAS wavefunction.
    A k-dependent band phase can mix Wannier orbitals between cell fragments.
    A fragment-local phase preserves energy only when CI is transformed
    consistently with the active orbitals.

    args:
        klas: instance of mrh.my_pyscf.pbc.mcscf.klasci 
              periodic LASCI object
        frag_atoms: one unit-cell fragment, specified by atom indices, 
                    AO-label strings, or AO indices when ``frags_by_AOs=True``

    kwargs:
        mo_coeff: np.ndarray or the list of np.arrays, Shape: (nkpts, nao, nmo)
            molecular orbitals for each k-point. If not provided, the orbitals 
            from klas._scf.mo_coeff are used.
        lo_coeff: np.ndarray or the list of np.arrays, Shape: (nkpts, nao, nmo)
                  orthonormal local orbitals.  meta-Lowdin AOs are used by default.
        fock: np.ndarray or the list of np.arrays, Shape: (nkpts, nao, nao)
            AO-basis Fock matrices used to canonicalize the localized active
            orbitals within each set of identical occupations.  The mean-field
            Fock matrices are used by default.
        mo_occ: np.array or the list of np.arrays, Shape: (nkpts, nmo)
            optional occupation labels.  Only active orbitals with the same
            occupation are mixed, as in the molecular implementation.  When
            omitted, every active orbital is assigned occupancy 1 so that the
            full localized active space is ordered by increasing Fock energy.
        freeze_cas_spaces: bool, optional, (default: True)
            If True, the active space is preserved and only the gauge of the
            active orbitals is changed.  If False, the active space is allowed to
            change, as in the molecular implementation.  But currently, this is not
            implemented for periodic systems and with the LAS framework.
        frags_by_AOs: see above.
        align_phases: bool, optional, (default: True)
            Enable core, active and virtual phase alignment, and virtual
            stabilization together. Individual options override this value
            when explicitly set; None inherits this value.
        align_active_phases: bool or None, optional, (default: None)
            Make the overlap of each active band with a fixed fragment local
            orbital real and positive at every k-point. The reference is chosen
            to maximize its smallest overlap across the k-points. This fixes
            phases only; it does not mix bands or optimize Wannier spreads.
            These phases can vary with k and therefore change the Wannier
            fragment spaces and the LAS energy at fixed CI. This option fixes
            the initial Wannier gauge; it is not a visualization operation.
            If no reference has nonzero overlap everywhere, a ValueError is
            raised. Supply suitable lo_coeff or disable phase alignment to use
            a separately constructed Wannier gauge. None inherits
            ``align_phases``.
        align_core_phases: bool or None, optional, (default: None)
            Make the largest AO coefficient of each core orbital real and
            positive independently at each k-point. Near-ties are resolved
            by fixed AO ordering. This fixes individual orbital phases only;
            it does not rotate or reorder the core orbitals. The core density
            and fixed-CI LAS energy are preserved. This option is independent
            of ``align_active_phases``. None inherits ``align_phases``.
        align_virtual_phases: bool or None, optional, (default: None)
            Make the largest AO coefficient of each virtual orbital real and
            positive independently at each k-point. Near-ties use fixed AO
            ordering. Only unit-modulus phases are applied, preserving orbital
            orthonormality, the virtual subspace and fixed-CI LAS energy.
            This runs after optional virtual stabilization and does not mix
            or reorder virtual orbitals. Use stabilize_virtuals=False for
            phase alignment alone. None inherits ``align_phases``.
        stabilize_virtuals: bool or None, optional, (default: None)
            Choose a reproducible basis within the supplied virtual space by
            projecting meta-Lowdin AOs and selecting the strongest remaining
            projection at each step. Fixed AO order resolves near-ties. The
            orthonormalization constructs a unitary rotation inside the input
            virtual space, preserving orthogonality to core and active orbitals. This removes arbitrary virtual-space rotations, including
            orbital phases, while preserving the core and active orbitals.
            The virtual orbitals are not ordered by Fock energy. None
            inherits ``align_phases``.

    returns:
        return_umat: bool, optional, (default: False)
            If True, also return the full k-point MO rotation matrices.
            Umat[k] is the unitary matrix that transforms the input orbitals at k-point k
            to the localized orbitals at k-point k.
        return_svals: bool, optional, (default: False)
            If True, also return the fragment-overlap singular values.
        sval_thresh: float, optional, (default: 1e-8)
            Minimum accepted singular value.  If any singular value is below this
            threshold, an error is raised. 
    '''
    # making sure that the unused args are not used in this function
    del spin, smults_f, nelec_f

    if align_active_phases is None:
        align_active_phases = align_phases
    if align_core_phases is None:
        align_core_phases = align_phases
    if stabilize_virtuals is None:
        stabilize_virtuals = align_phases
    if align_virtual_phases is None:
        align_virtual_phases = align_phases

    if not freeze_cas_spaces:
        msg = ("Periodic active-band localization always preserves the"
               "active space; freeze_cas_spaces must be True")
        raise NotImplementedError(msg)
    
    if mo_coeff is None: mo_coeff = klas.mo_coeff
    mo_coeff = np.asarray(mo_coeff)

    kmf = klas._scf
    cell = kmf.cell
    kpts = kmf.kpts
    nkpts = len(kpts)
    ncore = klas.ncore
    ncas = klas.ncas
    nocc = ncore + ncas

    # Some sanity checks for mo_coeff
    if mo_coeff.ndim != 3:
        msg = f"mo_coeff must have shape (nkpts, nao, nmo); got {mo_coeff.shape}"
        raise ValueError(msg)
    
    if mo_coeff.shape[0] != nkpts:
        msg = (f"mo_coeff contains {mo_coeff.shape[0]} k-points; "
               f"expected {nkpts}")
        raise ValueError(msg)
    
    nao, nmo = mo_coeff.shape[1:]
    if nao != cell.nao_nr():
        msg = (f"mo_coeff AO dimension is {nao}; expected {cell.nao_nr()}")
        raise ValueError(msg)

    if fock is None:
        fock = kmf.get_fock()
    fock = np.asarray(fock)
    if fock.shape != (nkpts, nao, nao):
        msg = (f"fock must have shape ({nkpts}, {nao}, {nao}); "
               f"got {fock.shape}")
        raise ValueError(msg)

    # Get the ovlp from the kmf object only, in case of the pseudo-potential
    # directly using the cell.pbc_intro might be dangerous.
    ovlp = np.asarray(kmf.get_ovlp(kpts=kpts))

    # Localize the orbitals
    if lo_coeff is None: lo_coeff = meta_lowdin_orbitals(cell, ovlp)
    lo_coeff = np.asarray(lo_coeff)

    if lo_coeff.ndim != 3 or lo_coeff.shape[:2] != (nkpts, nao):
        msg = (f"lo_coeff must have shape (nkpts, nao, nlo); got {lo_coeff.shape}")
        raise ValueError(msg)

    frag_orbs = _interpret_unit_cell_orbitals(cell, frag_atoms, 
                                              frags_by_AOs=frags_by_AOs
    )
    if frag_orbs is None:
        frag_orbs = np.arange(lo_coeff.shape[2])
    if np.any(frag_orbs >= lo_coeff.shape[2]):
        msg = ("Fragment AO index is outside the local-orbital space")
        raise IndexError(msg)
    
    if frag_orbs.size < ncas:
        msg = (f"Cannot localize {ncas} active bands using only "
               f"{frag_orbs.size} local orbitals")
        raise ValueError(msg)

    # Collect the active-band occupations for each k-point.  As in the
    # molecular localizer, the default assigns occupancy 1 to every active
    # orbital so that the complete fragment is subsequently canonicalized.
    active_occ = _active_occupations(mo_occ, nkpts, nmo, ncore, ncas)

    result_dtype = np.result_type(mo_coeff.dtype, lo_coeff.dtype, fock.dtype)
    mo_out = np.array(mo_coeff, dtype=result_dtype, copy=True)
    umat = np.zeros((nkpts, nmo, nmo), dtype=mo_out.dtype)
    svals_out = []
    fragment_overlaps = []

    for k in range(nkpts):
        c_act = mo_coeff[k, :, ncore:nocc]
        ortho_lo = lo_coeff[k][:, frag_orbs]
        _, svals, c_local, localized_occ = klas._svd(
            ortho_lo, c_act, s=ovlp[k], mo_occ=active_occ[k]
        )
        # _svd sorts all occupation sectors together by their singular
        # values. Restore the conventional occupied-to-virtual ordering
        # while applying the same permutation to the orbitals and their
        # singular values.
        occ_order = np.argsort(-localized_occ, kind='stable')
        c_local = c_local[:, occ_order]
        localized_occ = localized_occ[occ_order]
        svals = svals[occ_order]

        if len(svals) < ncas:
            msg = "Note sufficient AOs were selected to localize the active space."
            raise ValueError(msg)

        # Sanity check, if the active space have very minimal overlap with the target
        # fragment orbitals.
        svals = np.asarray(svals[:ncas])
        if np.min(svals) < sval_thresh:
            msg = (f"k-point {k}: fragment orbitals localization is poorer, "
                   f"singular values = {svals}")
            raise ValueError(msg)
        
        c_local = np.asarray(c_local[:, :ncas], dtype=result_dtype)
        localized_occ = np.asarray(localized_occ[:ncas])

        # Match molecular localize_init_guess: within every set of identical
        # occupations, diagonalize the localized Fock block and order the
        # resulting orbitals from lowest to highest energy.
        fock_local = c_local.conj().T @ fock[k] @ c_local
        for occupation in np.unique(localized_occ):
            idx = np.flatnonzero(localized_occ == occupation)
            fock_block = fock_local[np.ix_(idx, idx)]
            energy, rotation = np.linalg.eigh(fock_block)
            energy_order = np.argsort(energy)
            c_local[:, idx] = c_local[:, idx] @ rotation[:, energy_order]

        if align_active_phases:
            fragment_overlaps.append(ortho_lo.conj().T @ ovlp[k] @ c_local)
        mo_out[k, :, ncore:nocc] = c_local

        umat[k] = np.eye(nmo, dtype=mo_out.dtype)
        umat[k, ncore:nocc, ncore:nocc] = (c_act.conj().T @ ovlp[k] @ c_local)
        svals_out.append(svals)

    if align_active_phases and ncas:
        overlaps = np.asarray(fragment_overlaps)
        # Use one reference per band across the entire mesh. Choosing the
        # largest projection independently at each k can introduce sign jumps.
        scores = np.min(np.abs(overlaps), axis=0)
        best = np.max(scores, axis=0)
        # Resolve numerical ties by fragment order, independently of the
        # arbitrary phases or rotations of the input active orbitals.
        references = np.argmax(scores >= best[None, :] * (1 - 1e-10), axis=0)
        anchors = overlaps[:, references, np.arange(ncas)]
        if np.any(np.abs(anchors) < 1e-12):
            raise ValueError(
                "Cannot align active-band phases: no fragment local orbital "
                "has nonzero overlap at every k-point. Supply suitable "
                "lo_coeff or use align_active_phases=False with a separate Wannier gauge"
            )
        phases = anchors.conj() / np.abs(anchors)
        mo_out[:, :, ncore:nocc] *= phases[:, None, :]
        umat[:, :, ncore:nocc] *= phases[:, None, :]

    if align_core_phases and ncore:
        core = mo_out[:, :, :ncore]
        magnitude = np.abs(core)
        largest = np.max(magnitude, axis=1)
        # A fixed AO order resolves near-ties without depending on input phases.
        references = np.argmax(
            magnitude >= largest[:, None, :] * (1 - 1e-10), axis=1,
        )
        anchors = np.take_along_axis(
            core, references[:, None, :], axis=1,
        )[:, 0, :]
        if np.any(np.abs(anchors) == 0):
            raise ValueError("Cannot align the phase of a zero core orbital")
        phases = anchors.conj() / np.abs(anchors)
        core *= phases[:, None, :]
        umat[:, :, :ncore] *= phases[:, None, :]

    nvir = nmo - nocc
    if stabilize_virtuals and nvir:
        virtual_lo = meta_lowdin_orbitals(cell, ovlp)
        for k in range(nkpts):
            lo_k = virtual_lo[k]
            c_virtual = mo_coeff[k, :, nocc:]
            coordinates = lo_k.conj().T @ ovlp[k] @ c_virtual
            # Build a unitary rotation in the supplied virtual coordinates.
            # Working in AO space can amplify roundoff outside the virtual
            # span when an almost dependent projected AO is normalized.
            residual = coordinates.conj().T.copy()
            rotation = np.empty((nvir, nvir), dtype=mo_out.dtype)
            for column in range(nvir):
                norms = np.linalg.norm(residual, axis=0)
                # Pivot on the strongest remaining projection. Fixed AO order
                # resolves numerical ties independently of the input gauge.
                axis = np.flatnonzero(norms >= norms.max() * (1 - 1e-10))[0]
                vector = residual[:, axis].copy()
                previous = rotation[:, :column]
                for _ in range(2):
                    vector -= previous @ (previous.conj().T @ vector)
                vector /= np.linalg.norm(vector)
                rotation[:, column] = vector
                for _ in range(2):
                    residual -= np.outer(vector, vector.conj() @ residual)
            mo_out[k, :, nocc:] = c_virtual @ rotation
            umat[k, nocc:, nocc:] = rotation

    if align_virtual_phases and nvir:
        virtual = mo_out[:, :, nocc:]
        magnitude = np.abs(virtual)
        largest = np.max(magnitude, axis=1)
        references = np.argmax(
            magnitude >= largest[:, None, :] * (1 - 1e-10), axis=1,
        )
        anchors = np.take_along_axis(
            virtual, references[:, None, :], axis=1,
        )[:, 0, :]
        phases = anchors.conj() / np.abs(anchors)
        virtual *= phases[:, None, :]
        umat[:, :, nocc:] *= phases[:, None, :]

    # Check orthogonality of the output orbitals
    orthogonality_check(mo_out, ovlp)
    svals_out = np.asarray(svals_out)

    result = [mo_out]
    if return_umat:
        result.append(umat)
    if return_svals:
        result.append(svals_out)

    if len(result) == 1:
        return result[0]
    else:
        return tuple(result)
