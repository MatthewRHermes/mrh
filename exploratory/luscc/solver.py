from copy import deepcopy
from time import perf_counter

import numpy as np
from scipy.sparse.linalg import aslinearoperator
from pyscf import lib
from pyscf.csf_fci import csf_solver

from mrh.my_pyscf.lassi import LASSI
from mrh.my_pyscf.mcscf.addons import state_average_n_mix, get_h1e_zipped_fcisolver

from .excitations import apply_operator_string_fci
from .spin import exact_spin_basis, residual_gram


class LUSCC(LASSI):
    """Build a selected LUSCC product-state space and solve it with LASSI.

    ``las_or_lsi`` may be a LAS reference or a post-kernel LASSI reference.
    For a LASSI reference, significant product components are selected by
    ``threshold`` or ``top_m`` and every requested excitation is applied to
    each selected component.

    Hamiltonian and S^2 operators are supplied by LASSI. With ``target_spin``,
    the residual Gram matrix defines an exact-spin subspace, solved using
    LASSI's projected Davidson solver. Stage wall times are in ``timings``.
    ``operator_mode='auto'`` uses LASSI's in-memory assembly when its estimated
    memory fits; ``'matrix_free'`` forces operator contractions on every call.

    Unchanged spectator CI arrays and repeated local excitation images are
    shared by default. Treat prepared CI factors as immutable, or request
    ``share_spectator_ci=False`` for independent excitation copies.
    """

    def __init__(self, las_or_lsi, a_idxs, i_idxs, state=0, threshold=0.01,
                 top_m=None, opt=1, smult_si=None, share_spectator_ci=True,
                 target_spin=None, spin_tol=1e-10, lin_dep_tol=1e-10,
                 operator_mode='auto', **kwargs):
        if operator_mode not in ('auto', 'matrix_free'):
            raise ValueError("operator_mode must be 'auto' or 'matrix_free'")
        self.a_idxs = a_idxs
        self.i_idxs = i_idxs
        self._smult_si = smult_si
        self._share_spectator_ci = bool(share_spectator_ci)
        self.target_spin = target_spin
        self.spin_tol = spin_tol
        self.lin_dep_tol = lin_dep_tol
        self.spin_residual_norm = None
        self.spin_residual_eigenvalues = None
        self.timings = {}
        self.operator_mode = operator_mode
        self.exact_spin_operator_mode = None

        if isinstance(las_or_lsi, LASSI):
            self._ref_lsi = las_or_lsi
            self._lsi_state = state
            self._si_threshold = threshold
            self._top_m = top_m
            las = las_or_lsi._las
        else:
            self._ref_lsi = None
            self._lsi_state = 0
            self._si_threshold = threshold
            self._top_m = None
            las = las_or_lsi

        super().__init__(las, opt=opt, **kwargs)

    def _select_sig_indices(self):
        """Return selected product-state indices of the reference wavefunction."""
        if self._ref_lsi is None or self._ref_lsi.si is None:
            return [0]

        si = np.asarray(self._ref_lsi.si[:, self._lsi_state])
        if self._top_m is not None:
            nselect = min(self._top_m, len(si))
            return np.argsort(-np.abs(si))[:nselect].tolist()

        selected = np.where(np.abs(si) > self._si_threshold)[0].tolist()
        if not selected:
            selected = [int(np.argmax(np.abs(si)))]
        return selected

    def getAci(self, a_idx, i_idx, ref_ci=None, ref_nelecas_sub=None, _cache=None):
        """Apply one spin-orbital excitation to fragment CI product states."""
        frag_orbs_start = np.append(0, np.cumsum(self.ncas_sub[:-1]))
        if ref_ci is None:
            ref_ci = [frag_ci[0] for frag_ci in self.ci]
        if ref_nelecas_sub is None:
            ref_nelecas_sub = self.nelecas_sub

        # Excitations replace only active-fragment arrays. Sharing immutable
        # spectator arrays helps MRH identify equivalent product-state factors.
        aci = list(ref_ci) if self._share_spectator_ci else deepcopy(ref_ci)
        frag_ops = [[] for _ in range(self.nfrags)]
        for op_type, indices in (("ann", i_idx), ("cre", a_idx[::-1])):
            for index in indices:
                spatial = index % self.ncas
                spin = index // self.ncas
                ifrag = np.searchsorted(
                    frag_orbs_start, spatial, side="right") - 1
                local_orbital = spatial - frag_orbs_start[ifrag]
                frag_ops[ifrag].append((op_type, local_orbital, spin))

        nelecas_sub = deepcopy(list(ref_nelecas_sub))
        for ifrag, ci in enumerate(aci):
            if not frag_ops[ifrag]:
                continue
            key = (ifrag, id(ref_ci[ifrag]), tuple(ref_nelecas_sub[ifrag]),
                   tuple(frag_ops[ifrag]))
            if _cache is not None and key in _cache:
                ci, nelec = _cache[key]
            else:
                ci, nelec = apply_operator_string_fci(
                    ci, self.ncas_sub[ifrag], ref_nelecas_sub[ifrag],
                    frag_ops[ifrag])
                if ci is not None and np.any(ci):
                    ci = ci / np.linalg.norm(ci)
                if _cache is not None:
                    _cache[key] = ci, nelec
            if ci is None or not np.any(ci):
                return None, None
            aci[ifrag] = ci
            nelecas_sub[ifrag] = nelec
        return aci, nelecas_sub

    def prepare_states_(self):
        """Construct reference, excitation, and de-excitation root spaces."""
        from mrh.my_pyscf.mcscf.lasci import get_space_info
        from mrh.my_pyscf.lassi.citools import get_lroots, get_rootaddr_fragaddr

        selected = self._select_sig_indices()
        max_nroots = len(selected) * (1 + 2 * len(self.a_idxs))
        charges = np.zeros((max_nroots, self.nfrags), dtype=np.int32)
        spins = np.zeros_like(charges)
        smults = np.ones_like(charges)
        wfnsyms = np.zeros_like(charges)

        if self._ref_lsi is not None:
            ref = self._ref_lsi
            space_info = get_space_info(ref)
            nelec_frs = self.get_nelec_frs(ref)
            rootaddr, fragaddr = get_rootaddr_fragaddr(get_lroots(ref.ci))

            def ref_ci_nelec(index):
                iroot = rootaddr[index]
                ci = []
                for ifrag in range(self.nfrags):
                    ci_root = ref.ci[ifrag][iroot]
                    ci.append(ci_root[fragaddr[ifrag, index]]
                              if ci_root.ndim > 2 else ci_root)
                nelec = [tuple(nelec_frs[ifrag, iroot])
                         for ifrag in range(self.nfrags)]
                return ci, nelec, iroot
        else:
            ref = self._las
            space_info = get_space_info(ref)
            nelec_frs = self.get_nelec_frs(ref)
            rootaddr = None

            def ref_ci_nelec(index):
                ci = [self.ci[ifrag][index]
                      for ifrag in range(self.nfrags)]
                nelec = [tuple(nelec_frs[ifrag, index])
                         for ifrag in range(self.nfrags)]
                return ci, nelec, index

        ref_charges, ref_spins, ref_smults, ref_wfnsyms = space_info
        new_ci = [[] for _ in range(self.nfrags)]
        seen_rootspaces = set()
        self.nroots = 0

        # Preserve complete reference root spaces. Their local-root Cartesian
        # products are the original LASSI reference basis.
        for index in selected:
            _, _, iroot = ref_ci_nelec(index)
            if iroot in seen_rootspaces:
                continue
            seen_rootspaces.add(iroot)
            for ifrag in range(self.nfrags):
                new_ci[ifrag].append(ref.ci[ifrag][iroot])
            charges[self.nroots] = ref_charges[iroot]
            spins[self.nroots] = ref_spins[iroot]
            smults[self.nroots] = ref_smults[iroot]
            wfnsyms[self.nroots] = ref_wfnsyms[iroot]
            self.nroots += 1

        nref = self.nroots
        # Keep local-root views alive and reuse each local operator string.
        # The cache lasts only for this preparation; edits to a reference are
        # observed on the next call. No cached CI vector is modified in place.
        references = [ref_ci_nelec(index) for index in selected]
        excitation_cache = {} if self._share_spectator_ci else None
        for a_idx, i_idx in zip(self.a_idxs, self.i_idxs):
            for ref_ci, ref_nelec, _ in references:
                for a_op, i_op in ((a_idx, i_idx), (i_idx, a_idx)):
                    aci, nelec = self.getAci(
                        a_op, i_op, ref_ci=ref_ci,
                        ref_nelecas_sub=ref_nelec, _cache=excitation_cache)
                    if aci is None:
                        continue
                    for ifrag, ci in enumerate(aci):
                        new_ci[ifrag].append(ci)
                        charges[self.nroots, ifrag] = (
                            self.ncas_sub[ifrag] - sum(nelec[ifrag]))
                        spins[self.nroots, ifrag] = (
                            nelec[ifrag][0] - nelec[ifrag][1])
                        # This is only nominal metadata. LASSI detects impure
                        # fragment vectors and passes smult_fr=None to opt=1.
                        smults[self.nroots, ifrag] = (
                            abs(spins[self.nroots, ifrag]) + 1)
                    self.nroots += 1

        charges = charges[:self.nroots]
        spins = spins[:self.nroots]
        smults = smults[:self.nroots]
        wfnsyms = wfnsyms[:self.nroots]
        self.ci = new_ci
        self.weights = np.zeros(self.nroots)
        self.weights[:nref] = 1.0
        self.e_states_meaningless = True

        self.fciboxes = [get_h1e_zipped_fcisolver(state_average_n_mix(
            self._las,
            [csf_solver(self._las.mol, smult=smult).set(
                charge=charge, spin=spin, wfnsym=wfnsym)
             for charge, spin, smult, wfnsym in zip(
                 charges_frag, spins_frag, smults_frag, wfnsyms_frag)],
            self.weights).fcisolver)
            for charges_frag, spins_frag, smults_frag, wfnsyms_frag in zip(
                charges.T, spins.T, smults.T, wfnsyms.T)]

        nstates = int(np.sum(np.prod(get_lroots(self.ci), axis=0)))
        lib.logger.info(
            self, "Prepared %d LUSCC product states in %d rootspaces",
            nstates, self.nroots)
        return self

    def kernel(self, **kwargs):
        """Build the LUSCC states and delegate the complete solve to LASSI."""
        self.timings = {}
        start = perf_counter()
        self._time_stage('prepare_states', self.prepare_states_)
        if self.target_spin is not None:
            if self._smult_si is not None or kwargs.get("smult_si") is not None:
                raise ValueError("Use either target_spin=S or smult_si, not both")
            result = self._kernel_exact_spin(**kwargs)
            self.timings['total'] = perf_counter() - start
            return result
        if self._smult_si is not None:
            kwargs.setdefault("smult_si", self._smult_si)
            kwargs.setdefault("davidson_only", True)
        return super().kernel(**kwargs)

    def _time_stage(self, name, function):
        start = perf_counter()
        result = function()
        self.timings[name] = perf_counter() - start
        lib.logger.info(self, 'LUSCC %s wall time %.6f s', name, self.timings[name])
        return result

    def _kernel_exact_spin(self, **kwargs):
        """Diagonalize in the exact null space of (S^2-S(S+1))."""
        unsupported = set(kwargs) - {"mo_coeff", "veff_c", "h2eff_sub"}
        if unsupported:
            raise TypeError("Unsupported exact-spin kernel options: "
                            + ", ".join(sorted(unsupported)))
        h0, h1, h2 = self._time_stage('integrals', lambda: self.ham_2q(
            mo_coeff=kwargs.get("mo_coeff"), veff_c=kwargs.get("veff_c"),
            h2eff_sub=kwargs.get("h2eff_sub")))
        # LUSCC fragment vectors can be spin impure, so do not pass fragment
        # multiplicities (and do not invoke the Clebsch--Gordan basis path).
        hop, s2op, _, _, get_overlap = self._time_stage('operators', lambda:
            self._get_exact_spin_operators(h1, h2))
        # Build each rootspace-pair block once. Applying mop to an identity
        # rebuilds these same blocks once per column of the dense matrix.
        overlap = self._time_stage('overlap', get_overlap)
        residual = self._time_stage('residual', lambda:
                                    residual_gram(self, self.target_spin))
        spin_basis, keig = self._time_stage('spin_basis', lambda: exact_spin_basis(
            overlap, residual, self.target_spin,
            lin_dep_tol=self.lin_dep_tol, spin_tol=self.spin_tol))
        converged, energy, coeff, s2 = self._time_stage('projected_davidson', lambda:
            self.sisolver.kernel_projected(
                hop, s2op, spin_basis, nroots=1))
        self.converged = self.converged and converged
        self.e_roots = energy + h0
        self.si = coeff
        self.s2 = s2
        self.spin_residual_norm = np.sqrt(np.maximum(0.0, np.real(
            np.einsum("ip,ij,jp->p", coeff.conj(), residual, coeff))))
        self.spin_residual_eigenvalues = keig
        return self.e_roots, self.si, self.s2, self.spin_residual_norm

    def _get_exact_spin_operators(self, h1, h2):
        """Use LASSI's in-memory assembly when the spin solve fits in memory.

        Reserve eight raw-state matrices for H, S2, overlap, residual, and
        eigensolver/projection work, rather than budgeting H alone. This is
        an estimate (as in ordinary LASSI), not a hard bound on peak RSS.
        """
        from mrh.my_pyscf.lassi import op_o1
        from mrh.my_pyscf.lassi.citools import get_lroots

        nstates = int(np.sum(np.prod(get_lroots(self.ci), axis=0)))
        itemsize = np.dtype(np.result_type(h1.dtype, np.float64)).itemsize
        required = 8 * itemsize * nstates**2 / 1e6
        available = max(0., self.max_memory - lib.current_memory()[0])
        args = (self, h1, h2, self.ci, self.get_nelec_frs())
        kwargs = dict(smult_fr=None, disc_fr=self.get_disc_fr())
        if self.operator_mode == 'auto' and required <= available:
            self.exact_spin_operator_mode = 'dense'
            lib.logger.info(self, 'LUSCC in-memory operators: %.1f MB estimated, '
                            '%.1f MB available', required, available)
            ham, s2, overlap, _ = op_o1.ham(*args, **kwargs)
            hop, s2op, mop = map(aslinearoperator, (ham, s2, overlap))
            return hop, s2op, mop, ham.diagonal(), lambda: overlap
        self.exact_spin_operator_mode = 'matrix_free'
        lib.logger.info(self, 'LUSCC matrix-free operators: %.1f MB estimated, '
                        '%.1f MB available', required, available)
        return op_o1.gen_contract_op_si_hdiag(*args, **kwargs)

    def filter_spaces(self, las):
        return las


# Backward-compatible public name.
LSI_LUSCC = LUSCC
