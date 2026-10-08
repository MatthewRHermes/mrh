import numpy as np
from itertools import combinations

from pyscf import lib
from pyscf.fci import cistring

from mrh.my_pyscf.pbc.fci.csf_cplx import cplxCSFFCISolver as CSFFCISolver
from mrh.my_pyscf.mcscf.productstate import (
    ProductStateFCISolver as molProductStateFCISolver,
    ImpureProductStateFCISolver as molImpureProductStateFCISolver,
)

# Author: Bhavnesh Jangid

'''
# TODO-1: add multiple root testing for the PBCTransSymmImpureProductStateFCISolver class.
# The current implementation does support multiple roots, but it has not been tested with
# more than one root per fragment.
'''

class PBCProductStateFCISolver (molProductStateFCISolver):

    ci_dtype = np.complex128

    def make_rdm1s(self, ci, norb_f, nelec_f, **kwargs):
        """Return <p^+ q> for direct contraction with the LAS Hamiltonian."""
        dm1s = super().make_rdm1s(ci, norb_f, nelec_f, **kwargs)
        return tuple(np.asarray(dm).T for dm in dm1s)

    def _get_grad (self, h1eff, h2, ci, norb_f, nelec_f, orbsym=None,
            **kwargs):
        nj = np.cumsum (norb_f)
        ni = nj - norb_f
        zipper = [h1eff, ci, norb_f, nelec_f, self.fcisolvers, ni, nj]
        grad = []
        for h1e, c, no, ne, solver, i, j in zip (*zipper):
            nelec = self._get_nelec (solver, ne)
            nroots = solver.nroots
            h2e = h2[i:j,i:j,i:j,i:j]
            h2e = solver.absorb_h1e (h1e, h2e, no, nelec, 0.5)
            if nroots==1: c=c[None,:] # nroots, na, nb
            hc = [solver.contract_2e (h2e, col, no, nelec) for col in c]
            c, hc = np.asarray (c), np.asarray (hc)
            chc = np.dot (np.asarray (c).reshape (nroots,-1).conj (),
                          np.asarray (hc).reshape (nroots,-1).T)
            hc = hc - np.tensordot (chc, c, axes=1)
            if isinstance (solver, CSFFCISolver):
                #hc = solver.transformer.vec_det2csf (hc, normalize=False)
                creal = solver.transformer.vec_det2csf (hc.real, order='C', normalize=False)
                cimag = solver.transformer.vec_det2csf (hc.imag, order='C', normalize=False)
                cout = creal.astype(h1e.dtype)
                cout.real = creal
                cout.imag = cimag
                hc = cout
            # External degrees of freedom: not weighted, because I want
            # to converge all of the roots even if they don't contribute
            # to the mean field
            assert (hc.size == nroots*solver.transformer.ncsf)
            grad.append (hc.ravel ())
            # Internal degrees of freedom: weighted and lower-triangular
            # TODO: confirm the sign choice below before using this gradient
            # for something more advanced than convergence checking
            if nroots>1 and getattr (solver, 'weights', None) is not None:
                chc *= np.asarray (solver.weights)[:,None]
                chc -= chc.T
                grad.append (chc[np.tril_indices (nroots,k=-1)])
        return np.concatenate (grad)

    def make_rdm2 (self, ci, norb_f, nelec_f, **kwargs):
        norb = sum (norb_f)
        nj = np.cumsum (norb_f)
        ni = nj - norb_f
        dm1a, dm1b = self.make_rdm1s (ci, norb_f, nelec_f, **kwargs)
        dm2_f = []
        for i, j, c, no, ne, s in zip (ni, nj, ci, norb_f, nelec_f, self.fcisolvers):
            nelec = self._get_nelec (s, ne)
            dm = np.asarray(s.make_rdm2 (c, no, nelec))
            dm2_f.append ((i, j, dm))
        dtype = np.result_type (float, dm1a, dm1b,
                                *[dm for _, _, dm in dm2_f])
        dm2 = np.zeros ([norb,]*4, dtype=dtype)
        for i, j, dm in dm2_f:
            dm2[i:j,i:j,i:j,i:j] = dm
        dm1 = dm1a + dm1b
        for (i,j), (k,l) in combinations (zip (ni, nj), 2):
            d1_ij = dm1[i:j,i:j]
            d1a_ij = dm1a[i:j,i:j]
            d1b_ij = dm1b[i:j,i:j]
            d1_kl = dm1[k:l,k:l]
            d1a_kl = dm1a[k:l,k:l]
            d1b_kl = dm1b[k:l,k:l]
            d2 = np.multiply.outer (d1_ij, d1_kl)
            dm2[i:j,i:j,k:l,k:l] = d2
            dm2[k:l,k:l,i:j,i:j] = d2.transpose (2,3,0,1)
            d2  = np.multiply.outer (d1a_ij, d1a_kl)
            d2 += np.multiply.outer (d1b_ij, d1b_kl)
            d2_exch = -d2.transpose (0,3,2,1)
            dm2[i:j,k:l,k:l,i:j] = d2_exch
            dm2[k:l,i:j,i:j,k:l] = d2_exch.conj().transpose (1,0,3,2)
        return dm2


class ImpureProductStateFCISolver (PBCProductStateFCISolver):
    r'''Minimize the energy of an impure state:

    E = \sum_n1 w_n1 \sum_n2 w_n2 \sum_n3 w_n3 ... <n1n2n3...|H|n1n2n3...>

    over orthonormal sets of CI vectors {nK} for fragment K.'''

    __init__ = molImpureProductStateFCISolver.__init__


class PBCTransSymmImpureProductStateFCISolver(ImpureProductStateFCISolver):
    r'''
    Translation-symmetry adapted product-state solver.

    This class has currently been tested with one root per unit cell.  The
    implementation retains the generalized multi-root and state-averaged
    code paths.

    In this class, only the reference fragment is optimized. Any function
    (method) or local variable which ends in ``_ref`` represents a reference-cell
    quantity. The solver retains the conventional full fragment CI list;
    reference-only helpers extract ``ci_ref`` when performing the optimized
    reference-cell calculation.

    In this implementation, the
    translation phase is a scalar CI gauge.  If fragment ``f`` has
    ``phase_per_frag[f] = phi_f``, the translated CI vector and its CI
    residual or gradient transform as
        The CI vector:
            C_f = phi_f C_ref
        The CI residual or gradient:
            g_f = phi_f g_ref

    These quantities require phase multiplication when assembled from the
    reference cell.

    Ordinary one- and two-body density matrices do not:
    their phase cancels between the bra and ket.  The projected effective
    Hamiltonians also do not require phase multiplication.  Under the scalar-
    phase translation assumed here,
        The one-body density matrix:
            D_f = D_ref
        The Hamiltonian matrix elements:
            H_eff,f = H_eff,ref

    Applying the copied Hamiltonian to a phased CI vector therefore produces
    the correctly phased result automatically.  This convention assumes that
    translated active orbitals differ only by a common scalar phase.  Note that a
    general orbital rotation would instead require the corresponding basis
    transformation of operators and density matrices.

    TODO: Read the Mcweeny's paper on RDM assembly to make sure not making any mistake
    in the RDM assembly for the complex case.
    '''

    trans_sym = True

    def __init__(self, fcisolvers, stdout=None, verbose=0, lroots=None,
                 lweights=None, ref_cell=0, phase_per_frag=None,
                 pack_h1=None, pack_h2=None, **kwargs):
        '''
        For documentation see above, the new args are:
        ref_cell: int
            Index of the reference fragment.  The reference fragment is the only
            fragment that is optimized, and the other fragments are generated by
            translation of the reference fragment.
        phase_per_frag: np.array or list of np.array or None
            Optional translation phases for each fragment. Generated from the overlap of the 
            orbitals of the reference fragment with the orbitals of each other fragment.
            If None, all phases are set to one
        pack_h1: callable or None
            This function will transform the one-electron integrals into a packed form.
        pack_h2: callable or None
            This function will transform the two-electron integrals into a packed form.
        '''
        super().__init__(fcisolvers, stdout=stdout, verbose=verbose, lroots=lroots,
                         lweights=lweights, **kwargs)
        # Checks:
        if not isinstance(ref_cell, (int, np.integer)):
            msg = f"ref_cell must be an integer, got {type(ref_cell)}"
            raise TypeError(msg)
        if not 0 <= ref_cell < len(self.fcisolvers):
            msg = f"ref_cell must be in [0, {len(self.fcisolvers)}); got {ref_cell}"
            raise ValueError(msg)
        self.ref_cell = int(ref_cell)
        nroots_per_frag = [solver.nroots for solver in self.fcisolvers]
        if any(nroots != nroots_per_frag[self.ref_cell]
               for nroots in nroots_per_frag):
            msg = "translation-symmetric fragments must have identical "
            msg += f"local root counts; got {nroots_per_frag}"
            raise ValueError(msg)
        self.phase_per_frag = self._normalize_phase_per_frag(phase_per_frag)
        if (pack_h1 is None) != (pack_h2 is None):
            raise ValueError("pack_h1 and pack_h2 must be provided together")
        if pack_h1 is not None and not callable(pack_h1):
            raise TypeError("pack_h1 must be callable")
        if pack_h2 is not None and not callable(pack_h2):
            raise TypeError("pack_h2 must be callable")
        self.pack_h1 = pack_h1
        self.pack_h2 = pack_h2

    def _normalize_phase_per_frag(self, phase_per_frag):
        '''
        Validate fragment phases and fix the reference-cell phase to one.
        '''
        ncells = len(self.fcisolvers)
        if phase_per_frag is None:
            return np.ones(ncells, dtype=np.complex128)
        dtype = np.result_type(phase_per_frag)
        phase_per_frag = np.asarray(phase_per_frag, dtype=dtype)

        if phase_per_frag.shape != (ncells,):
            msg = f"phase_per_frag must have shape ({ncells},), got {phase_per_frag.shape}"
            raise ValueError(msg)
        
        magnitudes = np.abs(phase_per_frag)

        # Extreme sanity checks to avoid division by zero or NaN propagation
        if np.any(~np.isfinite(magnitudes)) or np.any(magnitudes == 0):
            raise ValueError("phase_per_frag must contain finite nonzero phases")
        if np.max(np.abs(magnitudes - 1.0)) >= 1e-8:
            raise ValueError("phase_per_frag entries must have unit magnitude")

        phase_per_frag = phase_per_frag / magnitudes
        phase_per_frag *= phase_per_frag[self.ref_cell].conjugate()
        phase_per_frag[self.ref_cell] = 1.0

        return phase_per_frag

    def _pack_ci(self, ci):
        '''
        In short: CI_tot -> CI_ref
        Select the reference CI and remove its stored translation phase.
        By reference CI, I mean the CI vector corresponding to the reference
        fragment/cell, which is the only one that is optimized.  The other fragments
        are generated by translation of the reference fragment.
        '''
        # Just for safety:
        if ci is None or ci[self.ref_cell] is None:
            return None
        ci_ref = np.asarray(ci[self.ref_cell])
        ci_ref /= self.phase_per_frag[self.ref_cell]
        return np.array(ci_ref, copy=True)

    def _unpack_cif(self, ci_ref):
        '''
        In short: CI_ref -> CI_tot

        Expand a reference cell CI vector into one independent vector per fragment.
        args:
            ci_ref: np.ndarray
                Reference-cell CI vector with shape ``(ndeta, ndetb)`` for
                one root or ``(nroots, ndeta, ndetb)`` for multiple roots.
        returns:
            ci_tot: list of np.ndarray
                List of CI vectors for each fragment, with the reference cell
                vector independently copied and multiplied by the stored
                fragment phase.
        '''
        ncells = len(self.fcisolvers)

        if ci_ref is None:
            return [None for _ in range(ncells)]

        ci_ref = np.asarray(ci_ref)
        if ci_ref.ndim == 2:
            nroots = 1
        elif ci_ref.ndim == 3:
            nroots = ci_ref.shape[0]
        else:
            msg = ("ci_ref must have shape (ndeta, ndetb) or "
                   f"(nroots, ndeta, ndetb); got {ci_ref.shape}")
            raise ValueError(msg)

        exptd_nroots = self.fcisolvers[self.ref_cell].nroots
        if nroots != exptd_nroots:
            msg = (f"ci_ref contains {nroots} roots, but the reference "
                   f"solver requires {exptd_nroots}")
            raise ValueError(msg)

        ci_tot = [np.array(self.phase_per_frag[ifrag] * ci_ref, copy=True)
                  for ifrag in range(ncells)]
        return ci_tot

    def _unpack_hfrag(self, h1eff_ref, h0eff_ref):
        '''
        In short: h1eff_ref, h0eff_ref -> h1eff, h0eff

        Assemble full-fragment effective Hamiltonians from the reference
        cell effective Hamiltonians.

        The effective Hamiltonians are invariant under the scalar CI gauge
        stored in ``phase_per_frag``.  They are therefore independently
        copied to every translated cell without phase multiplication.  A
        phased CI vector acquires the corresponding phase when the copied
        Hamiltonian acts on it.
        '''
        ncells = len(self.fcisolvers)
        h1eff = [np.array(h1eff_ref, copy=True)
                 for _ in range(ncells)]
        h0eff = [np.array(h0eff_ref, copy=True)
                 for _ in range(ncells)]
        return h1eff, h0eff

    def _make_ref_rdm1s(self, ci_ref, norb_f, nelec_f):
        '''
        Calculate spin-separated one-body RDMs for the reference cell.
        '''
        dtype = np.result_type(ci_ref)
        ref = self.ref_cell
        norb_ref = norb_f[ref]
        solver_ref = self.fcisolvers[ref]
        nelec_ref = self._get_nelec(solver_ref, nelec_f[ref])
        ci_solver = ci_ref
        if getattr(ci_solver, 'ndim', 3) == 3:
            ci_solver = list(ci_solver)

        dm1a_ref, dm1b_ref = solver_ref.make_rdm1s(ci_solver, norb_ref, nelec_ref,)

        dm1a_ref = np.asarray(dm1a_ref, dtype=dtype).T
        dm1b_ref = np.asarray(dm1b_ref, dtype=dtype).T

        if self.verbose >= lib.logger.DEBUG:
            assert np.allclose(dm1a_ref, dm1a_ref.conj().T,
                               atol=1e-8, rtol=0.0), \
                "dm1a_ref is not Hermitian"
            assert np.allclose(dm1b_ref, dm1b_ref.conj().T,
                               atol=1e-8, rtol=0.0), \
                "dm1b_ref is not Hermitian"

            nelec_ref = sum(nelec_ref)
            nelec_check = np.trace(dm1a_ref) + np.trace(dm1b_ref)
            assert np.isclose(nelec_check, nelec_ref, atol=1e-8, rtol=0.0), \
                f"nelec_ref ({nelec_ref}) does not match trace(dm1a_ref + dm1b_ref) ({nelec_check})"

        return dm1a_ref, dm1b_ref

    def _make_ref_rdm2(self, ci_ref, norb_f, nelec_f):
        '''
        Calculate the two-body RDM for the reference cell.
        '''
        ref = self.ref_cell
        norb_ref = norb_f[ref]
        dtype = np.result_type(ci_ref)
        solver_ref = self.fcisolvers[ref]
        nelec_ref = self._get_nelec(solver_ref, nelec_f[ref])
        ci_solver = ci_ref
        if getattr(ci_solver, 'ndim', 3) == 3:
            ci_solver = list(ci_solver)

        rdm2 = np.asarray(solver_ref.make_rdm2(ci_solver, norb_ref, nelec_ref), 
                          dtype=dtype)

        # Sanity checks for Hermiticity of the two-body RDM
        if self.verbose >= lib.logger.DEBUG:
            assert np.allclose(rdm2, rdm2.transpose(1,0,3,2).conj(),
                               atol=1e-10, rtol=1e-10), "RDM2 is not Hermitian"
        return rdm2

    def _unpack_rdm1s(self, dm1a_ref, dm1b_ref, norb_f, nelec_f):
        '''
        Assemble full one-body RDMs from the reference-cell blocks.
        return:
            dm1a: np.ndarray
                Full spin-up one-body RDM with shape (norb, norb)
                norb = ncells * norb_per_cell
            dm1b: similar to above.
        '''
        ref = self.ref_cell
        norb_ref = norb_f[ref]
        nelec_ref = self._get_nelec(self.fcisolvers[ref], nelec_f[ref],)
        norb = sum(norb_f)
        dtype = np.result_type(dm1a_ref.dtype, dm1b_ref.dtype)
        dm1a = np.zeros((norb, norb), dtype=dtype)
        dm1b = np.zeros((norb, norb), dtype=dtype)

        nj = np.cumsum(norb_f)
        ni = nj - norb_f
        for ifrag, (i, j, solver) in enumerate(zip(ni, nj, self.fcisolvers)):
            nelec = self._get_nelec(solver, nelec_f[ifrag])

            # In case after all of the transformation, the user is able
            # to reach at this stage.
            if norb_f[ifrag] != norb_ref \
                or tuple(nelec) != tuple(nelec_ref):
                msg = "translated fragments have inconsistent active spaces"
                raise ValueError(msg)

            dm1a[i:j, i:j] = dm1a_ref
            dm1b[i:j, i:j] = dm1b_ref
        return dm1a, dm1b

    def make_rdm1s(self, ci, norb_f, nelec_f, **kwargs):
        '''
        Build the full spin-resolved 1-RDMs from the translated CI list.
        '''
        ci_ref = self._pack_ci(ci)
        dm1a_ref, dm1b_ref = self._make_ref_rdm1s(ci_ref, norb_f, nelec_f)
        rdm1a, rdm1b = self._unpack_rdm1s(dm1a_ref, dm1b_ref, norb_f, nelec_f)
        return rdm1a, rdm1b

    def make_rdm1(self, ci, norb_f, nelec_f, **kwargs):
        '''
        Build the full spin-summed 1-RDM from the translated CI list.
        '''
        dm1a, dm1b = self.make_rdm1s(ci, norb_f, nelec_f, **kwargs)
        dm1 = dm1a + dm1b
        dm1 = 0.5 * (dm1 + dm1.conj().T)  # Ensure Hermiticity
        return dm1

    def make_rdm2(self, ci, norb_f, nelec_f, **kwargs):
        '''
        Build the full product-state 2-RDM from the translated CI list.
        '''
        
        ci_ref = self._pack_ci(ci)
        dm2_ref = self._make_ref_rdm2(ci_ref, norb_f, nelec_f,)
        dm1a_ref, dm1b_ref = self._make_ref_rdm1s(ci_ref, norb_f, nelec_f,)
        dm1a, dm1b = self._unpack_rdm1s(dm1a_ref, dm1b_ref, norb_f, nelec_f,)

        dm1 = dm1a + dm1b
        dm1 = 0.5 * (dm1 + dm1.conj().T)  # Ensure Hermiticity

        norb = sum(norb_f)
        dtype = np.result_type(dm2_ref.dtype, dm1.dtype)
        dm2 = np.zeros((norb,) * 4, dtype=dtype)
        nj = np.cumsum(norb_f)
        ni = nj - norb_f

        for i, j in zip(ni, nj):
            dm2[i:j, i:j, i:j, i:j] = dm2_ref

        for (i, j), (k, l) in combinations(zip(ni, nj), 2):
            d1_ij = dm1[i:j, i:j]
            d1a_ij = dm1a[i:j, i:j]
            d1b_ij = dm1b[i:j, i:j]
            d1_kl = dm1[k:l, k:l]
            d1a_kl = dm1a[k:l, k:l]
            d1b_kl = dm1b[k:l, k:l]

            d2 = np.multiply.outer(d1_ij, d1_kl)
            dm2[i:j, i:j, k:l, k:l] = d2
            dm2[k:l, k:l, i:j, i:j] = d2.transpose(2, 3, 0, 1)

            d2 = np.multiply.outer(d1a_ij, d1a_kl)
            d2 += np.multiply.outer(d1b_ij, d1b_kl)
            d2_exch = -d2.transpose(0, 3, 2, 1)
            dm2[i:j, k:l, k:l, i:j] = d2_exch
            dm2[k:l, i:j, i:j, k:l] = d2_exch.conj().transpose(1, 0, 3, 2)
        return dm2

    def energy_ref(self, h1_packed, h2_packed, ci_ref,
                   norb_f, nelec_f, ecore=0, **kwargs):
        '''
        Calculate the energy contribution associated with one cell.

        ``ecore`` is the total system core energy.  An equal ``1/ncell``
        share is included in the returned reference-cell energy.
        '''

        ncell = len(self.fcisolvers)
        norb_ref = norb_f[self.ref_cell]
        h1_packed = np.asarray(h1_packed)
        h2_packed = np.asarray(h2_packed)

        if h1_packed.shape == (ncell, norb_ref, norb_ref):
            h1_packed = np.stack([h1_packed, h1_packed], axis=0)

        h1_shape = (2, ncell, norb_ref, norb_ref)
        h2_shape = (ncell, ncell, ncell, norb_ref, norb_ref, norb_ref, norb_ref,)

        if h1_packed.shape != h1_shape:
            msg = (f"packed h1 must have shape {h1_shape}; "
                   f"got {h1_packed.shape}")
            raise ValueError(msg)
        
        if h2_packed.shape != h2_shape:
            msg = (f"packed h2 must have shape {h2_shape}; "
                   f"got {h2_packed.shape}")
            raise ValueError(msg)

        dm1a_ref, dm1b_ref = self._make_ref_rdm1s(ci_ref, norb_f, nelec_f,)
        dm1s_ref = np.stack([dm1a_ref, dm1b_ref], axis=0)
        dm1_ref = dm1a_ref + dm1b_ref
        dm2_ref = self._make_ref_rdm2(ci_ref, norb_f, nelec_f)

        e1 = np.einsum('spq,spq->', h1_packed[:, 0], dm1s_ref,)
        e2 = np.einsum('pqrs,pqrs->', h2_packed[0, 0, 0], dm2_ref,)

        for delta in range(1, ncell):
            e2 += np.einsum('pqrs,pq,rs->',
                            h2_packed[0, delta, delta], dm1_ref, dm1_ref,)
            for spin in range(2):
                e2 -= np.einsum('pqrs,ps,rq->',h2_packed[delta, delta, 0],
                                dm1s_ref[spin], dm1s_ref[spin],)

        return ecore / ncell + e1 + 0.5 * e2

    def energy_elec(self, h1, h2, ci, norb_f, nelec_f,
                    ecore=0, **kwargs):
        '''
        Evaluate the total energy from the translated CI list.
        '''
        if self.pack_h1 is None:
            return super().energy_elec(
                h1, h2, ci, norb_f, nelec_f, ecore=ecore, **kwargs,)

        h1_packed = self.pack_h1(h1)
        h2_packed = self.pack_h2(h2)
        ci_ref = self._pack_ci(ci)
        energy_ref = self.energy_ref(h1_packed, h2_packed, ci_ref, norb_f, nelec_f,
                                     ecore=ecore, **kwargs,)
        ncells = len(self.fcisolvers)
        return ncells * energy_ref

    def _unpack_grad(self, grad_ref):
        '''
        Assemble the packed full-fragment gradient from the reference.
        '''
        ref_solver = self.fcisolvers[self.ref_cell]
        nroots = ref_solver.nroots
        external_size = nroots * ref_solver.transformer.ncsf
        internal_size = 0
        if nroots > 1 and getattr(ref_solver, 'weights', None) is not None:
            internal_size = nroots * (nroots - 1) // 2

        grad_ref = np.asarray(grad_ref).reshape(-1)
        if grad_ref.size != external_size + internal_size:
            raise ValueError("reference gradient has an inconsistent size")

        grad = []
        for phase, solver in zip(self.phase_per_frag, self.fcisolvers):
            solver_external_size = solver.nroots * solver.transformer.ncsf
            solver_internal_size = 0
            if (solver.nroots > 1
                    and getattr(solver, 'weights', None) is not None):
                solver_internal_size = solver.nroots * (solver.nroots - 1) // 2

            if (solver_external_size != external_size
                    or solver_internal_size != internal_size):
                msg = ("translated fragment gradients have inconsistent sizes: "
                       f"expected ({external_size}, {internal_size}), "
                       f"got ({solver_external_size}, {solver_internal_size})")
                raise ValueError(msg)

            # Multiply by the phase.
            grad_external = phase * grad_ref[:external_size]
            grad.append(grad_external)
            if internal_size:
                grad.append(np.array(grad_ref[external_size:], copy=True))
        return np.concatenate(grad)

    def get_init_guess(self, ci, norb_f, nelec_f, h1, h2, nroots=None):
        '''
        Generate a reference guess and expand it into the full CI list.
        '''
        ci_ref = self._pack_ci(ci)
        ci_ref = self._get_ref_init_guess(ci_ref, norb_f, nelec_f, 
                                          h1, h2, nroots=nroots,)
        return self._unpack_cif(ci_ref)

    def project_hfrag(self, h1, h2, ci, norb_f, nelec_f,
                      ecore=0, dm1s=None, dm2=None, **kwargs):
        '''
        Assemble full effective Hamiltonians from the reference projection.
        '''
        h1eff_ref, h0eff_ref = self._project_ref_hfrag(
            h1, h2, ci, norb_f, nelec_f, ecore=ecore,
            dm1s=dm1s, dm2=dm2, **kwargs,)
        h1eff, h0eff = self._unpack_hfrag(h1eff_ref, h0eff_ref)
        return h1eff, h0eff, ci

    def _get_grad(self, h1eff, h2, ci, norb_f, nelec_f,**kwargs):
        '''
        Assemble the full CI gradient from its reference-cell block.
        '''
        i = sum(norb_f[:self.ref_cell])
        j = i + norb_f[self.ref_cell]
        h2_ref = h2[i:j, i:j, i:j, i:j]
        ci_ref = self._pack_ci(ci)
        h1eff_ref = h1eff[self.ref_cell]
        grad_ref = self._get_ref_grad(h1eff_ref, h2_ref, ci_ref, norb_f, nelec_f,
                                      **kwargs,)
        return self._unpack_grad(grad_ref)

    def _1shot_ref(self, h0eff_ref, h1eff_ref, h2, ci_ref,
                   norb_f, nelec_f, **kwargs):
        '''
        Optimize the reference-fragment CI vector once.
        '''
        ref = self.ref_cell
        i = sum(norb_f[:ref])
        j = i + norb_f[ref]
        norb = norb_f[ref]
        solver = self.fcisolvers[ref]
        nelec = self._get_nelec(solver, nelec_f[ref])
        h2_ref = h2[i:j, i:j, i:j, i:j]

        energy_ref, ci_ref = solver.kernel(h1eff_ref, h2_ref, norb, nelec, 
                                           ci0=ci_ref, ecore=h0eff_ref, **kwargs,)
        return energy_ref, np.array(ci_ref, copy=True)

    def kernel(self, h1, h2, norb_f, nelec_f, ecore=0, ci0=None,
               orbsym=None, conv_tol_grad=1e-4, conv_tol_self=1e-10,
               max_cycle_macro=50, **kwargs):
        '''
        Optimize the reference cell while retaining the full CI list.
        '''
        log = self.log
        converged = False
        energy_ref = 0.0
        energy_sigma = 0.0
        ci = ci0
        solver_ref = self.fcisolvers[self.ref_cell]
        i = sum(norb_f[:self.ref_cell])
        j = i + norb_f[self.ref_cell]
        h2_ref = h2[i:j, i:j, i:j, i:j]

        log.info('Entering translation-symmetric reference-cell CI iteration')
        
        for it in range(max_cycle_macro):
            # Full CI as the initial guess.
            ci = self.get_init_guess(ci, norb_f, nelec_f, h1, h2, **kwargs)
            h1eff_ref, h0eff_ref = self._project_ref_hfrag(h1, h2, ci, norb_f, nelec_f,
                                                           ecore=ecore, **kwargs,)
            # Curve out the reference cell CI vector.
            ci_ref = self._pack_ci(ci)
            grad_ref = self._get_ref_grad(h1eff_ref, h2_ref, ci_ref, norb_f, nelec_f, **kwargs,)

            grad_max = np.amax(np.abs(grad_ref))

            solver_converged = np.all(np.asarray(getattr(solver_ref, 'converged', False)))

            log.info('Cycle %d: max ref grad = %e ; sigma = %e ; ' 
                     'reference solver converged = %s', it, grad_max, energy_sigma, 
                     solver_converged,)
            
            if (grad_max < conv_tol_grad
                    and energy_sigma < conv_tol_self
                    and solver_converged and it > 0):
                converged = True
                break

            energy_ref, ci_ref = self._1shot_ref(h0eff_ref, h1eff_ref, h2, ci_ref,
                                                 norb_f, nelec_f, orbsym=orbsym, **kwargs,)

            ci = self._unpack_cif(ci_ref)

            # All translated fragment energies are identical, so their spread
            # is zero by construction.
            energy_sigma = 0.0

        conv_str = ['NOT converged', 'converged'][int(converged)]
        log.info('Translation-symmetric reference-cell CI iteration %s after %d '
                 'cycles', conv_str, it + 1,)

        energy_elec = self.energy_elec(h1, h2, ci, norb_f, nelec_f, 
                                       ecore=ecore, efinal=energy_ref, **kwargs,)
        
        return converged, energy_elec, ci

    def _project_ref_hfrag(self, h1, h2, ci, norb_f, nelec_f,
                           ecore=0, dm1s=None, dm2=None,
                           **kwargs):
        '''
        Project the full state and return the reference effective Hamiltonian.
        '''

        if dm1s is None:
            dm1s = np.stack(self.make_rdm1s(ci, norb_f, nelec_f), axis=0,)

        if dm2 is None:
            dm2 = self.make_rdm2(ci, norb_f, nelec_f)

        h1eff, h0eff, _ = super().project_hfrag(
            h1, h2, ci, norb_f, nelec_f, ecore=ecore,
            dm1s=dm1s, dm2=dm2, **kwargs,)

        return h1eff[self.ref_cell], h0eff[self.ref_cell]

    def _get_ref_grad(self, h1eff_ref, h2_ref, ci_ref, norb_f, nelec_f,
                      **kwargs):
        '''
        Calculate the reference-cell CI gradient from reference integrals.
        # TODO: Document the reference-gradient equation.
        '''

        ref = self.ref_cell
        norb = norb_f[ref]
        solver = self.fcisolvers[ref]
        nelec = self._get_nelec(solver, nelec_f[ref])
        nroots = solver.nroots

        ndeta = cistring.num_strings(norb, nelec[0])
        ndetb = cistring.num_strings(norb, nelec[1])

        ci_ref = np.asarray(ci_ref).reshape(nroots, ndeta, ndetb)

        h2eff_ref = solver.absorb_h1e(h1eff_ref, h2_ref, norb, nelec, 0.5,)

        hc = np.asarray([solver.contract_2e(h2eff_ref, root, norb, nelec) 
                         for root in ci_ref])
        chc = np.dot(ci_ref.reshape(nroots, -1).conj(), 
                     hc.reshape(nroots, -1).T)
        hc = hc - np.tensordot(chc, ci_ref, axes=1)

        if isinstance(solver, CSFFCISolver):
            # Transform the CI coefficients to the CSF basis
            hc_real = solver.transformer.vec_det2csf(hc.real, order='C', normalize=False,)
            hc_imag = solver.transformer.vec_det2csf(hc.imag, order='C', normalize=False,)
            hc_csf = hc_real.astype(h1eff_ref.dtype)
            hc_csf.real = hc_real
            hc_csf.imag = hc_imag
            hc = hc_csf

        assert hc.size == nroots * solver.transformer.ncsf

        grad = [hc.ravel()]

        if nroots > 1 and getattr(solver, 'weights', None) is not None:
            chc *= np.asarray(solver.weights)[:, None]
            chc -= chc.T
            grad.append(chc[np.tril_indices(nroots, k=-1)])

        return np.concatenate(grad)

    def _get_ref_init_guess(self, ci_ref, norb_f, nelec_f, h1, h2,
                            nroots=None):
        '''
        Preserve or generate the reference-cell CI initial guess.
        '''

        if ci_ref is not None:
            return np.array(ci_ref, copy=True)

        ref = self.ref_cell
        i = sum(norb_f[:ref])
        j = i + norb_f[ref]
        norb = norb_f[ref]
        solver = self.fcisolvers[ref]
        nelec = self._get_nelec(solver, nelec_f[ref])
        solver.norb = norb
        solver.nelec = nelec

        if h1.ndim < 3:
            h1 = np.stack([h1, h1], axis=0)
        h1_ref = h1[:, i:j, i:j]
        h2_ref = h2[i:j, i:j, i:j, i:j]

        solver.check_transformer_cache()

        if nroots is None:
            nroots = solver.nroots

        nroots = min(nroots, solver.transformer.ncsf)
        hdiag = solver.make_hdiag_csf(h1_ref, h2_ref, norb, nelec)
        ci_ref = solver.get_init_guess(norb, nelec, nroots, hdiag)
        ndeta = cistring.num_strings(norb, nelec[0])
        ndetb = cistring.num_strings(norb, nelec[1])
        return np.array(ci_ref, copy=True).reshape(nroots, ndeta, ndetb)
