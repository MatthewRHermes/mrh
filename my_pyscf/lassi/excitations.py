import copy
import numpy as np
import ctypes
import itertools
from scipy import linalg
from pyscf.lib import logger
from pyscf.fci.direct_spin1 import _unpack_nelec, trans_rdm1s, trans_rdm12s
from pyscf.scf.addons import canonical_orth_
from mrh.my_pyscf.mcscf.productstate import ProductStateFCISolver, state_average_fcisolver
from mrh.my_pyscf.fci import csf_solver, CSFFCISolver, direct_nosym_uhf
from mrh.my_pyscf.lassi import op_o0, op_o1
from mrh.my_pyscf.lassi.citools import get_lroots
from pyscf import lib
from pyscf.lib import temporary_env
from pyscf import __config__
op = (op_o0, op_o1)

LOWEST_REFOVLP_EIGVAL_THRESH = getattr (__config__, 'lassi_excitations_refovlp_eigval_thresh', 1e-9)
IMAG_SHIFT = getattr (__config__, 'lassi_excitations_imag_shift', 1e-6)
MAX_CYCLE_E0 = getattr (__config__, 'lassi_excitations_max_cycle_e0', 1)
CONV_TOL_E0 = getattr (__config__, 'lassi_excitations_conv_tol_e0', 1e-8)

def lowest_refovlp_eigpair (ham_pq, p=1, ovlp_thresh=LOWEST_REFOVLP_EIGVAL_THRESH, log=None):
    ''' Identify the lowest-energy eigenpair for which the eigenvector has nonzero overlap with
    the first p basis functions. '''
    e_all, u_all = linalg.eigh (ham_pq)
    w_pp = (u_all[:p,:].conj () * u_all[:p,:]).sum (0) / p
    w_q0q0 = u_all[p,:].conj () * u_all[p,:]
    w_pq0 = np.abs (u_all[:p,:].conj () * u_all[p,:][None,:]).sum (0)
    idx_valid = w_q0q0 > 0.1
    e_valid = e_all[idx_valid]
    u_valid = u_all[:,idx_valid]
    idx_choice = np.argmin (e_valid)
    if log is not None and log.verbose > logger.DEBUG:
        log.debug2 ("Debugging eigenpair selection")
        log.debug2 (" idx e w_pp w_q0q0 w_pq0")
        i0 = np.where (idx_valid)[0][idx_choice]
        for i in range (len (e_all)):
            line = ' {} {} {} {} {}'.format (i,e_all[i],w_pp[i],w_q0q0[i],w_pq0[i])
            if i==i0: line += ' selected'
            log.debug2 (line)
    return e_valid[idx_choice], u_valid[:,idx_choice]

def lowest_refovlp_eigval (ham_pq, p=1, ovlp_thresh=LOWEST_REFOVLP_EIGVAL_THRESH):
    ''' Return the lowest eigenvalue of the matrix ham_pq, whose corresponding
    eigenvector has nonzero overlap with the first basis p basis functions. '''
    return lowest_refovlp_eigpair (ham_pq, p=p, ovlp_thresh=ovlp_thresh)[0]

def lowest_refovlp_eigvec (ham_pq, p=1, ovlp_thresh=LOWEST_REFOVLP_EIGVAL_THRESH):
    ''' Return the eigenvector corresponding to the lowest eigenvalue of the matrix ham_pq
    which has nonzero overlap with the first basis p basis functions. '''
    return lowest_refovlp_eigpair (ham_pq, p=p, ovlp_thresh=ovlp_thresh)[1]

class ExcitationPSFCISolver (ProductStateFCISolver):
    '''Minimize the energy of a normalized wave function of the form

    |Psi> = |exc> si_exc + sum_i |ref(i)> si_ref(i)
    |ref(i)> = A prod_K |ci(ref(i))_K>
    |exc> = A prod_{K in excited} |ci(exc)_K> prod_{K not in excited} |ci(ref(0))_K>

    with {ci(ref(i))_K} fixed.'''

    def __init__(self, solvers_ref, ci_ref, norb_ref, nelec_ref, orbsym_ref=None,
                 wfnsym_ref=None, stdout=None, verbose=0, opt=0, ref_weights=None, 
                 crash_locmin=False, **kwargs):
        if isinstance (solvers_ref, ProductStateFCISolver):
            solvers_ref = [solvers_ref]
            ci_ref = [[c] for c in ci_ref]
        if ref_weights is None:
            ref_weights = [0.0,]*len (solvers_ref)
            ref_weights[0] = 1.0
        self.solvers_ref = solvers_ref
        self.ci_ref = ci_ref
        self.ref_weights = np.asarray (ref_weights)
        self.norb_ref = np.asarray (norb_ref)
        self.nelec_ref = nelec_ref
        self.orbsym_ref = orbsym_ref
        self.wfnsym_ref = wfnsym_ref
        self.crash_locmin = crash_locmin
        self.opt = opt
        ProductStateFCISolver.__init__(self, solvers_ref[0].fcisolvers, stdout=stdout,
                                       verbose=verbose)
        ci_ref_rf = [[c[i] for c in ci_ref] for i in range (len (self.solvers_ref))]
        self.dm1s_ref = np.asarray ([s.make_rdm1s (c, norb_ref, nelec_ref)
                                     for s, c in zip (self.solvers_ref, ci_ref_rf)])
        self.dm1s_ref = np.tensordot (self.ref_weights, self.dm1s_ref, axes=1)
        self.dm2_ref = np.asarray ([s.make_rdm2 (c, norb_ref, nelec_ref)
                                    for s, c in zip (self.solvers_ref, ci_ref_rf)])
        self.dm2_ref = np.tensordot (self.ref_weights, self.dm2_ref, axes=1)
        self.excited_frags = []
        self.fcisolvers = []
        self._e_q = []
        self._si_q = []

    def get_excited_orb_idx (self):
        nj = np.cumsum (self.norb_ref)
        ni = nj - self.norb_ref
        idx = np.zeros (nj[-1], dtype=bool)
        for ifrag, solver in zip (self.excited_frags, self.fcisolvers):
            i, j = ni[ifrag], nj[ifrag]
            idx[i:j] = True
        return idx

    def get_excited_h (self, h0, h1, h2):
        '''Reduce the CAS Hamiltonian to the current excited fragments. Only some
        fragments are excited at any given time (nexc <= ncas).

        Args:
            h0: float
                Constant part of the CAS Hamiltonian
            h1: ndarray of shape (2,ncas,ncas)
                Spin-separated 1-electron part of the CAS Hamiltonian
            h2: ndarray of shape [ncas,]*4
                2-electron part of the CAS Hamiltonian

        Args:
            h0: float
                Constant part of the excited-fragment Hamiltonian
            h1: ndarray of shape (2,nexc,nexc)
                Spin-separated 1-electron part of the excited-fragment Hamiltonian
            h2: ndarray of shape [nexc,]*4
                2-electron part of the excited-fragment Hamiltonian

        '''
        idx = self.get_excited_orb_idx ()
        dm1s = self.dm1s_ref.copy ()
        dm2 = self.dm2_ref.copy ()
        norb_excited = np.count_nonzero (idx)
        idx = list (np.where (idx)[0]) + list (np.where (~idx)[0])
        dm1s = dm1s[:,idx][:,:,idx]
        h1 = h1[idx][:,idx]
        dm2 = dm2[idx][:,idx][:,:,idx][:,:,:,idx]
        h2 = h2[idx][:,idx][:,:,idx][:,:,:,idx]
        norb_ref = [norb_excited,] + [n for ifrag, n in enumerate (self.norb_ref)
                                      if not (ifrag in self.excited_frags)]
        h1eff, h0eff = self.project_hfrag (h1, h2, self.ci_ref, norb_ref, self.nelec_ref, 
                                           ecore=h0, dm1s=dm1s, dm2=dm2)[:2]
        h0, h1 = h0eff[0], h1eff[0]
        h2 = h2[:norb_excited][:,:norb_excited][:,:,:norb_excited][:,:,:,:norb_excited]
        return h0, h1, h2

    def set_excited_fragment_(self, ifrag, nelec, smult, weights=None):
        '''Indicate that the a fragment is excited and set its quantum numbers in the
        P-space

        Args:
            ifrag: integer
                Index of excited fragment
            nelec: tuple of length 2
                Nelectron tuple of fragment ifrag in P-space
            smult: integer
                spin-multiplicity of fragment ifrag in P-space'''
        # TODO: point group symmetry
        nelec = _unpack_nelec (nelec)
        spin = nelec[0] - nelec[1]
        s_ref = self.solvers_ref[0].fcisolvers[ifrag]
        mol = s_ref.mol
        nelec_ref = _unpack_nelec (self.nelec_ref[ifrag])
        charge = (nelec_ref[0]+nelec_ref[1]) - (nelec[0]+nelec[1])
        nelec = tuple (nelec)
        fcisolver = csf_solver (mol, smult=smult).set (charge=charge, spin=spin,
                                                       nelec=nelec, norb=s_ref.norb)
        if hasattr (weights, '__len__') and len (weights) > 1:
            fcisolver = state_average_fcisolver (fcisolver, weights=weights)
        if ifrag in self.excited_frags:
            self.fcisolvers[self.excited_frags.index (ifrag)] = fcisolver
        else:
            self.excited_frags.append (ifrag)
            self.fcisolvers.append (fcisolver)
            idx = np.argsort (self.excited_frags)
            self.excited_frags = [self.excited_frags[i] for i in idx]
            self.fcisolvers = [self.fcisolvers[i] for i in idx]

    def space_delta (self, ci0, si0_p, ci1, si1_p, nroots):
        delta = 0
        for c0, c1 in zip (ci0, ci1):
            x0 = np.asarray (c0[:nroots]).reshape (nroots,-1) 
            x1 = np.asarray (c1[:nroots]).reshape (nroots,-1) 
            ovlp = x0.conj () @ x1.T
            ovlp = ovlp * si1_p[None,:]
            ovlp = ovlp.conj () * ovlp
            ovlp -= np.diag (si1_p.conj () * si1_p)
            delta = max (delta, ovlp.sum ())
        delta = max (delta, np.amax (np.abs (si1_p-si0_p)))
        return delta

    def kernel (self, h1, h2, ecore=0, ci0=None,
                conv_tol_space=1e-4, conv_tol_self=1e-6, max_cycle_macro=50,
                serialfrag=False, nroots=1, **kwargs):
        t0 = lib.logger.process_clock (), lib.logger.perf_counter ()
        h0, h1, h2 = self.get_excited_h (ecore, h1, h2)
        log = self.log
        norb_f = np.asarray ([self.norb_ref[ifrag] for ifrag in self.excited_frags])
        nelec_f = np.asarray ([self.nelec_ref[ifrag] for ifrag in self.excited_frags])
        orbsym = self.orbsym_ref
        if orbsym is not None:
            idx = self.get_excited_orb_idx ()
            orbsym = [orbsym[iorb] for iorb in range (norb_tot) if idx[iorb]]
        ci0 = self.get_init_guess (ci0, norb_f, nelec_f, h1, h2, nroots=3*nroots)
        ham_pq = self.get_ham_pq (h0, h1, h2, ci0)
        e, si = self.eig1 (ham_pq, ci0)
        disc_svals, u, si_p, si_q, vh = self.schmidt_trunc (si, ci0, nroots=nroots)
        ham_pq = self.truncrot_ham_pq (ham_pq, u, vh)
        ci1 = self.truncrot_ci (ci0, u, vh)
        hci_qspace = self.op_ham_pq_ref (h1, h2, ci1)
        hci_pspace_diag = self.op_ham_pp_diag (h1, h2, ci1, norb_f, nelec_f)
        tdm1s_f = self.get_tdm1s_f (ci1, ci1, norb_f, nelec_f)
        e, si0_p = 0, si_p
        disc_sval_max = max (list(disc_svals)+[0.0,])
        converged = False
        log.info ('Entering product-state fixed-point CI iteration')
        for it in range (max_cycle_macro):
            e_last = e
            space_delta = self.space_delta (ci0, si0_p, ci1, si_p, nroots)
            ci0, si0_p = ci1, si_p
            # Re-diagonalize in truncated space
            e, si = self.eig1 (ham_pq, ci0)
            _, u, si_p, si_q, vh = self.schmidt_trunc (si, ci0, nroots=nroots)
            log.debug ('Singular values in truncated space: {}'.format (si_p))
            ci1 = self.truncrot_ci (ci0, u, vh)
            log.info ('Cycle %d: |delta space| = %e ; e = %e, |delta e| = %e, max (discarded) = %e',
                      it, space_delta, e, e - e_last, disc_sval_max)
            if ((space_delta < conv_tol_space) and (abs (e-e_last) < conv_tol_self)):
                converged = True
                break
            ham_pq = self.truncrot_ham_pq (ham_pq, u, vh)
            hci_qspace = self.truncrot_hci_qspace (hci_qspace, u, vh)
            hci_pspace_diag = self.truncrot_hci_pspace_diag (hci_pspace_diag, u, vh)
            tdm1s_f = self.truncrot_tdm1s_f (tdm1s_f, u, vh)
            # Generate additional vectors and compute gradient
            hpq_xq = self.get_hpq_xq (hci_qspace, ci1, si_q)
            hpp_xp = self.get_hpp_xp (ci1, si_p, hci_pspace_diag, h0, h2, tdm1s_f, norb_f, nelec_f)
            grad = self._get_grad (ci1, si_p, hpq_xq, hpp_xp, nroots=nroots)
            ci2 = self.get_new_vecs (ci1, hpq_xq, hpp_xp, nroots=nroots)
            # Extend intermediates
            hci2_qspace = self.op_ham_pq_ref (h1, h2, ci2)
            hci2_pspace_diag = self.op_ham_pp_diag (h1, h2, ci2, norb_f, nelec_f)
            tdm1s_f_12 = self.get_tdm1s_f (ci1, ci2, norb_f, nelec_f)
            tdm1s_f_21 = self.get_tdm1s_f (ci2, ci1, norb_f, nelec_f)
            tdm1s_f_22 = self.get_tdm1s_f (ci2, ci2, norb_f, nelec_f)
            for ifrag in range (len (ci1)):
                ci1[ifrag] = np.append (ci1[ifrag], ci2[ifrag], axis=0)
                hci_qspace[ifrag] = np.append (hci_qspace[ifrag], hci2_qspace[ifrag], axis=0)
                hci_pspace_diag[ifrag] = np.append (hci_pspace_diag[ifrag], hci2_pspace_diag[ifrag], axis=0)
                tdm1s_f[ifrag] = np.append (
                    np.append (tdm1s_f[ifrag], tdm1s_f_12[ifrag], axis=1),
                    np.append (tdm1s_f_21[ifrag], tdm1s_f_22[ifrag], axis=1),
                    axis=0
                )
            ham_pq = self.update_ham_pq (ham_pq, h0, h1, h2, ci1, hci_qspace, hci_pspace_diag,
                                         tdm1s_f, norb_f, nelec_f)
            # Diagonalize and truncate
            _, si = self.eig1 (ham_pq, ci1)
            disc_svals, u, si_p, si_q, vh = self.schmidt_trunc (si, ci1, nroots=nroots)
            ham_pq = self.truncrot_ham_pq (ham_pq, u, vh)
            ci1 = self.truncrot_ci (ci1, u, vh)
            hci_qspace = self.truncrot_hci_qspace (hci_qspace, u, vh)
            hci_pspace_diag = self.truncrot_hci_pspace_diag (hci_pspace_diag, u, vh)
            tdm1s_f = self.truncrot_tdm1s_f (tdm1s_f, u, vh)
            log.debug ('Retained singular values: {}'.format (si_p))
            log.debug ('Discarded singular values: {}'.format (disc_svals))
            disc_sval_max = max (list (disc_svals) + [0.0,])
        conv_str = ['NOT converged','converged'][int (converged)]
        log.info (('Product_state fixed-point CI iteration {} after {} '
                   'cycles').format (conv_str, it))
        if not converged:
            ci1 = self.get_init_guess (ci1, norb_f, nelec_f, h1, h2, nroots=nroots)
            # Issue #86: see above, same problem
            self._debug_csfs (log, ci0, ci1, norb_f, nelec_f, grad, nroots=nroots)
        energy_elec = e
        ci1_active = ci1
        ci1 = [c for c in self.ci_ref]
        for ifrag, c in zip (self.excited_frags, ci1_active):
            ci1[ifrag] = np.asarray (c)
        t1 = self.log.timer ('ExcitationPSFCISolver kernel', *t0)
        return converged, energy_elec, ci1, disc_sval_max

    def get_nq (self):
        lroots = get_lroots ([self.ci_ref[ifrag] for ifrag in self.excited_frags])
        return np.prod (lroots, axis=0).sum ()

    def get_ham_pq (self, h0, h1, h2, ci_p):
        '''Build the model-space Hamiltonian matrix for the current state of the P-space.

        Args:
            h0: float
                Constant part of the excited-fragment Hamiltonian
            h1: ndarray of shape (2,nexc,nexc)
                Spin-separated 1-electron part of the excited-fragment Hamiltonian
            h2: ndarray of shape [nexc,]*4
                2-electron part of the excited-fragment Hamiltonian
            ci_p: list of ndarray of shape (p[i],ndeta[i],ndetb[i])
                CI vectors for the active fragments in the P-space

        Returns:
            ham_pq: ndarray of shape (np+nq,np+nq)
                Model space Hamiltonian matrix'''
        t0 = lib.logger.process_clock (), lib.logger.perf_counter ()
        excited_frags, nelec_ref = self.excited_frags, self.nelec_ref
        nelec_ref = [nelec_ref[ifrag] for ifrag in excited_frags]
        norb_ref = [self.norb_ref[ifrag] for ifrag in excited_frags]
        ci_q = [self.ci_ref[ifrag] for ifrag in excited_frags]
        ci_fr = [[cp,] + cq for cp, cq in zip (ci_p, ci_q)]
        nelec_frs_q = np.zeros ((len (excited_frags), len (self.solvers_ref), 2), dtype=int)
        for iq, solver_ref in enumerate (self.solvers_ref):
            fcisolvers = [solver_ref.fcisolvers[ifrag] for ifrag in excited_frags]
            for ifrag, (s, n) in enumerate (zip (fcisolvers, nelec_ref)):
                nelec_frs_q[ifrag,iq,:] = self._get_nelec (s, n)
        fcisolvers = self.fcisolvers
        nelec_fs_p = np.asarray ([list(self._get_nelec (s, n)) 
                                   for s, n in zip (fcisolvers, nelec_ref)])
        nelec_frs = np.append (nelec_fs_p[:,None,:], nelec_frs_q, axis=1)
        with temporary_env (self, ncas_sub=norb_ref, mol=fcisolvers[0].mol):
            ham_pq, _, ovlp_pq = op[self.opt].ham (self, h1, h2, ci_fr, nelec_frs, soc=0,
                                                   orbsym=self.orbsym_ref,
                                                   wfnsym=self.wfnsym_ref)[:3]
        t1 = self.log.timer ('get_ham_pq', *t0)
        return ham_pq + (h0*ovlp_pq)

    def update_ham_pq (self, ham_pq, h0, h1, h2, ci, hci_qspace, hci_pspace_diag, tdm1s_f,
                       norb_f, nelec_f):
        '''Build the model-space Hamiltonian matrix for the current state of the P-space, reusing
        as many intermediates as possible.

        Args:
            ham_pq: ndarray of shape (?+q,?+q)
                Hamiltonian matrix describing a previous P-space. Only used to extract the Q-space
                diagonal block
            h0: float
                Constant part of the excited-fragment Hamiltonian
            h1: ndarray of shape (2,nexc,nexc)
                Spin-separated 1-electron part of the excited-fragment Hamiltonian
            h2: ndarray of shape [nexc,]*4
                2-electron part of the excited-fragment Hamiltonian
            ci: list of ndarray of shape (p,ndeta[i],ndetb[i])
                CI vectors for the active fragments in the P-space
            hci_qspace: list of ndarray of shape (nroots,ndeta,ndetb,q)
                H|q> projected on <p| for all but one fragment, where <p| is the
                vectors of CI, i.e., the output of op_ham_pq_ref for ci.
            hci_pspace_diag: list of ndarray of shape (nroots,ndeta,ndetb)
                H(ifrag)|p(ifrag)>, where <p| is the vectors of CI; i.e., the output of
                op_ham_pp_diag for ci.
            tdm1s_f: list of ndarray of shape (p,p,2,norb[i],norb[i])
                Spin-separated transition one-body density matrices within each fragment for the
                vectors in ci.
            norb_f: list of length (nfrag)
                Number of orbitals in excited fragments
            nelec_f: list of length (nfrag)
                Reference number of electrons in excited fragments

        Returns:
            ham_pq: ndarray of shape (p+q,p+q)
                Model space Hamiltonian matrix
        '''
        #ref = self.get_ham_pq (h0, h1, h2, ci)
        t0 = lib.logger.process_clock (), lib.logger.perf_counter ()
        #return ref
        nfrags = len (ci)
        assert (nfrags == 2)
        old_ham_pq = ham_pq
        lroots = get_lroots (ci)
        p = np.prod (lroots)
        q = self.get_nq ()

        # q,q sector
        ham_pq = np.zeros ((p+q,p+q), dtype=old_ham_pq.dtype)
        ham_pq[-q:,-q:] = old_ham_pq[-q:,-q:]        
        #assert (np.amax (np.abs (ham_pq[p:,p:] - ref[p:,p:])) < 1e-6)

        # p,q sector
        h_pq = lib.einsum (
            'iab,jabq->ijq', ci[1].conj (), hci_qspace[1],
        )
        h_pq = h_pq.reshape (lroots[1]*lroots[0],q)
        ham_pq[:p,p:] = h_pq
        #assert (np.amax (np.abs (ham_pq[:p,p:] - ref[:p,p:])) < 1e-6)
        ham_pq[p:,:p] = h_pq.conj ().T
        #assert (np.amax (np.abs (ham_pq[p:,:p] - ref[p:,:p])) < 1e-6)

        # p,p sector - constant
        h_pp = np.zeros ((p,p), dtype=ham_pq.dtype)
        h_pp[np.diag_indices_from (h_pp)] = h0

        # p,p sector - semidiagonal
        h_pp = h_pp.reshape (lroots[1],lroots[0],lroots[1],lroots[0])
        h_pp_diag = [np.dot (c.reshape (l,-1).conj (), hc.reshape (l,-1).T)
                     for c, hc, l in zip (ci, hci_pspace_diag, lroots)]
        for i in range (lroots[1]):
            h_pp[i,:,i,:] += h_pp_diag[0]
        for i in range (lroots[0]):
            h_pp[:,i,:,i] += h_pp_diag[1]

        # p,p sector - offdiagonal
        i = norb_f[0]
        w = lib.einsum ('ijab,klcd,abcd->ijkl', tdm1s_f[1].sum (2), tdm1s_f[0].sum (2),
                        h2[i:,i:,:i,:i])
        w -= lib.einsum ('ijab,klcd,adcb->ijkl', tdm1s_f[1][:,:,0], tdm1s_f[0][:,:,0],
                        h2[i:,:i,:i,i:])
        w -= lib.einsum ('ijab,klcd,adcb->ijkl', tdm1s_f[1][:,:,1], tdm1s_f[0][:,:,1],
                        h2[i:,:i,:i,i:])
        h_pp += w.transpose (0,2,1,3)
        ham_pq[:p,:p] = h_pp.reshape (lroots[1]*lroots[0], lroots[1]*lroots[0])
        #try:
        #    assert (np.amax (np.abs (ham_pq[:p,:p] - ref[:p,:p])) < 1e-6), '{}-{}={}'.format (
        #        lib.fp (ham_pq[:p,:p]), lib.fp (ref[:p,:p]), lib.fp (ham_pq[:p,:p])-lib.fp (ref[:p,:p]))
        #except AssertionError as err:
        #    idx = np.argmax (np.abs (ham_pq[:p,:p]-ref[:p,:p]))
        #    print (lroots, idx, ham_pq[:p,:p].flat[idx], ref[:p,:p].flat[idx], (ham_pq[:p,:p]-ref[:p,:p]).flat[idx])
        #    raise (err)

        #try:
        #    assert (np.amax (np.abs (ham_pq - ref)) < 1e-6), '{}-{}={}'.format (
        #        lib.fp (ham_pq), lib.fp (ref), lib.fp (ham_pq)-lib.fp (ref))
        #except AssertionError as err:
        #    idx = np.argmax (np.abs (ham_pq-ref))
        #    print (lroots, idx, ham_pq.flat[idx], ref.flat[idx], ham_pq.flat[idx]-ref.flat[idx])
        #    raise (err)

        t1 = self.log.timer ('update_ham_pq', *t0)
        return ham_pq

    def op_ham_pq_ref (self, h1, h2, ci):
        '''Act the Hamiltonian on the reference CI vectors and project onto the current
        ground state of all but one active fragment, for each active fragment.

        Args:
            h1: ndarray of shape (2,nexc,nexc)
                1-electron part of the excited-fragment Hamiltonian
            h2: ndarray of shape [nexc,]*4
                2-electron part of the excited-fragment Hamiltonian
            ci: list of ndarray of shape (p,ndeta[i],ndetb[i])
                CI vectors of the active fragments in the P-space

        Returns:
            hci_f_pabq: list of ndarray of shape (p,ndeta[i],ndetb[i],q)
                Contains H|q>, projected onto <p| for all but one fragment, for each fragment.
                Vectors are multiplied by the sqrt of the weight of p.'''
        # TODO: point group symmetry
        t0 = lib.logger.process_clock (), lib.logger.perf_counter ()
        lroots = get_lroots (ci)
        nfrags = len (lroots)
        ci = [c.copy () for c in ci]
        # ZERO-STATE CLUDGE
        for ifrag in range (nfrags):
            if lroots[ifrag]==0:
                ci[ifrag] = np.zeros ([1,]+list(ci[ifrag].shape[1:]), dtype=ci[ifrag].dtype)
                ci[ifrag].flat[0] = 1.0
        excited_frags = [ifrag for ifrag in self.excited_frags]
        norb_f = [self.norb_ref[ifrag] for ifrag in excited_frags]
        nelec_f = np.asarray ([self.nelec_ref[ifrag] for ifrag in excited_frags])
        ci_fr_ket = [self.ci_ref[ifrag] for ifrag in excited_frags]
        nelec_frs_ket = np.zeros ((len (excited_frags), len (self.solvers_ref), 2), dtype=int)
        for iq, solver_ref in enumerate (self.solvers_ref):
            fcisolvers = [solver_ref.fcisolvers[ifrag] for ifrag in excited_frags]
            for ifrag, (s, n) in enumerate (zip (fcisolvers, nelec_f)):
                nelec_frs_ket[ifrag,iq,:] = self._get_nelec (s, n)
        ci_fr_bra = [[np.asarray (c)] for c in ci]
        nelec_rfs_bra = np.asarray ([[list(self._get_nelec (s, n))
                                     for s, n in zip (self.fcisolvers, nelec_f)]])
        nelec_frs_bra = nelec_rfs_bra.transpose (1,0,2)
        h_op = op[self.opt].contract_ham_ci
        with temporary_env (self, ncas_sub=norb_f, mol=self.fcisolvers[0].mol):
            hci_fr_pabq = h_op (self, h1, h2, ci_fr_ket, nelec_frs_ket, ci_fr_bra, nelec_frs_bra,
                                soc=0, orbsym=None, wfnsym=None)
        hci_f_pabq = [hc[0] for hc in hci_fr_pabq]
        # ZERO-STATE CLUDGE
        for ifrag in range (nfrags):
            if lroots[ifrag]!=0: continue
            for jfrag in range (nfrags):
                if ifrag==jfrag: continue
                hci = hci_f_pabq[jfrag]
                hci_f_pabq[jfrag] = np.zeros ([0,]+list(hci.shape[1:]), dtype=hci.dtype)
        t1 = self.log.timer ('op_ham_pq_ref', *t0)
        return hci_f_pabq

    def op_ham_pp_diag (self, h1, h2, ci, norb_f, nelec_f):
        ''' Act Hfrag[i]|ci[i]> for all fragments i 

        Args:
            h1: ndarray of shape (2,nexc,nexc)
                1-electron part of the excited-fragment Hamiltonian
            h2: ndarray of shape [nexc,]*4
                2-electron part of the excited-fragment Hamiltonian
            ci: list of ndarray of shape (p,ndeta[i],ndetb[i])
                CI vectors of the active fragments in the P-space
            norb_f: list of length (nfrag)
                Number of orbitals in excited fragments
            nelec_f: list of length (nfrag)
                Reference number of electrons in excited fragments

        Returns:
            hci: list of ndarray of shape (p,ndeta[i],ndetb[i])
                Hamiltonian-vector products
        '''
        t0 = lib.logger.process_clock (), lib.logger.perf_counter ()
        lroots = get_lroots (ci)
        nfrags = len (ci)
        nj = np.cumsum (norb_f)
        ni = nj - norb_f
        if h1.ndim < 3: h1 = np.stack ([h1,h1], axis=0)
        hci = []
        for ifrag in range (nfrags):
            nroots = lroots[ifrag]
            solver = self.fcisolvers[ifrag]
            if nroots == 0:
                na, nb = solver.transformer.ndeta, solver.transformer.ndetb
                hci.append (np.zeros ((0,na,nb)))
                continue
            i, j = ni[ifrag], nj[ifrag]
            norb, nelec = norb_f[ifrag], self._get_nelec (solver, nelec_f[ifrag])
            h1e = h1[:,i:j,i:j]
            h2e = h2[i:j,i:j,i:j,i:j]
            h2eff = solver.absorb_h1e (h1e, h2e, norb, nelec, 0.5)
            hc = []
            for iroot in range (nroots):
                c = ci[ifrag][iroot]
                hc.append (solver.contract_2e (h2eff, c, norb, nelec))
            hci.append (np.stack (hc, axis=0))
        t1 = self.log.timer ('op_ham_pp_diag', *t0)
        return hci

    def get_tdm1s_f (self, cibra, ciket, norb_f, nelec_f):
        '''Transition density matrices between the states of each fragment

        Args:
            cibra: list of ndarray of shape (pbra[i],ndeta[i],ndetb[i])
                Bra CI vectors
            ciket: list of ndarray of shape (pket[i],ndeta[i],ndetb[i])
                Ket CI vectors
            norb_f: list of length (nfrag)
                Number of orbitals in excited fragments
            nelec_f: list of length (nfrag)
                Reference number of electrons in excited fragments

        Returns:
            tdm1s_f: list of ndarray of shape (pbra[i],pket[i],2,norb_f[i],norb_f[i])
                Spin-separated 1-body transition density matrices between cibra and ciket
        '''
        t0 = lib.logger.process_clock (), lib.logger.perf_counter ()
        lroots_bra = get_lroots (cibra)
        lroots_ket = get_lroots (ciket)
        nfrags = len (cibra)
        tdm1s_f = []
        for ifrag in range (nfrags):
            solver = self.fcisolvers[ifrag]
            norb, nelec = norb_f[ifrag], self._get_nelec (solver, nelec_f[ifrag])
            nbra, nket = lroots_bra[ifrag], lroots_ket[ifrag]
            tdm1s = np.zeros ((nbra, nket, 2, norb, norb))
            b = cibra[ifrag]
            k = ciket[ifrag]
            for i, j in itertools.product (range (nbra), range (nket)):
                tdm1s[i,j,0], tdm1s[i,j,1] = trans_rdm1s (b[i], k[j], norb, nelec)
                tdm1s[i,j,0] = tdm1s[i,j,0].T
                tdm1s[i,j,1] = tdm1s[i,j,1].T
            assert (tdm1s.ndim==5)
            tdm1s_f.append (tdm1s)
        t1 = self.log.timer ('get_tdm1s_f', *t0)
        return tdm1s_f

    def eig1 (self, ham_pq, ci0, ovlp_thresh=1e-3):
        '''Diagonalize the coupled Hamiltonian for the lowest-energy eigensolution with substantial
        overlap on the reference state.

        Args:
            ham_pq: ndarray of shape (p+q,p+q)
                Hamiltonian matrix including both P and Q spaces
            ci0: list of ndarray of shape (p[i],ndeta[i],ndetb[i])
                CI vectors describing the P states of ham_pq

        Kwargs:
            ovlp_thresh: float
                Tolerance for identifying substantial overlap with the reference state

        Returns:
            e: float
                Total energy
            si: ndarray of shape (p+q,)
                SI vector corresponding to e
        '''

        lroots = get_lroots (ci0)
        p = np.prod (lroots)
        e, si = lowest_refovlp_eigpair (ham_pq, p=p, ovlp_thresh=ovlp_thresh, log=self.log)
        return e, si

    def schmidt_trunc (self, si, ci0, nroots=1):
        '''Perform the Schmidt decomposition on the P-space part of an si vector, truncate all but
        the highest nroots singular values, and correspondingly transform various intermediates.

        Args:
            si: ndarray of shape (p+?+q,)
                SI vector
            ci0: list of ndarray of shape (nroots+?,ndeta[i],ndetb[i])
                CI vectors describing the P states of ham_pq

        Kwargs:
            nroots: integer
                Number of roots for each fragment to retain; i.e., sqrt (p)

        Returns:
            disc_svals: ndarray of shape (p-nroots,)
                List of singular values discarded in the truncation
            u: ndarray of shape (nroots+?,nroots)
                nroots left-singular vectors
            si_p: ndarray of shape (nroots,)
                P-space part of the CI vector, in the Schmidt (diagonal) basis
            si_q: ndarray of shape (q,)
                Q-space part of the CI vector
            vh: ndarray of shape (nroots,nroots+?)
                nroots right-singular vectors
        '''
        t0 = lib.logger.process_clock (), lib.logger.perf_counter ()
        nfrags = len (ci0)
        assert (nfrags==2)
        lroots = get_lroots (ci0)
        p = np.prod (lroots)
        schmidt_vec = si[:p].reshape (lroots[1],lroots[0])
        u, svals, vh = linalg.svd (schmidt_vec)
        disc_svals = []
        if len (svals) > nroots:
            disc_svals = svals[nroots:]
            svals = svals[:nroots]
        u = u[:,:nroots]
        vh = vh[:nroots,:]
        si_p = svals
        si_q = si[p:]
        return disc_svals, u, si_p, si_q, vh

    def truncrot_ci (self, ci0, u, vh):
        ci1 = [np.tensordot (vh, ci0[0], axes=1),
               np.tensordot (u.conj ().T, ci0[1], axes=1)]
        return ci1

    def truncrot_ham_pq (self, ham_pq, u, vh):
        uh = u.conj ().T
        v = vh.conj ().T
        nroots = u.shape[1]
        lroots = [v.shape[0], u.shape[0]]
        p = np.prod (lroots)
        nstates = ham_pq.shape[1]
        h_pr = ham_pq[:p,:].reshape (lroots[1],lroots[0],nstates)
        h_pr = np.dot (uh, np.dot (vh, h_pr)).reshape (u.shape[1]*v.shape[1],nstates)
        ham_pq = np.append (h_pr, ham_pq[p:,:], axis=0)
        nstates = ham_pq.shape[0]
        h_rp = ham_pq[:,:p].reshape (nstates,lroots[1],lroots[0])
        h_rp = lib.einsum ('rij,ia,jb->rab', h_rp, u, v).reshape (nstates,u.shape[1]*v.shape[1])
        ham_pq = np.append (h_rp, ham_pq[:,p:], axis=1)
        return ham_pq

    def truncrot_hci_qspace (self, hci0_qspace, u, vh):
        hci1_qspace = [None, None]
        hci1_qspace[0] = np.tensordot (u.T, hci0_qspace[0], axes=1)
        hci1_qspace[1] = np.tensordot (vh.conj(), hci0_qspace[1], axes=1)
        return hci1_qspace

    def truncrot_hci_pspace_diag (self, hci0_pspace_diag, u, vh):
        hci1_pspace_diag = [None, None]
        hci1_pspace_diag[0] = np.tensordot (vh.conj(), hci0_pspace_diag[0], axes=1)
        hci1_pspace_diag[1] = np.tensordot (u.T, hci0_pspace_diag[1], axes=1)
        return hci1_pspace_diag

    def truncrot_tdm1s_f (self, tdm1s_f0, u, vh):
        tdm1s_f1 = [None, None]
        tdm1s_f1[0] = lib.einsum ('mu,uvsij,nv->mnsij', vh.conj (), tdm1s_f0[0], vh)
        tdm1s_f1[1] = lib.einsum ('um,uvsij,vn->mnsij', u, tdm1s_f0[1], u.conj ())
        return tdm1s_f1

    def get_hpq_xq (self, hci_f_pabq, ci0, si_q):
        '''Generate the P-row, Q-column part of the Hamiltonian-vector product projected into P'

        Args:
            hci_f_pabq: list of ndarrays of shape (nroots,ndeta[i],ndetb[i],q)
                H|q> projected on <p0| for all but one fragment, where <p0| is the first few
                vectors of CI, i.e., the output of op_ham_pq_ref for ci0.
            ci0: list of ndarrays of shape (nroots,ndeta[i],ndetb[i])
                CI vectors of the p-space
            si_q: ndarray of shape (nq,)
                Q-space part of the SI vector

        Returns:
            hci_f_pab: list of ndarrays of shape (nroots,ndeta[i],ndetb[i])
                Hamiltonian-vector product component, projected orthogonal to ci0
        '''
        t0 = lib.logger.process_clock (), lib.logger.perf_counter ()
        lroots = get_lroots (ci0)
        p = np.prod (lroots)
        hci_f_pab = []
        for ci, hci_pabq in zip (ci0, hci_f_pabq):
            hci_pab = np.dot (hci_pabq, si_q)
            hci_pr = np.tensordot (hci_pab, ci.conj (), axes=((1,2),(1,2)))
            hci_pab -= np.tensordot (hci_pr, ci, axes=1)
            hci_f_pab.append (hci_pab)
        t1 = self.log.timer ('get_hpq_xq', *t0)
        return hci_f_pab

    def get_hpp_xp (self, ci0, si_p, hci_pspace_diag, h0, h2, tdm1s_f, norb_f, nelec_f):
        '''Generate the P-row, P-column part of the Hamiltonian-vector product projected into P'

        Args:
            ci0: list of ndarrays of shape (nroots,ndeta[i],ndetb[i])
                CI vectors of the p-space in the Schmidt basis
            si_p: ndarray of shape (nroots)
                SI vector for the current P-space (requires Schmidt basis)
            hci_pspace_diag: list of ndarray of shape (p0,ndeta,ndetb)
                H(ifrag)|p0(ifrag)>, where <p0| is the vectors of ci0; i.e., the output of
                op_ham_pp_diag for ci0.
            h0: float
                Constant part of the excited-fragment Hamiltonian
            h2: ndarray of shape [nexc,]*4
                2-electron part of the excited-fragment Hamiltonian
            tdm1s_f: list of ndarray of shape (nroots,nroots,2,norb[i],norb[i])
                Spin-separated transition one-body density matrices within each fragment for ci0
            norb_f: list of length (nfrag)
                Number of orbitals in excited fragments
            nelec_f: list of length (nfrag)
                Reference number of electrons in excited fragments

        Returns:
            hci_f_pab: list of ndarrays of shape (nroots,ndeta[i],ndetb[i])
                Hamiltonian-vector product component, projected orthogonal to ci0
        '''
        t0 = lib.logger.process_clock (), lib.logger.perf_counter ()
        nfrags = len (ci0)
        lroots = get_lroots (ci0)
        assert (nfrags==2)
        assert (lroots[0]==lroots[1])
        nj = np.cumsum (norb_f)
        ni = nj - norb_f
        hci_f_pab = []
        e_pspace_diag = np.zeros ((nfrags, lroots[0]))
        for ifrag in range (nfrags):
            e = [np.dot (c.conj ().flat, hc.flat) for c, hc 
                 in zip (ci0[ifrag], hci_pspace_diag[ifrag])]
            e_pspace_diag[ifrag] = e[:]
        e_pspace_diag = e_pspace_diag.sum (0) - e_pspace_diag
        hci_f_pab = []
        for ifrag in range (nfrags):
            c = ci0[ifrag]
            hc = hci_pspace_diag[ifrag]
            e = e_pspace_diag[ifrag] + h0
            hc = si_p[:,None,None] * (hc + e[:,None,None]*c)
            hci_f_pab.append (hc)
        for ifrag, jfrag in itertools.permutations (range (nfrags), 2):
            c = ci0[ifrag]
            hc = hci_f_pab[ifrag]
            nroots = lroots[ifrag]
            solver = self.fcisolvers[ifrag]
            norb, nelec = norb_f[ifrag], self._get_nelec (solver, nelec_f[ifrag])
            tdm1s = tdm1s_f[jfrag]
            i0, i1 = ni[ifrag], nj[ifrag]
            j0, j1 = ni[jfrag], nj[jfrag]
            h2_j = h2[i0:i1,i0:i1,j0:j1,j0:j1]
            h2_k = h2[i0:i1,j0:j1,j0:j1,i0:i1].transpose (0,3,2,1)
            vj = np.tensordot (tdm1s, h2_j, axes=((-2,-1),(-2,-1)))
            vk = np.tensordot (tdm1s, h2_k, axes=((-2,-1),(-2,-1)))
            veff = vj.sum (2)[:,:,None,:,:] - vk
            # This is the part that requires the asserts at top
            # Generalizing it requires substantial refactoring of this entire file
            for iroot, jroot in itertools.product (range (nroots), repeat=2):
                v = veff[iroot,jroot]
                vc = direct_nosym_uhf.contract_1e (v, c[jroot], norb, nelec)
                hc[iroot] += si_p[jroot] * vc
            hci_f_pab[ifrag] = hc
        for ifrag in range (nfrags):
            c = ci0[ifrag]
            hc = hci_f_pab[ifrag]
            nroots = lroots[ifrag]
            chc = np.dot (np.asarray (c).reshape (nroots,-1).conj (),
                          np.asarray (hc).reshape (nroots,-1).T).T
            hc = hc - np.tensordot (chc, c, axes=1)
            hci_f_pab[ifrag] = hc
        t1 = self.log.timer ('get_hpp_xp', *t0)
        return hci_f_pab

    def _get_grad (self, ci1, si_p, hpq_xq, hpp_xp, nroots=None):
        ''' Compute the gradient of the target interacting energy

        Args:
            ci0: list of ndarray of shape (nroots,ndeta[i],ndetb[i])
                CI vectors
            si_p: ndarray of shape (p,)
                SI vector in Schmidt basis
            hpq_xq: list of ndarray of shape (nroots,ndeta[i],ndetb[i])
                P-row, Q-column part of the Hamiltonian-vector product
            hpp_xp: list of ndarray of shape (nroots,ndeta[i],ndetb[i])
                P-row, P-column part of the Hamiltonian-vector product

        Returns:
            grad: flat ndarray
                Gradient with respect to all nonredundant parameters
        '''
        t0 = lib.logger.process_clock (), lib.logger.perf_counter ()
        grad_ext = []
        grad_int = []
        for solver, c, hc1, hc2 in zip (self.fcisolvers, ci1, hpq_xq, hpp_xp):
            if nroots is None: nroots = solver.nroots
            hc = si_p[:,None,None] * (hc1 + hc2)
            if isinstance (solver, CSFFCISolver):
                c = solver.transformer.vec_det2csf (c, normalize=True)
                hc = solver.transformer.vec_det2csf (hc, normalize=False)
            chc = np.dot (c.conj (), hc.T)
            hc = hc - np.dot (chc.T, c)
            grad_ext.append (hc.flat)
            grad_int.append (chc)
        grad_int[0] -= grad_int[1].T
        grad_int[1] = -grad_int[0].T
        grad = []
        for i, e in zip (grad_int, grad_ext):
            grad.append (e)
            if nroots>1:
                grad.append (i[np.tril_indices (nroots, k=-1)])
        t1 = self.log.timer ('ExcitationPSFCISolver _get_grad', *t0)
        return np.concatenate (grad)

    def get_new_vecs (self, ci0, hpq_xq, hpp_xp, nroots=1):
        ''' Extend the P space: P + P' -> P

        Args:
            ci0: list of ndarray of shape (nroots,ndeta[i],ndetb[i])
                CI vectors
            hpq_xq: list of ndarray of shape (nroots,ndeta[i],ndetb[i])
                P-row, Q-column part of the Hamiltonian-vector product
            hpp_xp: list of ndarray of shape (nroots,ndeta[i],ndetb[i])
                P-row, P-column part of the Hamiltonian-vector product

        Kwargs:
            nroots: integer
                Number of states in each fragment

        Returns:
            ci1: list of ndarray of shape (nroots+?,ndeta[i],ndetb[i])
                CI vectors beginning with ci0, followed by new vectors
        '''

        t0 = lib.logger.process_clock (), lib.logger.perf_counter ()
        ci1 = []
        lroots = get_lroots (ci0)
        for s, c0, c1, c2, nx0 in zip (self.fcisolvers, ci0, hpq_xq, hpp_xp, lroots):
            x = np.concatenate ([c0,c1,c2],axis=0)
            if isinstance (s, CSFFCISolver):
                ndeta, ndetb = s.transformer.ndeta, s.transformer.ndetb
                x = s.transformer.vec_det2csf (x)
                x0 = s.transformer.vec_det2csf (c0)
            else:
                nx, ndeta, ndetb = x.shape
                x = x.reshape (nx, ndeta*ndetb)
                x0 = c0.reshape (len (c0), ndeta*ndetb)
            # normalize vecs
            x_norm = linalg.norm (x, axis=1)
            x = x[x_norm>0,:]
            x_norm = x_norm[x_norm>0]
            x /= x_norm[:,None]
            # canonical orthogonalization
            ovlp = x.conj () @ x.T
            evals, evecs = linalg.eigh (ovlp)
            x = canonical_orth_(ovlp).T @ x
            nx = len (x)
            # line up with c0
            ovlp = x0.conj () @ x.T
            u, svals, vh = linalg.svd (ovlp, full_matrices=True)
            vh[:nx0] = u @ vh[:nx0]
            x = vh[nx0:] @ x
            if isinstance (s, CSFFCISolver):
                assert (nx <= s.transformer.ncsf), '{}'.format (evals)
                x = s.transformer.vec_csf2det (x)
            assert (x.shape[0]+nx0 >= nroots), '{} {} {} {} {} {}'.format (
                x.shape, nx0, nroots, c0.shape, c1.shape, c2.shape
            )
            ci1.append (x.reshape (nx-nx0, ndeta, ndetb))
        t1 = self.log.timer ('get_new_vecs', *t0)
        return ci1

    def get_init_guess (self, ci0, norb_f, nelec_f, h1, h2, nroots=None):
        ''' Make sure initial guess is orthonormal '''
        ci0 = ProductStateFCISolver.get_init_guess (
            self, ci0, norb_f, nelec_f, h1, h2, nroots=nroots
        )
        for ifrag in range (len (ci0)):
            s = self.fcisolvers[ifrag]
            if isinstance (s, CSFFCISolver):
                ndeta, ndetb = s.transformer.ndeta, s.transformer.ndetb
                x0 = s.transformer.vec_det2csf (ci0[ifrag])
            else:
                nx, ndeta, ndetb = ci0[ifrag].shape
                x0 = ci0[ifrag].reshape (nx, ndeta*ndetb)
            ovlp = x0.conj () @ x0.T
            x = canonical_orth_(ovlp).T @ x0
            nx = len (x)
            ovlp = x0.conj () @ x.T
            u, svals, vh = linalg.svd (ovlp)
            k = len (svals)
            vh[:k,:k] = u[:k,:k] @ vh[:k,:k]
            x = vh @ x
            if isinstance (s, CSFFCISolver):
                ci0[ifrag] = s.transformer.vec_csf2det (x).reshape (nx, ndeta, ndetb)
            else:
                ci0[ifrag] = x.reshape (nx, ndeta, ndetb)
        return ci0



