import copy
import numpy as np
from scipy import linalg
from pyscf.fci import cistring
from pyscf.fci.direct_spin1 import _unpack_nelec
from pyscf.mcscf.addons import StateAverageFCISolver
from mrh.my_pyscf.mcscf.productstate import ProductStateFCISolver, ImpureProductStateFCISolver, state_average_fcisolver
from mrh.my_pyscf.fci import csf_solver, CSFFCISolver
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

def lowest_refovlp_eigval (ham_pq, ovlp_thresh=LOWEST_REFOVLP_EIGVAL_THRESH):
    ''' Return the lowest eigenvalue of the matrix ham_pq, whose corresponding
    eigenvector has nonzero overlap with the first basis function. '''
    e_all, u_all = linalg.eigh (ham_pq)
    w = u_all[0,:].conj () * u_all[0,:]
    idx_valid = w > ovlp_thresh
    e_valid = e_all[idx_valid]
    u_valid = u_all[:,idx_valid]
    idx_choice = np.argmin (e_valid)
    return e_valid[idx_choice]

def sort_ci0 (obj, ham_pq, ci0):
    '''Prepare guess CI vectors, guess energy, and Q-space Hamiltonian eigenvalues
    and eigenvectors. Sort ci0 so that the ENV |00...0> is the state with the
    minimum guess energy for the downfolded eigenproblem

    (h_pp + h_pq (e0 - e_q)^-1 h_qp) |p> = e0|p>

    NOTE: in the current LASSIS method, this isn't how the converged vectors should be
    sorted: this arrangement only for the initial guess.

    Args:
        ham_pq: ndarray of shape (p+q,p+q)
            Hamiltonian matrix in model-space basis. In the p-space, ENVs ascend in
            column-major order: |00...0>, |10...0>, |20...0>, ... |0100...0>, |1100...0>, ...
        ci0: list of ndarray
            CI vectors for the active fragments

    Returns:
        ci0: list of ndarray
            Resorted on each fragment so that |00...0> has the lowest downfolded
            guess energy
        e0_p: float
            Downfolded guess energy of |00...0>
        ham_pq: ndarray of shape (p+q,p+q)
            Copy of the input matrix sorted so that |00...0> is the first basis state and
            individual fragment states "i" are sorted in ascending order of the energy of
            |00...0i00...0>.'''
    # Find lowest-energy ENV, including VRV contributions
    log = lib.logger.new_logger (obj, obj.verbose)
    lroots = get_lroots (ci0)
    p = np.prod (lroots)
    h_pp = ham_pq[:p,:p]
    h_pq = ham_pq[:p,p:]
    h_qq = ham_pq[p:,p:]
    q = ham_pq.shape[-1] - p
    e_q, si_q = linalg.eigh (h_qq)
    def project_1p (ip):
        idx = np.ones (len (ham_pq), dtype=bool)
        idx[:p] = False
        idx[ip] = True
        return ham_pq[idx,:][:,idx]
    e_p = np.array ([lowest_refovlp_eigval (project_1p (i)) for i in range (p)])
    idxmin = np.argmin (e_p)
    e0_p = e_p[idxmin]
    h_pq = np.dot (h_pq, si_q)
                
    # ENV index to address
    idx = idxmin
    addr = []
    for ifrag, lroot in enumerate (lroots):
        idx, j = divmod (idx, lroot)
        addr.append (j)

    # Sort against this reference state
    nfrag = len (addr)
    e_p_arr = e_p.reshape (*lroots[::-1]).T
    h_pp = h_pp.reshape (*(list(lroots[::-1])*2))
    h_pq = ham_pq[:p,p:].reshape (*(list(lroots[::-1])+[q,]))
    ci1 = [c.copy () for c in ci0]
    for ifrag in range (nfrag):
        if lroots[ifrag]<2: continue
        e_p_slice = e_p_arr
        for jfrag in range (ifrag):
            e_p_slice = e_p_slice[addr[jfrag]]
        for jfrag in range (ifrag+1,nfrag):
            e_p_slice = e_p_slice[:,addr[jfrag]]
        sort_idx = np.argsort (e_p_slice)
        assert (sort_idx[0] == addr[ifrag])
        ci1[ifrag] = np.stack ([ci1[ifrag][i] for i in sort_idx], axis=0)
        dimorder = list(range(h_pp.ndim))
        dimorder.insert (0, dimorder.pop (nfrag-(1+ifrag)))
        dimorder.insert (1, dimorder.pop (2*nfrag-(1+ifrag)))
        h_pp = h_pp.transpose (*dimorder)
        h_pp = h_pp[sort_idx,...][:,sort_idx,...]
        dimorder = np.argsort (dimorder)
        h_pp = h_pp.transpose (*dimorder)
        dimorder = list(range(h_pq.ndim))
        dimorder.insert (0, dimorder.pop (nfrag-(1+ifrag)))
        h_pq = h_pq.transpose (*dimorder)
        h_pq = h_pq[sort_idx,...]
        dimorder = np.argsort (dimorder)
        h_pq = h_pq.transpose (*dimorder)
    h_pp = h_pp.reshape (p, p)
    h_pq = h_pq.reshape (p, q)
    ham_pq = np.block ([[h_pp, h_pq],[h_pq.conj ().T, h_qq]])
    return ci1, e0_p, ham_pq

class _vrvloop_env (object):
    def __init__(self, fciobj, vrvsolvers, e_q, si_q):
        self.fciobj = fciobj
        self.vrvsolvers = vrvsolvers
        self.e_q = e_q
        self.si_q = si_q
    def __enter__(self):
        self.fciobj.fcisolvers = self.vrvsolvers
        self.fciobj._e_q = self.e_q
        self.fciobj._si_q = self.si_q
    def __exit__(self, type, value, traceback):
        self.fciobj.revert_vrvsolvers_()

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
        self._deactivate_vrv = False # for testing
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

    def kernel (self, h1, h2, ecore=0, ci0=None,
                conv_tol_grad=1e-4, conv_tol_self=1e-6, max_cycle_macro=50,
                serialfrag=False, _add_vrv_energy=False, **kwargs):
        h0, h1, h2 = self.get_excited_h (ecore, h1, h2)
        norb_f = np.asarray ([self.norb_ref[ifrag] for ifrag in self.excited_frags])
        nelec_f = np.asarray ([self.nelec_ref[ifrag] for ifrag in self.excited_frags])
        orbsym = self.orbsym_ref
        if orbsym is not None:
            idx = self.get_excited_orb_idx ()
            orbsym = [orbsym[iorb] for iorb in range (norb_tot) if idx[iorb]]
        # TODO: point group symmetry; I probably also have to do something to wfnsym
        ci0, vrvsolvers, e_q, si_q = self.prepare_vrvsolvers_(h0, h1, h2, ci0=ci0)
        with _vrvloop_env (self, vrvsolvers, e_q, si_q):
            converged, energy_elec, ci1_active = ProductStateFCISolver.kernel (
                self, h1, h2, norb_f, nelec_f, ecore=h0, ci0=ci0, orbsym=orbsym,
                conv_tol_grad=conv_tol_grad, conv_tol_self=conv_tol_self,
                max_cycle_macro=max_cycle_macro, serialfrag=serialfrag, **kwargs
            )
            if _add_vrv_energy: # for a sanity check in unittests only
                energy_elec += self._energy_vrv (h1, h2, ci1_active)
        ci1 = [c for c in self.ci_ref]
        for ifrag, c in zip (self.excited_frags, ci1_active):
            ci1[ifrag] = np.asarray (c)
        return converged, energy_elec, ci1

    def project_hfrag (self, h1, h2, ci, norb_f, nelec_f, ecore=0, dm1s=None, dm2=None, **kwargs):
        h1eff, h0eff, ci = ProductStateFCISolver.project_hfrag (
            self, h1, h2, ci, norb_f, nelec_f, ecore=ecore, dm1s=dm1s, dm2=dm2, **kwargs
        )
        nj = np.cumsum (norb_f)
        ni = nj - norb_f
        # Project the part coupling the p and q rootspaces
        ci1 = [c for c in ci]
        if len (self._e_q) and not self._deactivate_vrv:
            ci0 = [np.asarray (c) for c in ci]
            hci_f_pabq = self.op_ham_pq_ref (h1, h2, ci0)
            zipper = zip (hci_f_pabq, self.fcisolvers, ci0, norb_f, nelec_f, h1eff, h0eff, ni, nj)
            for ifrag, (hci_pabq, solver, c, norb, nelec, h1e, h0e, i, j) in enumerate (zipper):
                h2e = h2[i:j,i:j,i:j,i:j]
                ne = self._get_nelec (solver, nelec)
                solver.v_qpab = np.tensordot (self._si_q, hci_pabq, axes=((0),(-1)))
                e0, ci1[ifrag] = solver.sort_ci (h0e, h1e, h2e, norb, ne, c)
                solver.denom_q = e0 - solver.e_q
                # The reason I do the above two lines here and not in the fragment-solver kernel is
                # that between this function and the start of the fragment-solver kernel, the
                # wrapper needs to compute the current gradient for the whole system. The fragment
                # solvers need to already know the correct e0 for that gradient calculation.
        return h1eff, h0eff, ci1

    def _energy_vrv (self, h1, h2, ci):
        # The only purpose this serves is for a sanity test in test_excitations.py
        # It can be reattached to energy_elec if I can figure out how to define a whole-system
        # energy for this step in LASSIS that makes sense.
        # If so, I should rewrite it so that it can be called outside of this class's kernel
        # function
        if (len (self._e_q) == 0) or self._deactivate_vrv: return 0
        ci0 = []
        denom_q = 0
        for c, solver in zip (ci, self.fcisolvers):
            t = solver.transformer
            c = np.asarray (c).reshape (-1, t.ndeta, t.ndetb)
            ci0.append (c)
            denom_q += solver.denom_q
        denom_q /= len (ci)
        lroots = get_lroots (ci0)
        p = np.prod (lroots)
        ham_pq = self.get_ham_pq (0, h1, h2, ci0)
        h_pp = ham_pq[:p,:p]
        h_pq = ham_pq[:p,p:]
        h_qq = ham_pq[p:,p:]
        h_qq = self._si_q.conj ().T @ h_qq @ self._si_q
        h_pq = np.dot (h_pq, self._si_q)
        idx = np.abs (denom_q) > 1e-16
        e_p = np.diag (np.dot (h_pq[:,idx].conj () / denom_q[None,idx], h_pq[:,idx].T))
        e_p = e_p.reshape (*lroots[::-1]).T
        for solver in self.fcisolvers:
            if hasattr (getattr (solver, 'weights', None), '__len__'):
                e_p = np.dot (solver.weights, e_p)
            else:
                e_p = e_p[0]
        return e_p

    def get_ham_pq (self, h0, h1, h2, ci_p):
        '''Build the model-space Hamiltonian matrix for the current state of the P-space.

        Args:
            h0: float
                Constant part of the excited-fragment Hamiltonian
            h1: ndarray of shape (2,nexc,nexc)
                Spin-separated 1-electron part of the excited-fragment Hamiltonian
            h2: ndarray of shape [nexc,]*4
                2-electron part of the excited-fragment Hamiltonian
            ci_p: list of ndarray
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
                                                   orbsym=self.orbsym_ref, wfnsym=self.wfnsym_ref)
        t1 = self.log.timer ('get_ham_pq', *t0)
        return ham_pq + (h0*ovlp_pq)

    def op_ham_pq_ref (self, h1, h2, ci):
        '''Act the Hamiltonian on the reference CI vectors and project onto the current
        ground state of all but one active fragment, for each active fragment.

        Args:
            h1: ndarray of shape (2,nexc,nexc)
                1-electron part of the excited-fragment Hamiltonian
            h2: ndarray of shape [nexc,]*4
                2-electron part of the excited-fragment Hamiltonian
            ci: list of ndarray
                CI vectors of the active fragments in the P-space

        Returns:
            hci_f_pabq: list of ndarray
                Contains H|q>, projected onto <p| for all but one fragment, for each fragment.
                Vectors are multiplied by the sqrt of the weight of p.'''
        # TODO: point group symmetry
        t0 = lib.logger.process_clock (), lib.logger.perf_counter ()
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
        # weights for p
        weights = getattr (self.fcisolvers[0], 'weights', np.array ([1.0]))
        for s in self.fcisolvers[1:]:
            w = getattr (s, 'weights', np.array ([1.0]))
            weights = np.multiply.outer (w, weights)
        hci_f_pabq = []
        for ifrag, hc in enumerate (hci_fr_pabq):
            # column-major order
            w = weights.sum (-ifrag-1).ravel ()
            assert (np.count_nonzero (w<0)==0)
            idx = w > 0
            hc = hc[0][idx] * np.sqrt (w[idx])[:,None,None,None]
            hci_f_pabq.append (hc)
        t1 = self.log.timer ('op_ham_pq_ref', *t0)
        return hci_f_pabq

    def sort_ci0 (self, ham_pq, ci0):
        return sort_ci0 (self, ham_pq, ci0)[:2]

    def prepare_vrvsolvers_(self, h0, h1, h2, ci0=None):
        do_sort_ci0 = (ci0 is None)
        norb_f = np.asarray ([self.norb_ref[ifrag] for ifrag in self.excited_frags])
        nelec_f = np.asarray ([self.nelec_ref[ifrag] for ifrag in self.excited_frags])
        ci0 = self.get_init_guess (ci0, norb_f, nelec_f, h1, h2)
        ham_pq = self.get_ham_pq (h0, h1, h2, ci0)
        p = np.prod (get_lroots (ci0))
        h_qq = ham_pq[p:,p:]
        e_q, si_q = linalg.eigh (h_qq)
        ci0_sorted, e0 = self.sort_ci0 (ham_pq, ci0)
        if do_sort_ci0: ci0 = ci0_sorted
        vrvsolvers = []
        for ix, solver in enumerate (self.fcisolvers):
            vrvsolvers.append (vrv_fcisolver (solver, e0, e_q, None, max_cycle_e0=MAX_CYCLE_E0,
                                              crash_locmin=self.crash_locmin))
        return ci0, vrvsolvers, e_q, si_q

    def revert_vrvsolvers_(self):
        self._e_q = []
        self._si_q = []

def vrvsolver_single_kernel (fciobj, h1e, h2e, norb, nelec, ecore=0, ci0=None, orbsym=None,
                             **kwargs):
    log = lib.logger.new_logger (fciobj, fciobj.verbose)
    max_cycle_e0 = fciobj.max_cycle_e0
    conv_tol_e0 = fciobj.conv_tol_e0
    e0_last = 0
    converged = False
    ket = ci0[0] if fciobj.nroots>1 else ci0
    e0 = fciobj.solve_e0 (ecore, h1e, h2e, norb, nelec, ket)
    ci1 = ci0
    fciobj.denom_q = e0 - fciobj.e_q
    log.debug ("Self-energy singularities in VRVSolver: {}".format (fciobj.e_q))
    log.debug ("e0 = %.8g", e0)
    log.debug ("Denominators in VRVSolver: {}".format (fciobj.denom_q))
    fciobj.test_locmin (e0, ci1, norb, nelec, ecore, h1e, h2e, warntag='Saddle-point initial guess')
    h2eff = fciobj.absorb_h1e (h1e, h2e, norb, nelec, 0.5)
    for it in range (max_cycle_e0):
        e, ci1 = fciobj.undressed_kernel (
            h1e, h2e, norb, nelec, ecore=ecore, ci0=ci1, orbsym=orbsym, **kwargs
        )
        # Subtract the vrv energy so that agreement between different fragments can
        # be checked in the impure-state case
        if isinstance (e, (list,tuple,np.ndarray)):
            for i in range (len (e)):
                hci = fciobj.undressed_contract_2e (h2eff, ci1[i], norb, nelec)
                e[i] = ecore + np.dot (ci1[i].ravel (), hci.ravel ())
        else:
            hci = fciobj.undressed_contract_2e (h2eff, ci1, norb, nelec)
            e = ecore + np.dot (ci1.ravel (), hci.ravel ())
        e0_last = e0
        e0 = fciobj.solve_e0 (ecore, h1e, h2e, norb, nelec, ket)
        fciobj.denom_q = e0 - fciobj.e_q
        log.debug ("e0 = %.8g", e0)
        log.debug ("Denominators in VRVSolver: {}".format (fciobj.denom_q))
        if abs(e0-e0_last)<conv_tol_e0:
            converged = True
            break
    fciobj.test_locmin (e0, ci1, norb, nelec, ecore, h1e, h2e)
    fciobj.converged = (converged and np.all (fciobj.converged))
    return e, ci1

class single_root_env:
    def __init__(self, fciobj, i, ci0, ci1):
        self.fciobj = fciobj
        self.i = i
        self.ci0 = ci0
        self.ci1 = ci1
        self.old_nroots = fciobj.nroots
        self.old_v_qpab = fciobj.v_qpab
        self.old_q_qab = fciobj.q_qab
    def __enter__(self):
        i = self.i
        self.fciobj.nroots = 1
        self.fciobj.v_qpab = self.old_v_qpab[:,i:i+1,...]
        ci0 = self.ci0[i]
        ci1 = np.asarray (self.ci1[:i])
        if i>0:
            self.fciobj.q_qab = ci1
            ci0 = self.fciobj.project_2e (ci0)
            ci0 /= linalg.norm (ci0)
        return ci0
    def __exit__(self, type, value, traceback):
        self.fciobj.nroots = self.old_nroots
        self.fciobj.v_qpab = self.old_v_qpab
        self.fciobj.q_qab = self.old_q_qab

def vrvsolver_loop_kernel (fciobj, h1e, h2e, norb, nelec, ecore=0, ci0=None, orbsym=None,
                           **kwargs):
    e = []
    ci1 = []
    conv = []
    for i in range (nroots):
        with single_root_env (fciobj, i, ci0, ci1) as ci0_i:
            ei, ci1i = vrvsolver_single_kernel (fciobj, h1e, h2e, norb, nelec, ecore=ecore,
                                                ci0=ci0_i, orbsym=orbsym, **kwargs)
            e.append (ei)
            ci1.append (ci1i)
            conv.append (fciobj.converged)
    fciobj.converged = all (conv)
    return e, ci1

class VRVDressedFCISolver (object):
    '''Minimize the energy of a wave function of the form

    |Psi> = |P> + sum_i |Q(i)>

    expressed in intermediate normalization <P|Psi> = 1, <P|Q(i)> = 0,
    <Q(i)|Q(j)> = delta_ij, with {|Q(i)>} fixed, using self-energy downfolding:

    _______________     _______
    |      |      |     |     |
    | H_PP | V_PQ |     | |P> |
    |      |      |     |     |
    --------------- = E -------
    |      |      |     |     |
    | V_QP | H_QQ |     | |Q> |
    |      |      |     |     |
    ---------------     -------
  
    ------>

    (H_PP + V_PQ (E - H_QQ)^-1 V_QP) |P> = E|P>

    The inverted quantity is sometimes called the "resolvent," R,

    (H_PP + V_PQ R_QQ(E) V_QP) |P> = E|P>

    hence the term "VRV".

    The self-consistency is only determined for the lowest energy. H_QQ is assumed
    to be diagonal for simplicity.

    Additional attributes:
        v_qpab: ndarray of shape (nq, np, ndeta, ndetb)
            Contains the CI vector V_PQ |Q> in the |P> Hilbert space
        e_q: ndarray of shape (nq,)
            Eigenenergies of the QQ sector of the Hamiltonian
        denom_q: ndarray of shape (nq,)
            Contains E-e_q with the current guess E solution of the self-consistent
            eigenproblem
        q_qab: ndarray of shape (nq, ndeta, ndetb)
            Intermediate used to project orthogonal to previous vectors (req'd due to nonlinear H)
        max_cycle_e0: integer
            Maximum number of cycles allowed to attempt to converge the self-consistent
            eigenproblem
        conv_tol_e0: float
            Convergence threshold for the self-consistent eigenenergy
    '''
    _keys = {'contract_vrv', 'base', 'v_qpab', 'denom_q', 'e_q', 'max_cycle_e0', 'conv_tol_e0',
             'charge', 'crash_locmin', 'imag_shift'}
    def __init__(self, fcibase, my_vrv, my_eq, my_e0, max_cycle_e0=MAX_CYCLE_E0,
                 conv_tol_e0=CONV_TOL_E0, crash_locmin=False):
        self.base = copy.copy (fcibase)
        if isinstance (fcibase, StateAverageFCISolver):
            self._undressed_class = fcibase._base_class
        else:
            self._undressed_class = fcibase.__class__
        self.__dict__.update (fcibase.__dict__)
        self.denom_q = 0
        self.imag_shift = IMAG_SHIFT
        self.e_q = my_eq
        self.v_qpab = my_vrv
        self.q_qab = None
        self.max_cycle_e0 = max_cycle_e0
        self.conv_tol_e0 = conv_tol_e0
        self.crash_locmin = crash_locmin
    def project_2e (self, ci):
        if self.q_qab is None or self.q_qab.shape[0]==0: return ci
        ovlp = np.tensordot (self.q_qab.conj (), ci, axes=((-2,-1),(-2,-1)))
        qci = ci - np.tensordot (ovlp, self.q_qab, axes=((0),(0)))
        return qci
    def get_projectors (self, h1e, eri, norb, nelec, link_index=None):
        h2eff = self.absorb_h1e (h1e, eri, norb, nelec, 0.5)
        if self.q_qab is None or self.q_qab.shape[0]==0: return self.q_qab, None, None
        hq_qab = np.stack ([self.contract_2e (h2eff, c, norb, nelec, link_index=link_index,
                                              project_q=False)
                            for c in self.q_qab], axis=0)
        ham_q = np.tensordot (self.q_qab.conj (), hq_qab, axes=((-2,-1),(-2,-1)))
        return self.q_qab, hq_qab, ham_q
    def project_hdiag (self, q_qr, hq_qr, ham_q):
        if q_qr is None: return 0
        hdiag_shape = q_qr.shape[1:]
        nq = q_qr.shape[0]
        q_qr = q_qr.reshape (nq, -1)
        hq_qr = hq_qr.reshape (nq, -1)
        hdiag = (np.dot (ham_q, q_qr) * q_qr.conj ())
        hdiag -= (q_qr.conj () * hq_qr + q_qr * hq_qr.conj ())
        return hdiag.sum (0).reshape (hdiag_shape)
    def project_pspace (self, addr, q_qr, hq_qr, ham_q):
        if q_qr is None: return 0
        nq = q_qr.shape[0]
        q_qr = q_qr.reshape (nq, -1)[:,addr]
        hq_qr = hq_qr.reshape (nq, -1)[:,addr]
        h0 = np.dot (q_qr.conj ().T, np.dot (ham_q, q_qr))
        h0 -= np.dot (q_qr.conj ().T, hq_qr)
        h0 -= np.dot (hq_qr.conj ().T, q_qr)
        return h0
    def contract_2e(self, eri, fcivec, norb, nelec, link_index=None, v_qpab=None, denom_q=None,
                    project_q=True, **kwargs):
        if project_q: fcivec = self.project_2e (fcivec)
        ci0 = self.undressed_contract_2e (eri, fcivec, norb, nelec, link_index, **kwargs)
        ci0 += self.contract_vrv (fcivec, v_qpab=v_qpab, denom_q=denom_q)
        if project_q: ci0 = self.project_2e (ci0)
        return ci0
    def contract_vrv (self, ket, v_qpab=None, denom_q=None):
        if v_qpab is None: v_qpab = self.v_qpab
        if denom_q is None: denom_q = self.denom_q
        if v_qpab is None: return np.zeros_like (ket)
        ket_shape = ket.shape
        idx = np.abs (denom_q) > 1e-16
        p = v_qpab.shape[1]
        q = np.count_nonzero (idx)
        if (not q) or (not p): return np.zeros_like (ket)
        v_qpab, denom_q = v_qpab[idx].reshape (q,p,-1), denom_q[idx]
        denom_q = denom_q + 1j*self.imag_shift
        denom_fac_q = np.real (1.0 / denom_q)
        rv_qp = np.ravel (np.dot (v_qpab.conj (), ket.ravel ()) * denom_fac_q[:,None])
        hket = np.dot (rv_qp, v_qpab.reshape(p*q,-1)).reshape (ket_shape)
        return hket
    def test_locmin (self, e0, ci, norb, nelec, h0e, h1e, h2e, warntag='Apparent local minimum'):
        log = lib.logger.new_logger (self, self.verbose)
        if self.v_qpab is not None:
            p, na, nb = self.v_qpab.shape[1:]
            ci = np.asarray (ci).reshape (-1,na,nb)[0]
        ket = ci if isinstance (ci, np.ndarray) else ci[0]
        vrvket = self.contract_vrv (ket)
        vrv = np.dot (ket.conj ().ravel (), vrvket.ravel ())
        if abs (vrv) < 1e-16: return False
        e_p = e0 - vrv
        h_qp = np.tensordot (self.v_qpab, ket, axes=2)
        de_pq = np.zeros_like (self.denom_q)
        idx = np.abs (self.denom_q) > 1e-16
        de_pq[idx] = np.diag (np.dot (h_qp.conj (), h_qp.T))[idx] / self.denom_q[idx]
        idx = np.abs (de_pq) > 1e-8
        e_q = self.e_q[idx]
        e_pq = np.append ([e_p,], e_q)
        h_diagmin = np.amin (e_pq)
        if e0-h_diagmin > 1e-8:
            if self.verbose >= lib.logger.DEBUG:
                log.warn ("%s in VRVSolver: min (hdiag) = %.6f < e0 = %.6f",
                          warntag, np.amin (e_pq), e0)
            log.debug ('e_p = %.6f ; vrv = %.6f', e_p, vrv)
            log.debug ('e_q = {}'.format (e_q))
            log.debug ('valid de_pq = {}'.format (de_pq[idx]))
            log.debug ('invalid de_pq = {}'.format (de_pq[~idx]))
            log.debug ('%d valid q poles ; %d invalid q poles',
                       np.count_nonzero (idx),
                       np.count_nonzero (~idx))
            log.debug ('valid denominators: {}'.format (self.denom_q[idx]))
            log.debug ('invalid denominators: {}'.format (self.denom_q[~idx]))
            h2_q = np.diag (np.dot (h_qp.conj (), h_qp.T))
            log.debug ('valid numerators: {}'.format (h2_q[idx]))
            log.debug ('invalid numerators: {}'.format (h2_q[~idx]))
            hket_p = self.undressed_contract_2e (self.absorb_h1e (h1e, h2e, norb, nelec, 0.5),
                                                 ket, norb, nelec)
            e_p_test = np.dot (np.ravel (ket), np.ravel (hket_p)) + h0e
            log.debug ('e_p error: %.6f', e_p_test - e_p)
            if self.crash_locmin:
                errstr = "locmin crashed as requested (crash_locmin=True)"
                log.error (errstr)
                raise RuntimeError (errstr)
            return True
        return False
    def solve_e0 (self, h0e, h1e, h2e, norb, nelec, ket):
        # TODO: figure out how to modify this for p>1
        log = lib.logger.new_logger (self, self.verbose)
        hket_p = self.undressed_contract_2e (self.absorb_h1e (h1e, h2e, norb, nelec, 0.5),
                                             ket, norb, nelec)
        e_p = np.dot (np.ravel (ket), np.ravel (hket_p)) + h0e
        if self.v_qpab is None: return e_p
        q, p = self.v_qpab.shape[0:2]
        v_q = np.dot (self.v_qpab.reshape (q,p,-1), np.ravel (ket)).T.ravel ()
        e_pq = np.append ([e_p,], list(self.e_q)*p)
        ham_pq = np.diag (e_pq)
        ham_pq[0,1:] = v_q
        ham_pq[1:,0] = v_q
        log.debug2 ('v_q = {}'.format (v_q))
        log.debug2 ('e_pq = {}'.format (e_pq))
        e0 = lowest_refovlp_eigval (ham_pq)
        return e0
    def sort_ci (self, h0e, h1e, h2e, norb, nelec, ci):
        if self.nroots==1: ci = [ci]
        e0 = [self.solve_e0 (h0e, h1e, h2e, norb, nelec, ket) for ket in ci]
        idx = np.argsort (e0)
        e0 = [e0[ix] for ix in idx]
        ci = [ci[ix] for ix in idx]
        den = e0[0] - self.e_q
        h2eff = self.absorb_h1e (h1e, h2e, norb, nelec, 0.5)
        e = [np.dot (ket.ravel(), self.contract_2e (h2eff, ket, norb, nelec, denom_q=den).ravel())
             for ket in ci]
        if self.nroots > 1:
            idx = np.argsort (e[1:])
            ci = [ci[0]] + [ci[1:][ix] for ix in idx]
        else:
            ci = ci[0]
        return e0[0], ci
    def undressed_kernel (self, *args, **kwargs):
        return self._undressed_class.kernel (self, *args, **kwargs)
    def undressed_contract_2e (self, *args, **kwargs):
        return self._undressed_class.contract_2e (self, *args, **kwargs)
    kernel = vrvsolver_single_kernel


def make_hdiag_det_vrv (fciobj, v_qpab=None, denom_q=None):
    # Untested!
    if v_qpab is None: v_qpab = fciobj.v_qpab
    if denom_q is None: denom_q = fciobj.denom_q 
    if v_qpab is None: return 0
    q, p, ndeta, ndetb = v_qpab.shape
    idx = np.abs (denom_q) > 1e-16
    p = v_qpab.shape[1]
    q = np.count_nonzero (idx)
    if (not q) or (not p): return np.zeros_like (v_qpab).sum ((0,1))
    v_qpab, denom_q = v_qpab[idx], denom_q[idx]
    denom_q = denom_q + 1j*fciobj.imag_shift
    denom_fac_q = np.real (1.0 / denom_q)
    rv_qp = (v_qpab.conj () * denom_fac_q[:,None,None,None])
    hdiag = (rv_qp * v_qpab).sum ((0,1))
    return hdiag

def make_hdiag_csf_vrv (fciobj, transformer=None, v_qpab=None, denom_q=None):
    if transformer is None: transformer = fciobj.transformer
    if v_qpab is None: v_qpab = fciobj.v_qpab
    if denom_q is None: denom_q = fciobj.denom_q 
    if v_qpab is None: return 0
    q, p, ndeta, ndetb = v_qpab.shape
    idx = np.abs (denom_q) > 1e-16
    p = v_qpab.shape[1]
    q = np.count_nonzero (idx)
    if (not q) or (not p): return np.zeros (transformer.ncsf, dtype=v_qpab.dtype)
    v_qpab, denom_q = v_qpab[idx].reshape (q*p,ndeta,ndetb), denom_q[idx]
    v_qpr = transformer.vec_det2csf (v_qpab, normalize=False).reshape (q,p,transformer.ncsf)
    denom_q = denom_q + 1j*fciobj.imag_shift
    denom_fac_q = np.real (1.0 / denom_q)
    rv_qp = (v_qpr.conj () * denom_fac_q[:,None,None])
    hdiag = (rv_qp * v_qpr).sum ((0,1))
    return hdiag

def pspace_det_vrv (fciobj, norb, nelec, det_addr, v_qpab=None, denom_q=None):
    # Untested!
    if v_qpab is None: v_qpab = fciobj.v_qpab
    if denom_q is None: denom_q = fciobj.denom_q 
    if v_qpab is None: return 0
    q, p, ndeta, ndetb = v_qpab.shape
    idx = np.abs (denom_q) > 1e-16
    p = v_qpab.shape[1]
    q = np.count_nonzero (idx)
    if (not q) or (not p): return np.zeros ((len(det_addr),len(det_addr)), dtype=v_qpab.dtype)
    neleca, nelecb = _unpack_nelec (nelec)
    ndeta = cistring.num_strings (norb, neleca)
    ndetb = cistring.num_strings (norb, nelecb)
    addra, addrb = divmod (det_addr, ndetb)
    jdx = np.ix_(idx,[True,]*p,addra,addrb)
    v_qpab, denom_q = v_qpab[jdx].reshape (q,p,len(det_addr)), denom_q[idx]
    denom_q = denom_q + 1j*fciobj.imag_shift
    denom_fac_q = np.real (1.0 / denom_q)
    rv_qpab = (v_qpab.conj () * denom_fac_q[:,None,None])
    h0 = np.tensordot (rv_qpab, v_qpab, axes=((0,1),(0,1)))
    return h0

def pspace_csf_vrv (fciobj, csf_addr, transformer=None, v_qpab=None, denom_q=None):
    if transformer is None: transformer = fciobj.transformer
    if v_qpab is None: v_qpab = fciobj.v_qpab
    if denom_q is None: denom_q = fciobj.denom_q 
    if v_qpab is None: return 0
    ncsf = len (csf_addr)
    q, p, ndeta, ndetb = v_qpab.shape
    idx = np.abs (denom_q) > 1e-16
    p = v_qpab.shape[1]
    q = np.count_nonzero (idx)
    if (not q) or (not p): return np.zeros ((ncsf, ncsf), dtype=v_qpab.dtype)
    v_qpab, denom_q = v_qpab[idx].reshape (q*p,ndeta,ndetb).transpose (1,2,0), denom_q[idx]
    v_rqp = transformer.vec_det2csf (v_qpab, order='F', normalize=False)
    v_rqp = v_rqp[csf_addr].reshape (ncsf, q, p)
    denom_q = denom_q + 1j*fciobj.imag_shift
    denom_fac_q = np.real (1.0 / denom_q)
    rv_rqp = (v_rqp.conj () * denom_fac_q[None,:,None])
    rv_rx, v_rx = rv_rqp.reshape (ncsf, q*p), v_rqp.reshape (ncsf, q*p)
    h0 = np.dot (rv_rx, v_rx.T)
    assert (h0.shape == (ncsf, ncsf))
    return h0

def vrv_fcisolver (fciobj, e0, e_q, v_qpab, max_cycle_e0=MAX_CYCLE_E0, conv_tol_e0=CONV_TOL_E0,
                   crash_locmin=False):
    if isinstance (fciobj, VRVDressedFCISolver):
        fciobj.v_qpab = v_qpab
        fciobj.e_q = e_q
        fciobj.max_cycle_e0 = max_cycle_e0
        fciobj.conv_tol_e0 = conv_tol_e0
        fciobj.crash_locmin = crash_locmin
        return fciobj
    # VRV should be injected below the state-averaged layer
    if isinstance (fciobj, StateAverageFCISolver):
        fciobj_class = fciobj._base_class
        weights = fciobj.weights
    else:
        fciobj_class = fciobj.__class__
        weights = None
    class FCISolver (VRVDressedFCISolver, fciobj_class):
        if isinstance (fciobj, CSFFCISolver):
            def make_hdiag_csf (self, *args, **kwargs):
                h1e, eri, norb, nelec = args[:4]
                link_index = kwargs.get ('link_index', None)
                q_qr, hq_qr, ham_q = self.get_projectors (
                    h1e, eri, norb, nelec, link_index=link_index
                )
                hdiag_csf = fciobj_class.make_hdiag_csf (self, *args, **kwargs)
                dhdiag_csf = make_hdiag_csf_vrv (self)
                qhdiag_csf = self.project_hdiag (q_qr, hq_qr, ham_q)
                return hdiag_csf + dhdiag_csf + qhdiag_csf
            def pspace (self, *args, **kwargs):
                h1e, eri, norb, nelec = args[:4]
                link_index = kwargs.get ('link_index', None)
                q_qr, hq_qr, ham_q = self.get_projectors (
                    h1e, eri, norb, nelec, link_index=link_index
                )
                csf_addr, h0 = fciobj_class.pspace (self, *args, **kwargs)
                dh0 = pspace_csf_vrv (self, csf_addr)
                qh0 = self.project_pspace (csf_addr, q_qr, hq_qr, ham_q)
                return csf_addr, h0 + dh0 + qh0
            def get_projectors (self, h1e, eri, norb, nelec, link_index=None):
                q_qab, hq_qab, ham_q = VRVDressedFCISolver.get_projectors (
                    self, h1e, eri, norb, nelec, link_index=link_index
                )
                if q_qab is None: return q_qab, hq_qab, ham_q
                t = self.transformer
                q_qr = t.vec_det2csf (q_qab)
                hq_qr = t.vec_det2csf (hq_qab, normalize=False)
                return q_qr, hq_qr, ham_q
        else:
            raise NotImplementedError ("Non-CSF version of excitation solver")
            def make_hdiag (self, *args, **kwargs):
                h1e, eri, norb, nelec = args[:4]
                link_index = kwargs.get ('link_index', None)
                q_qab, hq_qab, ham_q = self.get_projectors (
                    h1e, eri, norb, nelec, link_index=link_index
                )
                hdiag = fciobj_class.make_hdiag (self, *args, **kwargs)
                dhdiag = make_hdiag_det_vrv (self)
                qhdiag = self.project_hdiag (q_qab, hq_qab, ham_q)
                return hdiag + dhdiag + qhdiag
            def pspace (self, *args, **kwargs):
                h1e, eri, norb, nelec = args[:4]
                link_index = kwargs.get ('link_index', None)
                q_qab, hq_qab, ham_q = self.get_projectors (
                    h1e, eri, norb, nelec, link_index=link_index
                )
                det_addr, h0 = fciobj_class.pspace (self, h1e, eri, norb, nelec, **kwargs)
                dh0 = pspace_det_vrv (self, norb, nelec, det_addr)
                qh0 = self.project_pspace (det_addr, q_qab, hq_qab, ham_q)
                return det_addr, h0 + dh0 + qh0
    new_fciobj = FCISolver (fciobj, v_qpab, e_q, e0, max_cycle_e0=max_cycle_e0,
                            conv_tol_e0=conv_tol_e0, crash_locmin=crash_locmin)
    if weights is not None: new_fciobj = state_average_fcisolver (new_fciobj, weights=weights)
    return new_fciobj


