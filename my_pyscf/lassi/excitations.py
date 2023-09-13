import copy
import numpy as np
from scipy import linalg
from pyscf.fci import cistring
from pyscf.fci.direct_spin1 import _unpack_nelec
from pyscf.mcscf.addons import StateAverageFCISolver
from mrh.my_pyscf.mcscf.productstate import ProductStateFCISolver, ImpureProductStateFCISolver, state_average_fcisolver
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.lassi import op_o0, op_o1
from mrh.my_pyscf.lassi.citools import get_lroots
from pyscf import lib
from pyscf.lib import temporary_env
op = (op_o0, op_o1)

def only_ground_states (ci0):
    '''For a list of sequences of CI vectors in the same Hilbert space,
    generate a list in which all but the first element of each sequence
    is discarded.''' 
    # TODO: examine whether this should be considered a limitation
    ci1 = []
    for c in ci0:
        c = np.asarray (c)
        if c.ndim==3: c = c[0]
        ci1.append (c)
    return ci1

def lowest_refovlp_eigval (ham_pq, e_q=None, u_q=None, sceig_thresh=1e-4):
    ''' Return the lowest eigenvalue of the matrix ham_pq, whose corresponding
    eigenvector has nonzero overlap with the first basis function, as
    determined by self-consistency with the corresponding Green's function. '''
    h_pp = ham_pq[0,0]
    h_pq = ham_pq[0,1:]
    h_qq = ham_pq[1:,1:]
    if e_q is None or u_q is None:
        e_q, u_q = linalg.eigh (h_qq)
    h_pq = np.dot (h_pq, u_q)
    v_pq = h_pq.conj () * h_pq
    def sigma_pp (e0):
        denom = e0 - e_q
        idx = np.abs (denom) > 1e-16
        denom, numer = denom[idx], v_pq[idx]
        return np.dot (numer, 1/denom)
    e_all, u_all = linalg.eigh (ham_pq)
    err = np.array ([e - (h_pp + sigma_pp (e)) for e in e_all])
    idx_valid = np.abs (err) < sceig_thresh
    return np.amin (e_all[idx_valid])

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
                 wfnsym_ref=None, stdout=None, verbose=0, opt=0, ref_weights=None, **kwargs):
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
                                           ecore=h0, dm1s=dm1s, dm2=dm2)
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

    def kernel (self, h1, h2, ecore=0,
                conv_tol_grad=1e-4, conv_tol_self=1e-10, max_cycle_macro=50,
                **kwargs):
        h0, h1, h2 = self.get_excited_h (ecore, h1, h2)
        norb_f = np.asarray ([self.norb_ref[ifrag] for ifrag in self.excited_frags])
        nelec_f = np.asarray ([self.nelec_ref[ifrag] for ifrag in self.excited_frags])
        orbsym = self.orbsym_ref
        if orbsym is not None:
            idx = self.get_excited_orb_idx ()
            orbsym = [orbsym[iorb] for iorb in range (norb_tot) if idx[iorb]]
        # TODO: point group symmetry; I probably also have to do something to wfnsym
        ci0, vrvsolvers, e_q, si_q = self.prepare_vrvsolvers_(h0, h1, h2)
        with _vrvloop_env (self, vrvsolvers, e_q, si_q):
            converged, energy_elec, ci1_active = ProductStateFCISolver.kernel (
                self, h1, h2, norb_f, nelec_f, ecore=h0, ci0=ci0, orbsym=orbsym,
                conv_tol_grad=conv_tol_grad, conv_tol_self=conv_tol_self,
                max_cycle_macro=max_cycle_macro, **kwargs
            )
        ci1 = [c for c in self.ci_ref]
        for ifrag, c in zip (self.excited_frags, ci1_active):
            ci1[ifrag] = np.asarray (c)
        return converged, energy_elec, ci1

    def project_hfrag (self, h1, h2, ci, norb_f, nelec_f, ecore=0, dm1s=None, dm2=None, **kwargs):
        h1eff, h0eff = ProductStateFCISolver.project_hfrag (
            self, h1, h2, ci, norb_f, nelec_f, ecore=ecore, dm1s=dm1s, dm2=dm2, **kwargs
        )
        # Project the part coupling the p and q rootspaces
        if len (self._e_q) and not self._deactivate_vrv:
            ci0 = [np.asarray (c) for c in ci]
            ci0, e0, e_q, si_q = self.sort_ci0 (ecore, h1, h2, ci0)
            hci_f_abq = self.op_ham_pq_ref (h1, h2, ci0)
            self._e_q = e_q
            self._si_q = si_q
            for ifrag, (c, hci_abq, solver) in enumerate (zip (ci0, hci_f_abq, self.fcisolvers)):
                solver.v_qab = np.tensordot (si_q, hci_abq, axes=((0),(-1)))
                solver.e0_p = e0
                solver.e_q = e_q
                ci[ifrag] = c
        return h1eff, h0eff

    def energy_elec (self, h1, h2, ci, norb_f, nelec_f, ecore=0, **kwargs):
        energy_tot = ProductStateFCISolver.energy_elec (
            self, h1, h2, ci, norb_f, nelec_f, ecore=ecore, **kwargs
        )
        # Also compute the vrv perturbation energy
        if len (self._e_q) and not self._deactivate_vrv:
            ci0 = []
            # extract this from the putatively converged solver cycles
            # if you attempt to recalculate it, then it has to be reconverged i guess
            denom_q = 0
            for c, solver in zip (ci, self.fcisolvers):
                t = solver.transformer
                c = np.asarray (c).reshape (-1, t.ndeta, t.ndetb)
                ci0.append (c)
                denom_q += solver.denom_q
            denom_q /= len (ci)
            lroots = get_lroots (ci0)
            p = np.prod (lroots)
            ham_pq = self.get_ham_pq (ecore, h1, h2, ci0)
            idx = np.ones (len (ham_pq), dtype=bool)
            idx[1:p] = False
            e0, si = linalg.eigh (ham_pq[idx,:][:,idx])
            h_pp = ham_pq[:p,:p]
            h_pq = ham_pq[:p,p:]
            h_qq = ham_pq[p:,p:]
            h_qq = self._si_q.conj ().T @ h_qq @ self._si_q
            h_pq = np.dot (h_pq, self._si_q)
            ham_pq = ham_pq[idx,:][:,idx]
            idx = np.abs (denom_q) > 1e-16
            e_p = np.diag (np.dot (h_pq[:,idx].conj () / denom_q[None,idx], h_pq[:,idx].T))
            e_p = e_p.reshape (*lroots[::-1]).T
            for solver in self.fcisolvers:
                if hasattr (getattr (solver, 'weights', None), '__len__'):
                    e_p = np.dot (solver.weights, e_p)
                else:
                    e_p = e_p[0]
            energy_tot += e_p
        return energy_tot

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
            hci_f_abq: list of ndarray
                Contains H|q>, projected onto all but one fragment, for each fragment'''
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
        ci_fr_bra = [[c] for c in only_ground_states (ci)]
        nelec_rfs_bra = np.asarray ([[list(self._get_nelec (s, n))
                                     for s, n in zip (self.fcisolvers, nelec_f)]])
        nelec_frs_bra = nelec_rfs_bra.transpose (1,0,2)
        h_op = op[self.opt].contract_ham_ci
        with temporary_env (self, ncas_sub=norb_f, mol=self.fcisolvers[0].mol):
            hci_fr_pabq = h_op (self, h1, h2, ci_fr_ket, nelec_frs_ket, ci_fr_bra, nelec_frs_bra,
                                soc=0, orbsym=None, wfnsym=None)
        hci_f_abq = [hc[0][0] for hc in hci_fr_pabq]
        t1 = self.log.timer ('op_ham_pq_ref', *t0)
        return hci_f_abq

    def sort_ci0 (self, h0, h1, h2, ci0):
        '''Prepare guess CI vectors, guess energy, and Q-space Hamiltonian eigenvalues
        and eigenvectors. Sort ci0 so that the ENV |0000...> is the state with the
        minimum guess energy for the downfolded eigenproblem

        (h_pp + h_pq (e0 - e_q)^-1 h_qp) |p> = e0|p>

        Args:
            h0: float
                Constant part of the excited-fragment Hamiltonian
            h1: ndarray of shape (2,nexc,nexc)
                Spin-separated 1-electron part of the excited-fragment Hamiltonian
            h2: ndarray of shape [nexc,]*4
                2-electron part of the excited-fragment Hamiltonian
            ci0: list of ndarray
                CI vectors for the active fragments

        Returns:
            ci0: list of ndarray
                Resorted on each fragment so that |0000...> has the lowest downfolded
                guess energy
            e0_p: float
                Downfolded guess energy of |0000....>
            e_q: ndarray of shape (nq,)
                Eigenvalues of the Q-space Hamiltonian
            si_q: ndarray of shape (nq,nq)
                Eigenvectors of the Q-space Hamiltonian'''
        # Find lowest-energy ENV, including VRV contributions
        lroots = get_lroots (ci0)
        ham_pq = self.get_ham_pq (h0, h1, h2, ci0)
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
        #print ("Winner:")
        #lowest_refovlp_eigval (project_1p (idxmin))
        e0_p = e_p[idxmin]
        h_pq = np.dot (h_pq, si_q)
        def sigma_pp (e):
            denom = e - e_q
            idx = np.abs (denom) > 1e-16
            return np.dot (h_pq[:,idx].conj () / denom[None,idx], h_pq[:,idx].T)
        heff_pp = h_pp + sigma_pp (e0_p)
        assert (abs (heff_pp[idxmin,idxmin] - e0_p) < 1e-4)

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
        for ifrag in range (nfrag):
            if lroots[ifrag]<2: continue
            e_p_slice = e_p_arr
            for jfrag in range (ifrag):
                e_p_slice = e_p_slice[addr[jfrag]]
            for jfrag in range (ifrag+1,nfrag):
                e_p_slice = e_p_slice[:,addr[jfrag]]
            sort_idx = np.argsort (e_p_slice)
            assert (sort_idx[0] == addr[ifrag])
            ci0[ifrag] = np.stack ([ci0[ifrag][i] for i in sort_idx], axis=0)
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
        return ci0, e0_p, e_q, si_q

    def prepare_vrvsolvers_(self, h0, h1, h2):
        norb_f = np.asarray ([self.norb_ref[ifrag] for ifrag in self.excited_frags])
        nelec_f = np.asarray ([self.nelec_ref[ifrag] for ifrag in self.excited_frags])
        ci0 = self._check_init_guess (None, norb_f, nelec_f, h1, h2)
        ci0, e0, e_q, si_q = self.sort_ci0 (h0, h1, h2, ci0)
        vrvsolvers = []
        for ix, solver in enumerate (self.fcisolvers):
            vrvsolvers.append (vrv_fcisolver (solver, e0, e_q, None, max_cycle_e0=1))
        return ci0, vrvsolvers, e_q, si_q

    def revert_vrvsolvers_(self):
        self._e_q = []
        self._si_q = []

class VRVDressedFCISolver (object):
    '''Minimize the energy of a wave function of the form

    |Psi> = |P> + \sum_i |Q(i)>

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
        v_qab: ndarray of shape (nq, ndeta, ndetb)
            Contains the CI vector V_PQ |Q> in the |P> Hilbert space
        e_q: ndarray of shape (nq,)
            Eigenenergies of the QQ sector of the Hamiltonian
        e0_p: float
            Initial guess for the lowest energy solution of the self-consistent
            eigenproblem
        denom_q: ndarray of shape (nq,)
            Contains E-e_q with the current guess E solution of the self-consistent
            eigenproblem
        max_cycle_e0: integer
            Maximum number of cycles allowed to attempt to converge the self-consistent
            eigenproblem
        conv_tol_e0: float
            Convergence threshold for the self-consistent eigenenergy
    '''
    def __init__(self, fcibase, my_vrv, my_eq, my_e0, max_cycle_e0=100, conv_tol_e0=1e-8):
        self.base = copy.copy (fcibase)
        if isinstance (fcibase, StateAverageFCISolver):
            self._undressed_class = fcibase._base_class
        else:
            self._undressed_class = fcibase.__class__
        self.__dict__.update (fcibase.__dict__)
        keys = set (('contract_vrv', 'base', 'v_qab', 'denom_q', 'e_q', 'max_cycle_e0',
                     'conv_tol_e0', 'e0_p', 'charge'))
        self.denom_q = 0
        self.e0_p = my_e0
        self.e_q = my_eq
        self.v_qab = my_vrv
        self.max_cycle_e0 = max_cycle_e0
        self.conv_tol_e0 = conv_tol_e0
        self._keys = self._keys.union (keys)
        self.davidson_only = self.base.davidson_only = True
        # TODO: Relaxing this ^ requires accounting for pspace, precond, and/or hdiag
    def contract_2e(self, eri, fcivec, norb, nelec, link_index=None, **kwargs):
        ci0 = self.undressed_contract_2e (eri, fcivec, norb, nelec, link_index, **kwargs)
        ci0 += self.contract_vrv (fcivec)
        return ci0
    def contract_vrv (self, ket):
        v_qab, denom_q = self.v_qab, self.denom_q
        if v_qab is None: return np.zeros_like (ket)
        ket_shape = ket.shape
        idx = np.abs (denom_q) > 1e-16
        nq = np.count_nonzero (idx)
        if not nq: return np.zeros_like (ket)
        v_qab, denom_q = v_qab[idx].reshape (nq,-1), denom_q[idx]
        rv_q = np.dot (v_qab.conj (), ket.ravel ()) / denom_q
        hket = np.dot (rv_q, v_qab).reshape (ket_shape)
        return hket
    def test_locmin (self, e0, ci, warntag='Apparent local minimum'):
        log = lib.logger.new_logger (self, self.verbose)
        if self.v_qab is not None:
            na, nb = self.v_qab.shape[1:]
            ci = np.asarray (ci).reshape (-1,na,nb)[0]
        ket = ci if isinstance (ci, np.ndarray) else ci[0]
        vrvket = self.contract_vrv (ket)
        vrv = np.dot (ket.conj ().ravel (), vrvket.ravel ())
        if abs (vrv) < 1e-16: return False
        e_p = e0 - vrv
        h_pq = np.tensordot (self.v_qab, ket, axes=2)
        de_pq = np.zeros_like (h_pq)
        idx = np.abs (self.denom_q) > 1e-16
        de_pq[idx] = (h_pq.conj () * h_pq)[idx] / self.denom_q[idx]
        idx = np.abs (de_pq) > 1e-8
        e_q = self.e_q[idx]
        e_pq = np.append ([e_p,], e_q)
        h_diagmin = np.amin (e_pq)
        if e0>h_diagmin:
            log.warn ("%s in VRVSolver: min (hdiag) = %e < e0 = %e",
                      warntag, np.amin (e_pq - self.e0_p), e0 - self.e0_p)
            return True
        return False
    def solve_e0 (self, ket, e):
        vrv = np.dot (np.ravel (ket), np.ravel (self.contract_vrv (ket)))
        if vrv==0: return e
        e_p = e - vrv
        nq = self.v_qab.shape[0]
        v_q = np.dot (self.v_qab.reshape (nq,-1), np.ravel (ket))
        e_pq = np.append ([e_p,], self.e_q)
        ham_pq = np.diag (e_pq)
        ham_pq[0,1:] = v_q
        ham_pq[1:,0] = v_q
        return lowest_refovlp_eigval (ham_pq, e_q=self.e_q, u_q=np.eye(nq))
    def kernel (self, h1e, h2e, norb, nelec, ecore=0, ci0=None, orbsym=None, **kwargs):
        log = lib.logger.new_logger (self, self.verbose)
        max_cycle_e0 = self.max_cycle_e0
        conv_tol_e0 = self.conv_tol_e0
        e0_last = 0
        converged = False
        e0 = self.e0_p
        ci1 = ci0
        self.denom_q = e0 - self.e_q
        assert (not self.test_locmin (e0, ci1, warntag='Saddle-point initial guess'))
        warn_swap = True
        self.denom_q = e0 - self.e_q
        for it in range (max_cycle_e0):
            e, ci1 = self.undressed_kernel (
                h1e, h2e, norb, nelec, ecore=ecore, ci0=ci1, orbsym=orbsym, **kwargs
            )
            e0_last = e0
            if isinstance (e, (list,tuple,np.ndarray)):
                delta_e0 = np.array (e) - e0
                if warn_swap and np.argmin (np.abs (delta_e0)) != 0:
                    log.warn ("Possible rootswapping in H(E)|Psi> = E|Psi> fixed-point iteration")
                    warn_swap = False
                e0 = self.solve_e0 (ci1[0], e[0])
            else:
                e0 = self.solve_e0 (ci1, e)
            self.denom_q = e0 - self.e_q
            ci0 = ci1
            if abs(e0-e0_last)<conv_tol_e0:
                converged = True
                break
        assert (not self.test_locmin (e0, ci1))
        self.converged = (converged and self.converged)
        return e, ci1
    # I don't feel like futzing around with MRO
    def undressed_kernel (self, *args, **kwargs):
        return self._undressed_class.kernel (self, *args, **kwargs)
    def undressed_contract_2e (self, *args, **kwargs):
        return self._undressed_class.contract_2e (self, *args, **kwargs)

def vrv_fcisolver (fciobj, e0, e_q, v_qab, max_cycle_e0=100, conv_tol_e0=1e-8):
    if isinstance (fciobj, VRVDressedFCISolver):
        fciobj.v_qab = v_qab
        fciobj.e_q = e_q
        fciobj.max_cycle_e0 = max_cycle_e0
        fciobj.conv_tol_e0 = conv_tol_e0
        return fciobj
    # VRV should be injected below the state-averaged layer
    if isinstance (fciobj, StateAverageFCISolver):
        fciobj_class = fciobj._base_class
        weights = fciobj.weights
    else:
        fciobj_class = fciobj.__class__
        weights = None
    class FCISolver (VRVDressedFCISolver, fciobj_class):
        pass
    new_fciobj = FCISolver (fciobj, v_qab, e_q, e0, max_cycle_e0=max_cycle_e0,
                            conv_tol_e0=conv_tol_e0)
    if weights is not None: new_fciobj = state_average_fcisolver (new_fciobj, weights=weights)
    return new_fciobj


