import numpy as np
from scipy import linalg

# Probably missing imports
# I don't want to delete all this hard work but this is no longer useful for now.

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
        self.max_cycle_e0 = max_cycle_e0
        self.conv_tol_e0 = conv_tol_e0
        self.crash_locmin = crash_locmin
    def contract_2e(self, eri, fcivec, norb, nelec, link_index=None, v_qpab=None, denom_q=None,
                    **kwargs):
        ci0 = self.undressed_contract_2e (eri, fcivec, norb, nelec, link_index, **kwargs)
        ci0 += self.contract_vrv (fcivec, v_qpab=v_qpab, denom_q=denom_q)
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
    def kernel (self, h1e, h2e, norb, nelec, ecore=0, ci0=None, orbsym=None, **kwargs):
        log = lib.logger.new_logger (self, self.verbose)
        max_cycle_e0 = self.max_cycle_e0
        conv_tol_e0 = self.conv_tol_e0
        e0_last = 0
        converged = False
        ket = ci0[0] if self.nroots>1 else ci0
        e0 = self.solve_e0 (ecore, h1e, h2e, norb, nelec, ket)
        ci1 = ci0
        self.denom_q = e0 - self.e_q
        log.debug ("Self-energy singularities in VRVSolver: {}".format (self.e_q))
        log.debug ("e0 = %.8g", e0)
        log.debug ("Denominators in VRVSolver: {}".format (self.denom_q))
        self.test_locmin (e0, ci1, norb, nelec, ecore, h1e, h2e, warntag='Saddle-point initial guess')
        h2eff = self.absorb_h1e (h1e, h2e, norb, nelec, 0.5)
        for it in range (max_cycle_e0):
            e, ci1 = self.undressed_kernel (
                h1e, h2e, norb, nelec, ecore=ecore, ci0=ci1, orbsym=orbsym, **kwargs
            )
            # Subtract the vrv energy so that agreement between different fragments can
            # be checked in the impure-state case
            if isinstance (e, (list,tuple,np.ndarray)):
                for i in range (len (e)):
                    hci = self.undressed_contract_2e (h2eff, ci1[i], norb, nelec)
                    e[i] = ecore + np.dot (ci1[i].ravel (), hci.ravel ())
            else:
                hci = self.undressed_contract_2e (h2eff, ci1, norb, nelec)
                e = ecore + np.dot (ci1.ravel (), hci.ravel ())
            e0_last = e0
            e0 = self.solve_e0 (ecore, h1e, h2e, norb, nelec, ket)
            self.denom_q = e0 - self.e_q
            log.debug ("e0 = %.8g", e0)
            log.debug ("Denominators in VRVSolver: {}".format (self.denom_q))
            if abs(e0-e0_last)<conv_tol_e0:
                converged = True
                break
        self.test_locmin (e0, ci1, norb, nelec, ecore, h1e, h2e)
        self.converged = (converged and np.all (self.converged))
        return e, ci1
    def undressed_kernel (self, *args, **kwargs):
        return self._undressed_class.kernel (self, *args, **kwargs)
    def undressed_contract_2e (self, *args, **kwargs):
        return self._undressed_class.contract_2e (self, *args, **kwargs)

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
                hdiag_csf = fciobj_class.make_hdiag_csf (self, *args, **kwargs)
                dhdiag_csf = make_hdiag_csf_vrv (self)
                return hdiag_csf + dhdiag_csf
            def pspace (self, *args, **kwargs):
                csf_addr, h0 = fciobj_class.pspace (self, *args, **kwargs)
                dh0 = pspace_csf_vrv (self, csf_addr)
                return csf_addr, h0 + dh0
        else:
            raise NotImplementedError ("Non-CSF version of excitation solver")
            #def make_hdiag (self, *args, **kwargs):
            #    hdiag = fciobj_class.make_hdiag (self, *args, **kwargs)
            #    dhdiag = make_hdiag_det_vrv (self)
            #    return hdiag + dhdiag
            #def pspace (self, h1e, eri, norb, nelec, **kwargs):
            #    det_addr, h0 = fciobj_class.pspace (self, h1e, eri, norb, nelec, **kwargs)
            #    dh0 = pspace_det_vrv (self, norb, nelec, det_addr)
            #    return det_addr, h0 + dh0
    new_fciobj = FCISolver (fciobj, v_qpab, e_q, e0, max_cycle_e0=max_cycle_e0,
                            conv_tol_e0=conv_tol_e0, crash_locmin=crash_locmin)
    if weights is not None: new_fciobj = state_average_fcisolver (new_fciobj, weights=weights)
    return new_fciobj


