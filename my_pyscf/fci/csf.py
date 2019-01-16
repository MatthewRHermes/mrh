import numpy as np
import scipy
import ctypes
from pyscf import lib, ao2mo, __config__
from pyscf.fci import direct_spin1, cistring
from pyscf.fci.direct_spin1 import _unpack, _unpack_nelec, _get_init_guess, kernel_ms1, make_hdiag
from mrh.my_pyscf.fci.csdstring import make_csd_mask, make_econf_det_mask, get_nspin_dets, get_csdaddrs_shape, pretty_csdaddrs
from mrh.my_pyscf.fci.csfstring import transform_civec_det2csf, transform_civec_csf2det
from mrh.my_pyscf.fci.csfstring import transform_opmat_det2csf, transform_opmat_det2csf_pspace
from mrh.my_pyscf.fci.csfstring import count_all_csfs, make_econf_csf_mask, get_spin_evecs
from mrh.my_pyscf.fci.csfstring import get_csfvec_shape

libfci = lib.load_library('libfci')

def make_hdiag_csf (h1e, eri, norb, nelec, smult, csd_mask=None, hdiag_det=None):
    if hdiag_det is None:
        hdiag_det = make_hdiag (h1e, eri, norb, nelec)
    eri = ao2mo.restore(1, eri, norb)
    neleca, nelecb = _unpack_nelec (nelec)
    min_npair, npair_csd_offset, npair_dconf_size, npair_sconf_size, npair_sdet_size = get_csdaddrs_shape (norb, neleca, nelecb)
    _, npair_csf_offset, _, _, npair_csf_size = get_csfvec_shape (norb, neleca, nelecb, smult)
    npair_econf_size = npair_dconf_size * npair_sconf_size
    max_npair = nelecb
    ncsf_all = count_all_csfs (norb, neleca, nelecb, smult)
    ndeta_all = cistring.num_strings(norb, neleca)
    ndetb_all = cistring.num_strings(norb, nelecb)
    ndet_all = ndeta_all * ndetb_all
    hdiag_csf = np.ascontiguousarray (np.zeros (ncsf_all, dtype=np.float64))
    hdiag_csf_check = np.ones (ncsf_all, dtype=np.bool)
    for npair in range (min_npair, max_npair+1):
        ipair = npair - min_npair
        nconf = npair_econf_size[ipair]
        ndet = npair_sdet_size[ipair]
        ncsf = npair_csf_size[ipair]
        if ncsf == 0:
            continue
        nspin = neleca + nelecb - 2*npair
        csd_offset = npair_csd_offset[ipair]
        csf_offset = npair_csf_offset[ipair]
        hdiag_conf = np.ascontiguousarray (np.zeros ((nconf, ndet, ndet), dtype=np.float64))
        if csd_mask is None:
            det_addr = get_nspin_dets (norb, neleca, nelecb, nspin)
        else:
            det_addr = csd_mask[csd_offset:][:nconf*ndet].reshape (nconf, ndet, order='C')
        if ndet == 1:
            # Closed-shell singlets
            assert (ncsf == 1)
            hdiag_csf[csf_offset:][:nconf] = hdiag_det[det_addr.flat]
            hdiag_csf_check[csf_offset:][:nconf] = False
            continue
        umat = get_spin_evecs (nspin, neleca, nelecb, smult)
        ipair_check = 0
        for iconf in range (nconf):
            addr = det_addr[iconf]
            assert (len (addr) == ndet)
            addra, addrb = divmod (addr, ndetb_all)
            stra = cistring.addrs2str(norb, neleca, addra).astype (np.uint64)
            strb = cistring.addrs2str(norb, nelecb, addrb).astype (np.uint64)
            libfci.FCIpspace_h0tril(hdiag_conf[iconf].ctypes.data_as(ctypes.c_void_p),
                h1e.ctypes.data_as(ctypes.c_void_p),
                eri.ctypes.data_as(ctypes.c_void_p),
                stra.ctypes.data_as(ctypes.c_void_p),
                strb.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(norb), ctypes.c_int(ndet))
            hdiag_conf[iconf][np.diag_indices (ndet)] = hdiag_det[addr]
            hdiag_conf[iconf] = lib.hermi_triu(hdiag_conf[iconf])

        hdiag_conf = np.tensordot (hdiag_conf, umat, axes=1)
        hdiag_conf = (hdiag_conf * umat[np.newaxis,:,:]).sum (1)
        hdiag_csf[csf_offset:][:nconf*ncsf] = hdiag_conf.ravel (order='C')
        hdiag_csf_check[csf_offset:][:nconf*ncsf] = False
    assert (np.count_nonzero (hdiag_csf_check) == 0), np.count_nonzero (hdiag_csf_check)
    return hdiag_csf

def pspace (fci, h1e, eri, norb, nelec, smult, idx_sym=None, hdiag_det=None, hdiag_csf=None, csd_mask=None,
    econf_det_mask=None, econf_csf_mask=None, npsp=200):
    ''' Note that getting pspace for npsp CSFs is substantially more costly than getting it for npsp determinants,
    until I write code than can evaluate Hamiltonian matrix elements of CSFs directly. On the other hand
    a pspace of determinants contains many redundant degrees of freedom for the same reason. Therefore I have
    reduced the default pspace size by a factor of 2.'''
    if norb > 63:
        raise NotImplementedError('norb > 63')

    neleca, nelecb = _unpack_nelec(nelec)
    h1e = np.ascontiguousarray(h1e)
    eri = ao2mo.restore(1, eri, norb)
    nb = cistring.num_strings(norb, nelecb)
    if hdiag_det is None:
        hdiag_det = fci.make_hdiag(h1e, eri, norb, nelec)
    if hdiag_csf is None:
        hdiag_csf = fci.make_hdiag_csf(h1e, eri, norb, nelec, hdiag_det=hdiag_det)
    if idx_sym is None:
        idx_sym = np.ones (hdiag_csf.size, dtype=np.bool)
    csf_addr = np.arange (hdiag_csf.size, dtype=np.int)[idx_sym]
    if np.count_nonzero (idx_sym) > npsp:
        try:
            csf_addr = csf_addr[np.argpartition(hdiag_csf[csf_addr], npsp-1)[:npsp]]
        except AttributeError:
            csf_addr = csf_addr[np.argsort(hdiag_csf[csf_addr])[:npsp]]

    # To build 
    econf_addr = np.unique (econf_csf_mask[csf_addr])
    det_addr = np.concatenate ([np.nonzero (econf_det_mask == conf)[0] for conf in econf_addr])
    lib.logger.debug (fci, ("csf_solver.pspace: Lowest-energy %s CSFs correspond to %s configurations"
        " which are spanned by %s determinants"), npsp, econf_addr.size, det_addr.size)

    addra, addrb = divmod(det_addr, nb)
    stra = cistring.addrs2str(norb, neleca, addra)
    strb = cistring.addrs2str(norb, nelecb, addrb)
    npsp_det = len(det_addr)
    h0 = np.zeros((npsp_det,npsp_det))
    libfci.FCIpspace_h0tril(h0.ctypes.data_as(ctypes.c_void_p),
                            h1e.ctypes.data_as(ctypes.c_void_p),
                            eri.ctypes.data_as(ctypes.c_void_p),
                            stra.ctypes.data_as(ctypes.c_void_p),
                            strb.ctypes.data_as(ctypes.c_void_p),
                            ctypes.c_int(norb), ctypes.c_int(npsp_det))

    for i in range(npsp_det):
        h0[i,i] = hdiag_det[det_addr[i]]
    h0 = lib.hermi_triu(h0)

    if fci.verbose >= lib.logger.DEBUG: evals_before = scipy.linalg.eigh (h0)[0]

    h0, csf_addr = transform_opmat_det2csf_pspace (h0, econf_addr, norb, neleca, nelecb, smult,
        csd_mask, econf_det_mask, econf_csf_mask) 

    if fci.verbose >= lib.logger.DEBUG:
        lib.logger.debug2 (fci, "csf_solver.pspace: eigenvalues of h0 before transformation %s", evals_before)
        evals_after = scipy.linalg.eigh (h0)[0]
        lib.logger.debug2 (fci, "csf_solver.pspace: eigenvalues of h0 after transformation %s", evals_after)
        idx = [np.argmin (np.abs (evals_before - ev)) for ev in evals_after]
        resid = evals_after - evals_before[idx]
        lib.logger.debug2 (fci, "csf_solver.pspace: best h0 eigenvalue matching differences after transformation: %s", resid)
        lib.logger.debug (fci, "csf_solver.pspace: if the transformation of h0 worked the following number will be zero: %s", np.max (np.abs(resid)))

    # We got extra CSFs from building the configurations most of the time.
    if csf_addr.size > npsp:
        try:
            csf_addr_2 = np.argpartition(np.diag (h0), npsp-1)[:npsp]
        except AttributeError:
            csf_addr_2 = np.argsort(np.diag (h0))[:npsp]
        csf_addr = csf_addr[csf_addr_2]
        h0 = h0[np.ix_(csf_addr_2,csf_addr_2)]
    npsp_csf = csf_addr.size
    lib.logger.debug (fci, "csf_solver.pspace: asked for %s-CSF pspace; found %s CSFs", npsp, npsp_csf)

    return csf_addr, h0

def kernel(fci, h1e, eri, norb, nelec, smult=None, idx_sym=None, ci0=None,
           tol=None, lindep=None, max_cycle=None, max_space=None,
           nroots=None, davidson_only=None, pspace_size=None, max_memory=None,
           orbsym=None, wfnsym=None, ecore=0, **kwargs):
    if 'verbose' in kwargs:
        verbose = kwargs['verbose']
        kwargs.pop ('verbose')
    else: verbose = lib.logger.Logger (stdout=fci.stdout, verbose=fci.verbose)
    if (isinstance (verbose, lib.logger.Logger) and verbose.verbose >= lib.logger.WARN) or verbose >= lib.logger.WARN:
        fci.check_sanity()
    if nroots is None: nroots = fci.nroots
    if pspace_size is None: pspace_size = fci.pspace_size
    if davidson_only is None: davidson_only = fci.davidson_only
    nelec = _unpack_nelec(nelec, fci.spin)
    neleca, nelecb = nelec
    hdiag_det = fci.make_hdiag (h1e, eri, norb, nelec)
    hdiag_csf = fci.make_hdiag_csf (h1e, eri, norb, nelec, hdiag_det=hdiag_det)
    ncsf_all = count_all_csfs (norb, neleca, nelecb, smult)
    if idx_sym is None:
        idx_sym = np.ones (ncsf_all, dtype=np.bool)
    ncsf_sym = np.count_nonzero (idx_sym)
    nroots = min(ncsf_sym, nroots)
    if nroots is not None:
        assert (ncsf_sym >= nroots), "Can't find {} roots among only {} CSFs".format (nroots, ncsf_sym)
    link_indexa, link_indexb = _unpack(norb, nelec, None)
    na = link_indexa.shape[0]
    nb = link_indexb.shape[0]

    addr, h0 = fci.pspace(h1e, eri, norb, nelec, idx_sym=idx_sym, hdiag_det=hdiag_det, hdiag_csf=hdiag_csf, npsp=max(pspace_size,nroots))
    lib.logger.debug (fci, 'csf_solver.kernel: error of hdiag_csf: %s', np.amax (np.abs (hdiag_csf[addr]-np.diag (h0))))
    if pspace_size > 0:
        pw, pv = fci.eig (h0)
    else:
        pw = pv = None

    if pspace_size >= ncsf_sym and not davidson_only:
        if ncsf_sym == 1:
            civec = np.zeros (ncsf_all, dtype=np.float64) 
            civec[idx_sym] = pv[:,0].reshape (1,1)
            civec = transform_civec_csf2det (civec, norb, neleca, nelecb, smult, csd_mask=fci.csd_mask)[0]
            return pw[0]+ecore, civec
        elif nroots > 1:
            civec = np.empty((nroots,ncsf_all))
            civec[:,addr] = pv[:,:nroots].T
            civec = transform_civec_csf2det (civec, norb, neleca, nelecb, smult, csd_mask=fci.csd_mask)[0]
            return pw[:nroots]+ecore, [c.reshape(na,nb) for c in civec]
        elif abs(pw[0]-pw[1]) > 1e-12:
            civec = np.empty((ncsf_all))
            civec[addr] = pv[:,0]
            civec = transform_civec_csf2det (civec, norb, neleca, nelecb, smult, csd_mask=fci.csd_mask)[0]
            return pw[0]+ecore, civec.reshape(na,nb)

    precond = fci.make_precond(hdiag_csf, pw, pv, addr)
    '''
    fci.eci, fci.ci = \
            kernel_ms1(fci, h1e, eri, norb, nelec, ci0, None,
                       tol, lindep, max_cycle, max_space, nroots,
                       davidson_only, pspace_size, ecore=ecore, **kwargs)
    '''
    h2e = fci.absorb_h1e(h1e, eri, norb, nelec, .5)
    def hop_nosym(x):
        x_det = transform_civec_csf2det (x, norb, neleca, nelecb, smult, csd_mask=fci.csd_mask)[0]
        hx = fci.contract_2e(h2e, x_det, norb, nelec, (link_indexa,link_indexb))
        hx = transform_civec_det2csf (hx, norb, neleca, nelecb, smult, csd_mask=fci.csd_mask, do_normalize=False)[0]
        return hx.ravel()
    hop = hop_nosym
    if ncsf_sym < ncsf_all:
        def hop_sym (x):
            # Memory wastage
            x_csf = np.zeros (ncsf_all, dtype=x.dtype)
            x_csf[idx_sym] = x
            x_csf = hop_nosym (x_csf)
            return (x_csf[idx_sym])
        hop = hop_sym

    if ci0 is None:
        if hasattr(fci, 'get_init_guess'):
            def ci0 ():
                x0 = transform_civec_det2csf (fci.get_init_guess(norb, nelec, nroots, hdiag_det), 
                    norb, neleca, nelecb, smult, csd_mask=fci.csd_mask)[0]
                x0 = [x[idx_sym] for x in x0]
                return x0
                    
        else:
            def ci0():  # lazy initialization to reduce memory footprint
                x0 = []
                for i in range(nroots):
                    x = np.zeros(ncsf_sym)
                    x[addr[i]] = 1
                    x0.append(x)
                return x0
    else:
        if isinstance(ci0, np.ndarray) and ci0.size == na*nb:
            ci0 = [transform_civec_det2csf (ci0.ravel (), norb, neleca, nelecb, smult, csd_mask=fci.csd_mask)[0]]
        else:
            ci0 = np.asarray (ci0)
            ci0 = ci0.reshape (ci0.shape[0], na*nb)
            ci0 = [x for x in transform_civec_det2csf (ci0, norb, neleca, nelecb, smult, csd_mask=fci.csd_mask, vec_on_cols=False)[0]]

    if tol is None: tol = fci.conv_tol
    if lindep is None: lindep = fci.lindep
    if max_cycle is None: max_cycle = fci.max_cycle
    if max_space is None: max_space = fci.max_space
    if max_memory is None: max_memory = fci.max_memory
    tol_residual = getattr(fci, 'conv_tol_residual', None)

    #with lib.with_omp_threads(fci.threads):
        #e, c = lib.davidson(hop, ci0, precond, tol=fci.conv_tol, lindep=fci.lindep)
    e, c = fci.eig(hop, ci0, precond, tol=tol, lindep=lindep,
                       max_cycle=max_cycle, max_space=max_space, nroots=nroots,
                       max_memory=max_memory, verbose=verbose, follow_state=True,
                       tol_residual=tol_residual, **kwargs)
    ci = np.zeros ((nroots, ncsf_all), dtype=np.asarray (c).dtype)
    ci[:,idx_sym] = np.asarray (c)
    c = transform_civec_csf2det (ci, norb, neleca, nelecb, smult, csd_mask=fci.csd_mask, vec_on_cols=False)[0]
    if nroots > 1:
        return e+ecore, [ci.reshape(na,nb) for ci in c]
    else:
        return e+ecore, c.reshape(na,nb)

class FCISolver (direct_spin1.FCISolver):
    r''' get_init_guess uses csfstring.py and csdstring.py to construct a spin-symmetry-adapted initial guess, and the Davidson algorithm is carried
    out in the CSF basis. However, the ci attribute is put in the determinant basis at the end of it all, and "ci0" is also assumed
    to be in the determinant basis.'''

    pspace_size = getattr(__config__, 'fci_csf_FCI_pspace_size', 200)

    def __init__(self, mol=None, smult=None):
        self.smult = smult
        self.csd_mask = self.econf_det_mask = self.econf_csf_mask = None
        self.mask_cache = [0, 0, 0, 0]
        super().__init__(mol)

    def get_init_guess(self, norb, nelec, nroots, hdiag):
        ''' The existing _get_init_guess function will work in the csf basis if I pass it with na, nb = ncsf, 1. This might change in future PySCF versions though. '''
        neleca, nelecb = _unpack_nelec (nelec)
        self.check_mask_cache ()
        hdiag_csf = transform_civec_det2csf (hdiag, norb, neleca, nelecb, self.smult, csd_mask=self.csd_mask, do_normalize=False)[0]
        ncsf = count_all_csfs (norb, neleca, nelecb, self.smult)
        assert (ncsf >= nroots), "Can't find {} roots among only {} CSFs".format (nroots, ncsf)
        ci_csf = _get_init_guess (ncsf, 1, nroots, hdiag_csf)
        ci = transform_civec_csf2det (ci_csf, norb, neleca, nelecb, self.smult, csd_mask=self.csd_mask)[0]
        return ci

    def make_hdiag_csf (self, h1e, eri, norb, nelec, hdiag_det=None):
        self.check_mask_cache ()
        return make_hdiag_csf (h1e, eri, norb, nelec, self.smult, csd_mask=self.csd_mask, hdiag_det=hdiag_det)



    '''
    01/14/2019: Changing strategy; I'm now replacing the kernel and pspace functions instead of make_precond and eig
    '''

    def pspace (self, h1e, eri, norb, nelec, hdiag_det=None, hdiag_csf=None, npsp=200, **kwargs):
        self.check_mask_cache ()
        return pspace (self, h1e, eri, norb, nelec, self.smult, hdiag_det=hdiag_det,
            hdiag_csf=hdiag_csf, npsp=npsp, csd_mask=self.csd_mask, idx_sym=None,
            econf_det_mask=self.econf_det_mask, econf_csf_mask=self.econf_csf_mask)
        
    def kernel(self, h1e, eri, norb, nelec, ci0=None, **kwargs):
           #ci0=None, smult=None,
           #tol=None, lindep=None, max_cycle=None, max_space=None,
           #nroots=None, davidson_only=None, pspace_size=None, max_memory=None,
           #orbsym=None, wfnsym=None, ecore=0, **kwargs):
        self.norb = norb
        self.nelec = nelec
        if 'smult' in kwargs:
            self.smult = kwargs['smult']
            kwargs.pop ('smult')
        self.check_mask_cache ()
        return kernel (self, h1e, eri, norb, nelec, smult=self.smult,
            idx_sym=None, ci0=ci0, **kwargs)

    def check_mask_cache (self):
        assert (isinstance (self.smult, (int, np.number)))
        neleca, nelecb = _unpack_nelec (self.nelec)
        if self.mask_cache != [self.norb, neleca, nelecb, self.smult] or self.csd_mask is None:
            self.csd_mask = make_csd_mask (self.norb, neleca, nelecb)
            self.econf_det_mask = make_econf_det_mask (self.norb, neleca, nelecb, self.csd_mask)
            self.econf_csf_mask = make_econf_csf_mask (self.norb, neleca, nelecb, self.smult)
            self.mask_cache = [self.norb, neleca, nelecb, self.smult]
