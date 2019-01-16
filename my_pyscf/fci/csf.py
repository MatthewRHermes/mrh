import numpy as np
import scipy
import ctypes
from pyscf import lib, ao2mo
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


class FCISolver (direct_spin1.FCISolver):
    r''' get_init_guess uses csfstring.py and csdstring.py to construct a spin-symmetry-adapted initial guess, and the Davidson algorithm is carried
    out in the CSF basis. However, the ci attribute is put in the determinant basis at the end of it all, and "ci0" is also assumed
    to be in the determinant basis.'''

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

    def kernel(self, h1e, eri, norb, nelec, ci0=None, smult=None,
               tol=None, lindep=None, max_cycle=None, max_space=None,
               nroots=None, davidson_only=None, pspace_size=None, max_memory=None,
               orbsym=None, wfnsym=None, ecore=0, **kwargs):
        ''' Over the top of the existing kernel, I just need to set the parameters and cache values related to spin. '''
        if 'verbose' in kwargs:
            verbose = kwargs['verbose']
            kwargs.pop ('verbose')
        else: verbose = lib.logger.Logger (stdout=self.stdout, verbose=self.verbose)
        if (isinstance (verbose, lib.logger.Logger) and verbose.verbose >= lib.logger.WARN) or verbose >= lib.logger.WARN:
            self.check_sanity()
        self.norb = norb
        self.nelec = nelec
        if nroots is None: nroots = self.nroots
        if pspace_size is None: pspace_size = self.pspace_size
        if davidson_only is None: davidson_only = self.davidson_only
        nelec = _unpack_nelec(nelec, self.spin)
        neleca, nelecb = nelec
        if smult is not None:
            self.smult = smult
        self.check_mask_cache ()
        hdiag_det = self.make_hdiag (h1e, eri, norb, nelec)
        hdiag_csf = self.make_hdiag_csf (h1e, eri, norb, nelec, hdiag_det=hdiag_det)
        ncsf = count_all_csfs (norb, neleca, nelecb, self.smult)
        nroots = min(ncsf, nroots)
        if nroots is not None:
            assert (ncsf >= nroots), "Can't find {} roots among only {} CSFs".format (nroots, ncsf)
        link_indexa, link_indexb = _unpack(norb, nelec, None)
        na = link_indexa.shape[0]
        nb = link_indexb.shape[0]
    
        addr, h0 = self.pspace(h1e, eri, norb, nelec, hdiag_det, max(pspace_size,nroots))
        lib.logger.debug (self, 'csf_solver.kernel: error of hdiag_csf: %s', scipy.linalg.norm (hdiag_csf[addr]-np.diag (h0)))
        if pspace_size > 0:
            pw, pv = scipy.linalg.eigh (h0)
        else:
            pw = pv = None

        if pspace_size >= ncsf and not davidson_only:
            if ncsf == 1:
                civec = transform_civec_csf2det (pv[:,0].reshape (1,1), norb, neleca, nelecb, self.smult, csd_mask=self.csd_mask)[0]
                return pw[0]+ecore, civec
            elif nroots > 1:
                civec = np.empty((nroots,ncsf))
                civec[:,addr] = pv[:,:nroots].T
                civec = transform_civec_csf2det (civec, norb, neleca, nelecb, self.smult, csd_mask=self.csd_mask)[0]
                return pw[:nroots]+ecore, [c.reshape(na,nb) for c in civec]
            elif abs(pw[0]-pw[1]) > 1e-12:
                civec = np.empty((ncsf))
                civec[addr] = pv[:,0]
                civec = transform_civec_csf2det (civec, norb, neleca, nelecb, self.smult, csd_mask=self.csd_mask)[0]
                return pw[0]+ecore, civec.reshape(na,nb)
 
        precond = self.make_precond(hdiag_csf, pw, pv, addr)
        '''
        self.eci, self.ci = \
                kernel_ms1(self, h1e, eri, norb, nelec, ci0, None,
                           tol, lindep, max_cycle, max_space, nroots,
                           davidson_only, pspace_size, ecore=ecore, **kwargs)
        '''
        h2e = self.absorb_h1e(h1e, eri, norb, nelec, .5)
        def hop(x):
            x_det = transform_civec_csf2det (x, norb, neleca, nelecb, self.smult, csd_mask=self.csd_mask)[0]
            hx = self.contract_2e(h2e, x_det, norb, nelec, (link_indexa,link_indexb))
            hx = transform_civec_det2csf (hx, norb, neleca, nelecb, self.smult, csd_mask=self.csd_mask, do_normalize=False)[0]
            return hx.ravel()

        if ci0 is None:
            if hasattr(self, 'get_init_guess'):
                ci0 = lambda: transform_civec_det2csf (self.get_init_guess(norb, nelec, nroots, hdiag_det), 
                    norb, neleca, nelecb, self.smult, csd_mask=self.csd_mask)[0]
            else:
                def ci0():  # lazy initialization to reduce memory footprint
                    x0 = []
                    for i in range(nroots):
                        x = np.zeros(ncsf)
                        x[addr[i]] = 1
                        x0.append(x)
                    return x0
        else:
            if isinstance(ci0, np.ndarray) and ci0.size == na*nb:
                ci0 = [transform_civec_det2csf (ci0.ravel (), norb, neleca, nelecb, self.smult, csd_mask=self.csd_mask)[0]]
            else:
                ci0 = np.asarray (ci0)
                ci0 = ci0.reshape (ci0.shape[0], na*nb)
                ci0 = [x for x in transform_civec_det2csf (ci0, norb, neleca, nelecb, self.smult, csd_mask=self.csd_mask, vec_on_cols=False)[0]]

        if tol is None: tol = self.conv_tol
        if lindep is None: lindep = self.lindep
        if max_cycle is None: max_cycle = self.max_cycle
        if max_space is None: max_space = self.max_space
        if max_memory is None: max_memory = self.max_memory
        tol_residual = getattr(self, 'conv_tol_residual', None)

        #with lib.with_omp_threads(self.threads):
            #e, c = lib.davidson(hop, ci0, precond, tol=self.conv_tol, lindep=self.lindep)
        e, c = self.eig(hop, ci0, precond, tol=tol, lindep=lindep,
                           max_cycle=max_cycle, max_space=max_space, nroots=nroots,
                           max_memory=max_memory, verbose=verbose, follow_state=True,
                           tol_residual=tol_residual, **kwargs)
        c = transform_civec_csf2det (c, norb, neleca, nelecb, self.smult, csd_mask=self.csd_mask, vec_on_cols=False)[0]
        if nroots > 1:
            return e+ecore, [ci.reshape(na,nb) for ci in c]
        else:
            return e+ecore, c.reshape(na,nb)


    '''
    01/14/2019: Changing strategy; I'm now replacing the kernel and pspace functions instead of make_precond and eig
    '''

    def pspace (self, h1e, eri, norb, nelec, hdiag=None, npsp=400):
        ''' npsp will be taken as the minimum number of determinants to include
        the actual number of determinants included in pspace will be determined
        to span the electron configurations of the npsp lowest-energy determinants '''
        if norb > 63:
            raise NotImplementedError('norb > 63')
    
        neleca, nelecb = _unpack_nelec(nelec)
        h1e = np.ascontiguousarray(h1e)
        eri = ao2mo.restore(1, eri, norb)
        nb = cistring.num_strings(norb, nelecb)
        self.check_mask_cache ()
        if hdiag is None:
            self.make_hdiag(h1e, eri, norb, nelec)
        if hdiag.size < npsp:
            addr = np.arange(hdiag.size)
        else:
            try:
                addr = np.argpartition(hdiag, npsp-1)[:npsp]
            except AttributeError:
                addr = np.argsort(hdiag)[:npsp]

        # Now make sure that the addrs span full electron configurations
        addr_econfs = np.unique (self.econf_det_mask[addr])
        addr = np.concatenate ([np.nonzero (self.econf_det_mask == conf)[0] for conf in addr_econfs])
        lib.logger.debug (self, ("csf_solver.pspace: Lowest-energy %s determinants correspond to %s configurations"
            " which are spanned by %s determinants"), npsp, addr_econfs.size, addr.size)

        addra, addrb = divmod(addr, nb)
        stra = cistring.addrs2str(norb, neleca, addra)
        strb = cistring.addrs2str(norb, nelecb, addrb)
        npsp = len(addr)
        h0 = np.zeros((npsp,npsp))
        libfci.FCIpspace_h0tril(h0.ctypes.data_as(ctypes.c_void_p),
                                h1e.ctypes.data_as(ctypes.c_void_p),
                                eri.ctypes.data_as(ctypes.c_void_p),
                                stra.ctypes.data_as(ctypes.c_void_p),
                                strb.ctypes.data_as(ctypes.c_void_p),
                                ctypes.c_int(norb), ctypes.c_int(npsp))
    
        for i in range(npsp):
            h0[i,i] = hdiag[addr[i]]
        h0 = lib.hermi_triu(h0)

        if self.verbose >= lib.logger.DEBUG: evals_before = scipy.linalg.eigh (h0)[0]

        h0, csf_addr = transform_opmat_det2csf_pspace (h0, addr_econfs, norb, neleca, nelecb, self.smult,
            self.csd_mask, self.econf_det_mask, self.econf_csf_mask) 

        lib.logger.debug (self, "csf_solver.pspace: pspace of %s configurations corresponds to %s CSFs", addr_econfs.size, csf_addr.size)

        if self.verbose >= lib.logger.DEBUG:
            lib.logger.debug2 (self, "csf_solver.pspace: eigenvalues of h0 before transformation %s", evals_before)
            evals_after = scipy.linalg.eigh (h0)[0]
            lib.logger.debug2 (self, "csf_solver.pspace: eigenvalues of h0 after transformation %s", evals_after)
            idx = [np.argmin (np.abs (evals_before - ev)) for ev in evals_after]
            resid = evals_after - evals_before[idx]
            lib.logger.debug2 (self, "csf_solver.pspace: best h0 eigenvalue matching differences after transformation: %s", resid)
            lib.logger.debug (self, "csf_solver.pspace: if the transformation of h0 worked the following number will be zero: %s", scipy.linalg.norm (resid))
                
        return csf_addr, h0


    def check_mask_cache (self):
        assert (isinstance (self.smult, (int, np.number)))
        neleca, nelecb = _unpack_nelec (self.nelec)
        if self.mask_cache != [self.norb, neleca, nelecb, self.smult] or self.csd_mask is None:
            self.csd_mask = make_csd_mask (self.norb, neleca, nelecb)
            self.econf_det_mask = make_econf_det_mask (self.norb, neleca, nelecb, self.csd_mask)
            self.econf_csf_mask = make_econf_csf_mask (self.norb, neleca, nelecb, self.smult)
            self.mask_cache = [self.norb, neleca, nelecb, self.smult]
