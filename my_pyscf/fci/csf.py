import numpy as np
import scipy
from pyscf.lib import logger
from pyscf.fci import direct_spin1, cistring
from pyscf.fci.direct_spin1 import _unpack_nelec, _get_init_guess, kernel_ms1
from mrh.my_pyscf.fci.csdstring import make_csd_mask
from mrh.my_pyscf.fci.csfstring import transform_civec_det2csf, transform_civec_csf2det, transform_opmat_det2csf, count_all_csfs

class FCISolver (direct_spin1.FCISolver):
    r''' get_init_guess uses csfstring.py to construct a spin-symmetry-adapted initial guess
    So far, nothing else is different. For some reason, the changes to eig leads to numerical noise if
    a CASSCF problem is small enough to skip the Davidson algorithm (and PySCF FORCES it to skip the Davidson
    algorithm, unfortunately) '''

    def __init__(self, mol=None, smult=None):
        self.smult = smult
        self.csd_mask = None
        self.csd_mask_cache = [0, 0, 0]
        super().__init__(mol)

    def get_init_guess(self, norb, nelec, nroots, hdiag):
        neleca, nelecb = _unpack_nelec (nelec)
        na = cistring.num_strings(norb, neleca)
        nb = cistring.num_strings(norb, nelecb)
        assert (isinstance (self.smult, (int, np.number)))
        if self.csd_mask_cache != [norb, neleca, nelecb] or self.csd_mask is None:
            self.csd_mask = make_csd_mask (norb, neleca, nelecb)
            self.csd_mask_cache = [norb, neleca, nelecb]
        hdiag_csf = transform_civec_det2csf (hdiag, norb, neleca, nelecb, self.smult, csd_mask=self.csd_mask)[0]
        ncsf = count_all_csfs (norb, neleca, nelecb, self.smult)
        assert (ncsf >= nroots), "Can't find {} roots among only {} CSFs".format (nroots, ncsfs)
        ci_csf = _get_init_guess (ncsf, 1, nroots, hdiag_csf)
        ci = [transform_civec_csf2det (c, norb, neleca, nelecb, self.smult, csd_mask=self.csd_mask)[0].ravel () for c in ci_csf]
        return ci

    def kernel(self, h1e, eri, norb, nelec, ci0=None, smult=None,
               tol=None, lindep=None, max_cycle=None, max_space=None,
               nroots=None, davidson_only=None, pspace_size=None,
               orbsym=None, wfnsym=None, ecore=0, **kwargs):
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.norb = norb
        self.nelec = nelec
        neleca, nelecb = _unpack_nelec (nelec)
        if smult is not None:
            self.smult = smult
        assert (isinstance (self.smult, (int, np.number)))
        if nroots is not None:
            ncsf = count_all_csfs (norb, neleca, nelecb, self.smult)
            assert (ncsf >= nroots), "Can't find {} roots among only {} CSFs".format (nroots, ncsfs)
        if self.csd_mask_cache != [norb, neleca, nelecb] or self.csd_mask is None:
            self.csd_mask = make_csd_mask (norb, neleca, nelecb)
            self.csd_mask_cache = [norb, neleca, nelecb]
        self.eci, self.ci = \
                kernel_ms1(self, h1e, eri, norb, nelec, ci0, None,
                           tol, lindep, max_cycle, max_space, nroots,
                           davidson_only, pspace_size, ecore=ecore, **kwargs)
        return self.eci, self.ci

    def eig(self, op, x0=None, precond=None, **kwargs):
        r''' put the operator matrix in the csf basis!!! '''
        if isinstance(op, np.ndarray):
            assert (isinstance (self.smult, (int, np.number)))
            norb = self.norb
            neleca, nelecb = _unpack_nelec (self.nelec)
            if self.csd_mask_cache != [norb, neleca, nelecb] or self.csd_mask is None:
                self.csd_mask = make_csd_mask (norb, neleca, nelecb)
                self.csd_mask_cache = [norb, neleca, nelecb]
            self.converged = True
            op_csf = transform_opmat_det2csf (op, norb, neleca, nelecb, self.smult, csd_mask=self.csd_mask)
            e, ci = scipy.linalg.eigh (op_csf)
            ci = np.stack ([transform_civec_csf2det (ci[:,i], norb, neleca, nelecb,
                self.smult, csd_mask=self.csd_mask)[0].ravel () for i in range (ci.shape[-1])], axis=-1)
            return e, ci
        return super().eig (op, x0, precond, **kwargs)



