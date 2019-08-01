import numpy as np
import scipy
from pyscf import symm, __config__
from pyscf.lib import logger, davidson1
from pyscf.fci import direct_spin1_symm, cistring, direct_uhf
from pyscf.lib.numpy_helper import tag_array
from pyscf.fci.direct_spin1 import _unpack_nelec, _get_init_guess, kernel_ms1
from pyscf.fci.direct_spin1_symm import _gen_strs_irrep, _id_wfnsym
from mrh.my_pyscf.fci.csdstring import make_csd_mask, make_econf_det_mask, pretty_ddaddrs
from mrh.my_pyscf.fci.csfstring import transform_civec_det2csf, transform_civec_csf2det, transform_opmat_det2csf, count_all_csfs, make_econf_csf_mask, make_confsym
from mrh.my_pyscf.fci.csf import kernel, pspace, get_init_guess, make_hdiag_csf, make_hdiag_det, unpack_h1e_cs
'''
    MRH 03/24/2019
    IMPORTANT: this solver will interpret a two-component one-body Hamiltonian as [h1e_charge, h1e_spin] where
    h1e_charge = h^p_q (a'_p,up a_q,up + a'_p,down a_q,down)
    h1e_spin   = h^p_q (a'_p,up a_q,up - a'_p,down a_q,down)
    This is to preserve interoperability with the members of direct_spin1_symm, since there is no direct_uhf_symm in pyscf yet.
    Only with an explicitly CSF-based solver can such potentials be included in a calculation that retains S^2 symmetry.
    Multicomponent two-body integrals are currently not available (so this feature is only for use with, e.g., ROHF-CASSCF with 
    with some SOMOs outside of the active space or LASSCF with multiple nonsinglet fragments, not UHF-CASSCF).
'''


class FCISolver (direct_spin1_symm.FCISolver):
    r''' get_init_guess uses csfstring.py and csdstring.py to construct a spin-symmetry-adapted initial guess, and the Davidson algorithm is carried
    out in the CSF basis. However, the ci attribute is put in the determinant basis at the end of it all, and "ci0" is also assumed
    to be in the determinant basis.

    ...However, I want to also do point-group symmetry better than direct_spin1_symm...
    '''

    pspace_size = getattr(__config__, 'fci_csf_FCI_pspace_size', 200)

    def __init__(self, mol=None, smult=None):
        self.smult = smult
        self.csd_mask = self.econf_det_mask = self.econf_csf_mask = None
        self.mask_cache = [0, 0, 0, 0]
        self.confsym = None
        self.orbsym_cache = None
        super().__init__(mol)

    make_hdiag = make_hdiag_det

    def absorb_h1e (self, h1e, eri, norb, nelec, fac=1):
        h1e_c, h1e_s = unpack_h1e_cs (h1e)
        h2eff = super().absorb_h1e (h1e_c, eri, norb, nelec, fac)
        if h1e_s is not None:
            h2eff = tag_array (h2eff, h1e_s=h1e_s)
        return h2eff

    def contract_2e(self, eri, fcivec, norb, nelec, link_index=None, **kwargs):
        hc = super().contract_2e(eri, fcivec, norb, nelec, link_index, **kwargs)
        if hasattr (eri, 'h1e_s'):
           hc += direct_uhf.contract_1e ([eri.h1e_s, -eri.h1e_s], fcivec, norb, nelec, link_index)  
        return hc

    def make_hdiag_csf (self, h1e, eri, norb, nelec, hdiag_det=None):
        self.check_mask_cache ()
        return make_hdiag_csf (h1e, eri, norb, nelec, self.smult, csd_mask=self.csd_mask, hdiag_det=hdiag_det)

    def get_init_guess(self, norb, nelec, nroots, hdiag_csf):
        ''' The existing _get_init_guess function will work in the csf basis if I pass it with na, nb = ncsf, 1. This might change in future PySCF versions though. 

        ...For point-group symmetry, I pass the direct_spin1.py version of _get_init_guess with na, nb = ncsf_sym, 1 and hdiag_csf including only csfs of the right point-group symmetry.
        This should clean up the symmetry-breaking "noise" in direct_spin0_symm.py! '''
        wfnsym = _id_wfnsym(self, norb, nelec, self.orbsym, self.wfnsym)
        wfnsym_str = symm.irrep_id2name (self.mol.groupname, wfnsym)
        self.check_mask_cache ()
        idx_sym = self.confsym[self.econf_csf_mask] == wfnsym
        return get_init_guess (norb, nelec, nroots, hdiag_csf, smult=self.smult, csd_mask=self.csd_mask,
            wfnsym_str=wfnsym_str, idx_sym=idx_sym)

    def pspace (self, h1e, eri, norb, nelec, hdiag_det=None, hdiag_csf=None, npsp=200, **kwargs):
        self.check_mask_cache ()
        if 'wfnsym' in kwargs:
            idx_sym = self.confsym[self.econf_csf_mask] == kwargs['wfnsym']
        else:
            idx_sym = self.confsym[self.econf_csf_mask] == self.wfnsym
        return pspace (self, h1e, eri, norb, nelec, self.smult, hdiag_det=hdiag_det,
            hdiag_csf=hdiag_csf, npsp=npsp, csd_mask=self.csd_mask, idx_sym=idx_sym,
            econf_det_mask=self.econf_det_mask, econf_csf_mask=self.econf_csf_mask)

    def kernel(self, h1e, eri, norb, nelec, ci0=None, **kwargs):
        ''' Over the top of the existing kernel, I just need to set the parameters and cache values related to spin.

        ...and electron configuration point group '''
        if 'nroots' not in kwargs:
            nroots = self.nroots
            kwargs['nroots'] = nroots
        orbsym_back = self.orbsym
        if 'orbsym' not in kwargs:
            kwargs['orbsym'] = self.orbsym
        orbsym = kwargs['orbsym']
        wfnsym_back = self.wfnsym
        if 'wfnsym' not in kwargs:
            wfnsym = self.wfnsym
            kwargs['wfnsym'] = wfnsym
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.norb = norb
        self.nelec = nelec
        if 'smult' in kwargs:
            self.smult = kwargs['smult']
            kwargs.pop ('smult')

        # The order of the four things below is super sensitive
        self.orbsym = orbsym
        wfnsym = self.guess_wfnsym(norb, nelec, ci0, **kwargs)
        self.wfnsym = wfnsym
        kwargs['wfnsym'] = wfnsym
        self.check_mask_cache ()

        idx_sym = self.confsym[self.econf_csf_mask] == wfnsym
        e, c = kernel (self, h1e, eri, norb, nelec, smult=self.smult, idx_sym=idx_sym, ci0=ci0, **kwargs)
        self.eci, self.ci = e, c

        self.orbsym = orbsym_back
        self.wfnsym = wfnsym_back
        return e, c

    def check_mask_cache (self):
        assert (isinstance (self.smult, (int, np.number)))
        neleca, nelecb = _unpack_nelec (self.nelec)
        if self.mask_cache != [self.norb, neleca, nelecb, self.smult] or self.csd_mask is None:
            self.csd_mask = make_csd_mask (self.norb, neleca, nelecb)
            self.econf_det_mask = make_econf_det_mask (self.norb, neleca, nelecb, self.csd_mask)
            self.econf_csf_mask = make_econf_csf_mask (self.norb, neleca, nelecb, self.smult)
            self.mask_cache = [self.norb, neleca, nelecb, self.smult]
        if self.orbsym_cache is None or (not np.all (self.orbsym == self.orbsym_cache)):
            self.confsym = make_confsym (self.norb, neleca, nelecb, self.econf_det_mask, self.orbsym)
            self.orbsym_cache = np.array (self.orbsym, copy=True)

