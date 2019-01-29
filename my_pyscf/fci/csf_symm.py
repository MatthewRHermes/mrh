import numpy as np
import scipy
from pyscf import symm, __config__
from pyscf.lib import logger, davidson1
from pyscf.fci import direct_spin1_symm, cistring
from pyscf.fci.direct_spin1 import _unpack_nelec, _get_init_guess, kernel_ms1
from pyscf.fci.direct_spin1_symm import _gen_strs_irrep, _id_wfnsym
from mrh.my_pyscf.fci.csdstring import make_csd_mask, make_econf_det_mask, pretty_ddaddrs
from mrh.my_pyscf.fci.csfstring import transform_civec_det2csf, transform_civec_csf2det, transform_opmat_det2csf, count_all_csfs, make_econf_csf_mask
from mrh.my_pyscf.fci.csf import kernel, pspace, get_init_guess, make_hdiag_csf

def make_confsym (norb, neleca, nelecb, econf_det_mask, orbsym):
    strsa = cistring.gen_strings4orblist(range(norb), neleca)
    airreps = birreps = _gen_strs_irrep(strsa, orbsym)
    if neleca != nelecb:
        strsb = cistring.gen_strings4orblist(range(norb), nelecb)
        birreps = _gen_strs_irrep(strsb, orbsym)
    nconf, addr = np.unique (econf_det_mask, return_index=True)
    nconf = nconf.size
    # Note: econf_det_mask[addr] = np.arange (nconf)
    # All determinants of the same configuration have the same point group
    conf_addra = addr // len (birreps)
    conf_addrb = addr % len (birreps)
    confsym = airreps[conf_addra] ^ birreps[conf_addrb]
    return confsym

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

