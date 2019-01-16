import numpy as np
import scipy
from pyscf import symm
from pyscf.lib import logger, davidson1
from pyscf.fci import direct_spin1_symm, cistring
from pyscf.fci.direct_spin1 import _unpack_nelec, _get_init_guess, kernel_ms1
from pyscf.fci.direct_spin1_symm import _gen_strs_irrep, _id_wfnsym
from mrh.my_pyscf.fci.csdstring import make_csd_mask, make_econf_det_mask, pretty_ddaddrs
from mrh.my_pyscf.fci.csfstring import transform_civec_det2csf, transform_civec_csf2det, transform_opmat_det2csf, count_all_csfs, make_econf_csf_mask
from mrh.my_pyscf.fci.csf import pspace, kernel

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

def unpack_sym_ci (ci, idx, vec_on_cols=False):
    tot_len = idx.size
    sym_len = np.count_nonzero (idx)
    if isinstance (ci, list) or isinstance (ci, tuple):
        assert (ci[0].size == sym_len), '{} {}'.format (ci[0].size, sym_len)
        dummy = np.zeros ((len (ci), tot_len), dtype=ci[0].dtype)
        dummy[:,idx] = np.asarray (ci)[:,:]
        if isinstance (ci, list):
            ci = list (dummy)
        else:
            ci = tuple (dummy)
        return ci
    elif ci.ndim == 2:
        if vec_on_cols:
                ci = ci.T
        assert (ci.shape[1] == sym_len), '{} {}'.format (ci.shape, sym_len)
        dummy = np.zeros ((ci.shape[0], tot_len), dtype=ci.dtype)
        dummy[:,idx] = ci
        if vec_on_cols:
            dummy = dummy.T
        return dummy
    else:
        assert (ci.ndim == 1), ci.ndim
        dummy = np.zeros (tot_len, dtype=ci.dtype)
        dummy[idx] = ci
        return dummy

def pack_sym_ci (ci, idx, vec_on_cols=False):
    tot_len = idx.size
    sym_len = np.count_nonzero (idx)
    if isinstance (ci, list) or isinstance (ci, tuple):
        assert (ci[0].size == tot_len), '{} {}'.format (ci[0].size, tot_len)
        dummy = np.asarray (ci)[:,idx]
        if isinstance (ci, list):
            ci = list (dummy)
        else:
            ci = tuple (dummy)
        return ci
    elif ci.ndim == 2:
        if vec_on_cols:
            ci = ci.T
        assert (ci.shape[1] == sym_len), '{} {}'.format (ci.shape, sym_len)
        dummy = ci[:,idx]
        if vec_on_cols:
            dummy = dummy.T
        return dummy
    else:
        assert (ci.ndim == 1)
        return ci[idx]
    

class FCISolver (direct_spin1_symm.FCISolver):
    r''' get_init_guess uses csfstring.py and csdstring.py to construct a spin-symmetry-adapted initial guess, and the Davidson algorithm is carried
    out in the CSF basis. However, the ci attribute is put in the determinant basis at the end of it all, and "ci0" is also assumed
    to be in the determinant basis.

    ...However, I want to also do point-group symmetry better than direct_spin1_symm...
    '''

    def __init__(self, mol=None, smult=None):
        self.smult = smult
        self.csd_mask = self.econf_det_mask = self.econf_csf_mask = None
        self.mask_cache = [0, 0, 0, 0]
        self.confsym = None
        self.orbsym_cache = None
        super().__init__(mol)

    def get_init_guess(self, norb, nelec, nroots, hdiag):
        ''' The existing _get_init_guess function will work in the csf basis if I pass it with na, nb = ncsf, 1. This might change in future PySCF versions though. 

        ...For point-group symmetry, I pass the direct_spin1.py version of _get_init_guess with na, nb = ncsf_sym, 1 and hdiag_csf including only csfs of the right point-group symmetry.
        This should clean up the symmetry-breaking "noise" in direct_spin1_symm.py! '''
        neleca, nelecb = _unpack_nelec (nelec)
        wfnsym = _id_wfnsym(self, norb, nelec, self.wfnsym)
        wfnsym_str = symm.irrep_id2name (self.mol.groupname, wfnsym)
        self.check_mask_cache ()
        hdiag_csf = transform_civec_det2csf (hdiag, norb, neleca, nelecb, self.smult, csd_mask=self.csd_mask, do_normalize=False)[0]
        idx_sym = self.confsym[self.econf_csf_mask] == wfnsym
        ncsf_tot = count_all_csfs (norb, neleca, nelecb, self.smult)
        ncsf_sym = np.count_nonzero (idx_sym)
        assert (ncsf_sym >= nroots), "Can't find {} roots among only {} CSFs of symmetry {}".format (nroots, ncsf_sym, wfnsym_str)
        ci = _get_init_guess (ncsf_sym, 1, nroots, hdiag_csf[idx_sym])
        ci = unpack_sym_ci (ci, idx_sym)
        ci = transform_civec_csf2det (ci, norb, neleca, nelecb, self.smult, csd_mask=self.csd_mask)[0]
        return ci

    def kernel(self, h1e, eri, norb, nelec, ci0=None, smult=None,
               tol=None, lindep=None, max_cycle=None, max_space=None,
               nroots=None, davidson_only=None, pspace_size=None,
               orbsym=None, wfnsym=None, ecore=0, **kwargs):
        ''' Over the top of the existing kernel, I just need to set the parameters and cache values related to spin.

        ...and electron configuration point group '''
        if nroots is None: nroots = self.nroots
        orbsym_back = self.orbsym
        if orbsym is None: orbsym = self.orbsym
        wfnsym_back = self.wfnsym
        if wfnsym is None: wfnsym = self.wfnsym
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.norb = norb
        self.nelec = nelec
        if smult is not None:
            self.smult = smult

        # The order of the four things below is super sensitive
        self.orbsym = orbsym
        wfnsym = self.guess_wfnsym(norb, nelec, ci0, wfnsym, **kwargs)
        self.wfnsym = wfnsym
        self.check_mask_cache ()

        nroots_test = nroots or 1
        ncsf = np.count_nonzero (self.confsym[self.econf_csf_mask] == wfnsym)
        assert (ncsf >= nroots_test), "Can't find {} roots among only {} CSFs with symmetry {}\n{}\n{}".format (
            nroots_test, ncsf, wfnsym, self.confsym[self.econf_csf_mask], self.econf_csf_mask)
        e, c = kernel_ms1(self, h1e, eri, norb, nelec, ci0, None,
                          tol, lindep, max_cycle, max_space,
                          nroots, davidson_only, pspace_size,
                          ecore=ecore, **kwargs)
        self.eci, self.ci = e, c

        self.orbsym = orbsym_back
        self.wfnsym = wfnsym_back
        return e, c


    def make_precond(self, hdiag, pspaceig, pspaceci, addr):
        ''' I need to transform hdiag, pspaceci, and addr into the CSF basis
        addr is trickiest. I match the determinant address to the electron configuration,
        and from there match the electron configuration to the CSFs. '''

        norb, smult = self.norb, self.smult
        neleca, nelecb = _unpack_nelec (self.nelec)
        self.check_mask_cache ()
        wfnsym = self.wfnsym
        if isinstance (wfnsym, str):
            wfnsym = symm.irrep_name2id (self.mol.groupname, wfnsym)
        idx_sym = self.confsym[self.econf_csf_mask] == wfnsym

        hdiag = transform_civec_det2csf (hdiag, norb, neleca, nelecb, self.smult, csd_mask=self.csd_mask, do_normalize=False)[0]
        hdiag = pack_sym_ci (hdiag, idx_sym)
        if pspaceci is not None:
            pspaceci = transform_civec_det2csf (pspaceci, norb, neleca, nelecb, self.smult, csd_mask=self.csd_mask, vec_on_cols=True)[0]
            pspaceci = pack_sym_ci (pspaceci, idx_sym, vec_on_cols=True)
        addr = np.isin (self.econf_csf_mask, np.unique (self.econf_det_mask[addr]))
        addr = np.nonzero (addr & idx_sym)[0]

        return super().make_precond (hdiag, pspaceig, pspaceci, addr)

    def eig(self, op, x0=None, precond=None, **kwargs):
        r''' op and x0 need to be put in the CSF basis. '''
        norb = self.norb
        neleca, nelecb = _unpack_nelec (self.nelec)
        self.check_mask_cache ()
        wfnsym = self.wfnsym
        if isinstance (wfnsym, str):
            wfnsym = symm.irrep_name2id (self.mol.groupname, wfnsym)
        idx_sym = self.confsym[self.econf_csf_mask] == wfnsym
        if isinstance(op, np.ndarray):
            assert (isinstance (self.smult, (int, np.number)))
            self.converged = True
            op = transform_opmat_det2csf (op, norb, neleca, nelecb, self.smult, csd_mask=self.csd_mask)
            op = op[np.ix_(idx_sym,idx_sym)]
            e, ci = scipy.linalg.eigh (op)
            ci = unpack_sym_ci (ci, idx_sym, vec_on_cols=True)
            ci = transform_civec_csf2det (ci, norb, neleca, nelecb, self.smult, csd_mask=self.csd_mask, vec_on_cols=True)[0]
            return e, ci
        def op_csf (x):
            x = unpack_sym_ci (x, idx_sym)
            x = transform_civec_csf2det (x, norb, neleca, nelecb, self.smult, csd_mask=self.csd_mask)[0]
            hx = op (x)
            hx = transform_civec_det2csf (hx, norb, neleca, nelecb, self.smult, csd_mask=self.csd_mask, do_normalize=False)[0]
            hx = pack_sym_ci (hx, idx_sym) 
            return hx
        try:
            x0_csf = pack_sym_ci (transform_civec_det2csf (x0, norb, neleca, nelecb, self.smult, csd_mask=self.csd_mask)[0], idx_sym)
        except AttributeError as e:
            x0_csf = lambda: pack_sym_ci (transform_civec_det2csf (x0 (), norb, neleca, nelecb, self.smult, csd_mask=self.csd_mask)[0], idx_sym)
        self.converged, e, ci = \
                davidson1(lambda xs: [op_csf(x) for x in xs],
                              x0_csf, precond, lessio=self.lessio, **kwargs)
        ci = unpack_sym_ci (ci, idx_sym)
        ci = transform_civec_csf2det (ci, norb, neleca, nelecb, self.smult, csd_mask=self.csd_mask)[0]
        if kwargs['nroots'] == 1:
            self.converged = self.converged[0]
            e = e[0]
            ci = ci[0]

        return e, ci

    '''
    def eig(self, op, x0=None, precond=None, **kwargs):
        if isinstance(op, numpy.ndarray):
            self.converged = True
            return scipy.linalg.eigh(op)

        self.converged, e, ci = \
                lib.davidson1(lambda xs: [op(x) for x in xs],
                              x0, precond, lessio=self.lessio, **kwargs)
        if kwargs['nroots'] == 1:
            self.converged = self.converged[0]
            e = e[0]
            ci = ci[0]
        return e, ci
    '''

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

