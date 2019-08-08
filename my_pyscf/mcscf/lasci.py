from pyscf.mcscf import casci, casci_symm, df
from pyscf import symm, gto, scf, ao2mo, lib
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.scf import hf_as
from itertools import combinations
from scipy import linalg
import numpy as np
import time

# This must be locked to CSF solver for the forseeable future, because I know of no other way to handle spin-breaking potentials while retaining spin constraint
# There's a lot that will have to be checked in the future with spin-breaking stuff, especially if I still have the convention ms < 0, na <-> nb and h1e_s *= -1 

def LASCI (mf_or_mol, ncas_sub, nelecas_sub, **kwargs):
    if isinstance(mf_or_mol, gto.Mole):
        mf = scf.RHF(mf_or_mol)
    else:
        mf = mf_or_mol
    if mf.mol.symmetry: 
        las = LASCISymm (mf, ncas_sub, nelecas_sub, **kwargs)
    else:
        las = LASCINoSymm (mf, ncas_sub, nelecas_sub, **kwargs)
    if getattr (mf, 'with_df', None):
        las = density_fit (las, with_df = mf.with_df) 
    return las

class _DFLASCI: # Tag
    pass

def get_grad (las, mo_coeff=None, ci=None, fock=None, h1eff_sub=None, h2eff_sub=None, veff_sub=None):
    ''' Return energy gradient for 1) inactive-external orbital rotation and 2) CI relaxation.
    Eventually to include 3) intersubspace orbital rotation. '''
    if mo_coeff is None: mo_coeff = las.mo_coeff
    if ci is None: ci = las.ci
    if veff_sub is None: veff_sub = las.get_veff (mo_coeff=mo_coeff, ci=ci)
    if fock is None: fock = las.get_fock (veff_sub=veff_sub)
    if h1eff_sub is None: h1eff_sub = las.get_h1eff (mo_coeff, veff_sub=veff_sub)
    if h2eff_sub is None: h2eff_sub = las.get_h2eff (mo_coeff)
    nao, nmo = mo_coeff.shape
    ncore = las.ncore
    nocc = las.ncore + las.ncas
    nvirt = nmo - nocc

    # Inactive-external orbital rotation
    mo_inac = mo_coeff[:,:ncore]
    moH_virt = mo_coeff[:,nocc:].conjugate ().T
    gorb = 2 * moH_virt @ fock @ mo_inac

    # The CI part
    gci = []
    for isub, (h1eff, ci0, ncas, nelecas) in enumerate (zip (h1eff_sub, ci, las.ncas_sub, las.nelecas_sub)):
        eri_cas = las.get_h2eff_slice (h2eff_sub, isub, compact=8)
        max_memory = max(400, las.max_memory-lib.current_memory()[0])
        h1eff_c = (h1eff[0] + h1eff[1]) / 2
        h1eff_s = (h1eff[0] - h1eff[1]) / 2
        nel = nelecas
        # CI solver has enforced convention: na >= nb
        if nelecas[0] < nelecas[1]:
            nel = (nel[1], nel[0])
            h1eff_s *= -1
        h1e = (h1eff_c, h1eff_s)
        if getattr(las.fcisolver, 'gen_linkstr', None):
            linkstrl = las.fcisolver.gen_linkstr(ncas, nelecas, True)
            linkstr  = las.fcisolver.gen_linkstr(ncas, nelecas, False)
        else:
            linkstrl = linkstr  = None
        h2eff = las.fcisolver.absorb_h1e(h1e, eri_cas, ncas, nelecas, .5)
        hc0 = las.fcisolver.contract_2e(h2eff, ci0, ncas, nelecas, link_index=linkstrl).ravel()
        ci0 = ci0.ravel ()
        eci0 = ci0.dot(hc0)
        gci.append ((hc0 - ci0 * eci0).ravel ())

    # The external part. Semi-cumulant decomposition works between active/inactive but not among active subspaces
    dm1 = las.make_casdm1 (ci=ci)
    mo_cas = mo_coeff[:,ncore:nocc]
    moH = mo_coeff.conjugate ().T
    gx = moH @ fock @ mo_cas @ dm1
    dm2 = las.make_casdm2 (ci = ci)
    dm1_outer = np.multiply.outer (dm1, dm1)
    dm1_outer -= dm1_outer.transpose (0,3,2,1) / 2 
    dm2 -= dm1_outer
    dm2 = (dm2 + dm2.transpose (0,1,3,2)).reshape (las.ncas, las.ncas, -1)
    ix_i, ix_j = np.tril_indices (las.ncas)
    dm2 = dm2[:,:,(ix_i*las.ncas)+ix_j].reshape (las.ncas, -1).T
    gx += h2eff_sub @ dm2
    moH_inac = mo_inac.conjugate ().T
    gx[:ncore,:] -= 2 * moH_inac @ fock @ mo_cas
    gx = np.append (gx[:ncore,:], gx[nocc:,:], axis=0)

    return gorb.ravel (), np.concatenate (gci), gx.ravel ()

def density_fit (las, auxbasis=None, with_df=None):
    ''' Here I ONLY need to attach the tag and the df object because I put conditionals in LASCINoSymm to make my life easier '''
    las_class = las.__class__
    if with_df is None:
        if (getattr(las._scf, 'with_df', None) and
            (auxbasis is None or auxbasis == las._scf.with_df.auxbasis)):
            with_df = las._scf.with_df
        else:
            with_df = df.DF(las.mol)
            with_df.max_memory = las.max_memory
            with_df.stdout = las.stdout
            with_df.verbose = las.verbose
            with_df.auxbasis = auxbasis
    class DFLASCI (las_class, _DFLASCI):
        def __init__(self, my_las):
            self.__dict__.update(my_las.__dict__)
            #self.grad_update_dep = 0
            self.with_df = with_df
            self._keys = self._keys.union(['with_df'])
    return DFLASCI (las)

def h1e_for_cas (las, mo_coeff=None, ncas=None, ncore=None, nelecas=None, ci=None, ncas_sub=None, nelecas_sub=None, spin_sub=None, veff_sub=None):
    ''' Effective one-body Hamiltonians (plural) for a LASCI problem

    Args:
        las: a LASCI object

    Kwargs:
        mo_coeff: ndarray of shape (nao,nmo)
            Orbital coefficients ordered on the columns as: 
            core orbitals, subspace 1, subspace 2, ..., external orbitals
        ncas: integer
            As in PySCF's existing CASCI/CASSCF implementation
        nelecas: sequence of 2 integers
            As in PySCF's existing CASCI/CASSCF implementation
        ci: list of ndarrays of length (nsub)
            CI coefficients
            used to generate 1-RDMs in active subspaces; overrides casdm0_sub
        ncas_sub: ndarray of shape (nsub)
            Number of active orbitals in each subspace
        nelecas_sub: ndarray of shape (nsub,2)
            na, nb in each subspace
        spin_sub: ndarray of shape (nsub)
            Total spin quantum numbers in each subspace
        veff_sub: ndarray of shape (nsub+1, 2, nao, nao)
            If you precalculated this, pass it to save on calls to get_jk

    Returns:
        h1e: list like [ndarray of shape (2, isub, isub) for isub in ncas_sub]
            Spin-separated 1-body Hamiltonian operator for each active subspace
    '''
    if mo_coeff is None: mo_coeff = las.mo_coeff
    if ncas is None: ncas = las.ncas
    if ncore is None: ncore = las.ncore
    if ncas_sub is None: ncas_sub = las.ncas_sub
    if nelecas_sub is None: nelecas_sub = las.nelecas_sub
    if spin_sub is None: spin_sub = las.spin_sub
    if ncore is None: ncore = las.ncore
    if ci is None: ci = las.ci
    if veff_sub is None: veff_sub = las.get_veff (mo_coeff=mo_coeff, ci=ci)
    mo_cas = [las.get_mo_slice (idx, mo_coeff) for idx in range (len (ncas_sub))]
    moH_cas = [mo.conjugate ().T for mo in mo_cas]
    h1e = las.get_hcore ()[None,:,:] + veff_sub.sum (0) # JK of inactive orbitals
    veff_sub = veff_sub[1:] if veff_sub.shape[0] > 1 else [0 for isub in len (ncas_sub)] # JK of various active subspaces
    # Has to be a list, not array, because different subspaces have different ncas
    h1e = [np.tensordot (moH, np.dot (h1e - veff_self, mo), axes=((1),(1))).transpose (1,0,2)
        for moH, veff_self, mo in zip (moH_cas, veff_sub, mo_cas)]
    return h1e

def kernel (las, mo_coeff=None, ci0=None, casdm0_sub=None, conv_tol_grad=1e-4, verbose=lib.logger.NOTE):
    if mo_coeff is None: mo_coeff = las.mo_coeff
    log = lib.logger.new_logger(las, verbose)
    t0 = (time.clock(), time.time())
    log.debug('Start LASCI')

    h2eff_sub = las.get_h2eff (mo_coeff)
    t1 = log.timer('integral transformation to LAS space', *t0)

    # In the first cycle, I may pass casdm0_sub instead of ci0. Therefore, I need to work out this get_veff call separately.
    if ci0 is not None:
        veff_sub = las.get_veff (mo_coeff=mo_coeff, ci=ci0)
    elif casdm0_sub is not None:
        dm1_core = mo_coeff[:,:las.ncore] @ mo_coeff[:,:las.ncore].conjugate ().T
        dm1s_sub = [np.stack ([dm1_core, dm1_core], axis=0)]
        for idx, casdm1s in enumerate (casdm0_sub):
            mo = las.get_mo_slice (idx, mo_coeff=mo_coeff)
            moH = mo.conjugate ().T
            dm1s_sub.append (np.tensordot (mo, np.dot (casdm1s, moH), axes=((1),(1))).transpose (1,0,2))
        dm1s_sub = np.stack (dm1s_sub, axis=0)
        veff_sub = las.get_veff (dm1s=dm1s_sub)
    t1 = log.timer('LASCI initial get_veff', *t1)

    converged = False
    ci1 = ci0
    for it in range (las.max_cycle):
        e_cas, ci1 = ci_cycle (las, mo_coeff, ci1, veff_sub, h2eff_sub, log)
        t1 = log.timer ('LASCI ci_cycle', *t1)

        veff_old_sub = veff_sub.copy ()
        veff_sub = las.get_veff (mo_coeff=mo_coeff, ci=ci1)
        t1 = log.timer ('LASCI get_veff', *t1)

        e_tot = las.energy_nuc () + las.energy_elec (mo_coeff=mo_coeff, ci=ci1, h2eff=h2eff_sub, veff_sub=veff_sub)
        print ("LASCI energy after ci step only: {:.15g}".format (e_tot))

        mo_energy, mo_coeff = inac_scf_cycle (las, mo_coeff, ci1, veff_sub, h2eff_sub, log)
        t1 = log.timer ('LASCI hf_as cycle', *t1)

        veff_old_sub = veff_sub.copy ()
        veff_sub = las.get_veff (mo_coeff=mo_coeff, ci=ci1)
        t1 = log.timer ('LASCI get_veff', *t1)

        e_tot = las.energy_nuc () + las.energy_elec (mo_coeff=mo_coeff, ci=ci1, h2eff=h2eff_sub, veff_sub=veff_sub)
        gorb, gci, gx = las.get_grad (mo_coeff=mo_coeff, ci=ci1, h2eff_sub=h2eff_sub, veff_sub=veff_sub)
        norm_gorb = linalg.norm (gorb) if gorb.size else 0.0
        norm_gci = linalg.norm (gci) if gci.size else 0.0
        norm_gx = linalg.norm (gx) if gx.size else 0.0
        lib.logger.info (las, 'LASCI %d E = %.15g ; |g_int| = %.15g ; |g_ci| = %.15g ; |g_ext| = %.15g', it+1, e_tot, norm_gorb, norm_gci, norm_gx)
        t1 = log.timer ('LASCI post-cycle energy & gradient', *t1)
        
        if (norm_gorb < conv_tol_grad or norm_gorb*10 < norm_gx) and norm_gci < conv_tol_grad:
            converged = True
            break
        
    return converged, e_tot, mo_energy, mo_coeff, e_cas, ci1

def ci_cycle (las, mo, ci0, veff_sub, h2eff_sub, log):
    if ci0 is None: ci0 = [None for idx in range (len (las.ncas_sub))]
    # CI problems
    t1 = (time.clock(), time.time())
    h1eff_sub = las.get_h1eff (mo, veff_sub=veff_sub)
    ncas_cum = np.cumsum ([0] + las.ncas_sub.tolist ()) + las.ncore
    e_cas = []
    ci1 = []
    for isub, (ncas, nelecas, spin, h1eff, fcivec) in enumerate (zip (las.ncas_sub, las.nelecas_sub, las.spin_sub, h1eff_sub, ci0)):
        eri_cas = las.get_h2eff_slice (h2eff_sub, isub, compact=8)
        max_memory = max(400, las.max_memory-lib.current_memory()[0])
        h1eff_c = (h1eff[0] + h1eff[1]) / 2
        h1eff_s = (h1eff[0] - h1eff[1]) / 2
        nel = nelecas
        # CI solver has enforced convention: na >= nb
        if nelecas[0] < nelecas[1]:
            nel = (nel[1], nel[0])
            h1eff_s *= -1
        h1e = (h1eff_c, h1eff_s)
        wfnsym = orbsym = None
        if hasattr (las, 'wfnsym_sub') and hasattr (mo, 'orbsym'):
            wfnsym = las.wfnsym_sub[isub]
            i = ncas_cum[isub]
            j = ncas_cum[isub+1]
            orbsym = mo.orbsym[i:j]
            wfnsym_str = wfnsym if isinstance (wfnsym, str) else symm.irrep_id2name (las.mol.groupname, wfnsym)
            log.info ("LASCI subspace {} with irrep {}".format (isub, wfnsym_str))
        e_sub, fcivec = las.fcisolver.kernel(h1e, eri_cas, ncas, nel,
                                               ci0=fcivec, verbose=log,
                                               max_memory=max_memory,
                                               ecore=0, smult=spin,
                                               wfnsym=wfnsym, orbsym=orbsym)
        e_cas.append (e_sub)
        ci1.append (fcivec)
        t1 = log.timer ('FCI solver for subspace {}'.format (isub), *t1)
    return e_cas, ci1

def inac_scf_cycle (las, mo, ci0, veff_sub, h2eff_sub, log):
    casdm1 = las.make_casdm1 (ci=ci0)
    casdm2 = las.make_casdm2 (ci=ci0)
    ncas = las.ncas
    nocc = las.ncore + ncas
    eri_cas = h2eff_sub[las.ncore:nocc].reshape (ncas*ncas, -1)
    ix_i, ix_j = np.tril_indices (ncas)
    eri_cas = eri_cas[(ix_i*ncas)+ix_j,:]
    mf = hf_as.metaclass (las._scf)
    mf.max_cycle = 50
    mf.build_frozen_from_mo (mo, las.ncore, ncas, frozdm1=casdm1, frozdm2=casdm2, eri_fo=eri_cas)
    mf.mo_coeff = mo
    mf.kernel ()
    assert (mf.converged), 'inac scf cycle not converged'
    return mf.mo_energy, mf.mo_coeff
    '''
    # unactive MOs
    idx_unac = np.zeros (mo.shape[-1], dtype=np.bool_)
    idx_unac[:las.ncore] = True
    idx_unac[las.ncore+las.ncas:] = True
    if getattr (mo, 'orbsym', None) is not None:
        orbsym_unac = np.asarray (mo.orbsym)[idx_unac].tolist ()
    else:
        orbsym_unac = None
    mo_unac = mo[:,idx_unac]
    moH_unac = mo_unac.conjugate ().T
    fock_unac = moH_unac @ las.get_fock (veff_sub=veff_sub) @ mo_unac
    mo_energy, u = las._eig (fock_unac, 0, 0, orbsym_unac)
    idx_sort = np.argsort (mo_energy)
    mo_energy = mo_energy[idx_sort]
    u = u[:,idx_sort]
    if orbsym_unac is not None: orbsym_unac = np.asarray (orbsym_unac)[idx_sort].tolist ()
    mo1 = mo.copy ()
    ncore = las.ncore
    nocc = ncore + las.ncas
    mo1[:,:ncore] = mo_unac @ u[:,:ncore]
    mo1[:,nocc:] = mo_unac @ u[:,ncore:]
    if hasattr (las, 'wfnsym') and hasattr (mo, 'orbsym'):
        orbsym = mo.orbsym
        orbsym[:las.ncore] = orbsym_unac[:las.ncore]
        orbsym[las.ncore+las.ncas:] = orbsym_unac[las.ncore:]
        mo1 = lib.tag_array (mo1, orbsym=orbsym)
        #mo1 = casci_symm.label_symmetry_(las, mo1, None)
    t1 = log.timer ('Unactive orbital Fock diagonalization', *t1)
    return mo_energy, mo1, e_cas, ci1
    '''

def get_fock (las, mo_coeff=None, ci=None, eris=None, casdm1=None, verbose=None, veff_sub=None):
    ''' f_pq = h_pq + (g_pqrs - g_psrq/2) D_rs, AO basis
    Note the difference between this and h1e_for_cas: h1e_for_cas only has
    JK terms from electrons outside the "current" active subspace; get_fock
    includes JK from all electrons. This is also NOT the "generalized Fock matrix"
    of orbital gradients (but it can be used in calculating those if you do a
    semi-cumulant decomposition).
    The "eris" kwarg does not do anything and is retained only for backwards
    compatibility (also why I don't just call las.make_rdm1) '''
    if veff_sub is not None:
        return las.get_hcore () + veff_sub.sum ((0,1))/2 # spin-adapted component
    if mo_coeff is None: mo_coeff = las.mo_coeff
    if ci is None: ci = las.ci
    if casdm1 is None: casdm1 = las.make_casdm1 (ci=ci)
    mo_cas = mo_coeff[:,las.ncore:][:,:las.ncas]
    moH_cas = mo_cas.conjugate ().T
    mo_core = mo_coeff[:,:las.ncore]
    moH_core = mo_core.conjugate ().T
    dm1 = (2 * mo_core @ moH_core) + (mo_cas @ casdm1 @ moH_cas)
    if isinstance (las, _DFLASCI):
        vj, vk = las.with_df.get_jk(dm1, hermi=1)
    else:
        vj, vk = las._scf.get_jk(las.mol, dm1, hermi=1)
    fock = las.get_hcore () + vj - (vk/2)
    return fock

class LASCINoSymm (casci.CASCI):

    def __init__(self, mf, ncas, nelecas, ncore=None, spin_sub=None, **kwargs):
        ncas_tot = sum (ncas)
        nel_tot = [0, 0]
        for nel in nelecas:
            if isinstance (nel, (int, np.integer)):
                nb = nel // 2
                na = nb + (nel % 2)
            else:
                na, nb = nel
            nel_tot[0] += na
            nel_tot[1] += nb
        super().__init__(mf, ncas=ncas_tot, nelecas=nel_tot, ncore=ncore)
        if spin_sub is None: spin_sub = [0 for sub in ncas]
        self.ncas_sub = np.asarray (ncas)
        self.nelecas_sub = np.asarray (nelecas)
        self.spin_sub = np.asarray (spin_sub)
        self.conv_tol_grad = 1e-4
        self.max_cycle = 50
        keys = set(('ncas_sub', 'nelecas_sub', 'spin_sub', 'conv_tol_grad', 'max_cycle'))
        self._keys = set(self.__dict__.keys()).union(keys)
        self.fcisolver = csf_solver (self.mol, smult=0)

    def get_mo_slice (self, idx, mo_coeff=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        mo = mo_coeff[:,self.ncore:]
        for offs in self.ncas_sub[:idx]:
            mo = mo[:,offs:]
        mo = mo[:,:self.ncas_sub[idx]]
        return mo

    def ao2mo (self, mo_coeff=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        mo_cas = mo_coeff[:,self.ncore:self.ncore+self.ncas]
        mo = [mo_coeff, mo_cas, mo_cas, mo_cas]
        if getattr (self, 'with_df', None) is not None:
            eri = self.with_df.ao2mo (mo, compact=True)
        elif getattr (self._scf, '_eri', None) is not None:
            eri = ao2mo.incore.general (self._scf._eri, mo, compact=True)
        else:
            eri = ao2mo.outcore.general_iofree (self.mol, mo, compact=True)
        eri = eri.reshape (mo_coeff.shape[1], -1)
        return eri

    def get_h2eff_slice (self, h2eff, idx, compact=None):
        ncas_cum = np.cumsum ([0] + self.ncas_sub.tolist ())
        i = ncas_cum[idx] 
        j = ncas_cum[idx+1]
        ncore = self.ncore
        nocc = ncore + self.ncas
        eri = h2eff[ncore:nocc,:].reshape (self.ncas*self.ncas, -1)
        ix_i, ix_j = np.tril_indices (self.ncas)
        eri = eri[(ix_i*self.ncas)+ix_j,:]
        eri = ao2mo.restore (1, eri, self.ncas)[i:j,i:j,i:j,i:j]
        if compact: eri = ao2mo.restore (compact, eri, j-i)
        return eri

    get_h1eff = get_h1cas = h1e_for_cas = h1e_for_cas
    get_h2eff = ao2mo
    '''
    def get_h2eff (self, mo_coeff=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if isinstance (self, _DFLASCI):
            mo_cas = mo_coeff[:,self.ncore:][:,:self.ncas]
            return self.with_df.ao2mo (mo_cas)
        return self.ao2mo (mo_coeff)
    '''

    get_fock = get_fock
    get_grad = get_grad

    def kernel(self, mo_coeff=None, ci0=None, casdm0_sub=None, conv_tol_grad=None, verbose=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        else:
            self.mo_coeff = mo_coeff
        if ci0 is None: ci0 = self.ci
        if verbose is None: verbose = self.verbose
        if conv_tol_grad is None: conv_tol_grad = self.conv_tol_grad
        log = lib.logger.new_logger(self, verbose)

        if self.verbose >= lib.logger.WARN:
            self.check_sanity()
        self.dump_flags(log)

        self.converged, self.e_tot, self.mo_energy, self.mo_coeff, self.e_cas, self.ci = \
                kernel(self, mo_coeff, ci0=ci0, verbose=verbose, casdm0_sub=casdm0_sub, conv_tol_grad=conv_tol_grad)

        '''
        if self.canonicalization:
            self.canonicalize_(mo_coeff, self.ci,
                               sort=self.sorting_mo_energy,
                               cas_natorb=self.natorb, verbose=log)

        if getattr(self.fcisolver, 'converged', None) is not None:
            self.converged = np.all(self.fcisolver.converged)
            if self.converged:
                log.info('CASCI converged')
            else:
                log.info('CASCI not converged')
        else:
            self.converged = True
        self._finalize()
        '''
        return self.e_tot, self.e_cas, self.ci, self.mo_coeff, self.mo_energy

    def make_casdm1s_sub (self, ci=None, ncas_sub=None, nelecas_sub=None, **kwargs):
        ''' Spin-separated 1-RDMs in the MO basis for each subspace in sequence '''
        if ci is None: ci = self.ci
        if ncas_sub is None: ncas_sub = self.ncas_sub
        if nelecas_sub is None: nelecas_sub = self.nelecas_sub
        if ci is None:
            return [np.zeros ((2,ncas,ncas)) for ncas in ncas_sub]
        casdm1s = []
        for idx, (ci_i, ncas, nelecas) in enumerate (zip (ci, ncas_sub, nelecas_sub)):
            if ci_i is None:
                dm1a = dm1b = np.zeros ((ncas, ncas))
            else:   
                nel = (nelecas[1], nelecas[0]) if nelecas[1] > nelecas[0] else nelecas
                dm1a, dm1b = self.fcisolver.make_rdm1s (ci_i, ncas, nel)
                if nelecas[1] > nelecas[0]: dm1a, dm1b = dm1b, dm1a
            casdm1s.append (np.stack ([dm1a, dm1b], axis=0))
        return casdm1s

    def make_casdm2_sub (self, ci=None, ncas_sub=None, nelecas_sub=None, **kwargs):
        ''' Spin-separated 1-RDMs in the MO basis for each subspace in sequence '''
        if ci is None: ci = self.ci
        if ncas_sub is None: ncas_sub = self.ncas_sub
        if nelecas_sub is None: nelecas_sub = self.nelecas_sub
        casdm2 = []
        for idx, (ci_i, ncas, nelecas) in enumerate (zip (ci, ncas_sub, nelecas_sub)):
            nel = (nelecas[1], nelecas[0]) if nelecas[1] > nelecas[0] else nelecas
            casdm2.append (self.fcisolver.make_rdm2 (ci_i, ncas, nel))
        return casdm2

    def make_rdm1s_sub (self, mo_coeff=None, ci=None, ncas_sub=None, nelecas_sub=None, include_core=False, **kwargs):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if ci is None: ci = self.ci
        if ncas_sub is None: ncas_sub = self.ncas_sub
        if nelecas_sub is None: nelecas_sub = self.nelecas_sub
        ''' Same as make_casdm1s_sub, but in the ao basis '''
        casdm1s_sub = self.make_casdm1s_sub (ci=ci, ncas_sub=ncas_sub, nelecas_sub=nelecas_sub, **kwargs)
        rdm1s = []
        for idx, casdm1s in enumerate (casdm1s_sub):
            mo = self.get_mo_slice (idx, mo_coeff=mo_coeff)
            moH = mo.conjugate ().T
            rdm1s.append (np.tensordot (mo, np.dot (casdm1s, moH), axes=((1),(1))).transpose (1,0,2))
        if include_core and self.ncore:
            mo_core = mo_coeff[:,:self.ncore]
            moH_core = mo_core.conjugate ().T
            dm_core = mo_core @ moH_core
            rdm1s = [np.stack ([dm_core, dm_core], axis=0)] + rdm1s
        return np.stack (rdm1s, axis=0)

    def make_rdm1_sub (self, **kwargs):
        return self.make_rdm1s_sub (**kwargs).sum (1)

    def make_rdm1s (self, mo_coeff=None, ncore=None, **kwargs):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if ncore is None: ncore = self.ncore
        mo = mo_coeff[:,:ncore]
        moH = mo.conjugate ().T
        dm_core = mo @ moH
        dm_cas = self.make_rdm1s_sub (mo_coeff=mo_coeff, **kwargs).sum (0)
        return dm_core[None,:,:] + dm_cas

    def make_rdm1 (self, **kwargs):
        return self.make_rdm1s (**kwargs).sum (0)

    def make_casdm1s (self, **kwargs):
        ''' Make the full-dimensional casdm1s spanning the collective active space '''
        casdm1s_sub = self.make_casdm1s_sub (**kwargs)
        casdm1a = linalg.block_diag (*[dm[0] for dm in casdm1s_sub])
        casdm1b = linalg.block_diag (*[dm[1] for dm in casdm1s_sub])
        return np.stack ([casdm1a, casdm1b], axis=0)

    def make_casdm1 (self, **kwargs):
        ''' Spin-sum make_casdm1s '''
        return self.make_casdm1s (**kwargs).sum (0)

    def make_casdm2 (self, ci=None, ncas_sub=None, nelecas_sub=None, **kwargs):
        ''' Make the full-dimensional casdm2 spanning the collective active space '''
        if ci is None: ci = self.ci
        if ncas_sub is None: ncas_sub = self.ncas_sub
        if nelecas_sub is None: nelecas_sub = self.nelecas_sub
        ncas = sum (ncas_sub)
        ncas_cum = np.cumsum ([0] + ncas_sub.tolist ())
        casdm2s_sub = self.make_casdm2_sub (ci=ci, ncas_sub=ncas_sub, nelecas_sub=nelecas_sub, **kwargs)
        casdm2 = np.zeros ((ncas,ncas,ncas,ncas))
        # Diagonal 
        for isub, dm2 in enumerate (casdm2s_sub):
            i = ncas_cum[isub]
            j = ncas_cum[isub+1]
            casdm2[i:j, i:j, i:j, i:j] = dm2
        # Off-diagonal
        casdm1s_sub = self.make_casdm1s_sub (ci=ci)
        for (isub1, dm1s1), (isub2, dm1s2) in combinations (enumerate (casdm1s_sub), 2):
            i = ncas_cum[isub1]
            j = ncas_cum[isub1+1]
            k = ncas_cum[isub2]
            l = ncas_cum[isub2+1]
            dma1, dmb1 = dm1s1[0], dm1s1[1]
            dma2, dmb2 = dm1s2[0], dm1s2[1]
            # Coulomb slice
            dm2_view = casdm2[i:j, i:j, k:l, k:l]
            dm2_view = np.multiply.outer (dma1+dmb1, dma2+dmb2)
            casdm2[k:l, k:l, i:j, i:j] = dm2_view.transpose (2,3,0,1)
            # Exchange slice
            dm2_view = casdm2[i:j, k:l, k:l, i:j]
            dm2_view = -(np.multiply.outer (dma1, dma2) + np.multiply.outer (dmb1, dmb2)).transpose (0,2,3,1)
            casdm2[k:l, i:j, i:j, k:l] = dm2_view.transpose (2,3,0,1)
        return casdm2 

    def get_veff(self, mol=None, dm1s=None, hermi=1, **kwargs):
        ''' Returns a spin-separated veff! If dm1s isn't provided, builds from self.mo_coeff, self.ci etc. '''
        if mol is None: mol = self.mol
        nao = mol.nao_nr ()
        if dm1s is None: dm1s = self.make_rdm1s_sub (include_core=True, **kwargs).reshape (-1, nao, nao)
        else:
            dm1s = np.asarray (dm1s)
            if dm1s.ndim == 4:
                dm1s = dm1s.reshape (dm1s.shape[0]*2, nao, nao)
            assert (dm1s.ndim == 3 and dm1s.shape[0] % 2 == 0), 'Requires an even number of density matrices (a1,b1,a2,b2,...)!'
        if isinstance (self, _DFLASCI):
            vj, vk = self.with_df.get_jk(dm1s, hermi=hermi)
        else:
            vj, vk = self._scf.get_jk(mol, dm1s, hermi=hermi)
        vj = vj.reshape (-1,2,nao,nao).transpose (1,0,2,3)
        vk = vk.reshape (-1,2,nao,nao).transpose (1,0,2,3)
        vj = vj[0] + vj[1]
        veffa = vj - vk[0]
        veffb = vj - vk[1]
        return np.stack ([veffa, veffb], axis=1)

    def energy_elec (self, mo_coeff=None, ncore=None, ncas=None, ncas_sub=None, nelecas_sub=None, ci=None, h2eff=None, veff_sub=None, **kwargs):
        ''' Since the LASCI energy cannot be calculated as simply as ecas + ecore, I need this function '''
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if ncore is None: ncore = self.ncore
        if ncas is None: ncas = self.ncas
        if ncas_sub is None: ncas_sub = self.ncas_sub
        if nelecas_sub is None: nelecas_sub = self.nelecas_sub
        if ci is None: ci = self.ci
        if veff_sub is None: veff_sub = self.get_veff (mo_coeff=mo_coeff, ci=ci)
        if h2eff is None: h2eff = self.get_h2eff (mo_coeff)

        # 1-body veff terms
        h1e = self.get_hcore ()[None,:,:] + veff_sub.sum (0)/2
        dm1s = self.make_rdm1s (mo_coeff=mo_coeff, ncore=ncore, ci=ci, ncas_sub=ncas_sub, nelecas_sub=nelecas_sub)
        energy_elec = e1 = (h1e * dm1s).sum ()

        # 2-body cumulant terms
        casdm1s_sub = self.make_casdm1s_sub (ci=ci, ncas_sub=ncas_sub, nelecas_sub=nelecas_sub)
        casdm2_sub = self.make_casdm2_sub (ci=ci, ncas_sub=ncas_sub, nelecas_sub=nelecas_sub)
        e2 = 0
        for isub, (dm1s, dm2) in enumerate (zip (casdm1s_sub, casdm2_sub)):
            dm1a, dm1b = dm1s[0], dm1s[1]
            dm1 = dm1a + dm1b
            cdm2 = dm2 - np.multiply.outer (dm1, dm1)
            cdm2 += np.multiply.outer (dm1a, dm1a).transpose (0,3,2,1)
            cdm2 += np.multiply.outer (dm1b, dm1b).transpose (0,3,2,1)
            eri = self.get_h2eff_slice (h2eff, isub)
            te2 = np.tensordot (eri, cdm2, axes=4) / 2
            energy_elec += te2
            e2 += te2

        e0 = self.energy_nuc ()
        return energy_elec

class LASCISymm (casci_symm.CASCI, LASCINoSymm):

    def __init__(self, mf, ncas, nelecas, ncore=None, spin_sub=None, wfnsym_sub=None, **kwargs):
        LASCINoSymm.__init__(self, mf, ncas, nelecas, ncore=ncore, spin_sub=spin_sub)
        if wfnsym_sub is None: wfnsym_sub = [0 for icas in self.ncas_sub]
        self.wfnsym_sub = wfnsym_sub
        keys = set(('wfnsym_sub'))
        self._keys = set(self.__dict__.keys()).union(keys)

    make_rdm1s = LASCINoSymm.make_rdm1s
    make_rdm1 = LASCINoSymm.make_rdm1
    get_veff = LASCINoSymm.get_veff
    get_h1eff = get_h1cas = h1e_for_cas 

    @property
    def wfnsym (self):
        ''' This now returns the product of the irreps of the subspaces '''
        wfnsym = 0
        for ir in self.wfnsym_sub: wfnsym ^= ir
        return wfnsym
    @wfnsym.setter
    def wfnsym (self, ir):
        raise RuntimeError ("Cannot assign the whole-system symmetry of a LASCI wave function. Address the individual subspaces at lasci.wfnsym_sub instead.")

    def kernel(self, mo_coeff=None, ci0=None, casdm0_sub=None, verbose=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if ci0 is None:
            ci0 = self.ci

        # Initialize/overwrite mo_coeff.orbsym. Don't pass ci0 because it's not the right shape
        lib.logger.info (self, "LASCI lazy hack note: lines below reflect the point-group symmetry of the whole molecule but not of the individual subspaces")
        self.fcisolver.wfnsym = self.wfnsym
        mo_coeff = self.mo_coeff = casci_symm.label_symmetry_(self, mo_coeff, None)
        return LASCINoSymm.kernel(self, mo_coeff=mo_coeff, ci0=ci0, casdm0_sub=casdm0_sub, verbose=verbose)



























        

