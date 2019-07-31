from pyscf.mcscf import casci, casci_symm, df
from pyscf.lib import logger
from pyscf import symm, gto, scf
from mrh.my_pyscf import csf_solver
from itertools import combinations
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
    else
        las = LASCINoSymm (mf, ncas_sub, nelecas_sub, **kwargs)
    if getattr (mf, 'with_df', None):
        las = density_fit (las, with_df = mf.with_df) 
    return las

class _DFLASCI: # Tag
    pass

def density_fit (lasci, auxbasis=None, with_df=None):
    ''' Here I ONLY need to attach the tag and the df object because I put conditionals in LASCINoSymm to make my life easier '''
    lasci_class = lasci.__class__
    if with_df is None:
        if (getattr(lasci._scf, 'with_df', None) and
            (auxbasis is None or auxbasis == lasci._scf.with_df.auxbasis)):
            with_df = lasci._scf.with_df
        else:
            with_df = df.DF(lasci.mol)
            with_df.max_memory = lasci.max_memory
            with_df.stdout = lasci.stdout
            with_df.verbose = lasci.verbose
            with_df.auxbasis = auxbasis
    class DFLASCI (lasci_class, _DFLASCI):
        def __init__(self, my_lasci):
            self.__dict__.update(my_lasci.__dict__)
            #self.grad_update_dep = 0
            self.with_df = with_df
            self._keys = self._keys.union(['with_df'])
    return DFLASCI (lasci)

def h1e_for_cas (lasci, mo_coeff=None, ncas=None, ncore=None, nelecas=None, ci=None, casdm0_sub=None, ncas_sub=None, nelecas_sub=None, spin_sub=None)
    ''' Effective one-body Hamiltonians (plural) for a LASCI problem

    Args:
        lasci: a LASCI object

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
        casdm0_sub: list of ndarrays of length (nsub), each with shape (ncas,ncas)
            initial guess for the 1-RDMs in the active subspace bases
            overriden by ci coefficients
        ncas_sub: ndarray of shape (nsub)
            Number of active orbitals in each subspace
        nelecas_sub: ndarray of shape (nsub,2)
            na, nb in each subspace
        spin_sub: ndarray of shape (nsub)
            Total spin quantum numbers in each subspace
        
    Returns:
        h1e: list like [ndarray of shape (2, isub, isub) for isub in ncas_sub]
            Spin-separated 1-body Hamiltonian operator for each active subspace
    '''
    if mo_coeff is None: mo_coeff = lasci.mo_coeff
    if ncas is None: ncas = lasci.ncas
    if ncore is None: ncore = lasci.ncore
    if ncas_sub is None: ncas = lasci.ncas_sub
    if nelecas_sub is None: nelecas = lasci.nelecas_sub
    if spin_sub is None: spin = lasci.spin_sub
    if ncore is None: ncore = lasci.ncore
    if ci is None: ci = lasci.ci

    h1e_sum = lasci.get_hcore () + lasci.get_veff ()
    h1e_sum = np.stack ([h1e_sum, h1e_sum], axis=0)    

    dm1s_sub = casdm0_sub
    if ci is not None:
        dm1s_sub = lasci.make_rdm1s_sub (mo_coeff=mo_coeff, ci_sub=ci, ncas_sub=ncas_sub, nelecas_sub=nelecas_sub)
    if dm1s_sub is not None:
        veff_sub = [np.stack (lasci.get_veff (dm1s=dm1s), axis=0) for dm1s in dm1s_sub]
    else:
        veff_sub = [np.zeros (2,ncas,ncas) for ncas in ncas_sub]
    h1e_sum += sum (veff_sub) # Should only sum the list, not the arrays individually
    h1e = []
    for isub, veff_self in enumerate (veff_sub):
        mo = lasci.get_mo_slice (isub, mo_coeff=mo_coeff)
        moH = mo.conjugate ().T
        h1e.append (moH @ (h1e_sum - veff_self) @ mo)
    return h1e

def kernel (lasci, mo_coeff=None, ci0=None, casdm0_sub=None, verbose=logger.NOTE):
    if mo_coeff is None: mo_coeff = lasci.mo_coeff
    log = logger.new_logger(lasci, verbose)
    t0 = (time.clock(), time.time())
    log.debug('Start LASCI')

    h1eff_sub = lasci.get_h1eff (mo_coeff, casdm0_sub=casdm0_sub)
    t1 = log.timer('effective h1es in LAS space', *t0)
    h2eff_sub = lasci.get_h2eff (mo_coeff)
    t1 = log.timer('integral transformation to LAS space', *t1)

    e_cas = []
    ci1 = []
    for isub, (ncas, nelecas, spin, h1eff) in enumerate (zip (lasci.ncas_sub, lasci.nelecas_sub, lasci.spin_sub, h1eff_sub)):
        h2eff = self.get_h2eff_slice (h2eff_sub, isub)
        fcivec = ci0[isub] if ci0 is not None else None
        max_memory = max(400, casci.max_memory-lib.current_memory()[0])
        h1eff_c = (h1eff[0] + h1eff[1]) / 2
        h1eff_s = (h1eff[0] - h1eff[1]) / 2
        h1e = [h1eff_c, h1eff_s]
        wfnsym = orbsym = None
        if hasattr (lasci, 'wfnsym') and hasattr (mo_coeff, 'orbsym'):
            wfnsym = lasci.wfnsym_sub[isub]
            i = lasci.ncore + np.cumsum (lasci.ncas_sub[:isub])
            j = i + ncas
            orbsym = mo_coeff.orbsym[i:j]
            wfnsym_str = wfnsym if isinstance (wfnsym, str) else symm.irrep_id2name (lasci.mol.groupname, wfnsym)
            log.info ("LASCI subspace {} with irrep {}".format (isub, wfnsym_str))
        e_sub, fcivec = lasci.fcisolver.kernel(h1e, eri_cas, ncas, nelecas,
                                               ci0=fcivec, verbose=log,
                                               max_memory=max_memory,
                                               ecore=0, smult=spin,
                                               wfnsym=wfnsym, orbsym=orbsym)
        e_cas.append (e_sub)
        ci1.append (fcivec)
        t1 = log.timer ('FCI solver for subspace {}'.format (isub), *t1)
    e_tot = lasci.energy_elec (mo_coeff=mo_coeff, ci=ci1, h2eff=h2eff_sub)
    return e_tot, e_cas, ci1

class LASCINoSymm (casci.CASCI):

    def __init__(self, mf_or_mol, ncas, nelecas, ncore=None, spin_sub=None):
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
        super().__init__(mf_or_mol, ncas=ncas_tot, nelecas=nel_tot, ncore=ncore)
        if spin_sub is None: spin_sub = [0 for sub in ncas]
        self.ncas_sub = np.asarray (ncas)
        self.nelecas_sub = np.asarray (nelecas)
        self.spin_sub = np.asarray (spin_sub)
        keys = set(('ncas_sub', 'nelecas_sub', 'spin_sub'))
        self._keys = set(self.__dict__.keys()).union(keys)
        self.fcisolver = csf_solver (mf_or_mol, smult=0)

    def get_mo_slice (self, idx, mo_coeff=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        mo = mo_coeff[:,self.ncore:]
        for offs in self.ncas[:idx]:
            mo = mo[:,offs:]
        mo = mo[:,:self.ncas[idx]]
        return mo

    def get_h2eff_slice (self, h2eff, idx):
        ncas_cum = np.cumsum ([0] + self.ncas_sub.tolist ())
        i = ncas_cum[idx] 
        j = ncas_cum[idx+1]
        return h2eff[i:j,i:j,i:j,i:j]

    get_h1eff = get_h1cas = h1e_for_cas = h1e_for_cas
    def get_h2eff (self, mo_coeff=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if isinstance (self, _DFLASCI):
            mo_cas = mo_coeff[:,self.ncore:][:,:self.ncas]
            return self.with_df.ao2mo (mo_cas)
        return self.ao2mo (mo_coeff)

    def kernel(self, mo_coeff=None, ci0=None, casdm0_sub=None, verbose=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        else:
            self.mo_coeff = mo_coeff
        if ci0 is None:
            ci0 = self.ci
        log = logger.new_logger(self, verbose)

        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags(log)

        self.e_tot, self.e_cas, self.ci = \
                kernel(self, mo_coeff, ci0=ci0, verbose=log, casdm0_sub=casdm0_sub)

        if self.canonicalization:
            self.canonicalize_(mo_coeff, self.ci,
                               sort=self.sorting_mo_energy,
                               cas_natorb=self.natorb, verbose=log)

        if getattr(self.fcisolver, 'converged', None) is not None:
            self.converged = numpy.all(self.fcisolver.converged)
            if self.converged:
                log.info('CASCI converged')
            else:
                log.info('CASCI not converged')
        else:
            self.converged = True
        self._finalize()
        return self.e_tot, self.e_cas, self.ci, self.mo_coeff, self.mo_energy

    def make_rdm1s_sub (self, mo_coeff=None, ci=None, ncas_sub=None, nelecas_sub=None, **kwargs):
        ''' Spin-separated 1-RDMs in the AO basis for each subspace in sequence '''
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if ci is None: ci = self.ci
        if ncas_sub is None: ncas = self.ncas_sub
        if nelecas_sub is None: nelecas = self.nelecas_sub
        dm1s = []
        for idx, (ci_i, ncas, nelecas) in enumerate (zip (ci, ncas_sub, nelecas_sub)):
            mo = self.get_mo_slice (idx, mo_coeff=mo_coeff)
            moH = mo.conjugate ().T
            dm1a, dm1b = self.fcisolver.make_rdm1s (ci_i, ncas, nelecas)
            dm1s.append (np.stack ([moH @ dm @ mo for dm in (dm1a, dm1b)], axis=0))
        return np.stack (dm1s, axis=0)

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

    def make_casdm2 (self, ci=None, ncas_sub=None, nelecas_sub=None, **kwargs):
        ''' Make the full-dimensional casdm2 spanning the collective active spaces '''
        if ci is None: ci = self.ci
        if ncas_sub is None: ncas_sub = self.ncas_sub
        if nelecas_sub is None: nelecas_sub = self.nelecas_sub
        ncas = sum (ncas_sub)
        ncas_cum = np.cumsum ([0] + ncas_sub.tolist ())
        casdm2 = np.zeros ((ncas,ncas,ncas,ncas))
        # Diagonal 
        for isub, (fcivec, ncas, nelecas) in enumerate (zip (ci, ncas_sub, nelecas_sub)):
            i = ncas_cum[isub]
            j = ncas_cum[isub+1]
            dm2_view = casdm2[i:j,i:j,i:j,i:j]
            dm2_view = self.fcisolver.make_rdm2 (self.mol, fcivec, ncas, nelecas)
        # Off-diagonal
        for (isub1, (ci1, ncas1, nelecas1)), (isub2, (ci2, ncas2, nelecas2)) in combinations (
          enumerate (zip (ci, ncas_sub, nelecas_sub)), 2):
            i = ncas_cum[isub1]
            j = ncas_cum[isub1+1]
            k = ncas_cum[isub2]
            l = ncas_cum[isub2+1]
            dma1, dmb1 = self.fcisolver.make_rdm1s (self.mol, ci1, ncas1, nelecas1)
            dma2, dmb2 = self.fcisolver.make_rdm1s (self.mol, ci2, ncas2, nelecas2)
            # Coulomb slice
            dm2_view = casdm2[i:j, i:j, k:l, k:l]
            dm2_view = np.multiply.outer (dma1+dmb1, dma2+dmb2)
            casdm2[k:l, k:l, i:j, i:j] = dm2_view.transpose (2,3,0,1)
            # Exchange slice
            dm2_view = casdm2[i:j, k:l, k:l, i:j]
            dm2_view = -(np.multiply.outer (dma1, dma2) + np.multiply.outer (dmb1, dmb2)).transpose (0,2,3,1)
            casdm2[k:l, i:j, i:j, k:l] = dm2_view.transpose (2,3,0,1)
        return casdm2 

    def get_veff(self, mol=None, dm1s=None, hermi=1):
        ''' Returns a spin-separated veff! If dm1s isn't provided, assumes you want the core '''
        if mol is None: mol = self.mol
        if dm1s is None:
            mocore = self.mo_coeff[:,:self.ncore]
            dm1s = numpy.dot(mocore, mocore.T)
            dm1s = np.stack ([dm1s, dm1s], axis=0)
        if isinstance (self, _DFLASCI):
            vj, vk = self.with_df.get_jk(mol, dm, hermi=hermi)
        else:
            vj, vk = self._scf.get_jk(mol, dm, hermi=hermi)
        vj = vj[0] + vj[1]
        return vj - vk[0], vj - vk[1]

    def energy_elec (self, mo_coeff=None, ncore=None, ncas=None, ncas_sub=None, nelecas_sub=None, ci=None, h2eff=None, **kwargs):
        ''' Since the LASCI energy cannot be calculated as simply as ecas + ecore, I need this function '''
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if ncore is None: ncore = self.ncore
        if ncas is None: ncas = self.ncas
        if ncas_sub is None: ncas = self.ncas_sub
        if nelecas_sub is None: nelecas = self.nelecas_sub
        if ci is None: ci = self.ci
        if h2eff is None: h2eff = self.get_h2eff (mo_coeff)

        dm1s = self.make_rdm1s (mo_coeff=mo_coeff, ncore=ncore, ci=ci, ncas_sub=ncas_sub, nelecas_sub=nelecas_sub)
        h1e = self.get_hcore ()[None,:,:] + np.stack (self.get_veff (dm1s=dm1s), axis=0)/2
        energy_elec = (h1e * dm1s).sum ()

        # 2-body cumulant terms
        for isub, (norb, nel, ci_i) in enumerate (zip (ncas_sub, nelecas_sub, ci)):
            dm1a, dm1b = self.fcisolver.make_rdm1s (self.mol, ci_i, norb, nel)
            dm1 = dm1a + dm1b
            dm2 = self.fcisolver.make_rdm2 (self.mol, ci_i, norb, nel)
            dm2 -= np.multiply.outer (dm1, dm1)
            dm2 += np.multiply.outer (dm1a, dm1a).transpose (0,3,2,1)
            dm2 += np.multiply.outer (dm1b, dm1b).transpose (0,3,2,1)
            eri = self.get_h2eff_slice (h2eff, isub)
            energy_elec += (eri * dm2).sum () / 2

        return energy_elec

class LASCISymm (casci_symm.CASCI, LASCINoSymm):

    def __init__(self, mf_or_mol, ncas, nelecas, ncore=None, spin_sub=None, wfnsym_sub=None):
        LASCINoSymm.__init__(self, mf_or_mol, ncas, nelecas, ncore=ncore, spin_sub=spin_sub)
        if wfnsym_sub is None: wfnsym_sub = [0 for icas in self.ncas_sub]
        self.wfnsym_sub = wfnsym_sub

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
        logger.info ("LASCI lazy hack note: lines below reflect the point-group symmetry of the whole molecule but not of the individual subspaces")
        self.fcisolver.wfnsym = self.wfnsym
        mo_coeff = self.mo_coeff = casci_symm.label_symmetry_(self, mo_coeff, None)
        return LASCINoSymm.kernel(self, mo_coeff=mo_coeff, ci0=ci0, casdm0_sub=casdm0_sub, verbose=verbose)



























        

