import os
import numpy as np
from scipy import linalg
from pyscf import gto, scf, mcscf, ao2mo, lib, df
from pyscf.lib import logger
from pyscf.fci.direct_spin1 import _unpack_nelec
from pyscf.mcscf.addons import _state_average_mcscf_solver
from mrh.my_pyscf.mcscf import _DFLASCI, lasci_sync, lasci
import copy, json

from mrh.my_pyscf.gpu import libgpu

class ImpurityMole (gto.Mole):
    def __init__(self, las, stdout=None, output=None):
        gto.Mole.__init__(self)
        self._las = las
        self._imporb_coeff = None
        self.verbose = las.verbose
        self.max_memory = las.max_memory
        self.atom.append (('H', (0, 0, 0)))
        if stdout is None and output is None:
            self.stdout = las.stdout
        elif stdout is not None:
            self.stdout = stdout
        elif output is not None:
            self.output = output
        self.spin = None
        self._imporb_coeff = np.array ([0])
        self.build ()
        add_gpu = {"use_gpu":las.use_gpu}
        self.__dict__.update(add_gpu)

    def _update_space_(self, imporb_coeff, nelec_imp):
        self._imporb_coeff = imporb_coeff
        nelec_imp = _unpack_nelec (nelec_imp)
        self.nelectron = sum (nelec_imp)
        self.spin = nelec_imp[0] - nelec_imp[1]

    def get_imporb_coeff (self): return self._imporb_coeff
    def nao_nr (self, *args, **kwargs): return self._imporb_coeff.shape[-1]
    def nao (self): return self._imporb_coeff.shape[-1]
    def dumps (mol):
        '''Subclassing this to eliminate annoying warning message
        '''
        exclude_keys = set(('output', 'stdout', '_keys', 'use_gpu',
                            # Constructing in function loads
                            'symm_orb', 'irrep_id', 'irrep_name',
                            # LASSCF hook to rest of molecule
                            '_las'))
        nparray_keys = set(('_atm', '_bas', '_env', '_ecpbas',
                            '_symm_orig', '_symm_axes',
                            # Definition of fragment in LASSCF context
                            '_imporb_coeff'))

        moldic = dict(mol.__dict__)
        for k in exclude_keys:
            if k in moldic:
                del (moldic[k])
        for k in nparray_keys:
            if isinstance(moldic[k], np.ndarray):
                moldic[k] = moldic[k].tolist()
        moldic['atom'] = repr(mol.atom)
        moldic['basis']= repr(mol.basis)
        moldic['ecp' ] = repr(mol.ecp)

        try:
            return json.dumps(moldic)
        except TypeError:
            def skip_value(dic):
                dic1 = {}
                for k,v in dic.items():
                    if (v is None or
                        isinstance(v, (str, bool, int, float))):
                        dic1[k] = v
                    elif isinstance(v, np.integer):
                        dic1[k] = int (v)
                    elif isinstance(v, (list, tuple)):
                        dic1[k] = v   # Should I recursively skip_vaule?
                    elif isinstance(v, set):
                        dic1[k] = list(v)
                    elif isinstance(v, dict):
                        dic1[k] = skip_value(v)
                    else:
                        msg =('Function mol.dumps drops attribute %s because '
                              'it is not JSON-serializable' % k)
                        assert (False)
                        warnings.warn(msg)
                return dic1
            return json.dumps(skip_value(moldic), skipkeys=True)



class ImpuritySCF (scf.hf.SCF):

    def _is_mem_enough (self, df_naux=None):
        nao = self.mol.nao ()
        if df_naux is not None:
            return 4*df_naux*nao*(nao+1)/1e6+lib.current_memory()[0] < self.max_memory*.95
        else:
            return 2*(nao**4)/1e6+lib.current_memory()[0] < self.max_memory*.95

    def _update_space_(self, imporb_coeff, nelec_imp):
        '''Syntactic sugar for updating the impurity orbital subspace in the encapsulated
        ImpurityMole object.'''
        self.mol._update_space_(imporb_coeff, nelec_imp)

    def _update_impham_1_(self, veff, dm1s, e_tot=None):
        '''Update energy_nuc (), get_hcore (), and the two-electron integrals in either _eri or
        with_df to correspond to the current full-system total energy, the current full-system
        state-averaged Fock matrix, and the current impurity orbitals, respectively. I.E.,
        energy_nuc () and get_hcore () will have double-counting after this call, which can be
        subtracted subsequently by a call to _update_impham_2_.

        Args:
            veff : ndarray of shape (2,nao,nao)
                Full-system spin-separated effective potential in AO basis
            dm1s : ndarray of shape (2,nao,nao)
                Full-system spin-separated 1-RDM in AO basis

        Kwargs:
            e_tot : float
                Full-system LASSCF total energy; defaults to value stored on parent LASSCF object
        '''
        if e_tot is None: e_tot = self.mol._las.e_tot
        imporb_coeff = self.mol.get_imporb_coeff ()
        nimp = self.mol.nao ()
        mf = self.mol._las._scf
        # Two-electron integrals
        log = logger.new_logger (self, self.verbose)
        t0 = (logger.process_clock(), logger.perf_counter())
        conv_eris_mem_error = MemoryError (("Conventional two-electron integrals in asynchronous "
                                            "LASSCF (integral-direct algorithm is not yet "
                                            "supported)"))
        df_eris_mem_error = MemoryError (("Density-fitted two-electron integrals in asynchronous "
                                          "LASSCF (outcore algorithm is not yet supported"))
        if getattr (mf, 'with_df', None) is not None:
            # TODO: impurity outcore cderi
            if not self._is_mem_enough (df_naux = mf.with_df.get_naoaux ()):
                raise df_eris_mem_error
            _cderi = np.empty ((mf.with_df.get_naoaux (), nimp*(nimp+1)//2),
                               dtype=imporb_coeff.dtype)
            ijmosym, mij_pair, moij, ijslice = ao2mo.incore._conc_mos (imporb_coeff, imporb_coeff,
                                                                        compact=True)
            b0 = 0
            for eri1 in mf.with_df.loop ():
                b1 = b0 + eri1.shape[0]
                eri2 = _cderi[b0:b1]
                eri2 = ao2mo._ao2mo.nr_e2 (eri1, moij, ijslice, aosym='s2', mosym=ijmosym,
                                           out=eri2)
                b0 = b1
            if getattr (self, 'with_df', None) is not None:
                self.with_df._cderi = _cderi
            else:
                self._cderi = _cderi
                self._eri = np.dot (_cderi.conj ().T, _cderi)
        else:
            if getattr (mf, '_eri', None) is None:
                if not mf._is_mem_enough ():
                    raise conv_eris_mem_error
                mf._eri = mf.mol.intor('int2e', aosym='s8')
            if not self._is_mem_enough ():
                raise conv_eris_mem_error
            self._eri = ao2mo.full (mf._eri, imporb_coeff, 4)
        t0 = log.timer ("Two-electron integrals in embedding subspace", *t0)
        # External mean-field; potentially spin-broken
        h1s = mf.get_hcore ()[None,:,:] + veff
        h1s = np.dot (imporb_coeff.conj ().T, np.dot (h1s, imporb_coeff)).transpose (1,0,2)
        self._imporb_h1 = h1s.sum (0) / 2
        self._imporb_h1_sz = (h1s[0] - h1s[1]) / 2
        self._imporb_h0 = e_tot 

    def _update_impham_2_(self, mo_docc, mo_dm, dm1s, dm2, eri_dm=None):
        '''Update energy_nuc () and get_hcore* () to subtract double-counting from the electrons
        inside the current impurity space; I.E., following a call to _update_impham_1_.

        Args:
            mo_docc : ndarray of shape (nimp, *a)
                Doubly-occupied molecular orbitals in the impurity orbital basis
            mo_dm : ndarray of shape (nimp, *b)
                Partially-occupied molecular orbitals in the impurity orbital basis
            dm1s : ndarray of shape (2, *b, *b)
                Spin-separated 1-RDM in mo_dm basis
            dm2 : ndarray of shape (*b, *b, *b, *b)
                Spin-summed 2-RDM in mo_dm basis

        Kwargs:
            eri_dm : ndarray of shape (*b, *b, *b, *b)
                ERIs in mo_dm basis
        '''
        dm1 = dm1s.sum (0)
        dm2 -= np.multiply.outer (dm1, dm1)
        dm2 += np.multiply.outer (dm1s[0], dm1s[0]).transpose (0,3,2,1)
        dm2 += np.multiply.outer (dm1s[1], dm1s[1]).transpose (0,3,2,1)
        dm1s = np.dot (mo_dm, np.dot (dm1s, mo_dm.conj ().T)).transpose (1,0,2)
        dm1s += (mo_docc @ mo_docc.conj ().T)[None,:,:]
        vj, vk = self.get_jk (dm=dm1s)
        veff = vj.sum (0)[None,:,:] - vk
        self._imporb_h1 -= veff.sum (0) / 2
        self._imporb_h1_sz -= (veff[0] - veff[1]) / 2
        h1eff = self.get_hcore_spinsep ()
        h1eff += veff * .5
        self._imporb_h0 -= np.dot (h1eff.ravel (), dm1s.ravel ())
        self._imporb_h0 -= np.dot (eri_dm.ravel (), dm2.ravel ()) * .5

    def get_hcore (self, *args, **kwargs):
        return self._imporb_h1

    def get_hcore_sz (self):
        return self._imporb_h1_sz

    def get_hcore_spinsep (self):
        h1c = self.get_hcore ()
        h1s = self.get_hcore_sz ()
        return np.stack ([h1c+h1s, h1c-h1s], axis=0)

    def get_ovlp (self):
        return np.eye (self.mol.nao ())

    def energy_nuc (self):
        return self._imporb_h0

    def get_fock (self, h1e=None, s1e=None, vhf=None, dm=None, cycle=1, diis=None,
        diis_start_cycle=None, level_shift_factor=None, damp_factor=None):

        if vhf is None: vhf = self.get_veff (self.mol, dm)
        vhf[0] += self.get_hcore_sz ()
        vhf[1] -= self.get_hcore_sz ()
        return scf.rohf.get_fock (self, h1e=h1e, s1e=s1e, vhf=vhf, dm=dm, cycle=cycle, diis=diis,
            diis_start_cycle=diis_start_cycle, level_shift_factor=level_shift_factor,
            damp_factor=damp_factor)

    def energy_elec (self, dm=None, h1e=None, vhf=None):
        if dm is None: dm = self.make_rdm1 ()
        e_elec, e_coul = super().energy_elec (dm=dm, h1e=h1e, vhf=vhf)
        e_elec += (self.get_hcore_sz () * (dm[0] - dm[1])).sum ()
        return e_elec, e_coul

class ImpurityROHF (scf.rohf.ROHF, ImpuritySCF):
    get_hcore = ImpuritySCF.get_hcore
    get_ovlp = ImpuritySCF.get_ovlp
    get_fock = ImpuritySCF.get_fock
    energy_nuc = ImpuritySCF.energy_nuc
    energy_elec = ImpuritySCF.energy_elec

class ImpurityRHF (scf.hf.RHF, ImpuritySCF):
    get_hcore = ImpuritySCF.get_hcore
    get_ovlp = ImpuritySCF.get_ovlp
    get_fock = ImpuritySCF.get_fock
    energy_nuc = ImpuritySCF.energy_nuc
    energy_elec = ImpuritySCF.energy_elec

def ImpurityHF (mol):
    if mol.spin == 0: return ImpurityRHF (mol)
    else: return ImpurityROHF (mol)

# Monkeypatch the monkeypatch from mc1step.py
def _fake_h_for_fast_casci(casscf, mo, eris):
    mc = copy.copy(casscf)
    mc.mo_coeff = mo
    ncore = casscf.ncore
    nocc = ncore + casscf.ncas

    mo_core = mo[:,:ncore]
    mo_cas = mo[:,ncore:nocc]
    core_dm = np.dot(mo_core, mo_core.T) # *2 implicit in hcore = h1(up) + h1(down)
    energy_core = casscf.energy_nuc_r ()
    h1 = casscf.get_hcore_rs ()
    hcore = h1.sum (1)
    energy_core += np.einsum('ij,rji->r', core_dm, hcore)
    energy_core += eris.vhf_c[:ncore,:ncore].trace ()
    h1eff = np.tensordot (mo_cas.conj (), np.dot (h1, mo_cas), axes=((0),(2))).transpose (1,2,0,3)
    h1eff += eris.vhf_c[None,None,ncore:nocc,ncore:nocc]
    mc.get_h1eff = lambda *args: (h1eff, energy_core)

    eri_cas = eris.ppaa[ncore:nocc,ncore:nocc,:,:].copy()
    mc.get_h2eff = lambda *args: eri_cas
    return mc

# I sadly had to copy-and-paste this function from casci.py due to inapplicable
#   1) logging commands and
#   2) error checks
# that could not be monkeypatched out.
def casci_kernel(casci, mo_coeff=None, ci0=None, verbose=logger.NOTE, envs=None):
    '''CASCI solver

    Args:
        casci: CASCI or CASSCF object

        mo_coeff : ndarray
            orbitals to construct active space Hamiltonian
        ci0 : ndarray or custom types
            FCI sovler initial guess. For external FCI-like solvers, it can be
            overloaded different data type. For example, in the state-average
            FCI solver, ci0 is a list of ndarray. In other solvers such as
            DMRGCI solver, SHCI solver, ci0 are custom types.

    kwargs:
        envs: dict
            The variable envs is created (for PR 807) to passes MCSCF runtime
            environment variables to SHCI solver. For solvers which do not
            need this parameter, a kwargs should be created in kernel method
            and "envs" pop in kernel function
    '''
    if mo_coeff is None: mo_coeff = casci.mo_coeff
    log = logger.new_logger(casci, verbose)
    t0 = (logger.process_clock(), logger.perf_counter())
    log.debug('Start CASCI')

    ncas = casci.ncas
    nelecas = casci.nelecas

    # 2e
    eri_cas = casci.get_h2eff(mo_coeff)
    t1 = log.timer('integral transformation to CAS space', *t0)

    # 1e
    h1eff, energy_core = casci.get_h1eff(mo_coeff)
    log.debug('core energy = {}'.format (energy_core))
    t1 = log.timer('effective h1e in CAS space', *t1)

    if h1eff.shape[-1] != ncas:
        raise RuntimeError('Active space size error. nmo=%d ncore=%d ncas=%d' %
                           (mo_coeff.shape[1], casci.ncore, ncas))

    # FCI
    #max_memory = max(400, casci.max_memory-lib.current_memory()[0])
    # Issue #54: count memory here, or in FCISolver?
    e_tot, fcivec = casci.fcisolver.kernel(h1eff, eri_cas, ncas, nelecas,
                                           ci0=ci0, verbose=log,
                                           #max_memory=max_memory,
                                           ecore=energy_core)

    t1 = log.timer('FCI solver', *t1)
    e_cas = e_tot - energy_core
    return e_tot, e_cas, fcivec

# This is the really tricky part
class ImpuritySolver ():

    def _push_keyframe (self, kf1, mo_coeff=None, ci=None):
        '''Generate the whole-system MO and CI vectors corresponding to the current state of this
        ImpurityCASSCF instance.

        Args:
            kf1 : object of :class:`LASKeyframe`
                Copied to output and modified; not altered in-place

        Kwargs:
            mo_coeff : ndarray of shape (nimp, nimp)
                The current IO basis MO coefficients
            ci : list of ndarrays
                CI vectors

        Returns:
            kf2 : object of :class:`LASKeyframe`
                Contains updated whole-molecule data corresponding to mo_coeff and ci.
        '''
        if mo_coeff is None: mo_coeff=self.mo_coeff
        if ci is None: ci=self.ci
        log = logger.new_logger (self, self.verbose)
        kf2 = kf1.copy ()
        kf2.frags = set (self._ifrags)
        imporb_coeff = self.mol.get_imporb_coeff ()
        mo_self = imporb_coeff @ mo_coeff
        las = self.mol._las

        # active orbital part should be easy
        ci = self.ci if len (self._ifrags)>1 else [self.ci,]
        idx = []
        for ix, ifrag in enumerate (self._ifrags):
            kf2.ci[ifrag] = ci[ix]
            i = las.ncore + sum (las.ncas_sub[:ifrag])
            j = i + las.ncas_sub[ifrag]
            idx.extend (list (range (i,j)))
        kf2.mo_coeff[:,idx] = mo_self[:,self.ncore:self.ncore+self.ncas]

        # Unentangled inactive orbitals
        s0 = las._scf.get_ovlp ()
        ncore_unent = las.ncore - self.ncore
        assert (ncore_unent>=0), '{} {}'.format (las.ncore, self.ncore)
        if las.ncore:
            mo_full_core = kf2.mo_coeff[:,:las.ncore]
            ovlp = mo_full_core.conj ().T @ s0 @ imporb_coeff
            proj = ovlp @ ovlp.conj ().T
            evals, u = linalg.eigh (-proj)
            try:
                assert (ncore_unent==0 or np.amax (np.abs (evals[-ncore_unent:]))<1e-4)
            except AssertionError as err:
                log.warn ("push_keyframe imporb problem: ncore_unent = %d but max |evals[-ncore_unent:]| = %e",
                          ncore_unent, np.amax (np.abs (evals[-ncore_unent:])))
            if ncore_unent>0: kf2.mo_coeff[:,:ncore_unent] = mo_full_core @ u[:,-ncore_unent:]
            kf2.mo_coeff[:,ncore_unent:las.ncore] = mo_self[:,:self.ncore]
        
        # Canonicalize unentangled inactive orbitals
        # Be careful not to touch kf2.h2eff_sub or kf2.fock1 until we're done
        f0 = las.get_fock (mo_coeff=kf2.mo_coeff, ci=kf2.ci, veff=kf2.veff)
        if ncore_unent>0:
            mo_i = kf2.mo_coeff[:,:ncore_unent]
            f0_ij = mo_i.conj ().T @ f0 @ mo_i
            w, u = linalg.eigh (f0_ij)
            kf2.mo_coeff[:,:ncore_unent] = mo_i @ u

        # Unentangled virtual orbitals
        nvirt_full = kf2.mo_coeff.shape[1] - las.ncore - las.ncas
        nvirt_self = mo_coeff.shape[1] - self.ncore - self.ncas
        nvirt_unent = nvirt_full - nvirt_self
        if nvirt_full:
            assert (nvirt_unent>=0), '{} {}'.format (nvirt_full, nvirt_self)
            mo_full_virt = kf2.mo_coeff[:,las.ncore+las.ncas:]
            ovlp = mo_full_virt.conj ().T @ s0 @ imporb_coeff
            proj = ovlp @ ovlp.conj ().T
            evals, u = linalg.eigh (-proj)
            try:
                assert (nvirt_unent==0 or np.amax (np.abs (evals[-nvirt_unent:]))<1e-4)
            except AssertionError as err:
                log.warn ("push_keyframe imporb problem: nvirt_unent = %d but max |evals[-nvirt_unent:]| = %e",
                          nvirt_unent, np.amax (np.abs (evals[-nvirt_unent:])))
            if nvirt_unent>0:
                kf2.mo_coeff[:,-nvirt_unent:] = mo_full_virt @ u[:,-nvirt_unent:]
                kf2.mo_coeff[:,las.ncore+las.ncas:-nvirt_unent] = mo_self[:,self.ncore+self.ncas:]
                # Canonicalize unentangled virtual orbitals
                mo_a = kf2.mo_coeff[:,-nvirt_unent:]
                f0_ab = mo_a.conj ().T @ f0 @ mo_a
                w, u = linalg.eigh (f0_ab)
                kf2.mo_coeff[:,-nvirt_unent:] = mo_a @ u
            else:
                kf2.mo_coeff[:,las.ncore+las.ncas:] = mo_self[:,self.ncore+self.ncas:]

        return kf2

    def _pull_keyframe_(self, kf, max_size='mid'):
        '''Update this impurity solver, and all encapsulated impurity objects all the way down,
        with a new IO basis set, the corresponding Hamiltonian, and initial guess MO coefficients
        and CI vectors based on new whole-molecule data.

        Args:
            kf : object of :class:`LASKeyframe`
                Contains whole-molecule MO coefficients, CI vectors, and intermediate arrays

        Kwargs:
            max_size : str or int
                Control size of impurity subspace
        '''
        fo_coeff, nelec_f = self._imporb_builder (kf.mo_coeff, kf.dm1s, kf.veff, kf.fock1,
                                                  max_size=max_size)
        self._update_space_(fo_coeff, nelec_f)
        self._update_trial_state_(kf.mo_coeff, kf.ci, veff=kf.veff, dm1s=kf.dm1s)
        self._update_impurity_hamiltonian_(kf.mo_coeff, kf.ci, h2eff_sub=kf.h2eff_sub,
                                           veff=kf.veff, dm1s=kf.dm1s)
        if hasattr (self, '_max_stepsize'): self._max_stepsize = None # PySCF issue #1762

    _update_keyframe_ = _pull_keyframe_

    def _update_space_(self, imporb_coeff, nelec_imp):
        '''Syntactic sugar for updating the impurity orbital subspace in the encapsulated
        ImpurityMole object.'''
        self.mol._update_space_(imporb_coeff, nelec_imp)

    def _update_trial_state_(self, mo_coeff, ci, veff, dm1s):
        '''Project whole-molecule MO coefficients and CI vectors into the
        impurity space and store on self.mo_coeff; self.ci.'''
        las = self.mol._las
        mf = las._scf
        log = logger.new_logger(self, self.verbose)

        ci = [ci[ifrag] for ifrag in self._ifrags]
        if len (self._ifrags)==1: ci = ci[0]
        self.ci = ci

        # Project mo_coeff and ci keyframe into impurity space and cache
        imporb_coeff = self.mol.get_imporb_coeff ()
        # Inactive orbitals
        mo_core = mo_coeff[:,:las.ncore]
        s0 = mf.get_ovlp ()
        ovlp = imporb_coeff.conj ().T @ s0 @ mo_core
        evals, self.mo_coeff = linalg.eigh (-(ovlp @ ovlp.conj().T))
        self.ncore = np.count_nonzero (evals<-.5)
        evals = -evals[:self.ncore]
        if (self.ncore>0) and not (np.allclose (evals,1.0)):
            idx = np.argmax (np.abs (evals-1.0))
            log.warn ("pull_keyframe imporb problem: <i|P_emb|i> = %e", evals[idx])
        # Active and virtual orbitals (note self.ncas must be set at construction)
        nocc = self.ncore + self.ncas
        mo_las = []
        for ifrag in self._ifrags:
            i = las.ncore + sum (las.ncas_sub[:ifrag])
            j = i + las.ncas_sub[ifrag]
            mo_las.append (mo_coeff[:,i:j])
        mo_las = np.concatenate (mo_las, axis=1)
        ovlp = (imporb_coeff @ self.mo_coeff[:,self.ncore:]).conj ().T @ s0 @ mo_las
        u, svals, vh = linalg.svd (ovlp)
        if (self.ncas>0) and not (np.allclose (svals[:self.ncas],1)):
            idx = np.argmax (np.abs (svals[:self.ncas]-1.0))
            log.warn ("pull_keyframe imporb problem: <imp|active> = %e", svals[idx])
        u[:,:self.ncas] = u[:,:self.ncas] @ vh
        self.mo_coeff[:,self.ncore:] = self.mo_coeff[:,self.ncore:] @ u

        # Canonicalize core and virtual spaces
        fock = las.get_fock (veff=veff, dm1s=dm1s)
        fock = imporb_coeff.conj ().T @ fock @ imporb_coeff
        if self.ncore:
            mo_core = self.mo_coeff[:,:self.ncore]
            fock_core = mo_core.conj ().T @ fock @ mo_core
            w, c = linalg.eigh (fock_core)
            self.mo_coeff[:,:self.ncore] = mo_core @ c
        if (self.mo_coeff.shape[1] - nocc):
            mo_virt = self.mo_coeff[:,nocc:]
            fock_virt = mo_virt.conj ().T @ fock @ mo_virt
            w, c = linalg.eigh (fock_virt)
            self.mo_coeff[:,nocc:] = mo_virt @ c

    def _update_impurity_hamiltonian_(self, mo_coeff, ci, h2eff_sub=None, e_states=None, veff=None,
                                      dm1s=None, casdm1rs=None, casdm2rs=None, weights=None):
        '''Update the Hamiltonian data contained within this impurity solver and all encapsulated
        impurity objects'''
        las = self.mol._las
        gpu = las.use_gpu
        if h2eff_sub is None: h2eff_sub = las.ao2mo (mo_coeff)
        if e_states is None: e_states = las.energy_nuc () + las.states_energy_elec (
            mo_coeff=mo_coeff, ci=ci, h2eff=h2eff_sub)
        e_tot = np.dot (las.weights, e_states)
        if dm1s is None: dm1s = las.make_rdm1s (mo_coeff=mo_coeff, ci=ci)
        if veff is None: veff = las.get_veff (dm=dm1s, spin_sep=True)
        nocc = self.ncore + self.ncas

        # Default these to the "CASSCF" way of making them
        if weights is None: weights = self.fcisolver.weights
        if casdm1rs is None or casdm2rs is None:
            casdm1rs, casdm2rs = self.fcisolver.states_make_rdm12s (self.ci,self.ncas,self.nelecas)
            casdm1rs = np.stack (casdm1rs, axis=1)
            casdm2rs = np.stack (casdm2rs, axis=1)

        # Set underlying SCF object Hamiltonian to state-averaged Heff
        self._scf._update_impham_1_(veff, dm1s, e_tot=e_tot)
        casdm2sr = casdm2rs.transpose (1,0,2,3,4,5)
        casdm2r = casdm2sr[0] + casdm2sr[1] + casdm2sr[1].transpose (0,3,4,1,2) + casdm2sr[2]
        casdm1s = np.tensordot (weights, casdm1rs, axes=1)
        casdm2 = np.tensordot (weights, casdm2r, axes=1)
        eri_cas = ao2mo.restore (1, self.get_h2cas (self.mo_coeff), self.ncas)
        mo_core = self.mo_coeff[:,:self.ncore]
        mo_cas = self.mo_coeff[:,self.ncore:nocc]
        if gpu: libgpu.libgpu_set_update_dfobj_(gpu, 1)
        self._scf._update_impham_2_(mo_core, mo_cas, casdm1s, casdm2, eri_cas)

        # Set state-separated Hamiltonian 1-body
        mo_cas_full = mo_coeff[:,las.ncore:][:,:las.ncas]
        dm1rs_full = las.states_make_casdm1s (ci=ci)
        dm1s_full = np.tensordot (weights, dm1rs_full, axes=1)
        dm1rs_stateshift = dm1rs_full - dm1s_full
        for ifrag in self._ifrags:
            i = sum (las.ncas_sub[:ifrag])
            j = i + las.ncas_sub[ifrag]
            dm1rs_stateshift[:,:,i:j,:] = dm1rs_stateshift[:,:,:,i:j] = 0
        bmPu = getattr (h2eff_sub, 'bmPu', None)
        vj_r = self.get_vj_ext (mo_cas_full, dm1rs_stateshift.sum(1), bmPu=bmPu)
        vk_rs = self.get_vk_ext (mo_cas_full, dm1rs_stateshift, bmPu=bmPu)
        vext = vj_r[:,None,:,:] - vk_rs
        self._imporb_h1_stateshift = vext

        # Set state-separated Hamiltonian 0-body
        mo_core = self.mo_coeff[:,:self.ncore]
        mo_cas = self.mo_coeff[:,self.ncore:][:,:self.ncas]
        dm_core = 2*(mo_core @ mo_core.conj ().T)
        vj, vk = self._scf.get_jk (dm=dm_core)
        veff_core = vj - vk*.5
        e2_core = ((veff_core @ mo_core) * mo_core.conj ()).sum ()
        h1_rs = self.get_hcore_rs ()
        e1_core = np.tensordot (np.dot (h1_rs.sum (1), mo_core), mo_core[:,:].conj (), axes=2)
        h1_rs = self.get_hcore_rs () + veff_core[None,None,:,:]
        h1_rs = lib.einsum ('rsij,ip,jq->rspq', h1_rs, mo_cas.conj (), mo_cas)
        e1_cas = (h1_rs * casdm1rs).sum ((1,2,3))
        e2_cas = np.tensordot (casdm2r, eri_cas, axes=4)*.5
        e_states_core = e1_core + e2_core
        e_states_cas = e1_cas + e2_cas
        e_states_elec = e_states_core + e_states_cas
        e_states_nuc = e_states - e_states_elec
        self._imporb_h0_stateshift = e_states_nuc - self._scf.energy_nuc ()

    def get_vj_ext (self, mo_ext, dm1rs_ext, bmPu=None):
        output_shape = list (dm1rs_ext.shape[:-2]) + [self.mol.nao (), self.mol.nao ()]
        dm1 = dm1rs_ext.reshape (-1, mo_ext.shape[1], mo_ext.shape[1])
        if bmPu is not None:
            log = logger.new_logger (self, self.verbose)
            t_vj = (logger.process_clock(), logger.perf_counter())
            bPuu = np.tensordot (bmPu, mo_ext, axes=((0),(0)))
            rho = np.tensordot (dm1, bPuu, axes=((1,2),(1,2)))
            bPii = self._scf._cderi
            vj = lib.unpack_tril (np.tensordot (rho, bPii, axes=((-1),(0))))
            t_vj = log.timer("vj ext", *t_vj)    
        else: # Safety case: AO-basis SCF driver
            imporb_coeff = self.mol.get_imporb_coeff ()
            dm1 = np.dot (mo_ext, np.dot (dm1, mo_ext.conj().T)).transpose (1,0,2)
            vj = self.mol._las._scf.get_j (dm=dm1)
            vj = np.dot (imporb_coeff.conj ().T, np.dot (vj, imporb_coeff)).transpose (1,0,2)
        return vj.reshape (*output_shape) 

    def get_vk_ext (self, mo_ext, dm1rs_ext, bmPu=None):
        output_shape = list (dm1rs_ext.shape[:-2]) + [self.mol.nao (), self.mol.nao ()]
        dm1 = dm1rs_ext.reshape (-1, mo_ext.shape[1], mo_ext.shape[1])
        imporb_coeff = self.mol.get_imporb_coeff ()
        if bmPu is not None:
            log = logger.new_logger (self, self.verbose)
            t_vk = (logger.process_clock(), logger.perf_counter())
            biPu = np.tensordot (imporb_coeff, bmPu, axes=((0),(0)))
            vuiP = np.tensordot (dm1, biPu, axes=((-1),(-1)))
            vk = np.tensordot (vuiP, biPu, axes=((-3,-1),(-1,-2)))
            t_vk = log.timer("vk ext", *t_vk)    
        else: # Safety case: AO-basis SCF driver
            dm1 = np.dot (mo_ext, np.dot (dm1, mo_ext.conj().T)).transpose (1,0,2)
            #vk = self.mol._las._scf.get_k (dm=dm1) 
            _,vk = self.mol._las._scf.get_jk (dm=dm1) #TODO: for gpu run, this has to be written as get_k (it is now written as get_jk)
            vk = np.dot (imporb_coeff.conj ().T, np.dot (vk, imporb_coeff)).transpose (1,0,2)
        return vk.reshape (*output_shape)
            
    def get_hcore_rs (self):
        return self._scf.get_hcore_spinsep ()[None,:,:,:] + self._imporb_h1_stateshift

    def energy_nuc_r (self):
        return self._scf.energy_nuc () + self._imporb_h0_stateshift

class ImpurityCASSCF (mcscf.mc1step.CASSCF, ImpuritySolver):

    # make sure the fcisolver flag dump goes to the fragment output file,
    # not the main output file
    def dump_flags (self, verbose=None):
        with lib.temporary_env (self.fcisolver, stdout=self.stdout):
            mcscf.mc1step.CASSCF.dump_flags(self, verbose=verbose)

    def get_h1eff (self, mo_coeff=None, ncas=None, ncore=None):
        ''' must needs change the dimension of h1eff '''
        assert (False)
        h1_avg_spinless, energy_core = self.h1e_for_las (mo_coeff, ncas, ncore)[1]
        mo_cas = mo_coeff[:,ncore:][:,:ncas]
        h1_avg_sz = mo_cas.conj ().T @ self._scf.get_hcore_sz () @ mo_cas
        h1_avg = np.stack ([h1_avg_spinless + h1_avg_sz, h1_avg_spinless - h1_avg_sz], axis=0)
        h1 += mo_cas.conj ().T @ self.get_hcore_stateshift () @ mo_cas
        return h1, energy_core

    def update_casdm (self, mo, u, fcivec, e_cas, eris, envs={}):
        ''' inject the stateshift h1 into envs '''
        mou = mo @ u[:,self.ncore:][:,:self.ncas]
        h1_stateshift = self.get_hcore_rs () - self.get_hcore ()[None,None,:,:]
        h1_stateshift = np.tensordot (mou.conj ().T, np.dot (h1_stateshift, mou),
                                   axes=((1),(2))).transpose (1,2,0,3)
        envs['h1_stateshift'] = h1_stateshift
        return super().update_casdm (mo, u, fcivec, e_cas, eris, envs=envs)

    def solve_approx_ci (self, h1, h2, ci0, ecore, e_cas, envs):
        ''' get the stateshifted h1 from envs '''
        h1 = h1[None,None,:,:] + envs['h1_stateshift']
        return super().solve_approx_ci (h1, h2, ci0, ecore, e_cas, envs)

    def casci (self, mo_coeff, ci0=None, eris=None, verbose=None, envs=None):
        # I sadly had to copy-and-paste this function from mc1step.py due to inapplicable
        #   1) logging commands and
        #   2) error checks
        # that could not be monkeypatched out.
        log = logger.new_logger(self, verbose)
        if eris is None:
            fcasci = copy.copy(self)
            fcasci.ao2mo = self.get_h2cas
        else:
            fcasci = _fake_h_for_fast_casci(self, mo_coeff, eris)

        e_tot, e_cas, fcivec = casci_kernel(fcasci, mo_coeff, ci0, log,
                                            envs=envs)
        #if not isinstance(e_cas, (float, numpy.number)):
        #    raise RuntimeError('Multiple roots are detected in fcisolver.  '
        #                       'CASSCF does not know which state to optimize.\n'
        #                       'See also  mcscf.state_average  or  mcscf.state_specific  for excited states.')
        #elif numpy.ndim(e_cas) != 0:
            # This is a workaround for external CI solver compatibility.
        #    e_cas = e_cas[0]

        if envs is not None and log.verbose >= logger.INFO:
            log.debug('CAS space CI energy = {}'.format (e_cas))

            if getattr(self.fcisolver, 'spin_square', None):
                try:
                    ss = self.fcisolver.spin_square(fcivec, self.ncas, self.nelecas)
                except NotImplementedError:
                    ss = None
            else:
                ss = None

            if 'imicro' in envs:  # Within CASSCF iteration
                if ss is None:
                    log.info('macro iter %3d (%3d JK  %3d micro), '
                             'CASSCF E = %#.15g  dE = % .8e',
                             envs['imacro'], envs['njk'], envs['imicro'],
                             e_tot, e_tot-envs['elast'])
                else:
                    log.info('macro iter %3d (%3d JK  %3d micro), '
                             'CASSCF E = %#.15g  dE = % .8e  S^2 = %.7f',
                             envs['imacro'], envs['njk'], envs['imicro'],
                             e_tot, e_tot-envs['elast'], ss[0])
                if 'norm_gci' in envs and envs['norm_gci'] is not None:
                    log.info('               |grad[o]|=%5.3g  '
                             '|grad[c]|=%5.3g  |ddm|=%5.3g  |maxRot[o]|=%5.3g',
                             envs['norm_gorb0'],
                             envs['norm_gci'], envs['norm_ddm'], envs['max_offdiag_u'])
                else:
                    log.info('               |grad[o]|=%5.3g  |ddm|=%5.3g  |maxRot[o]|=%5.3g',
                             envs['norm_gorb0'], envs['norm_ddm'], envs['max_offdiag_u'])
            else:  # Initialization step
                if ss is None:
                    log.info('CASCI E = %#.15g', e_tot)
                else:
                    log.info('CASCI E = %#.15g  S^2 = %.7f', e_tot, ss[0])
        return e_tot, e_cas, fcivec

    def _finalize(self):
        # I sadly had to copy-and-paste this function from casci.py due to inapplicable
        #   1) logging commands and
        #   2) error checks
        # that could not be monkeypatched out.
        log = logger.Logger(self.stdout, self.verbose)
        if log.verbose >= logger.NOTE and getattr(self.fcisolver, 'spin_square', None):
            if isinstance(self.e_cas, (float, np.number)):
                try:
                    ss = self.fcisolver.spin_square(self.ci, self.ncas, self.nelecas)
                    log.note('CASCI E = %#.15g  E(CI) = %#.15g  S^2 = %.7f',
                             self.e_tot, self.e_cas, ss[0])
                except NotImplementedError:
                    log.note('CASCI E = %#.15g  E(CI) = %#.15g',
                             self.e_tot, self.e_cas)
            elif callable (getattr (self.fcisolver, 'states_spin_square')):
                ss = self.fcisolver.states_spin_square (self.ci, self.ncas, self.nelecas)
                for i, e in enumerate(self.e_cas):
                    log.note('CASCI state %3d  E = %#.15g  E(CI) = %#.15g  S^2 = %.7f',
                             i, self.e_states[i], e, ss[0][i])
            else:
                for i, e in enumerate(self.e_cas):
                    try:
                        ss = self.fcisolver.spin_square(self.ci[i], self.ncas, self.nelecas)
                        log.note('CASCI state %3d  E = %#.15g  E(CI) = %#.15g  S^2 = %.7f',
                                 i, self.e_states[i], e, ss[0])
                    except NotImplementedError:
                        log.note('CASCI state %3d  E = %#.15g  E(CI) = %#.15g',
                                 i, self.e_states[i], e)

        else:
            if isinstance(self.e_cas, (float, np.number)):
                log.note('CASCI E = %#.15g  E(CI) = %#.15g', self.e_tot, self.e_cas)
            else:
                for i, e in enumerate(self.e_cas):
                    log.note('CASCI state %3d  E = %#.15g  E(CI) = %#.15g',
                             i, self.e_states[i], e)
        return self

    def rotate_orb_cc (self, mo, fcivec, fcasdm1, fcasdm2, eris, x0_guess=None,
                       conv_tol_grad=1e-4, max_stepsize=None, verbose=None):
        ''' Intercept fcasdm1 and replace it with fully-separated casdm1rs '''
        try:
            casdm1rs = np.stack (self.fcisolver.states_make_rdm1s (fcivec(), self.ncas,
                                                                   self.nelecas), axis=1)
        except AttributeError as e:
            casdm1rs = self.fcisolver.make_rdm1s (fcivec(), self.ncas, self.nelecas)[None,:,:,:]
        my_fcasdm1 = lambda:casdm1rs
        return super().rotate_orb_cc (mo, fcivec, my_fcasdm1, fcasdm2, eris, x0_guess=x0_guess,
                                      conv_tol_grad=conv_tol_grad, max_stepsize=max_stepsize,
                                      verbose=verbose)

    def gen_g_hop (self, mo, u, casdm1rs, casdm2, eris):
        weights = self.fcisolver.weights
        casdm1 = np.tensordot (weights, casdm1rs.sum (1), axes=1)
        g_orb, gorb_update, h_op, h_diag = super().gen_g_hop (mo, u, casdm1, casdm2, eris)
        ncore = self.ncore
        ncas = self.ncas
        nelecas = self.nelecas
        nocc = ncore + ncas
        nao, nmo = mo.shape
        nroots = self.fcisolver.nroots

        h1_rs = lib.einsum ('ip,rsij,jq->rspq', mo.conj (), self.get_hcore_rs (), mo)
        h1 = mo.conj ().T @ self.get_hcore () @ mo

        def g1_correction (dm1_rs, dm1, u=1):
            g1_rs = np.zeros_like (h1_rs)
            h1u_rs = np.dot (h1_rs, u).transpose (0,1,3,2)
            h1u_rs = np.dot (h1u_rs, u).transpose (0,1,3,2)
            h1u_rs = h1u_rs[:,:,:,ncore:nocc]
            h1u = (np.dot (np.dot (h1, u).T, u).T)[:,ncore:nocc]
            g1_rs[...,ncore:nocc] = lib.einsum ('rsik,rskj->rsij', h1u_rs, dm1_rs)
            g1 = np.tensordot (weights, g1_rs.sum (1), axes=1)
            g1[:,ncore:nocc] -= h1u @ dm1
            return g1

        # Return 1: the macrocycle gradient (odd matrix)
        g1 = g1_correction (casdm1rs, casdm1)
        g_orb += self.pack_uniq_var (g1 - g1.T)

        # Return 2: the microcycle gradient as a function of u and fcivec (odd matrix)
        def my_gorb_update (u, fcivec):
            g_orb_u = gorb_update (u, fcivec)
            try:
                casdm1rs = np.stack (self.fcisolver.states_make_rdm1s (fcivec, ncas, nelecas),
                                     axis=1)
            except AttributeError as e:
                casdm1rs = self.fcisolver.make_rdm1s (fcivec, ncas, nelecas)[None,:,:,:]
            casdm1 = np.tensordot (weights, casdm1rs.sum (1), axes=1)
            g1_u = g1_correction (casdm1rs, casdm1, u=u)
            g_orb_u += self.pack_uniq_var (g1_u - g1_u.T)
            return g_orb_u

        # Return 3: the diagonal elements of the Hessian (even matrix)
        g2 = np.zeros_like (g1)
        g2[:,ncore:nocc] = lib.einsum ('r,rspp,rsqq->pq', weights, h1_rs, casdm1rs)
        g2[:,ncore:nocc] -= lib.einsum ('pp,qq->pq', h1, casdm1)
        h1_rs_cas = h1_rs[:,:,ncore:nocc,ncore:nocc]
        h1_cas = h1[ncore:nocc,ncore:nocc]
        g2_cas = g2[ncore:nocc,ncore:nocc]
        g2_cas[:,:] -= lib.einsum ('r,rspq,rspq->pq', weights, h1_rs_cas, casdm1rs)
        g2_cas[:,:] += h1_cas*casdm1
        g2 = g2 + g2.T
        g1_diag = g1.diagonal ()
        g2 -= g1_diag + g1_diag.reshape (-1,1)
        idx = np.arange (nmo)
        g2[idx,idx] += g1_diag * 2
        h_diag += self.pack_uniq_var (g2)

        # Return 4: the Hessian as a function (odd matrix)
        def my_h_op (x):
            x1 = self.unpack_uniq_var (x)
            x1_cas = x1[:,ncore:nocc]
            hx = np.zeros_like (g1)
            hx[:,ncore:nocc] = lib.einsum ('r,rsik,kl,rslj->ij', weights, h1_rs, x1_cas, casdm1rs)
            hx[:,ncore:nocc] -= h1 @ x1_cas @ casdm1
            hx -= (g1 + g1.T) @ x1 / 2
            return h_op (x) + self.pack_uniq_var (hx - hx.T)

        return g_orb, my_gorb_update, my_h_op, h_diag

class ImpurityLASCI_HessianOperator (lasci_sync.LASCI_HessianOperator):
    def _init_dms_(self, casdm1frs, casdm2fr):
        lasci_sync.LASCI_HessianOperator._init_dms_(self, casdm1frs, casdm2fr)
        ncore, nocc, nroots = self.ncore, self.nocc, self.nroots
        self.dm1rs = np.stack ([self.dm1s,]*nroots, axis=0)
        self.dm1rs[:,:,ncore:nocc,ncore:nocc] = self.casdm1rs

    def _init_ham_(self, h2eff_sub, veff):
        lasci_sync.LASCI_HessianOperator._init_ham_(self, h2eff_sub, veff)
        las, mo_coeff, ncore, nocc = self.las, self.mo_coeff, self.ncore, self.nocc
        h1rs = np.dot (las.get_hcore_rs (), mo_coeff)
        h1rs = np.tensordot (mo_coeff.conj (), h1rs, axes=((0),(2))).transpose (1,2,0,3)
        hcore = mo_coeff.conj ().T @ las.get_hcore () @ mo_coeff
        dh1rs = h1rs - hcore[None,None,:,:]
        # _init_ci_ and ci_response_diag
        for ix, h1rs in enumerate (self.h1frs):
            i = sum (self.ncas_sub[:ix])
            j = i + self.ncas_sub[ix]
            h1rs[:,:,:,:] += dh1rs[:,:,i:j,i:j]
        # _init_orb_ and orbital_response 
        self.h1rs = self.h1s[None,:,:,:] + dh1rs
        # ci_response_offdiag
        self.h1rs_cas = self.h1s_cas[None,:,:,:] + dh1rs[:,:,:,ncore:nocc]
        # Energy reportback
        self.e_tot += np.einsum ('rspq,rspq,r->', dh1rs, self.dm1rs, self.weights)

    def _init_orb_(self):
        ncore, nocc = self.ncore, self.nocc
        lasci_sync.LASCI_HessianOperator._init_orb_(self)
        for w, h1s, casdm1s in zip (self.weights, self.h1rs, self.casdm1rs):
            dh1s = h1s[:,ncore:nocc,ncore:nocc] - self.h1s[:,ncore:nocc,ncore:nocc]
            self.fock1[:,ncore:nocc] += w * (dh1s[0] @ casdm1s[0] + dh1s[1] @ casdm1s[1])

    def _get_Horb_diag (self):
        # It's unclear that this is even necessary...
        Hdiag = 0
        for w, h, d in zip (self.weights, self.h1rs, self.dm1rs):
            with lib.temporary_env (self, h1s=h, dm1s=d):
                Hdiag += w * lasci_sync.LASCI_HessianOperator._get_Horb_diag (self)
        return Hdiag

    def ci_response_offdiag (self, kappa1, h1frs_prime):
        ncore, nocc, ncas_sub = self.ncore, self.nocc, self.ncas_sub
        kappa1_cas = kappa1[ncore:nocc,:]
        dh1rs_cas = self.h1rs_cas - self.h1s_cas[None,:,:,:]
        dh1_core = -np.tensordot (kappa1_cas, dh1rs_cas, axes=((1),(2)))
        dh1_core = dh1_core.transpose (1,2,0,3) + dh1_core.transpose (1,2,3,0)
        for i, h1rs in enumerate (h1frs_prime):
            j = sum (ncas_sub[:i])
            k = j + ncas_sub[i]
            h1rs[:,:,:,:] += dh1_core[:,:,j:k,j:k]
        return lasci_sync.LASCI_HessianOperator.ci_response_offdiag (
            self, kappa1, h1frs_prime)

    def orbital_response (self, kappa1, odm1s, ocm2, tdm1rs, tcm2, veff_prime):
        kappa2 = lasci_sync.LASCI_HessianOperator.orbital_response (
            self, kappa1, odm1s, ocm2, tdm1rs, tcm2, veff_prime
        )
        h1rs = self.h1rs - self.h1s[None,:,:,:]
        odm1rs = -np.dot (self.dm1rs, kappa1)
        odm1rs += odm1rs.transpose (0,1,3,2)
        edm1rs = odm1rs + tdm1rs
        for w, h, d in zip (self.weights, h1rs, edm1rs):
            fock1 = h[0] @ d[0] + h[1] @ d[1]
            kappa2 += w * (fock1 - fock1.T)
        return kappa2

class ImpurityLASCI (lasci.LASCINoSymm, ImpuritySolver):
    _hop = ImpurityLASCI_HessianOperator

    def _update_impurity_hamiltonian_(self, mo_coeff, ci, h2eff_sub=None, e_states=None, veff=None,
                                      dm1s=None, casdm1rs=None, casdm2rs=None, weights=None):
        if weights is None: weights = self.weights
        if casdm1rs is None: casdm1rs = self.states_make_casdm1s (ci=self.ci)
        if casdm2rs is None: 
            casdm2frs = self.states_make_casdm2s_sub (ci=self.ci)
            nroots = len (casdm1rs)
            ncas = casdm1rs[0][0].shape[0]
            casdm2rs = np.zeros ((nroots,3,ncas,ncas,ncas,ncas), dtype=casdm1rs[0][0].dtype)
            for d2, d1 in zip (casdm2rs, casdm1rs):
                d1d1_aa = np.multiply.outer (d1[0], d1[0])
                d2[0] = d1d1_aa - d1d1_aa.transpose (0,3,2,1)
                d2[1] = np.multiply.outer (d1[0], d1[1])
                d1d1_bb = np.multiply.outer (d1[1], d1[1])
                d2[2] = d1d1_bb - d1d1_bb.transpose (0,3,2,1)
            for ifrag, d2f in enumerate (casdm2frs):
                i = sum (self.ncas_sub[:ifrag])
                j = i + self.ncas_sub[ifrag]
                casdm2rs[:,:,i:j,i:j,i:j,i:j] = d2f[:]
        ImpuritySolver._update_impurity_hamiltonian_(
            self, mo_coeff, ci, h2eff_sub=h2eff_sub, e_states=e_states, veff=veff, dm1s=dm1s,
            casdm1rs=casdm1rs, casdm2rs=casdm2rs, weights=weights
        )

    def get_grad_orb (las, **kwargs):
        gorb = lasci.LASCINoSymm.get_grad_orb (las, **kwargs)
        mo_coeff = kwargs.get ('mo_coeff', self.mo_coeff)
        hermi = kwargs.get ('hermi', -1)
        nao, nmo = las.mo_coeff.shape
        ncore, ncas = las.ncore, las.ncas
        nocc = ncore + ncas
        mo_cas = mo_coeff[:,ncore:nocc]
        dh1_rs = np.dot (self.get_hcore_rs () - self.get_hcore ()[None,None,:,:], mo_cas)
        dh1_rs = np.tensordot (mo_coeff.conj (), dh1_rs, axes=((0),(2))).transpose (1,2,0,3)
        casdm1rs = las.states_make_casdm1s (ci=ci)
        f = np.zeros ((nmo,nmo), dtype=gorb.dtype)
        for w, h, d in zip (las.weights, dh1_rs, casdm1rs):
            f[:,ncore:nocc] += w * (h[0] @ d[0] + h[1] @ d[1])
        if hermi == -1:
            return gorb + f - f.T
        elif hermi == 1:
            return gorb + .5*(f+f.T)
        elif hermi == 0:
            return gorb + f
        else:
            raise ValueError ("kwarg 'hermi' must = -1, 0, or +1")

    def h1e_for_las (las, **kwargs):
        h1e_fr = lasci.LASCINoSymm.h1e_for_las (las, **kwargs)
        mo_coeff = kwargs.get ('mo_coeff', self.mo_coeff)
        ncas_sub = kwargs.get ('ncas_sub', self.ncas_sub)
        dh1_rs = np.dot (self.get_hcore_rs () - self.get_hcore ()[None,None,:,:], mo_coeff)
        dh1_rs = np.tensordot (mo_coeff.conj (), dh1_rs, axes=((0),(2))).transpose (1,2,0,3)
        for ix in range (len (ncas_sub)):
            i = sum (ncas_sub[:ix])
            j = i + ncas_sub[ix]
            h1e_fr[ix] += dh1_rs[:,:,i:j,i:j]
        return h1e_fr

    def states_energy_elec (self, **kwargs):
        energy_elec = lasci.LASCINoSymm.states_energy_elec (self, **kwargs)
        mo_coeff = kwargs.get ('mo_coeff', self.mo_coeff)
        ci = kwargs.get ('ci', self.ci)
        ncore = kwargs.get ('ncore', self.ncore)
        ncas = kwargs.get ('nncas', self.ncas)
        ncas_sub = kwargs.get ('ncas_sub', self.ncas_sub)
        nelecas_sub = kwargs.get ('nelecas_sub', self.nelecas_sub)
        casdm1frs = kwargs.get ('casdm1frs', self.states_make_casdm1s_sub (
            ci=ci, ncas_sub=ncas_sub, nelecas_sub=nelecas_sub
        ))
        casdm1rs = self.states_make_casdm1s (ci=ci, ncas_sub=ncas_sub, nelecas_sub=nelecas_sub,
                                             casdm1frs=casdm1frs)
        nao, nmo = mo_coeff.shape
        nocc = ncore + ncas
        mo_cas = mo_coeff[:,ncore:nocc]
        dh1_rs = np.dot (self.get_hcore_rs () - self.get_hcore ()[None,None,:,:], mo_cas)
        dh1_rs = np.tensordot (mo_cas.conj (), dh1_rs, axes=((0),(2))).transpose (1,2,0,3)
        enuc_r = self.energy_nuc_r ()
        for ix, (h, d) in enumerate (zip (dh1_rs, casdm1rs)):
            energy_elec[ix] += np.dot (h.ravel (), d.ravel ())
            energy_elec[ix] += enuc_r[ix] - self.energy_nuc ()
        return energy_elec

    def energy_elec (self, **kwargs):
        energy_elec = self.states_energy_elec (**kwargs)
        return np.dot (self.weights, energy_elec)


def get_impurity_casscf (las, ifrag, imporb_builder=None):
    output = getattr (las.mol, 'output', None)
    # MRH: checking for '/dev/null' specifically as a string is how mol.build does it
    if not ((output is None) or (output=='/dev/null')): output += '.{}'.format (ifrag)
    imol = ImpurityMole (las, output=output)
    imf = ImpurityHF (imol)
    if isinstance (las, _DFLASCI):
        imf = imf.density_fit ()
    imc = ImpurityCASSCF (imf, las.ncas_sub[ifrag], las.nelecas_sub[ifrag])
    if isinstance (las, _DFLASCI):
        imc = df.density_fit (imc)
    imc = _state_average_mcscf_solver (imc, las.fciboxes[ifrag])
    imc._ifrags = [ifrag,]
    if imporb_builder is not None:
        imporb_builder.log = logger.new_logger (imc, imc.verbose)
    imc._imporb_builder = imporb_builder
    params = getattr (las, 'impurity_params', {})
    glob = {key: val for key, val in params.items () if isinstance (key, str)}
    imc.__dict__.update (glob)
    imc.__dict__.update (params.get (ifrag, {}))
    return imc

def get_pair_lasci (las, frags, inherit_df=False):
    stdout_dict = stdout = getattr (las, '_flas_stdout', None)
    if stdout is not None: stdout = stdout.get (frags, None)
    output = getattr (las.mol, 'output', None)
    if not ((output is None) or (output=='/dev/null')):
        output = output + '.' + '.'.join ([str (s) for s in frags])
    imol = ImpurityMole (las, output=output, stdout=stdout)
    if stdout is None and output is not None and stdout_dict is not None:
        stdout_dict[frags] = imol.stdout
    imf = ImpurityHF (imol)
    if inherit_df and isinstance (las, _DFLASCI):
        imf = imf.density_fit ()
    ncas_sub = [las.ncas_sub[i] for i in frags]
    nelecas_sub = [las.nelecas_sub[i] for i in frags]
    ilas = ImpurityLASCI (imf, ncas_sub, nelecas_sub, use_gpu=las.use_gpu)
    if inherit_df and isinstance (las, _DFLASCI):
        ilas = lasci.density_fit (ilas, with_df=imf.with_df)
    charges, spins, smults, wfnsyms = lasci.get_space_info (las)
    ilas.state_average_(weights=las.weights, charges=charges[:,frags], spins=spins[:,frags],
                        smults=smults[:,frags], wfnsyms=wfnsyms[:,frags])
    def imporb_builder (mo_coeff, dm1s, veff, fock1, **kwargs):
        idx = np.zeros (mo_coeff.shape[1], dtype=bool)
        for ix in frags:    
            i = las.ncore + sum (las.ncas_sub[:ix])
            j = i + las.ncas_sub[ix]
            idx[i:j] = True
        fo_coeff = mo_coeff[:,idx]
        nelec_f = sum ([sum (n) for n in nelecas_sub])
        return fo_coeff, nelec_f
    ilas._imporb_builder = imporb_builder
    ilas._ifrags = frags
    ilas.conv_tol_grad = 'DEFAULT'
    ilas.min_cycle_macro = 1
    params = getattr (las, 'relax_params', {})
    glob = {key: val for key, val in params.items () if isinstance (key, str)}
    glob = {key: val for key, val in glob.items () if key not in ('frozen', 'frozen_ci')}
    ilas.__dict__.update (glob)
    loc = params.get (tuple (frags), {})
    loc = {key: val for key, val in loc.items () if key not in ('frozen', 'frozen_ci')}
    ilas.__dict__.update (loc)
    return ilas

if __name__=='__main__':
    from mrh.tests.lasscf.c2h6n4_struct import structure as struct
    mol = struct (1.0, 1.0, '6-31g', symmetry=False)
    mol.verbose = 5
    mol.output = 'lasscf_async_crunch.log'
    mol.build ()
    mf = scf.RHF (mol).run ()
    from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
    las = LASSCF (mf, (4,4), ((4,0),(0,4)), spin_sub=(5,5))
    mo = las.localize_init_guess ((list (range (3)), list (range (9,12))), mf.mo_coeff)
    las.state_average_(weights=[1,0,0,0,0],
                       spins=[[0,0],[2,0],[-2,0],[0,2],[0,-2]],
                       smults=[[1,1],[3,1],[3,1],[1,3],[1,3]])
    las.conv_tol_grad = 1e-7
    las.kernel (mo)
    print (las.converged)
    from mrh.my_pyscf.mcscf.lasci import get_grad_orb
    if not callable (getattr (las, 'get_grad_orb', None)):
        from functools import partial
        las.get_grad_orb = partial (get_grad_orb, las)

    ###########################
    # Build the embedding space
    from mrh.my_pyscf.mcscf.lasscf_async.split import get_impurity_space_constructor
    get_imporbs_0 = get_impurity_space_constructor (las, 0, list (range (3)))
    ###########################

    ###########################
    # Build the impurity method object
    from mrh.my_pyscf.mcscf.lasscf_async.keyframe import LASKeyframe
    imc = get_impurity_casscf (las, 0, imporb_builder=get_imporbs_0)
    kf1 = LASKeyframe (las, las.mo_coeff, las.ci)
    imc._update_keyframe_(kf1)
    ###########################

    ###########################
    # Futz up the guess
    imc.ci = None
    kappa = (np.random.rand (*imc.mo_coeff.shape)-.5) * np.pi / 100
    kappa -= kappa.T
    umat = linalg.expm (kappa)
    imc.mo_coeff = imc.mo_coeff @ umat
    ###########################

    imc.kernel ()
    print (imc.converged, imc.e_tot, las.e_tot, imc.e_tot-las.e_tot)
    for t, r in zip (imc.e_states, las.e_states):
        print (t, r, t-r)
    kf2 = imc._push_keyframe (kf1)
    from mrh.my_pyscf.mcscf.lasscf_async.keyframe import approx_keyframe_ovlp
    print (approx_keyframe_ovlp (las, kf1, kf2))
