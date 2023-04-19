import numpy as np
from scipy import linalg
from pyscf import gto, scf, mcscf, ao2mo, lib, df
from pyscf.fci.direct_spin1 import _unpack_nelec
import copy

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

    def _update_space_(self, imporb_coeff, nelec_imp):
        self._imporb_coeff = imporb_coeff
        nelec_imp = _unpack_nelec (nelec_imp)
        self.nelectron = sum (nelec_imp)
        self.spin = nelec_imp[0] - nelec_imp[1]

    def get_imporb_coeff (self): return self._imporb_coeff
    def nao_nr (self, *args, **kwargs): return self._imporb_coeff.shape[-1]
    def nao (self): return self._imporb_coeff.shape[-1]

class ImpuritySCF (scf.hf.SCF):
    def _update_impham_1_(self, veff, dm1s, e_tot=None):
        ''' after this function, get_hcore () and energy_nuc () functions return
            full-system state-averaged fock and e_tot, respectively '''
        if e_tot is None: e_tot = self.mol._las.e_tot
        imporb_coeff = self.mol.get_imporb_coeff ()
        nimp = self.mol.nao ()
        mf = self.mol._las._scf
        # Two-electron integrals
        if getattr (mf, '_eri', None) is not None:
            self._eri = ao2mo.full (mf._eri, imporb_coeff, 4)
        if getattr (mf, 'with_df', None) is not None:
            # TODO: impurity outcore cderi
            self.with_df._cderi = np.empty ((mf.with_df.get_naoaux (), nimp*(nimp+1)//2),
                                            dtype=imporb_coeff.dtype)
            ijmosym, mij_pair, moij, ijslice = ao2mo.incore._conc_mos (imporb_coeff, imporb_coeff,
                                                                        compact=True)
            b0 = 0
            for eri1 in mf.with_df.loop ():
                b1 = b0 + eri1.shape[0]
                eri2 = self._cderi[b0:b1]
                eri2 = ao2mo._ao2mo.nr_e2 (eri1, moij, ijslice, aosym='s2', mosym=ijmosym,
                                           out=eri2)
                b0 = b1
        # External mean-field; potentially spin-broken
        h1s = mf.get_hcore ()[None,:,:] + veff
        h1s = np.dot (imporb_coeff.conj ().T, np.dot (h1s, imporb_coeff)).transpose (1,0,2)
        self._imporb_h1 = h1s.sum (0) / 2
        self._imporb_h1_sz = (h1s[0] - h1s[1]) / 2
        self._imporb_h0 = e_tot 

    def _update_impham_2_(self, mo_docc, mo_dm, dm1s, dm2, eri_dm=None):
        ''' after this function, get_hcore () and energy_nuc () functions return
            the state-averaged hcore and e0, respectively '''
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
    core_dm = np.dot(mo_core, mo_core.T) * 2
    energy_core = casscf.energy_nuc_r ()
    hcore = casscf.get_hcore ()
    energy_core += np.einsum('ij,ji', core_dm, hcore)
    energy_core += eris.vhf_c[:ncore,:ncore].trace ()
    h1 = casscf.get_hcore_rs ()
    h1eff = np.tensordot (mo_cas.conj (), np.dot (h1, mo_cas), axes=((0),(2))).transpose (1,2,0,3)
    h1eff += eris.vhf_c[None,None,ncore:nocc,ncore:nocc]
    
    mc.get_h1eff = lambda *args: (h1eff, energy_core)

    eri_cas = eris.ppaa[ncore:nocc,ncore:nocc,:,:].copy()
    mc.get_h2eff = lambda *args: eri_cas
    return mc


# This is the really tricky part
class ImpurityCASSCF (mcscf.mc1step.CASSCF):

    def _update_keyframe_(self, mo_coeff, ci, h2eff_sub=None, e_states=None):
        # Project mo_coeff and ci keyframe into impurity space and cache
        las = self.mol._las
        if h2eff_sub is None: h2eff_sub = las.ao2mo (mo_coeff)
        if e_states is None: e_states = las.energy_nuc () + las.energy_elec (
            mo_coeff=mo_coeff, ci=ci, h2eff=h2eff_sub)
        e_tot = np.dot (las.weights, e_states)
        mf = las._scf
        _ifrag = self._ifrag
        imporb_coeff = self.mol.get_imporb_coeff ()
        self.ci = ci[_ifrag]
        # Inactive orbitals
        mo_core = mo_coeff[:,:las.ncore]
        s0 = mf.get_ovlp ()
        ovlp = imporb_coeff.conj ().T @ s0 @ mo_core
        self.mo_coeff, svals, vh = linalg.svd (ovlp)
        self.ncore = np.count_nonzero (np.isclose (svals, 1))
        # Active and virtual orbitals (note self.ncas must be set at construction)
        nocc = self.ncore + self.ncas
        i = las.ncore + sum (las.ncas_sub[:_ifrag])
        j = i + las.ncas_sub[_ifrag]
        mo_las = mo_coeff[:,i:j]
        ovlp = (imporb_coeff @ self.mo_coeff[:,self.ncore:]).conj ().T @ s0 @ mo_las
        u, svals, vh = linalg.svd (ovlp)
        assert (np.allclose(svals[:self.ncas], 1))
        u[:,:self.ncas] = u[:,:self.ncas] @ vh
        self.mo_coeff[:,self.ncore:] = self.mo_coeff[:,self.ncore:] @ u
        # Set underlying SCF object Hamiltonian to state-averaged Heff
        casdm1rs, casdm2rs = self.fcisolver.states_make_rdm12s (self.ci, self.ncas, self.nelecas)
        casdm1rs = np.stack (casdm1rs, axis=1)
        casdm2sr = np.stack (casdm2rs, axis=0)
        casdm2r = casdm2sr[0] + casdm2sr[1] + casdm2sr[1].transpose (0,3,4,1,2) + casdm2sr[2]
        casdm1s = np.tensordot (self.fcisolver.weights, casdm1rs, axes=1)
        casdm2 = np.tensordot (self.fcisolver.weights, casdm2r, axes=1)
        #casdm1s, casdm2s = self.fcisolver.make_rdm12s (self.ci, self.ncas, self.nelecas)
        #casdm2 = casdm2s[0] + casdm2s[1] + casdm2s[1].transpose (2,3,0,1) + casdm2s[2]
        eri_cas = ao2mo.restore (1, self.get_h2eff (self.mo_coeff), self.ncas)
        mo_core = self.mo_coeff[:,:self.ncore]
        mo_cas = self.mo_coeff[:,self.ncore:nocc]
        self._scf._update_impham_2_(mo_core, mo_cas, casdm1s, casdm2, eri_cas)
        # Canonicalize core and virtual spaces
        dm1s = np.dot (mo_cas, np.dot (casdm1s, mo_cas.conj ().T)).transpose (1,0,2)
        dm1s += np.dot (mo_core, mo_core.conj ().T)[None,:,:]
        fock = self._scf.get_fock (dm=dm1s)
        mo_core = self.mo_coeff[:,:self.ncore]
        fock_core = mo_core.conj ().T @ fock @ mo_core
        w, c = linalg.eigh (fock_core)
        self.mo_coeff[:,:self.ncore] = mo_core @ c
        mo_virt = self.mo_coeff[:,nocc:]
        fock_virt = mo_virt.conj ().T @ fock @ mo_virt
        w, c = linalg.eigh (fock_virt)
        self.mo_coeff[:,nocc:] = mo_virt @ c
        # Set state-separated Hamiltonian 1-body
        mo_cas_full = mo_coeff[:,las.ncore:][:,:las.ncas]
        dm1rs_full = las.states_make_casdm1s (ci=ci)
        dm1s_full = np.tensordot (self.fcisolver.weights, dm1rs_full, axes=1)
        dm1rs_stateshift = dm1rs_full - dm1s_full
        i = sum (las.ncas_sub[:_ifrag])
        j = i + las.ncas_sub[_ifrag]
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
        e0_states = e_states - e2_core
        h1_rs = self.get_hcore_rs ()
        e1_core = np.tensordot (np.dot (h1_rs.sum (1), mo_core), mo_core[:,:].conj (), axes=2)
        e0_states -= e1_core
        h1_rs = self.get_hcore_rs () + veff_core[None,None,:,:]
        h1_rs = lib.einsum ('rsij,ip,jq->rspq', h1_rs, mo_cas.conj (), mo_cas)
        e1_cas = (h1_rs * casdm1rs).sum ((1,2,3))
        e2_cas = np.tensordot (casdm2r, eri_cas, axes=4)*.5
        e0_states -= e1_cas + e2_cas
        self._imporb_h0_stateshift = e0_states - self._scf.energy_nuc ()

    def get_vj_ext (self, mo_ext, dm1rs_ext, bmPu=None):
        output_shape = list (dm1rs_ext.shape[:-2]) + [self.mol.nao (), self.mol.nao ()]
        dm1 = dm1rs_ext.reshape (-1, mo_ext.shape[1], mo_ext.shape[1])
        if bmPu is not None:
            bPuu = np.tensordot (bmPu, mo_ext, axes=((0),(0)))
            rho = np.tensordot (dm1, bPuu, axes=((1,2),(1,2)))
            bPii = self._scf.with_df._cderi
            vj = lib.unpack_tril (np.tensordot (rho, bPii, axes=((-1),(0))))
        else: # Safety case: AO-basis SCF driver
            dm1 = np.dot (mo_ext.conj ().T, np.dot (dm1, mo_ext)).transpose (1,0,2)
            vj = self.mol._las.scf.get_j (dm1)
        return vj.reshape (*output_shape) 

    def get_vk_ext (self, mo_ext, dm1rs_ext, bmPu=None):
        output_shape = list (dm1rs_ext.shape[:-2]) + [self.mol.nao (), self.mol.nao ()]
        dm1 = dm1rs_ext.reshape (-1, mo_ext.shape[1], mo_ext.shape[1])
        imporb_coeff = self.mol.get_imporb_coeff ()
        if bmPu is not None:
            biPu = np.tensordot (imporb_coeff, bmPu, axes=((0),(0)))
            vuiP = np.tensordot (dm1, biPu, axes=((-1),(-1)))
            vk = np.tensordot (biPu, vuiP, axes=((-2,-1),(-1,-3)))
        else: # Safety case: AO-basis SCF driver
            dm1 = np.dot (mo_ext.conj ().T, np.dot (dm1, mo_ext)).transpose (1,0,2)
            vk = self.mol._las.scf.get_k (dm1).reshape (*output_shape)
        return vk.reshape (*output_shape)

    def get_hcore_rs (self):
        return self._scf.get_hcore_spinsep ()[None,:,:,:] + self._imporb_h1_stateshift

    def energy_nuc_r (self):
        return self._scf.energy_nuc () + self._imporb_h0_stateshift

    def get_h1eff (self, mo_coeff=None, ncas=None, ncore=None):
        ''' must needs change the dimension of h1eff '''
        assert (False)
        h1_avg_spinless, energy_core = self.h1e_for_cas (mo_coeff, ncas, ncore)[1]
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
        from pyscf.mcscf import mc1step
        with lib.temporary_env (mc1step, _fake_h_for_fast_casci=_fake_h_for_fast_casci):
            try:
                e_tot, e_cas, fcivec = super().casci (mo_coeff, ci0=ci0, eris=eris, verbose=verbose,
                                                      envs=envs)
            except AssertionError as e:
                print (type (ci0))
                raise (e)
        return e_tot, e_cas, fcivec

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
        dm1_rs = np.asarray ([[np.eye (nmo),]*2,]*nroots)
        dm1_rs[:,:,ncore:nocc,ncore:nocc] = casdm1rs
        dm1_rs[:,:,nocc:,:] = dm1_rs[:,:,:,nocc:] = 0
        dm1 = np.tensordot (weights, dm1_rs.sum (1), axes=1)

        # Return 1: the macrocycle gradient (odd matrix)
        g1 = np.tensordot (weights, lib.einsum ('rsik,rskj->rsij', h1_rs, dm1_rs).sum (1), axes=1)
        g1 -= h1 @ dm1
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
            dm1_rs[:,:,ncore:nocc,ncore:nocc] = casdm1rs
            h1_rs = lib.einsum ('ip,rsij,jq->rspq', u.conj(), h1_rs, u)
            h1 = u.conj ().T @ h1 @ u
            g1_u = np.tensordot (weights, lib.einsum ('rsik,rskj->rsij', h1_rs, dm1_rs).sum (1),
                                 axes=1)
            g1_u -= h1 @ dm1
            g_orb_u += self.pack_uniq_var (g1_u - g1_u.T)

        # Return 3: the diagonal elements of the Hessian (even matrix)
        # Return 4: the Hessian as a function (odd matrix)

        return g_orb, gorb_update, h_op, h_diag

if __name__=='__main__':
    from mrh.tests.lasscf.c2h6n4_struct import structure as struct
    mol = struct (1.0, 1.0, '6-31g', symmetry=False)
    mol.verbose = 5
    mol.output = 'lasscf_async_crunch.log'
    mol.build ()
    mf = scf.RHF (mol).density_fit ().run ()
    from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
    las = LASSCF (mf, (4,4), ((4,0),(0,4)), spin_sub=(5,5))
    mo = las.localize_init_guess ((list (range (3)), list (range (9,12))), mf.mo_coeff)
    las.state_average_(weights=[1,0,0,0,0],
                       spins=[[0,0],[2,0],[-2,0],[0,2],[0,-2]],
                       smults=[[1,1],[3,1],[3,1],[1,3],[1,3]])
    las.kernel (mo)

    ###########################
    from mrh.my_pyscf.mcscf.lasci import get_grad_orb
    from mrh.my_pyscf.mcscf.lasscf_async_split import LASImpurityOrbitalCallable
    dm1s = las.make_rdm1s ()
    veff = las.get_veff (dm1s=dm1s, spin_sep=True)
    fock1 = get_grad_orb (las, hermi=0)
    get_imporbs_0 = LASImpurityOrbitalCallable (las, 0, list (range (3)))
    fo_coeff, nelec_fo = get_imporbs_0 (las.mo_coeff, dm1s, veff, fock1)
    ###########################

    imol = ImpurityMole (las)
    imol._update_space_(fo_coeff, nelec_fo)
    imf = ImpurityHF (imol).density_fit ()
    imf._update_impham_1_(veff, dm1s, e_tot=las.e_tot)
    imc = df.density_fit (ImpurityCASSCF (imf, 4, 4))
    from pyscf.mcscf.addons import _state_average_mcscf_solver
    imc = _state_average_mcscf_solver (imc, las.fciboxes[0])
    imc._ifrag = 0
    imc._update_keyframe_(las.mo_coeff, las.ci)
    imc.kernel ()
    print (imc.converged, imc.e_tot, las.e_tot, imc.e_tot-las.e_tot)
