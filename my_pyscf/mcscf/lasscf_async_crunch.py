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
    def _update_heff_(self, veff, dm1s, e_tot=None):
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

    def _subtract_self_energy_(self, mo_docc, mo_dm, dm1s, dm2, eri_dm=None):
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
    energy_core = casscf.energy_nuc()
    hcore = casscf.get_hcore ()
    energy_core += np.einsum('ij,ji', core_dm, hcore)
    energy_core += eris.vhf_c[:ncore,:ncore].trace ()
    h1 = casscf.get_hcore_ci ()
    h1eff = np.tensordot (mo_cas.conj (), np.dot (h1, mo_cas), axes=((0),(2))).transpose (1,2,0,3)
    h1eff += eris.vhf_c[None,None,ncore:nocc,ncore:nocc]
    
    mc.get_h1eff = lambda *args: (h1eff, energy_core)

    eri_cas = eris.ppaa[ncore:nocc,ncore:nocc,:,:].copy()
    mc.get_h2eff = lambda *args: eri_cas
    return mc


# This is the really tricky part
class ImpurityCASSCF (mcscf.mc1step.CASSCF):

    def _update_keyframe_(self, mo_coeff, ci):
        # Project mo_coeff and ci keyframe into impurity space and cache
        las = self.mol._las
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
        # Subtract self-energy
        casdm1s, casdm2s = self.fcisolver.make_rdm12s (self.ci, self.ncas, self.nelecas)
        casdm2 = casdm2s[0] + casdm2s[1] + casdm2s[1].transpose (2,3,0,1) + casdm2s[2]
        eri_cas = ao2mo.restore (1, self.get_h2eff (self.mo_coeff), self.ncas)
        mo_core = self.mo_coeff[:,:self.ncore]
        mo_cas = self.mo_coeff[:,self.ncore:nocc]
        self._scf._subtract_self_energy_(mo_core, mo_cas, casdm1s, casdm2, eri_cas)
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

    def _update_hcore_cishift_(self, mo_coeff, ci, casdm1rs=None, h2eff_sub=None):
        las = self.mol._las
        if casdm1rs is None: casdm1rs = las.states_make_casdm1s (ci=ci)
        if h2eff_sub is None: h2eff_sub = las.ao2mo (mo_coeff)
        casdm1s = np.tensordot (las.weights, casdm1rs, axes=1)
        dm1rs = casdm1rs - casdm1s
        i = sum (las.ncas_sub[:self._ifrag])
        j = i + las.ncas_sub[self._ifrag]
        dm1rs[:,:,i:j,:] = dm1rs[:,:,:,i:j] = 0.0
        mo_cas = mo_coeff[:,las.ncore:][:,:las.ncas]
        bmPu = getattr (h2eff_sub, 'bmPu', None)
        vj_r = self.get_vj_ext (mo_cas, dm1rs.sum(1), bmPu=bmPu)
        vk_rs = self.get_vk_ext (mo_cas, dm1rs, bmPu=bmPu)
        vext = vj_r[:,None,:,:] - vk_rs
        self._imporb_h1_cishift = vext

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

    def get_hcore_cishift (self):
        return self._imporb_h1_cishift

    def get_hcore_ci (self):
        return self._scf.get_hcore_spinsep ()[None,:,:,:] + self.get_hcore_cishift ()

    def get_h1eff (self, mo_coeff=None, ncas=None, ncore=None):
        ''' must needs change the dimension of h1eff '''
        assert (False)
        h1_avg_spinless, energy_core = self.h1e_for_cas (mo_coeff, ncas, ncore)[1]
        mo_cas = mo_coeff[:,ncore:][:,:ncas]
        h1_avg_sz = mo_cas.conj ().T @ self._scf.get_hcore_sz () @ mo_cas
        h1_avg = np.stack ([h1_avg_spinless + h1_avg_sz, h1_avg_spinless - h1_avg_sz], axis=0)
        h1 += mo_cas.conj ().T @ self.get_hcore_cishift () @ mo_cas
        return h1, energy_core

    def update_casdm (self, mo, u, fcivec, e_cas, eris, envs={}):
        ''' inject the cishift h1 into envs '''
        mou = mo @ u[:,self.ncore:][:,:self.ncas]
        h1_cishift = self.get_hcore_ci () - self.get_hcore ()[None,None,:,:]
        h1_cishift = np.tensordot (mou.conj ().T, np.dot (h1_cishift, mou),
                                   axes=((1),(2))).transpose (1,2,0,3)
        envs['h1_cishift'] = h1_cishift
        return super().update_casdm (mo, u, fcivec, e_cas, eris, envs=envs)

    def solve_approx_ci (self, h1, h2, ci0, ecore, e_cas, envs):
        ''' get the cishifted h1 from envs '''
        h1 = h1[None,None,:,:] = envs['h1_cishift']
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


if __name__=='__main__':
    from mrh.tests.lasscf.c2h6n4_struct import structure as struct
    mol = struct (1.0, 1.0, '6-31g', symmetry=False)
    mol.verbose = 5
    mol.output = 'lasscf_async_crunch.log'
    mol.build ()
    mf = scf.RHF (mol).density_fit ().run ()
    from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
    las = LASSCF (mf, (4,4), (4,4), spin_sub=(1,1))
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
    imf._update_heff_(veff, dm1s, e_tot=las.e_tot)
    imc = df.density_fit (ImpurityCASSCF (imf, 4, 4))
    imc._ifrag = 0
    imc.fcisolver = las.fciboxes[0]
    imc._update_keyframe_(las.mo_coeff, las.ci)
    imc._update_hcore_cishift_(las.mo_coeff, las.ci)
    imc.kernel ()
    print (imc.converged, las.e_tot, imc.e_tot, imc.e_tot-las.e_tot)
