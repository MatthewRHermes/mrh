from pyscf import gto, scf, mcscf, ao2mo
from pyscf.fci.direct_spin1 import _unpack_nelec

class ImpurityMole (gto.Mole):
    def __init__(self, las, nelec_imp):
        self._las = las
        self.verbose = las.verbose
        self.max_memory = las.max_memory
        self._update_nelec_imp_(nelec_imp)
        self.atom.append (('H', (0, 0, 0)))
        if stdout is None and output is None:
            self.stdout = las.stdout
        elif stdout is not None
            self.stdout = stdout
        elif output is not None:
            self.output = output
        self.build ()

    def _update_space (self, imporb_coeff, nelec_imp):
        self._imporb_coeff = imporb_coeff
        nelec_imp = _unpack_nelec (nelec_imp)
        self.nelectron = sum (nelec_imp)
        self.spin = nelec_imp[0] - nelec_imp[1]

    def get_imporb_coeff (self): return self._imporb_coeff
    def nao_nr (self): return self._imporb_coeff.shape[-1]
    def nao (self): return self._imporb_coeff.shape[-1]

class ImpuritySCF (scf.hf.SCF):
    def _update_heff_(self, veff, dm1s, de=0):
        imporb_coeff = self.mol.get_imporb_coeff ()
        nimp = self.mol.nao ()
        mf = self.mol._las._scf
        # Two-electron integrals
        if hasattr (mf, '_eri'):
            self._eri = ao2mo.full (mf._eri, imporb_coeff, 4)
        if hasattr (mf, 'with_df'):
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
        smo = mf.get_ovlp () @ imporb_coeff
        dm1s = np.dot (smo.conj ().T, np.dot (dm1s, smo)).transpose (1,0,2)
        vj, vk = self.get_jk (dm1s)
        veff1 = vj.sum (0)[None,:,:] - vk
        h1s -= veff1
        self._imporb_h1 = h1s.sum (0) / 2
        self._imporb_h1_sz = (h1s[0] - h1s[1]) / 2
        # Constant
        de -= np.dot ((h1s + (veff1*.5)).ravel (), dm1s.ravel ())
        self._imporb_h0 = mf.mol.energy_nuc () + de 

    def get_hcore (self):
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
        return super().get_fock (h1e=h1e, s1e=s1e, vhf=vhf, dm=dm, cycle=cycle, diis=diis,
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

# This is the really tricky part
def ImpurityCASSCF (mcscf.mc1step.CASSCF):

    def _update_keyframe (self, mo_coeff, ci):
        # Project mo_coeff and ci keyframe into impurity space and cache
        las = self.mol._las
        mf = las._scf
        ifrag = self._ifrag
        imporb_coeff = self.mol.get_imporb_coeff ()
        self.ci = ci[_ifrag]
        # Inactive orbitals
        mo_core = mo_coeff[:,:las.ncore]
        s0 = mf.get_ovlp ()
        ovlp = imporb_coeff.conj ().T @ s0 @ mo_core
        self.mo_coeff, svals, vh = linalg.svd (ovlp)
        assert (self.mo_coeff.shape == imporb_coeff.shape)
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
        # Canonicalize core and virtual spaces
        casdm1s, casdm2s = self.fcisolver.make_rdm12s (ci, self.ncas, self.nelecas)
        mo_core = self.mo_coeff[:,:self.ncore]
        mo_cas = self.mo_coeff[:,self.ncore:nocc]
        dm1s = np.dot (mo_cas, np.dot (casdm1s, mo_cas.conj ().T)).transpose (1,0,2)
        dm1s += (mo_core @ mo_core.conj ().T)[None,:,:]
        fock = self._scf.get_fock (dm=dm1s)
        mo_core = self.mo_coeff[:,:self.ncore]
        fock_core = mo_core.conj ().T @ fock @ mo_core
        w, c = linalg.eigh (fock_core)
        self.mo_coeff[:,:self.ncore] = mo_core @ c
        mo_virt = self.mo_coeff[:,nocc:]
        fock_virt = mo_virt.conj ().T @ fock @ mo_virt
        w, c = linalg.eigh (fock_virt)
        self.mo_coeff[:,nocc:] = mo_virt @ c
        # Two-body correction to _scf._imporb_h0 is better computed here
        eri_cas = self.get_h2eff (mo_cas)
        casdm1 = casdm1s[0] + casdm1s[1]
        casdm2 = casdm2s[0] + casdm2s[1] + casdm2s[1].transpose (2,3,0,1) + casdm2s[2]
        casdm2 -= np.multiply.outer (casdm1, casdm1)
        casdm2 += np.multiply.outer (casdm1s[0], casdm1s[0]).transpose (0,3,2,1)
        casdm2 += np.multiply.outer (casdm1s[1], casdm1s[1]).transpose (0,3,2,1)
        de = np.dot (eri_cas.ravel (), casdm2.ravel ()) / 2
        self._scf._imporb_h0 -= de

    def _update_hcore_cishift (self, mo_coeff, ci, casdm1rs=None, h2eff_sub=None):
        las = self.mol._las
        if casdm1rs is None: casdm1rs = las.states_make_casdm1s (ci=ci)
        casdm1s = np.tensordot (las.weights, casdm1rs, axes=1)
        dm1rs = casdm1rs - casdm1s
        i = sum (las.ncas_sub[:self._ifrag])
        j = i + las.ncas_sub[self._ifrag]
        dm1rs = np.stack (dm1rs[:,:,:i,:], dm1rs[:,:,j:,:], axis=2)
        dm1rs = np.stack (dm1rs[:,:,:,:i], dm1rs[:,:,:,j:], axis=3)
        mo_cas = mo_coeff[:,las.ncore:][:,:las.ncas]
        mo_olas = np.stack (mo_cas[:,:i], mo_cas[:,j:], axis=1)
        bPmu = getattr (h2eff_sub, 'bPmu')
        if bPmu is not None:
            bPmu = np.stack (bPmu[...,:i], bPmu[...,j:], axis=-1)
            vj_r = self.get_vj_ext (mo_olas, dm1rs, bPmu=bPmu).sum (1)
            vk_rs = self.get_vk_ext (mo_olas, dm1rs, bPmu=bPmu)
            vext = vj_r[:,None,:,:] - vk_rs
        else:
            raise NotImplementedError ("Non-DF version")
        self._imporb_h1_cishift = vext

    def get_vj_ext (self, mo_ext, dm1rs_ext, bPmu=None):
        if bPmu is not None:
            bPuu = np.tensordot (bPmu, mo_ext, axes=((1),(0)))
            return np.tensordot (bPuu, dm1rs_ext, axes=((1,2),(-2,-1)))
        else: # Safety case: AO-basis SCF driver
            output_shape = list (dm1rs.shape[:-2]) + [self.mol.nao (), self.mol.nao ()]
            dm1 = dm1rs.reshape (-1, mo_ext.shape[1], mo_ext.shape[1])
            dm1 = np.dot (mo_ext.conj ().T, np.dot (dm1, mo_ext)).transpose (1,0,2)
            return self.mol._las.scf.get_j (dm1).reshape (*output_shape)

    def get_vk_ext (self, mo_ext, dm1rs_ext, bPmu=None):
        imporb_coeff = self.mol.get_imporb_coeff ()
        if bPmu is not None:
            bPiu = np.tensordot (bPmu, imporb_coeff, axes=((1),(0)))
            vuPi = np.tensordot (dm1rs_ext, bPiu, axes=((-1),(-1)))
            return np.tensordot (bPiu, vuPi, axes=((0,2),(-2,-3)))
        else: # Safety case: AO-basis SCF driver
            output_shape = list (dm1rs.shape[:-2]) + [self.mol.nao (), self.mol.nao ()]
            dm1 = dm1rs.reshape (-1, mo_ext.shape[1], mo_ext.shape[1])
            dm1 = np.dot (mo_ext.conj ().T, np.dot (dm1, mo_ext)).transpose (1,0,2)
            return self.mol._las.scf.get_k (dm1).reshape (*output_shape)
            
    def get_hcore_cishift (self):
        return self._imporb_h1_cishift

    def get_hcore_ci (self):
        return self._scf.get_hcore_spinsep ()[None,:,:,:] + self.get_hcore_cishift ()

    #def casci (self, mo_coeff, ci0=None, eris=None, verbose=None, envs=None):
    #    mo_cas = mo_coeff[:,mc.ncore:][:,:mc.ncas]
    #    h1e_s_amo[:,:] = amoH @ h1e_s @ amo
    #    return super().casci (mo_coeff, ci0=ci0, eris=eris, verbose=verbose, envs=envs)






