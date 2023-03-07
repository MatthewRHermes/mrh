from pyscf import gto, scf, mcscf, ao2mo
from pyscf.fci.direct_spin1 import _unpack_nelec

class ImpurityMole (gto.Mole):
    def __init__(self, las, nelec_imp):
        self.las = las
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

    def _update_nelec_imp_(self, nelec_imp):
        nelec_imp = _unpack_nelec (nelec_imp)
        self.nelectron = sum (nelec_imp)
        self.spin = nelec_imp[0] - nelec_imp[1]
        

class ImpuritySCF (scf.hf.SCF):
    def _update_space_(self, imporb_coeff, nelec_imp, veff, dm1s, de=0):
        self.mol._update_nelec_imp (nelec_imp)
        nimp = self._nimp = imporb_coeff.shape[1]
        mf = self.mol.las._scf
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

    def get_ovlp (self):
        return np.eye (self._nimp)

    def energy_nuc (self):
        return self._imporb_h0

    def get_fock (self, h1e=None, s1e=None, vhf=None, dm=None, cycle=1, diis=None,
        diis_start_cycle=None, level_shift_factor=None, damp_factor=None):

        if vhf is None: vhf = self.get_veff (self.mol, dm)
        vhf[0] += self._imporb_h1_sz
        vhf[1] -= self._imporb_h1_sz
        return super().get_fock (h1e=h1e, s1e=s1e, vhf=vhf, dm=dm, cycle=cycle, diis=diis,
            diis_start_cycle=diis_start_cycle, level_shift_factor=level_shift_factor,
            damp_factor=damp_factor)

    def energy_elec (self, dm=None, h1e=None, vhf=None):
        if dm is None: dm = self.make_rdm1 ()
        e_elec, e_coul = super().energy_elec (dm=dm, h1e=h1e, vhf=vhf)
        e_elec += (self._imporb_h1_sz * (dm[0] - dm[1])).sum ()
        return e_elec, e_coul

class ImpurityROHF (scf.rohf.ROHF, ImpuritySCF):
    get_hcore = ImpuritySCF.get_hcore
    get_ovlp = ImpuritySCF.get_hcore
    get_fock = ImpuritySCF.get_hcore
    energy_nuc = ImpuritySCF.get_hcore
    energy_elec = ImpuritySCF.get_hcore

class ImpurityRHF (scf.hf.RHF, ImpuritySCF):
    get_hcore = ImpuritySCF.get_hcore
    get_ovlp = ImpuritySCF.get_hcore
    get_fock = ImpuritySCF.get_hcore
    energy_nuc = ImpuritySCF.get_hcore
    energy_elec = ImpuritySCF.get_hcore

def ImpurityHF (mol):
    if mol.spin == 0: return ImpurityRHF (mol)
    else: return ImpurityROHF (mol)

# This is the really tricky part
def ImpurityCASSCF (mcscf.mc1step.CASSCF):
    def _update_space_(self, mo_coeff, ci, ifrag, imporb_coeff, nelec_imp, veff, dm1s, de=0):
        self._scf._update_space_(imporb_coeff, nelec_imp, veff, dm1s, de=de)
        # Two-body correction to _scf._imporb_h0 is better computed here
        eri_cas = self.get_h2eff (np.eye (self.ncas))
        casdm1s, casdm2s = self.fcisolver.make_rdm12s (ci, self.ncas, self.nelecas)
        casdm1 = casdm1s[0] + casdm1s[1]
        casdm2 = casdm2s[0] + casdm2s[1] + casdm2s[1].transpose (2,3,0,1) + casdm2s[2]
        casdm2 -= np.multiply.outer (casdm1, casdm1)
        casdm2 += np.multiply.outer (casdm1s[0], casdm1s[0]).transpose (0,3,2,1)
        casdm2 += np.multiply.outer (casdm1s[1], casdm1s[1]).transpose (0,3,2,1)
        de = np.dot (eri_cas.ravel (), casdm2.ravel ()) / 2
        self._scf._imporb_h0 -= de

    #def casci (self, mo_coeff, ci0=None, eris=None, verbose=None, envs=None):
    #    mo_cas = mo_coeff[:,mc.ncore:][:,:mc.ncas]
    #    h1e_s_amo[:,:] = amoH @ h1e_s @ amo
    #    return super().casci (mo_coeff, ci0=ci0, eris=eris, verbose=verbose, envs=envs)






