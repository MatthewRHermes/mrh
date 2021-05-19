# RDM-based variant of LASSCF in which internal electronic structure of each
# localized active subspace is decoupled from orbital rotation and the kernel
# for obtaining RDMs given LAS Hamiltonians can be subclassed arbitrarily

import numpy as np
from scipy import linalg, sparse
from mrh.my_pyscf.mcscf import lasscf_o0

class LASSCF_UnitaryGroupGenerators (lasscf_o0.LASSCF_UnitaryGroupGenerators):
    ''' spoof away CI degrees of freedom '''
    def __init__(self, las, mo_coeff, *args):
        lasscf_o0.LASSCF_UnitaryGroupGenerators.__init__(
            self, las, mo_coeff, None)
    def _init_ci (self, las, mo_coeff, ci):
        pass
    def pack (self, kappa):
        return kappa[self.uniq_orb_idx]
    def unpack (self, x):
        kappa = np.zeros ((self.nmo, self.nmo), dtype=x.dtype)
        kappa[self.uniq_orb_idx] = x[:self.nvar_orb]
        kappa = kappa - kappa.T
        return kappa
    @property
    def ncsf_sub (self): return np.array ([0])

class LASSCFSymm_UnitaryGroupGenerators (LASSCF_UnitaryGroupGenerators):
    def __init__(self, las, mo_coeff, *args):
        lasscf_o0.LASSCFSymm_UnitaryGroupGenerators.__init__(
            self, las, mo_coeff, None)
    _init_orb = lasscf_o0.LASSCFSymm_UnitaryGroupGenerators._init_orb

class LASSCF_HessianOperator (lasscf_o0.LASSCF_HessianOperator):
    def __init__(self, las, ugg, mo_coeff=None, casdm1frs=None, casdm2fr=None,
            ncore=None, ncas_sub=None, nelecas_sub=None, h2eff_sub=None, veff=None,
            do_init_eri=True):
        if mo_coeff is None: mo_coeff = las.mo_coeff
        if casdm1frs is None: casdm1frs = las.casdm1frs
        if casdm2fr is None: casdm2fr = las.casdm2fr
        if ncore is None: ncore = las.ncore
        if ncas_sub is None: ncas_sub = las.ncas_sub
        if nelecas_sub is None: nelecas_sub = las.nelecas_sub
        if h2eff_sub is None: h2eff_sub = las.get_h2eff (mo_coeff)
        self.las = las
        self.ah_level_shift = las.ah_level_shift
        self.ugg = ugg
        self.mo_coeff = mo_coeff
        self.ncore = ncore
        self.ncas_sub = ncas_sub
        self.nelecas_sub = nelecas_sub
        self.ncas = ncas = sum (ncas_sub)
        self.nao = nao = mo_coeff.shape[0]
        self.nmo = nmo = mo_coeff.shape[-1]
        self.nocc = nocc = ncore + ncas
        self.nroots = nroots = las.nroots
        self.weights = las.weights
        self.bPpj = None
        # Spoof away CI: fixed zeros
        self._tdm1frs = np.zeros ((len (ncas_sub), nroots, 2, ncas, ncas))
        self._tcm2 = np.zeros ([ncas,]*4)

        self._init_dms_(casdm1frs, casdm2fr)
        self._init_ham_(h2eff_sub, veff)
        self._init_orb_()
        if do_init_eri: self._init_eri_()

    def _matvec (self, x):
        kappa1 = self.ugg.unpack (x)

        # Effective density matrices, veffs, and overlaps from linear response
        odm1rs, ocm2 = self.make_odm1s2c_sub (kappa1)
        veff_prime = self.get_veff_prime (odm1rs)

        # Responses!
        kappa2 = self.orbital_response (kappa1, odm1rs, ocm2, veff_prime)

        # LEVEL SHIFT!!
        kappa3 = self.ugg.unpack (self.ah_level_shift * np.abs (x))
        kappa2 += kappa3
        return self.ugg.pack (kappa2)

    def get_veff_prime (self, odm1rs):
        # Spoof away CI by wrapping call
        fn = lasscf_o0.LASSCF_HessianOperator.get_veff_Heff
        return fn (self, odm1rs, self._tdm1frs)[0]

    def orbital_response (self, kappa1, odm1rs, ocm2, veff_prime):
        # Spoof away CI by wrapping call
        fn = lasscf_o0.LASSCF_HessianOperator.orbital_response
        t1, t2 = self._tdm1frs, self._tcm2
        return fn (self, kappa1, odm1rs, ocm2, t1, t2, veff_prime)

    def update_mo_eri (self, x, h2eff_sub):
        kappa = self.ugg.unpack (x)
        umat = linalg.expm (kappa/2)
        mo1 = self._update_mo (umat)
        h2eff_sub = self._update_h2eff_sub (mo1, umat, h2eff_sub)
        return mo1, h2eff_sub

    def get_grad (self): return self.ugg.pack (self.fock1 - self.fock1.T)

    def get_prec (self):
        Hdiag = self._get_Horb_diag () + self.ah_level_shift
        Hdiag[np.abs (Hdiag)<1e-8] = 1e-8
        return sparse.linalg.LinearOperator (self.shape, matvec=(lambda x:x/Hdiag), dtype=self.dtype)

if __name__ == '__main__':
    from pyscf import gto, scf, lib
    from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
    xyz = '''H 0.0 0.0 0.0
             H 1.0 0.0 0.0
             H 0.2 3.9 0.1
             H 1.159166 4.1 -0.1'''
    mol = gto.M (atom = xyz, basis = '6-31g', output='lasscf_rdm.log',
        verbose=lib.logger.INFO)
    mf = scf.RHF (mol).run ()
    las = LASSCF (mf, (2,2), (2,2), spin_sub=(1,1))
    frag_atom_list = ((0,1),(2,3))
    mo_loc = las.localize_init_guess (frag_atom_list, mf.mo_coeff)
    las.max_cycle_macro = 3
    las.kernel (mo_loc)

    mo = las.mo_coeff
    casdm1frs = las.states_make_casdm1s_sub ()
    casdm2fr = las.states_make_casdm2_sub ()    

    ugg_test = LASSCF_UnitaryGroupGenerators (las, mo)
    hop_test = LASSCF_HessianOperator (las, ugg_test, casdm1frs=casdm1frs,
        casdm2fr=casdm2fr)

    ugg_ref = las.get_ugg ()
    hop_ref = las.get_hop (ugg=ugg_ref)
     
    g_test = hop_test.get_grad ()
    g_ref = hop_ref.get_grad ()[:g_test.size]
    print ('gradient test:', linalg.norm (g_test-g_ref), linalg.norm (g_ref))

    x_test = np.random.rand (ugg_test.nvar_tot)
    x_ref = np.zeros (ugg_ref.nvar_tot)
    x_ref[:ugg_ref.nvar_orb] = x_test[:]

    prec_test = hop_test.get_prec ()(x_test)
    prec_ref = hop_ref.get_prec ()(x_ref)[:prec_test.size]
    print ('preconditioner test:', linalg.norm (prec_test-prec_ref),
        linalg.norm (prec_ref))

    hx_test = hop_test._matvec (x_test)
    hx_ref = hop_ref._matvec (x_ref)[:hx_test.size]
    print ('hessian test:', linalg.norm (hx_test-hx_ref), linalg.norm (hx_ref))


