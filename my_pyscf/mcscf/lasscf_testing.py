import numpy as np
from scipy import linalg
from mrh.my_pyscf.mcscf import lasci
from pyscf.mcscf import mc_ao2mo

# An implementation that carries out vLASSCF, but without utilizing Schmidt decompositions
# or "fragment" subspaces, so that the orbital-optimization part scales no better than
# CASSCF. Eventually to be modified into a true all-PySCF implementation of vLASSCF

class LASSCF_UnitaryGroupGenerators (lasci.LASCI_UnitaryGroupGenerators):

    def _init_orb (self, las, mo_coeff, ci):
        lasci.LASCI_UnitaryGroupGenerators._init_orb (self, las, mo_coeff, ci)
        ncore, ncas = las.ncore, las.ncas
        nmo = mo_coeff.shape[-1]
        nocc = ncore + ncas
        self.uniq_orb_idx[ncore:nocc,:ncore] = True
        self.uniq_orb_idx[nocc:,ncore:nocc] = True
        if self.frozen is not None:
            idx[self.frozen,:] = idx[:,self.frozen] = False

class LASSCFSymm_UnitaryGroupGenerators (LASSCF_UnitaryGroupGenerators):
    _init_orb = lasci.LASCISymm_UnitaryGroupGenerators._init_orb
    # TODO: test that the "super()" command in the above points to
    # the correct parent class

class LASSCF_HessianOperator (lasci.LASCI_HessianOperator):
    # Required modifications for Hx: [I forgot about 3) at first]
    #   1) cache CASSCF-type eris and paaa - init_df
    #   2) increase range of ocm2 - make_odm1s2c_sub
    #   3) extend veff to active-unactive sector - split_veff
    #   4) dot the above three together - orbital_response
    #   5) TODO: get_veff using DF needs to be extended as well
    # Required modifications for API:
    #   6) broader ERI rotation - update_mo_ci_eri
    # Possible modifications:
    #   7) current prec may not be "good enough" - get_prec
    #   8) define "gx" in this context - get_gx 

    def _init_df (self):
        lasci.LASCI_HessianOperator._init_df (self)
        self.cas_type_eris = mc_ao2mo._ERIS (self.las, self.mo_coeff,
            method='incore', level=2) # level=2 -> ppaa, papa only
            # level=1 computes more stuff; it's only useful if I
            # want the honest hdiag in get_prec ()


    def make_odm1s2c_sub (self, kappa):
        # This is tricky, because in the parent I transposed ocm2 to make it 
        # symmetric upon index permutation, but if I do the same thing here then
        # I have an array of shape [nmo,]*4 (too big).
        # My options are 
        #   -1) Ignore it and just hope the optimization converges without this
        #       term (lol it won't)
        #   0) Just make an extremely limited implementation for small molecules
        #       only like the laziest motherfucker in existence
        #   1) Tag with kappa and mar the whole design of the class
        #   2) Extend one index and distinguish btwn cas-cas and other blocks
        #   3) Rewrite the whole thing to not transpose in the first place
        # Obviously 3) is the cleanest. But 2) lets me be more "pythonic" and
        #   I'll go with that for now 
        ncore, nocc = self.ncore, self.nocc
        odm1fs, ocm2_cas = lasci.LASCI_HessianOperator.make_odm1s2c_sub (self, kappa)
        kappa_ext = kappa[ncore:nocc,:].copy ()
        kappa_ext[:,ncore:nocc] = 0.0
        ocm2 = -np.dot (self.cascm2, kappa_ext) 
        ocm2[:,:,:,ncore:nocc] += ocm2_cas
        ocm2 = np.asfortranarray (ocm2) # Make largest index slowest-moving
        return odm1fs, ocm2

    def split_veff (self, veff_mo, dm1s_mo):
        veff_c = veff_mo.copy ()
        ncore = self.ncore
        nocc = self.nocc
        sdm = dm1s_mo[0] - dm1s_mo[1]
        sdm_ra = sdm[:,ncore:nocc]
        sdm_ar = sdm[ncore:nocc,:].copy ()
        sdm_ar[:,ncore:nocc] = 0.0
        veff_s = np.zeros_like (veff_c)
        vk_pa = veff_s[:,ncore:nocc]
        for p, v1 in enumerate (vk_pa):
            praa = self.cas_type_eris.ppaa[p]
            para = self.cas_type_eris.papa[p]
            paaa = praa[ncore:nocc]
            v1[:]  = np.tensordot (sdm_ra, praa, axes=2)
            v1[:] += np.tensordot (sdm_ar, para, axes=2)
        veff_s[:,:] *= -0.5
        vk_aa = vk_pa[ncore:nocc]
        veff_s[ncore:nocc,:] = vk_pa.T
        assert (np.allclose (veff_s, veff_s.T)), vk_aa-vk_aa.T
        veffa = veff_c + veff_s
        veffb = veff_c - veff_s
        return np.stack ([veffa, veffb], axis=0)

    def orbital_response (self, odm1fs, ocm2, tdm1frs, tcm2, veff_prime):
        ncore, nocc, nmo = self.ncore, self.nocc, self.nmo
        ocm2_cas = ocm2[:,:,:,ncore:nocc] 
        gorb = lasci.LASCI_HessianOperator.orbital_response (self, odm1fs,
            ocm2_cas, tdm1frs, tcm2, veff_prime)
        f1_prime = np.zeros ((self.nmo, self.nmo), dtype=self.dtype)
        for p, f1 in enumerate (f1_prime):
            praa = self.cas_type_eris.ppaa[p]
            para = self.cas_type_eris.papa[p]
            paaa = praa[ncore:nocc]
            # g_pabc d_qabc + g_prab d_qrab + g_parb d_qarb + g_pabr d_qabr (Formal)
            #        d_cbaq          d_abqr          d_aqbr          d_qabr (Symmetry of ocm2)
            # g_pcba d_abcq + g_prab d_abqr + g_parc d_aqcr + g_pbcr d_qbcr (Relabel)
            #                                                 g_pbrc        (Symmetry of eri)
            # g_pcba d_abcq + g_prab d_abqr + g_parc d_aqcr + g_pbrc d_qbcr (Final)
            for i, j in ((0, ncore), (nocc, nmo)): # Don't double-count
                ra, ar, cm = praa[i:j], para[:,i:j], ocm2[:,:,:,i:j]
                assert (linalg.norm (cm) == 0.0)
                f1[i:j] += np.tensordot (paaa, cm, axes=((0,1,2),(2,1,0))) # last index external
                f1[ncore:nocc] += np.tensordot (ra, cm, axes=((0,1,2),(3,0,1))) # third index external
                f1[ncore:nocc] += np.tensordot (ar, cm, axes=((0,1,2),(0,3,2))) # second index external
                f1[ncore:nocc] += np.tensordot (ar, cm, axes=((0,1,2),(1,3,2))) # first index external
        return gorb + (f1_prime - f1_prime.T)

    def update_mo_ci_eri (self, x, h2eff_sub):
        # mo and ci are fine, but h2eff sub simply has to be replaced
        mo1, ci1 = lasci.LASCI_HessianOperator.update_mo_ci_eri (self, x, h2eff_sub)[:2]
        return mo1, ci1, self.las.ao2mo (mo1)


class LASSCFNoSymm (lasci.LASCINoSymm):
    get_ugg = LASSCF_UnitaryGroupGenerators
    get_hop = LASSCF_HessianOperator
    def split_veff (self, veff, h2eff_sub, mo_coeff=None, ci=None, casdm1s_sub=None): 
        # This needs to actually do the veff, otherwise the preconditioner is broken
        # Eventually I can remove this, once I've implemented Schmidt decomposition etc. etc.
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if ci is None: ci = self.ci
        if casdm1s_sub is None: casdm1s_sub = self.make_casdm1s_sub (ci = ci)
        mo_cas = mo_coeff[:,self.ncore:][:,:self.ncas]
        dm1s_cas = linalg.block_diag (*[dm[0] - dm[1] for dm in casdm1s_sub])
        dm1s = mo_cas @ dm1s_cas @ mo_cas.conj ().T
        veff_c = veff.copy ()
        veff_s = -self._scf.get_k (self.mol, dm1s, hermi=1)/2
        veff_a = veff_c + veff_s
        veff_b = veff_c - veff_s
        veff = np.stack ([veff_a, veff_b], axis=0)
        dm1s = self.make_rdm1s (mo_coeff=mo_coeff, ci=ci)
        vj, vk = self._scf.get_jk (self.mol, dm1s, hermi=1)
        veff_a = vj[0] + vj[1] - vk[0]
        veff_b = vj[0] + vj[1] - vk[1]
        veff_test = np.stack ([veff_a, veff_b], axis=0)
        assert (np.allclose (veff, veff_test))
        return veff

        
class LASSCFSymm (lasci.LASCISymm):
    get_ugg = LASSCFSymm_UnitaryGroupGenerators    
    get_hop = LASSCF_HessianOperator
    split_veff = LASSCFNoSymm.split_veff

if __name__ == '__main__':
    from pyscf import scf, lib
    from mrh.tests.lasscf.c2h4n4_struct import structure as struct
    mo0 = np.loadtxt ('/home/herme068/gits/mrh/tests/lasscf/test_lasci_mo.dat')
    ci00 = np.loadtxt ('/home/herme068/gits/mrh/tests/lasscf/test_lasci_ci0.dat')
    ci01 = np.loadtxt ('/home/herme068/gits/mrh/tests/lasscf/test_lasci_ci1.dat')
    ci0 = None #[[ci00], [-ci01.T]]
    dr_nn = 2.0
    mol = struct (dr_nn, dr_nn, '6-31g', symmetry=False)
    mol.verbose = lib.logger.DEBUG
    mol.output = 'lasscf_testing.log'
    mol.spin = 8 
    mol.build ()
    mf = scf.RHF (mol).run ()
    mc = LASSCFNoSymm (mf, (4,4), ((4,0),(4,0)), spin_sub=(5,5))
    mc.kernel (mo0, ci0)


