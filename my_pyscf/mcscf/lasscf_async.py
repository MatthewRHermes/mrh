import numpy as np
from scipy import linalg
from pyscf import lib
from pyscf.mcscf import mc1step
from mrh.my_pyscf.mcscf import lasci
from mrh.my_pyscf.mcscf.lasscf_async_crunch import get_impurity_casscf
from mrh.my_pyscf.mcscf.lasscf_async_keyframe import LASKeyframe
from mrh.my_pyscf.mcscf.lasscf_async_combine import combine_o0

def kernel (las, mo_coeff=None, ci0=None, conv_tol_grad=1e-4,
            assert_no_dupes=False, imporb_builders=None, verbose=lib.logger.NOTE):
    '''
    Kwargs:
        imporb_builders : callable of length nfrags
            The functions which produce localized impurity orbitals surrounding each
            active subspace. In a given keyframe, the impurity subspaces should contain
            some inactive and virtual orbitals but should be unentangled (i.e., contain
            an integer number of electrons). The calling pattern is
            imporb_coeff_i, nelec_imp_i = imporb_builders[i] (mo_coeff, dm1s, veff, fock1)
            Args:
                mo_coeff : ndarray of shape (nao,nmo)
                dm1s : ndarray of shape (2,nao,nao)
                veff : ndarray of shape (2,nao,nao)
                fock1 : ndarray of shape (nmo,nmo)
            Returns:
                imporb_coeff_i : ndarray of shape (nao,*)
                nelec_imp_i : tuple of length 2

    '''
    if mo_coeff is None: mo_coeff = las.mo_coeff
    if assert_no_dupes: las.assert_no_duplicates ()
    if (ci0 is None or any ([c is None for c in ci0]) or
      any ([any ([c2 is None for c2 in c1]) for c1 in ci0])):
        ci0 = las.get_init_guess_ci (mo_coeff, h2eff_sub, ci0)
    if (ci0 is None or any ([c is None for c in ci0]) or
      any ([any ([c2 is None for c2 in c1]) for c1 in ci0])):
        raise RuntimeError ("failed to populate get_init_guess")
    nfrags = len (las.ncas_sub)
    log = lib.logger.new_logger(las, verbose)
    t0 = (lib.logger.process_clock(), lib.logger.perf_counter())
    kf0 = las.get_keyframe (mo_coeff, ci0) 

    ###############################################################################################
    ################################## Begin actual kernel logic ##################################
    ###############################################################################################





    converged = False
    it = 0
    kf1 = kf0
    impurities = [get_impurity_casscf (las, i) for i in range (nfrags)]
    for it in range (las.max_cycle_macro):
        # 1. Divide into fragments
        for impurity, imporb_builder in zip (impurities, imporb_builders):
            impurity._pull_keyframe_(kf1)

        # 2. CASSCF on each fragment
        kf2_list = []
        for impurity in impurities:
            impurity.kernel ()
            kf2_list.append (impurity._push_keyframe (kf1))

        # 3. Combine from fragments
        kf1 = combine_o0 (las, kf2_list)

        # Break if converged




    ###############################################################################################
    ################################### End actual kernel logic ###################################
    ###############################################################################################

    t1 = log.timer ('LASSCF {} macrocycles'.format (it), *t1)
    e_tot = las.energy_nuc () + las.energy_elec (mo_coeff=mo_coeff, ci=ci1, h2eff=h2eff_sub,
                                                 veff=veff)
    # TODO: I'm guessing this is the only place anywhere that I insist on having an array like
    # (nroots,2,nao,nao). That's bad and should be designed around, since nroots can get stupidly
    # large and the matrices objectively cannot have more than ncas linearly-independent d.o.f.
    # distinguishing them.
    veff_a = np.stack ([las.fast_veffa ([d[state] for d in casdm1frs], h2eff_sub,
                                        mo_coeff=mo_coeff, ci=ci1, _full=True)
                        for state in range (las.nroots)], axis=0)
    veff_c = (veff.sum (0) - np.einsum ('rsij,r->ij', veff_a, las.weights))/2
    veff = veff_c[None,None,:,:] + veff_a
    veff = lib.tag_array (veff, c=veff_c, sa=np.einsum ('rsij,r->sij', veff, las.weights))
    e_states = las.energy_nuc () + np.array (las.states_energy_elec (mo_coeff=mo_coeff, ci=ci1,
                                                                     h2eff=h2eff_sub, veff=veff))
    # This crap usually goes in a "_finalize" function
    log.info ('LASSCF %s after %d cycles', ('not converged', 'converged')[converged], it+1)
    log.info ('LASSCF E = %.15g ; |g_int| = %.15g ; |g_ci| = %.15g ; |g_ext| = %.15g', e_tot,
              norm_gorb, norm_gci, norm_gx)
    t1 = log.timer ('LASSCF final energy', *t1)
    mo_coeff, mo_energy, mo_occ, ci1, h2eff_sub = las.canonicalize (mo_coeff, ci1, veff=veff.sa,
                                                                    h2eff_sub=h2eff_sub)
    t1 = log.timer ('LASSCF canonicalization', *t1)
    t0 = log.timer ('LASSCF kernel function', *t0)

    e_cas = None # TODO: get rid of this worthless, meaningless variable
    return converged, e_tot, e_states, mo_energy, mo_coeff, e_cas, ci1, h2eff_sub, veff

# TODO: Refactor localize_init_guess and _svd to be less silly
class LASSCFNoSymm (lasci.LASCINoSymm):
    get_grad_orb = lasci.get_grad_orb
    def get_keyframe (self, mo_coeff=None, ci=None):
        if mo_coeff is None: mo_coeff=self.mo_coeff
        if ci is None: ci=self.ci
        return LASKeyframe (self, mo_coeff, ci)
    as_scanner = mc1step.as_scanner

class LASSCFSymm (lasci.LASCISymm):
    get_grad_orb = lasci.get_grad_orb
    get_keyframe = LASSCFNoSymm.get_keyframe
    as_scanner = mc1step.as_scanner




