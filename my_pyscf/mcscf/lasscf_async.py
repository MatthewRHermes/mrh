import numpy as np
from scipy import linalg
from pyscf import lib

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
    log = lib.logger.new_logger(las, verbose)
    t0 = (lib.logger.process_clock(), lib.logger.perf_counter())

    h2eff_sub = las.get_h2eff (mo_coeff)
    t1 = log.timer ('LASSCF initial AO2MO', *t0)
    veff = las.get_veff (dm1s = las.make_rdm1 (mo_coeff=mo_coeff, ci=ci0))
    casdm1s_sub = las.make_casdm1s_sub (ci=ci0)
    casdm1frs = las.states_make_casdm1s_sub (ci=ci0)
    veff = las.split_veff (veff, h2eff_sub, mo_coeff=mo_coeff, ci=ci0, casdm1s_sub=casdm1s_sub)
    t1 = log.timer ('LASSCF initial get_veff', *t1)

    converged = False
    it = 0
    nfrags = len (las.ncas_sub)
    for it in range (las.max_cycle_macro):
        pass
        # 1. Divide into fragments
        # 2. CASSCF on each fragment
        # 3. Combine from fragments

    t1 = log.timer ('LASSCF {} macrocycles'.format (it), *t1)
    e_tot = las.energy_nuc () + las.energy_elec (mo_coeff=mo_coeff, ci=ci1, h2eff=h2eff_sub,
                                                 veff=veff)
    veff_a = np.stack ([las.fast_veffa ([d[state] for d in casdm1frs], h2eff_sub,
                                        mo_coeff=mo_coeff, ci=ci1, _full=True)
                        for state in range (las.nroots)], axis=0)
    veff_c = (veff.sum (0) - np.einsum ('rsij,r->ij', veff_a, las.weights))/2
    veff = veff_c[None,None,:,:] + veff_a
    veff = lib.tag_array (veff, c=veff_c, sa=np.einsum ('rsij,r->sij', veff, las.weights))
    e_states = las.energy_nuc () + np.array (las.states_energy_elec (mo_coeff=mo_coeff, ci=ci1,
                                                                     h2eff=h2eff_sub, veff=veff))
    log.info ('LASSCF %s after %d cycles', ('not converged', 'converged')[converged], it+1)
    log.info ('LASSCF E = %.15g ; |g_int| = %.15g ; |g_ci| = %.15g ; |g_ext| = %.15g', e_tot,
              norm_gorb, norm_gci, norm_gx)
    t1 = log.timer ('LASSCF final energy', *t1)
    mo_coeff, mo_energy, mo_occ, ci1, h2eff_sub = las.canonicalize (mo_coeff, ci1, veff=veff.sa,
                                                                    h2eff_sub=h2eff_sub)
    t1 = log.timer ('LASSCF canonicalization', *t1)
    t0 = log.timer ('LASSCF kernel function', *t0)

    return converged, e_tot, e_states, mo_energy, mo_coeff, e_cas, ci1, h2eff_sub, veff

