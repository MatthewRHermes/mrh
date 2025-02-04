# This is the original lasscf_async kernel, used prior to July 2024, which synchronously optimized
# the active-orbital--active-orbital rotation degrees of freedom and required all impurity problems
# to finish before combining them.

import itertools
import numpy as np
from scipy import linalg
from pyscf import lib
from mrh.my_pyscf.mcscf.lasscf_async import keyframe, combine
from mrh.my_pyscf.mcscf.lasscf_async.split import get_impurity_space_constructor
from mrh.my_pyscf.mcscf.lasscf_async.crunch import get_impurity_casscf

def kernel (las, mo_coeff=None, ci0=None, conv_tol_grad=1e-4,
            assert_no_dupes=False, verbose=lib.logger.NOTE, frags_orbs=None,
            **kwargs):
    log = lib.logger.new_logger(las, verbose)
    t_setup = (lib.logger.process_clock(), lib.logger.perf_counter())    
    if mo_coeff is None: mo_coeff = las.mo_coeff
    if assert_no_dupes: las.assert_no_duplicates ()
    h2eff_sub = las.get_h2eff (mo_coeff)
    if (ci0 is None or any ([c is None for c in ci0]) or
      any ([any ([c2 is None for c2 in c1]) for c1 in ci0])):
        ci0 = las.get_init_guess_ci (mo_coeff, h2eff_sub, ci0)
    if (ci0 is None or any ([c is None for c in ci0]) or
      any ([any ([c2 is None for c2 in c1]) for c1 in ci0])):
        raise RuntimeError ("failed to populate get_init_guess")
    if frags_orbs is None: frags_orbs = getattr (las, 'frags_orbs', None)
    imporb_builders = [get_impurity_space_constructor (las, i, frag_orbs=frag_orbs)
                       for i, frag_orbs in enumerate (frags_orbs)]
    nfrags = len (las.ncas_sub)
    t0 = (lib.logger.process_clock(), lib.logger.perf_counter())
    kf0 = las.get_keyframe (mo_coeff, ci0) 
    las._flas_stdout = None # TODO: more elegant model for this

    ###############################################################################################
    ################################## Begin actual kernel logic ##################################
    ###############################################################################################





    converged = False
    it = 0
    kf1 = kf0
    impurities = [get_impurity_casscf (las, i, imporb_builder=builder)
                  for i, builder in enumerate (imporb_builders)]
    ugg = las.get_ugg ()
    t1 = log.timer_debug1 ('impurity solver construction', *t0)
    t_setup = log.timer( "LASSCF setup", *t_setup)
    # GRAND CHALLENGE: replace rigid algorithm below with dynamic task scheduling
    for it in range (las.max_cycle_macro):

        t_macro = (lib.logger.process_clock(), lib.logger.perf_counter())    
        # 1. Divide into fragments
        for impurity in impurities: 
            impurity._pull_keyframe_(kf1)
            t_macro = log.timer("Pull keyframe for fragment",*t_macro)
        # 2. CASSCF on each fragment
        kf2_list = []
        for impurity in impurities: 
            impurity.kernel ()
            t_macro = log.timer("Fragment CASSCF",*t_macro)
            kf2_list.append (impurity._push_keyframe (kf1))
            t_macro = log.timer("Push keyframe for fragment",*t_macro)

        # 3. Combine from fragments. TODO: smaller chunks instead of one whole-molecule function
        kf1 = combine.combine_o0 (las, kf2_list)
        t_macro = log.timer("Recombination",*t_macro)

        # Evaluate status and break if converged
        e_tot = las.energy_nuc () + las.energy_elec (
            mo_coeff=kf1.mo_coeff, ci=kf1.ci, h2eff=kf1.h2eff_sub, veff=kf1.veff)
        gvec = las.get_grad (ugg=ugg, kf=kf1)
        norm_gvec = linalg.norm (gvec)
        log.info ('LASSCF macro %d : E = %.15g ; |g| = %.15g', it, e_tot, norm_gvec)
        t_macro = log.timer("Energy and gradient calculation",*t_macro)
        t1 = log.timer ('one LASSCF macro cycle', *t1)
        las.dump_chk (mo_coeff=kf1.mo_coeff, ci=kf1.ci)
        if norm_gvec < conv_tol_grad:
            converged = True
            break





    ###############################################################################################
    ################################### End actual kernel logic ###################################
    ###############################################################################################

    if getattr (las, '_flas_stdout', None) is not None: las._flas_stdout.close ()
    # TODO: more elegant model for this
    mo_coeff, ci1, h2eff_sub, veff = kf1.mo_coeff, kf1.ci, kf1.h2eff_sub, kf1.veff
    t1 = log.timer ('LASSCF {} macrocycles'.format (it), *t0)
    e_tot = las.energy_nuc () + las.energy_elec (mo_coeff=mo_coeff, ci=ci1, h2eff=h2eff_sub,
                                                 veff=veff)
    e_states = las.energy_nuc () + np.array (las.states_energy_elec (mo_coeff=mo_coeff, ci=ci1,
                                                                     h2eff=h2eff_sub, veff=veff))
    # This crap usually goes in a "_finalize" function
    log.info ('LASSCF %s after %d cycles', ('not converged', 'converged')[converged], it+1)
    log.info ('LASSCF E = %.15g ; |g| = %.15g', e_tot,
              norm_gvec)
    t1 = log.timer ('LASSCF final energy', *t1)
    mo_coeff, mo_energy, mo_occ, ci1, h2eff_sub = las.canonicalize (mo_coeff, ci1, veff=veff,
                                                                    h2eff_sub=h2eff_sub)
    t1 = log.timer ('LASSCF canonicalization', *t1)
    t0 = log.timer ('LASSCF kernel function', *t0)

    e_cas = None # TODO: get rid of this worthless, meaningless variable
    return converged, e_tot, e_states, mo_energy, mo_coeff, e_cas, ci1, h2eff_sub, veff


def patch_kernel (las):
    class PatchedLAS (las.__class__):
        _kern = kernel
    return lib.view (las, PatchedLAS)


