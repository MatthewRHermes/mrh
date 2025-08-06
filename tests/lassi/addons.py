import numpy as np
from scipy import linalg
import itertools
from pyscf import lib
from mrh.my_pyscf.mcscf.lasci import get_space_info
from mrh.my_pyscf.lassi.citools import get_lroots
from mrh.my_pyscf.lassi.spaces import SingleLASRootspace
from mrh.my_pyscf.lassi.op_o1.utilities import lst_hopping_index
from mrh.my_pyscf.lassi.lassis import coords, grad_orb_ci_si, hessian_orb_ci_si
from mrh.my_pyscf.lassi import op_o0, op_o1
from mrh.my_pyscf.fci.spin_op import mup
from mrh.my_pyscf.lassi.lassi import LINDEP_THRESH
from pyscf.scf.addons import canonical_orth_
from mrh.util.la import vector_error
op = (op_o0, op_o1)

def describe_interactions (nelec_frs):
    hopping_index = lst_hopping_index (nelec_frs)
    symm_index = np.all (hopping_index.sum (0) == 0, axis=0)
    onep_index = symm_index & (np.abs (hopping_index).sum ((0,1)) == 2)
    twop_index = symm_index & (np.abs (hopping_index).sum ((0,1)) == 4)
    twoc_index = twop_index & (np.abs (hopping_index.sum (1)).sum (0) == 4)
    ocos_index = twop_index & (np.abs (hopping_index.sum (1)).sum (0) == 2)
    ones_index = twop_index & (np.abs (hopping_index.sum (1)).sum (0) == 0)
    twoc2_index = twoc_index & (np.count_nonzero (hopping_index.sum (1), axis=0) == 2)
    twoc3_index = twoc_index & (np.count_nonzero (hopping_index.sum (1), axis=0) == 3)
    twoc4_index = twoc_index & (np.count_nonzero (hopping_index.sum (1), axis=0) == 4)
    interactions = ['null', '1c', '1s', '1c1s', '2c_2', '2c_3', '2c_4']
    interidx = (onep_index.astype (int) + 2*ones_index.astype (int)
                + 3*ocos_index.astype (int) + 4*twoc2_index.astype (int)
                + 5*twoc3_index.astype (int) + 6*twoc4_index.astype (int))
    return interactions, interidx

# TODO: SOC generalization!
def case_contract_hlas_ci (ks, las, h0, h1, h2, ci_fr, nelec_frs, si_bra=None, si_ket=None):
    interactions, interidx = describe_interactions (nelec_frs)
    nelec = nelec_frs

    spaces = [SingleLASRootspace (las, m, s, c, 0) for c,m,s,w in zip (*get_space_info (las))]

    lroots = get_lroots (ci_fr)
    lroots_prod = np.prod (lroots, axis=0)
    nj = np.cumsum (lroots_prod)
    ni = nj - lroots_prod
    ndim = nj[-1]
    sivec_bra = np.random.rand (ndim)
    sivec_ket = np.random.rand (ndim)
    if si_bra is not None:
        if np.issubdtype (np.asarray (si_bra).dtype, np.integer):
            sivec_bra[:] = 0
            sivec_bra[np.asarray (si_bra)] = 1
        elif isinstance (si_bra, str):
            if 'flat' in si_bra.lower ():
                sivec_bra[:] = 1
    if si_ket is not None:
        if np.issubdtype (np.asarray (si_ket).dtype, np.integer):
            sivec_ket[:] = 0
            sivec_ket[np.asarray (si_ket)] = 1
        elif isinstance (si_ket, str):
            if 'flat' in si_ket.lower ():
                sivec_ket[:] = 1
    for opt in range (2):
        ham, _, ovlp = op[opt].ham (las, h1, h2, ci_fr, nelec)[:3]
        ham += h0 * ovlp
        hket_fr_pabq = op[opt].contract_ham_ci (las, h1, h2, ci_fr, nelec, h0=h0)
        for f, (ci_r, hket_r_pabq) in enumerate (zip (ci_fr, hket_fr_pabq)):
            current_order = list (range (las.nfrags)) + [las.nfrags]
            current_order.insert (0, current_order.pop (las.nfrags-1-f))
            for r, (ci, hket_pabq) in enumerate (zip (ci_r, hket_r_pabq)):
                if ci.ndim < 3: ci = ci[None,:,:]
                proper_shape = np.append (lroots[::-1,r], ndim)
                current_shape = proper_shape[current_order]
                to_proper_order = list (np.argsort (current_order))
                hket_pq = lib.einsum ('rab,pabq->rpq', ci.conj (), hket_pabq)
                hket_pq = hket_pq.reshape (current_shape)
                hket_pq = hket_pq.transpose (*to_proper_order)
                hket_pq = hket_pq.reshape ((lroots_prod[r], ndim))
                hket_ref = ham[ni[r]:nj[r]]
                for s, (k, l) in enumerate (zip (ni, nj)):
                    hket_pq_s = hket_pq[:,k:l]
                    hket_ref_s = hket_ref[:,k:l]
                    # TODO: opt>0 for things other than single excitation
                    #if opt>0 and not spaces[r].is_single_excitation_of (spaces[s]): continue
                    #elif opt==1: print (r,s, round (lib.fp (hket_pq_s)-lib.fp (hket_ref_s),2))
                    with ks.subTest (opt=opt, frag=f, bra_space=r, ket_space=s,
                                       intyp=interactions[interidx[r,s]],
                                       dneleca=nelec[:,r,0]-nelec[:,s,0],
                                       dnelecb=nelec[:,r,1]-nelec[:,s,1]):
                        pass
                        #ks.assertAlmostEqual (lib.fp (hket_pq_s), lib.fp (hket_ref_s), 8)
        hket_ref = np.dot (ham, sivec_ket)
        hket_fr_pabq = op[opt].contract_ham_ci (las, h1, h2, ci_fr, nelec, h0=h0,
                                                si_bra=sivec_bra, si_ket=sivec_ket)
        for f, (ci_r, hket_r_pabq) in enumerate (zip (ci_fr, hket_fr_pabq)):
            for r, (ci, hket_pabq) in enumerate (zip (ci_r, hket_r_pabq)):
                if ci.ndim < 3: ci = ci[None,:,:]
                with ks.subTest (opt=opt, frag=f, bra_space=r, nelec=nelec[f,r]):
                    h_test = lib.einsum ('pab,pab->', hket_pabq, ci.conj ())
                    h_ref = np.dot (sivec_bra[ni[r]:nj[r]].conj (), hket_ref[ni[r]:nj[r]])
                    ks.assertAlmostEqual (h_test, h_ref, 8)
    return hket_fr_pabq

def case_contract_op_si (ks, las, h1, h2, ci_fr, nelec_frs, soc=0):
    ham, s2, ovlp = op[1].ham (las, h1, h2, ci_fr, nelec_frs, soc=soc)[:3]
    ops = op[1].gen_contract_op_si_hdiag (las, h1, h2, ci_fr, nelec_frs, soc=soc)
    ham_op, s2_op, ovlp_op, ham_diag = ops[:4]
    with ks.subTest ('hdiag'):
        ks.assertAlmostEqual (lib.fp (ham.diagonal ()), lib.fp (ham_diag), 7)
    nstates = ham.shape[0]
    x = np.random.rand (nstates)
    if soc:
        x = x + 1j*np.random.rand (nstates)
    with ks.subTest ('ham_op'):
        ks.assertAlmostEqual (lib.fp (ham_op (x)), lib.fp (ham @ x), 7)
    with ks.subTest ('s2_op'):
        ks.assertAlmostEqual (lib.fp (s2_op (x)), lib.fp (s2 @ x), 7)
    with ks.subTest ('ovlp_op'):
        ks.assertAlmostEqual (lib.fp (ovlp_op (x)), lib.fp (ovlp @ x), 7)

def debug_contract_op_si (ks, las, h1, h2, ci_fr, nelec_frs, soc=0):
    nroots = nelec_frs.shape[1]
    interactions, interidx = describe_interactions (nelec_frs)
    ham, s2, ovlp = op[1].ham (las, h1, h2, ci_fr, nelec_frs, soc=soc)[:3]
    np.save ('nelec_frs.npy', nelec_frs)
    ops = op[1].gen_contract_op_si_hdiag (las, h1, h2, ci_fr, nelec_frs, soc=soc)
    ham_op, s2_op, ovlp_op, ham_diag = ops[:4]
    lroots = get_lroots (ci_fr)
    lroots_prod = np.prod (lroots, axis=0)
    nj = np.cumsum (lroots_prod)
    ni = nj - lroots_prod    
    nstates = ham.shape[0]
    for r in range (nroots):
        i, j = ni[r], nj[r]
        with ks.subTest ('hdiag', root=r, nelec_fs=nelec_frs[:,r,:]):
            #print (ham.diagonal ()[i:j], ham_diag[i:j])
            ks.assertAlmostEqual (lib.fp (ham.diagonal ()[i:j]), lib.fp (ham_diag[i:j]), 7)
    x = np.random.rand (nstates)
    if soc:
        x = x + 1j*np.random.rand (nstates)
    for myop, ref, lbl in ((ham_op, ham, 'ham'), (s2_op, s2, 's2'), (ovlp_op, ovlp, 'ovlp')):
        test = myop (np.eye (nstates))
        for s in range (nroots):
            k, l = ni[s], nj[s]
            test = myop (np.eye (nstates)[:,k:l])
            for r in range (nroots):
                i, j = ni[r], nj[r]
                with ks.subTest (lbl, blk=(r,s), intyp=interactions[interidx[r,s]]):
                    ks.assertAlmostEqual (lib.fp (test[i:j]), lib.fp (ref[i:j,k:l]), 7)
        #for r,s in itertools.product (range (nroots), repeat=2):
        #    i, j = ni[r], nj[r]
        #    with ks.subTest (lbl, blk=(r,s), intyp=interactions[interidx[r,s]]):
        #        ks.assertAlmostEqual (lib.fp (test[i:j,k:l]), lib.fp (ref[i:j,k:l]), 7)

def case_lassis_fbf_2_model_state (ks, lsi):
    seen_fr = np.zeros ((lsi.nfrags,lsi.nroots), dtype=int)
    nlas = lsi.ncas_sub
    smult_fr = lsi.get_smult_fr ()
    with ks.subTest ('get_ref_fbf_rootspaces'):
        ci_ref = lsi.get_ci_ref ()
        for i in range (lsi.nfrags):
            idx, nelec_rs = lsi.get_ref_fbf_rootspaces (i)
            seen_fr[i,idx] += 1
            ci_ref_fp = lib.fp (ci_ref[i]) 
            for j, ne in zip (idx, nelec_rs):
                ci_test_fp = lib.fp (mup (lsi.ci[i][j], nlas[i], ne, smult_fr[i][j]))
                ks.assertAlmostEqual (ci_test_fp, ci_ref_fp, 9)
    with ks.subTest ('get_sf_fbf_rootspaces'):
        for i in range (lsi.nfrags):
            for s in range (2):
                idx, nelec_rs = lsi.get_sf_fbf_rootspaces (i, s)
                seen_fr[i,idx] += 1
                ci_ref = lsi.ci_spin_flips[i][s]
                ci_ref_fp = lib.fp (ci_ref) if ci_ref is not None else 0
                for j, ne in zip (idx, nelec_rs):
                    ci_test_fp = lib.fp (mup (lsi.ci[i][j], nlas[i], ne, smult_fr[i][j]))
                    ks.assertAlmostEqual (ci_test_fp, ci_ref_fp, 9)
    with ks.subTest ('get_ch_fbf_rootspaces'):
        for i,a in itertools.product (range (lsi.nfrags), repeat=2):
            for s in range (4):
                idx, nelec_i_rs, nelec_a_rs = lsi.get_ch_fbf_rootspaces (i,a,s)
                seen_fr[i,idx] += 1
                seen_fr[a,idx] += 1
                ci_ref = lsi.ci_charge_hops[i][a][s][0]
                ci_ref_fp = lib.fp (ci_ref) if ci_ref is not None else 0
                for j, ne in zip (idx, nelec_i_rs):
                    ci_test_fp = lib.fp (mup (lsi.ci[i][j], nlas[i], ne, smult_fr[i][j]))
                    ks.assertAlmostEqual (ci_test_fp, ci_ref_fp, 9)
                ci_ref = lsi.ci_charge_hops[i][a][s][1]
                ci_ref_fp = lib.fp (ci_ref) if ci_ref is not None else 0
                for b, ne in zip (idx, nelec_a_rs):
                    ci_test_fp = lib.fp (mup (lsi.ci[a][b], nlas[a], ne, smult_fr[a][b]))
                    ks.assertAlmostEqual (ci_test_fp, ci_ref_fp, 9)
    with ks.subTest ('comprehensive covering'):
        ks.assertTrue (np.all (seen_fr==1))

def case_lassis_fbfdm (ks, lsi):
    for ifrag in range (lsi.nfrags):
        ovlp = lsi.get_fbf_ovlp (ifrag)
        dm1 = lsi.make_fbfdm1 (ifrag)
        with ks.subTest ('hermiticity', ifrag=ifrag):
            ks.assertAlmostEqual (lib.fp (ovlp), lib.fp (ovlp.T), 9)
            ks.assertAlmostEqual (lib.fp (dm1), lib.fp (dm1.T), 9)
        with ks.subTest ('positive-semidefiniteness', ifrag=ifrag):
            evals, evecs = linalg.eigh (ovlp)
            ks.assertTrue (evals[0]>-1e-4)
            evals, evecs = linalg.eigh (dm1)
            ks.assertTrue (evals[0]>-1e-4)
        with ks.subTest ('normalization', ifrag=ifrag):
            x = canonical_orth_(ovlp, thr=LINDEP_THRESH)
            xdx = x.conj ().T @ dm1 @ x
            ks.assertAlmostEqual ((dm1*ovlp).sum (), 1.0, 9)

def case_lassis_grads (ks, lsis, s2=(0,2)):
    if not hasattr (s2, '__len__'): s2 = (s2,s2)
    if lsis.converged:
        i = np.where (np.abs (lsis.s2 - s2[0]) < 1e-4)[0][0]
        de = lsis.e_roots - lsis.e_roots[i]
        idx = np.abs (lsis.s2 - s2[1]) < 1e-4
        idx &= np.abs (lsis.e_roots - lsis.e_roots[i]) > 1e-4
        j = np.where (idx)[0][0]
        assert (i!=j)
        si = (lsis.si[:,i] + lsis.si[:,j]) * np.sqrt (0.5)
    else:
        si = lsis.si[:,0]
    g_all = grad_orb_ci_si.get_grad (lsis, si=si, pack=True)
    ugg = coords.UnitaryGroupGenerators (
        lsis,
        lsis.mo_coeff,
        lsis.get_ci_ref (),
        lsis.ci_spin_flips,
        lsis.ci_charge_hops,
        si
    )
    np.random.seed (1)
    x0 = np.random.rand (ugg.nvar_tot)
    x0 = ugg.pack (*ugg.unpack (x0)) # apply some projections
    assert (len (x0) == len (g_all))
    sec_lbls = ['orb', 'ci_ref', 'ci_sf', 'ci_ch', 'si_avg', 'si_ext']
    sec_offs = ugg.get_sector_offsets ()
    e0 = lsis.energy_tot (*ugg.update_wfn (np.zeros_like (x0)))
    for (i, j), lbl in zip (sec_offs, sec_lbls):
        if i==j: continue
        if np.amax (np.abs (g_all[i:j]))<1e-8: continue
        with ks.subTest (lbl):
            x1 = np.zeros_like (x0)
            x1[i:j] = x0[i:j]
            err_last = np.finfo (float).tiny
            e1_ref_last = np.finfo (float).tiny
            err_table = '{:s}\n'.format (lbl)
            e1_test = np.dot (x1, g_all)
            # NOTE: this starting point is empirical. I don't know the scale of the convergence
            # plateau a priori.
            for p in range (6,20):
                div = 2**p
                x2 = x1 / div
                e1_ref = lsis.energy_tot (*ugg.update_wfn (x2)) - e0
                e1_ref -= (lsis.energy_tot (*ugg.update_wfn (-x2)) - e0)
                e1_ref *= .5 * div
                err = (e1_test - e1_ref) / e1_ref
                rel_err = err / err_last
                conv = 1.0 - e1_ref/e1_ref_last
                err_table += '{:e} {:e} {:e} {:e} {:e}\n'.format (1/div, e1_ref, err, conv, rel_err)
                if (abs (conv) < 0.01) and (abs (rel_err-.25) < 0.01):
                    break
                err_last = err + np.finfo (float).tiny
                e1_ref_last = e1_ref
            ks.assertAlmostEqual (rel_err, .25, delta=0.01, msg=err_table)

def case_lassis_hessian (ks, lsis):
    if lsis.converged:
        de = lsis.e_roots - lsis.e_roots[0]
        i = np.where (de>1e-4)[0][0]
        si = (lsis.si[:,0] + lsis.si[:,i]) * np.sqrt (0.5)
    else:
        si = lsis.si[:,0]
    #si[:] = 0
    #si[0] = 1
    ci_ref = lsis.get_ci_ref ()
    #for ci_i in ci_ref:
    #    ci_i[:,:] = 0
    #    ci_i[0,0] = 1
    g0_sec = list (grad_orb_ci_si.get_grad (lsis, ci_ref=ci_ref, si=si, pack=False))
    ugg = coords.UnitaryGroupGenerators (
        lsis,
        lsis.mo_coeff,
        ci_ref, #lsis.get_ci_ref (),
        lsis.ci_spin_flips,
        lsis.ci_charge_hops,
        si
    )
    g0 = ugg.pack (*g0_sec)
    g0_debug = grad_orb_ci_si.get_grad (lsis, *ugg.update_wfn (np.zeros_like (g0)), pack=True)
    h_op = hessian_orb_ci_si.HessianOperator (ugg)
    np.random.seed (1)
    x0 = np.random.rand (ugg.nvar_tot)
    x0 = ugg.pack (*ugg.unpack (x0)) # apply some projections
    assert (len (x0) == len (g0))
    sec_lbls = ['orb', 'ci_ref', 'ci_sf', 'ci_ch', 'si_avg', 'si_ext']
    sec_offs = ugg.get_sector_offsets ()
    nao, nmo = lsis.mo_coeff.shape
    ncas, ncore = sum (lsis.ncas_sub), lsis.ncore
    nocc = ncore + ncas
    # A lot rides on this correction being accurate
    def orbital_frame_correction (xi):
        i, j = sec_offs[0]
        gorb = g0_sec[0]
        xorb = ugg.unpack (xi)[0]
        dgorb = (xorb @ gorb - gorb @ xorb) / 2
        dgi = [dgorb,] + g0_sec[1:]
        dgi = ugg.pack (*dgi)
        dgi[j:] = 0
        return dgi
    for (i, j), lbl0 in zip (sec_offs, sec_lbls):
        if i==j: continue
        with ks.subTest ("sanity", sector=lbl0):
            ks.assertLess (np.amax (np.abs (g0_debug[i:j]-g0[i:j])), 1e-9)
        x1 = np.zeros_like (x0)
        x1[i:j] = x0[i:j]
        dg1_ref = orbital_frame_correction (x1)
        g1_test = h_op (x1)
        err_last = [np.finfo (float).tiny,]*len(sec_lbls)
        err_table = ['\n{:s} {:s}\n'.format (lbl1, lbl0) for lbl1 in sec_lbls]
        rel_err = [1,]*len(sec_lbls)
        g1_ref_last = np.zeros_like (x1)
        brk = [False for lbl in sec_lbls]
        # NOTE: this starting point is empirical. I don't know the scale of the convergence plateau
        # a priori.
        for p in range (9,20):
            div = 2**p
            if (all (brk)): break
            x2 = x1 / div
            g1_ref = grad_orb_ci_si.get_grad (lsis, *ugg.update_wfn (x2))#, pack=True) - g0
            g1_ref = ugg.pack (*g1_ref) - g0
            mg1_ref = grad_orb_ci_si.get_grad (lsis, *ugg.update_wfn (-x2))#, pack=True) - g0
            mg1_ref = ugg.pack (*mg1_ref) - g0
            g1_ref -= mg1_ref
            g1_ref *= .5 * div
            g1_ref += dg1_ref
            for z, (k,l) in enumerate (sec_offs):
                if k==l:
                    rel_err[z] = .25
                    brk[z] = True
                if brk[z]: continue
                g2_test = np.zeros_like (g1_test)
                g2_ref = np.zeros_like (g1_ref)
                g2_ref_last = np.zeros_like (g1_ref_last)
                g2_test[k:l] = g1_test[k:l]
                g2_ref[k:l] = g1_ref[k:l]
                g2_ref_last[k:l] = g1_ref_last[k:l]
                err = vector_error (g2_test, g2_ref, err_type='rel', ang_units='deg')
                err_table[z] += '{:e} {:e} {:e} {:e} {:.1f}\n'.format (
                    1/div, linalg.norm (g2_ref), linalg.norm (g2_test), err[0], err[1]
                )
                err = err[0]
                rel_err[z] = (err / err_last[z])
                if linalg.norm (g2_test) < 1e-16:
                    err = linalg.norm (g2_ref)
                    rel_err[z] = (err / err_last[z])
                err_last[z] = err + np.finfo (float).tiny
                conv = vector_error (g2_ref, g2_ref_last, err_type='rel')[0]
                if (conv < 0.01) and (abs (rel_err[z]-.25) < 0.01):
                    brk[z] = True
            g1_ref_last = g1_ref
        for rel_err_i, err_table_i, lbl1 in zip (rel_err, err_table, sec_lbls):
            with ks.subTest ((lbl1,lbl0)):
                ks.assertAlmostEqual (rel_err_i, .25, delta=0.01, msg=err_table_i)

def _compare_lassis_wfn (ks, nfrags, wfn0, wfn1, lbl=''):
    mo0, cir0, cis0, cic0, si0 = wfn0
    mo1, cir1, cis1, cic1, si1 = wfn1
    with ks.subTest ('mo_coeff ' + lbl):
        ks.assertAlmostEqual (lib.fp (mo0), lib.fp (mo1))
    with ks.subTest ('ci_ref ' + lbl):
        ks.assertAlmostEqual (lib.fp (cir0), lib.fp (cir1))
    with ks.subTest ('si ' + lbl):
        ks.assertAlmostEqual (lib.fp (si0), lib.fp (si1))
    with ks.subTest ('ci_sf ' + lbl):
        for i in range (nfrags):
            for j in range (2):
                ks.assertEqual (cis0[i][j] is None, cis1[i][j] is None)
                if cis0[i][j] is not None:
                    ks.assertAlmostEqual (lib.fp (cis0[i][j]), lib.fp (cis1[i][j]))
    with ks.subTest ('ci_ch ' + lbl):
        for i in range (nfrags):
            for j in range (nfrags):
                for k in range (4):
                    for l in range (2):
                        ks.assertEqual (cic0[i][j][k][l] is None, cic1[i][j][k][l] is None)
                        if cic0[i][j][k][l] is not None:
                            ks.assertAlmostEqual (lib.fp (cic0[i][j][k][l]),
                                                  lib.fp (cic1[i][j][k][l]))

def case_lassis_ugg (ks, lsis):
    ugg = coords.UnitaryGroupGenerators (
        lsis,
        lsis.mo_coeff,
        lsis.get_ci_ref (),
        lsis.ci_spin_flips,
        lsis.ci_charge_hops,
        lsis.si
    )
    ### print out dimension addresses
    #x0 = np.arange (ugg.nvar_tot) + 1
    #kappa, ci_ref, ci_sf, ci_ch, si = ugg.unpack (x0)
    #kappa -= 1
    #np.save ('kappa.npy', kappa)
    #print (lsis.ncore, lsis.ncas_sub)
    #assert (False)
    ###
    wfn0 = lsis.mo_coeff, lsis.get_ci_ref (), lsis.ci_spin_flips, lsis.ci_charge_hops, lsis.si
    wfn1 = ugg.update_wfn (np.zeros (ugg.nvar_tot))
    _compare_lassis_wfn (ks, lsis.nfrags, wfn0, wfn1, 'zero step')
    x0 = np.random.rand (ugg.nvar_tot)
    x0 = ugg.pack (*ugg.unpack (x0)) # apply some projections
    wfn0 = ugg.unpack (x0)
    x1 = ugg.pack (*wfn0)
    wfn1 = ugg.unpack (x1)
    ks.assertAlmostEqual (lib.fp (x0), lib.fp (x1))
    _compare_lassis_wfn (ks, lsis.nfrags, wfn0, wfn1, 'repeated application')
    ugg = coords.UnitaryGroupGenerators (
        lsis,
        lsis.mo_coeff,
        lsis.get_ci_ref (),
        lsis.ci_spin_flips,
        lsis.ci_charge_hops,
        lsis.si[:,0]
    )
    x0 = np.random.rand (ugg.nvar_tot)
    x0 = ugg.pack (*ugg.unpack (x0)) # apply some projections
    e_tot = lsis.energy_tot (*ugg.update_wfn (x0))
    with ks.subTest ('energy minimization'):
        ks.assertLessEqual (lsis.e_roots[0], e_tot)
    with ks.subTest ('contract_hlas_ci'):
        h0, h1, h2 = lsis.ham_2q ()
        hci_fr = case_contract_hlas_ci (ks, lsis, h0, h1, h2, lsis.ci, lsis.get_nelec_frs ())
        # Just to syntax-debug this...
        hci_ref, hci_sf, hci_ch = coords.sum_hci (lsis, hci_fr)
        mo0, _, _, _, si0 = ugg.unpack (x0)
        x1 = ugg.pack (mo0, hci_ref, hci_sf, hci_ch, si0)

def eri_sector_indexes (nlas):
    faddr = []
    for i, n in enumerate (nlas):
        faddr += [i,]*n
    faddr = np.asarray (faddr)
    norb = sum (nlas)
    eri_idx = np.zeros ((4,norb,norb,norb,norb), dtype=int)
    eri_idx[0] = faddr[:,None,None,None]
    eri_idx[1] = faddr[None,:,None,None]
    eri_idx[2] = faddr[None,None,:,None]
    eri_idx[3] = faddr[None,None,None,:]
    eri_idx = eri_idx.reshape (4,norb**4)
    sorted_frags = np.sort (eri_idx, axis=0)
    nfrag = (sorted_frags[1:] != sorted_frags[:-1]).sum (0) + 1
    idx_j = (nfrag>1) & (
        (eri_idx[0]==eri_idx[1]) | (eri_idx[2]==eri_idx[3])
    )
    idx_k = (nfrag>1) & (
        (eri_idx[1]==eri_idx[2]) | (eri_idx[0]==eri_idx[3])
    )
    idx_pp = (nfrag>1) & (
        (eri_idx[0]==eri_idx[2]) | (eri_idx[1]==eri_idx[3])
    )
    idx_pph = (idx_j & idx_pp)
    idx_j = (idx_j & (~idx_pp))
    idx_k = (idx_k & (~idx_pp))
    idx = {'pph': idx_pph.reshape ([norb,]*4),
           'j': idx_j.reshape ([norb,]*4),
           'k': idx_k.reshape ([norb,]*4),
           'pp': idx_pp.reshape ([norb,]*4)}
    nfrag = nfrag.reshape ([norb,]*4)
    return nfrag, idx

def fuzz_sivecs (si, amp=0.00001):
    return si + amp*np.random.rand(*si.shape)




