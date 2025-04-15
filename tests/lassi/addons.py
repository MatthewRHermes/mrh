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
    hopping_index, zerop_index, onep_index = lst_hopping_index (nelec_frs)
    symm_index = np.all (hopping_index.sum (0) == 0, axis=0)
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
def case_contract_hlas_ci (ks, las, h0, h1, h2, ci_fr, nelec_frs):
    interactions, interidx = describe_interactions (nelec_frs)
    nelec = nelec_frs

    spaces = [SingleLASRootspace (las, m, s, c, 0) for c,m,s,w in zip (*get_space_info (las))]

    lroots = get_lroots (ci_fr)
    lroots_prod = np.prod (lroots, axis=0)
    nj = np.cumsum (lroots_prod)
    ni = nj - lroots_prod
    ndim = nj[-1]
    si_bra = np.random.rand (ndim)
    si_ket = np.random.rand (ndim)
    for opt in range (2):
        ham, _, ovlp = op[opt].ham (las, h1, h2, ci_fr, nelec)[:3]
        ham += h0 * ovlp
        hket_fr_pabq = op[opt].contract_ham_ci (las, h1, h2, ci_fr, nelec, ci_fr, nelec, h0=h0)
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
                    #elif opt==1: print (r,s, round (lib.fp (hket_pq_s)-lib.fp (hket_ref_s),3))
                    with ks.subTest (opt=opt, frag=f, bra_space=r, ket_space=s,
                                       intyp=interactions[interidx[r,s]],
                                       dneleca=nelec[:,r,0]-nelec[:,s,0],
                                       dnelecb=nelec[:,r,1]-nelec[:,s,1]):
                        ks.assertAlmostEqual (lib.fp (hket_pq_s), lib.fp (hket_ref_s), 8)
        h_ref = np.dot (si_bra.conj (), np.dot (ham, si_ket))
        hket_fr_pabq = op[opt].contract_ham_ci (las, h1, h2, ci_fr, nelec, ci_fr, nelec, h0=h0,
                                                si_bra=si_bra, si_ket=si_ket)
        for f, (ci_r, hket_r_pabq) in enumerate (zip (ci_fr, hket_fr_pabq)):
            h_test = 0
            for r, (ci, hket_pabq) in enumerate (zip (ci_r, hket_r_pabq)):
                if ci.ndim < 3: ci = ci[None,:,:]
                with ks.subTest (opt=opt, frag=f, bra_space=r, nelec=nelec[f,r]):
                    h_test += lib.einsum ('pab,pab->', hket_pabq, ci.conj ())
            with ks.subTest (opt=opt, frag=f, bra_space=r, nelec=nelec[f,r]):
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

def case_lassis_grads (ks, lsis):
    if lsis.converged:
        de = lsis.e_roots - lsis.e_roots[0]
        i = np.where (de>1e-4)[0][0]
        si = (lsis.si[:,0] + lsis.si[:,i]) * np.sqrt (0.5)
    else:
        si = si[:,0]
    g_all = grad_orb_ci_si.get_grad (lsis, si=si, pack=True)
    ugg = coords.UnitaryGroupGenerators (
        lsis,
        lsis.mo_coeff,
        lsis.get_ci_ref (),
        lsis.ci_spin_flips,
        lsis.ci_charge_hops,
        si
    )
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
            div = 1.0
            err_last = np.finfo (float).tiny
            err_table = '{:s}\n'.format (lbl)
            for p in range (20):
                x2 = x1 / div
                e1_test = np.dot (x2, g_all)
                e1_ref = lsis.energy_tot (*ugg.update_wfn (x2)) - e0
                err = (e1_test - e1_ref) / e1_ref
                err_table += '{:e} {:e} {:e}\n'.format (1/div, e1_ref, err)
                rel_err = err / err_last
                err_last = err + np.finfo (float).tiny
                div *= 2
            ks.assertAlmostEqual (rel_err, .5, 1, msg=err_table)

def case_lassis_hessian (ks, lsis):
    if lsis.converged:
        de = lsis.e_roots - lsis.e_roots[0]
        i = np.where (de>1e-4)[0][0]
        si = (lsis.si[:,0] + lsis.si[:,i]) * np.sqrt (0.5)
    else:
        si = si[:,0]
    #si[:] = 0
    #si[0] = 1
    ci_ref = lsis.get_ci_ref ()
    #for ci_i in ci_ref:
    #    ci_i[:,:] = 0
    #    ci_i[0,0] = 1
    g0 = grad_orb_ci_si.get_grad (lsis, ci_ref=ci_ref, si=si, pack=True)
    ugg = coords.UnitaryGroupGenerators (
        lsis,
        lsis.mo_coeff,
        ci_ref, #lsis.get_ci_ref (),
        lsis.ci_spin_flips,
        lsis.ci_charge_hops,
        si
    )
    g0_debug = grad_orb_ci_si.get_grad (lsis, *ugg.update_wfn (np.zeros_like (g0)), pack=True)
    ks.assertLess (np.amax (np.abs (g0_debug-g0)), 1e-10, msg='sanity fail')
    h_op = hessian_orb_ci_si.HessianOperator (ugg)
    x0 = np.random.rand (ugg.nvar_tot)
    x0 = ugg.pack (*ugg.unpack (x0)) # apply some projections
    assert (len (x0) == len (g0))
    sec_lbls = ['orb', 'ci_ref', 'ci_sf', 'ci_ch', 'si_avg', 'si_ext']
    sec_offs = ugg.get_sector_offsets ()
    for (i, j), lbl0 in zip (sec_offs, sec_lbls):
        if i==j: continue
        x1 = np.zeros_like (x0)
        x1[i:j] = x0[i:j]
        div = 1.0
        h_op_x1 = h_op (x1)
        err_last = [np.finfo (float).tiny,]*len(sec_lbls)
        err_table = ['\n{:s} {:s}\n'.format (lbl1, lbl0) for lbl1 in sec_lbls]
        rel_err = [1,]*len(sec_lbls)
        for p in range (20):
            x2 = x1 / div
            g1_test = h_op_x1 / div
            #g1_test = h_op (x2)
            g1_ref = grad_orb_ci_si.get_grad (lsis, *ugg.update_wfn (x2))#, pack=True) - g0
            g1_ref = ugg.pack (*g1_ref) - g0
            for z, (k,l) in enumerate (sec_offs):
                if k==l:
                    rel_err[z] = .5
                    continue
                g2_test = np.zeros_like (g1_test)
                g2_ref = np.zeros_like (g1_ref)
                g2_test[k:l] = g1_test[k:l]
                g2_ref[k:l] = g1_ref[k:l]
                err = vector_error (g2_test, g2_ref, err_type='rel', ang_units='deg')
                err_table[z] += '{:e} {:e} {:e} {:e} {:.1f}\n'.format (
                    1/div, linalg.norm (g2_ref), linalg.norm (g2_test), err[0], err[1]
                )
                err = err[0]
                rel_err[z] = (err / err_last[z])
                if linalg.norm (g2_test) < 1e-16:
                    err = linalg.norm (g2_ref)
                    rel_err[z] = (err / err_last[z]) * 2
                err_last[z] = err + np.finfo (float).tiny
            div *= 2
        for rel_err_i, err_table_i, lbl1 in zip (rel_err, err_table, sec_lbls):
            with ks.subTest ((lbl1,lbl0)):
                ks.assertAlmostEqual (rel_err_i, .5, 1, msg=err_table_i)




