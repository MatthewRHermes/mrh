import numpy as np
import itertools
from pyscf import lib
from mrh.my_pyscf.mcscf.lasci import get_space_info
from mrh.my_pyscf.lassi.citools import get_lroots
from mrh.my_pyscf.lassi.spaces import SingleLASRootspace
from mrh.my_pyscf.lassi.op_o1.utilities import lst_hopping_index
from mrh.my_pyscf.lassi import op_o0, op_o1
from mrh.my_pyscf.fci.spin_op import mup

op = (op_o0, op_o1)

def case_contract_hlas_ci (ks, las, h0, h1, h2, ci_fr, nelec_frs):
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

    nelec = nelec_frs

    spaces = [SingleLASRootspace (las, m, s, c, 0) for c,m,s,w in zip (*get_space_info (las))]

    lroots = get_lroots (ci_fr)
    lroots_prod = np.prod (lroots, axis=0)
    nj = np.cumsum (lroots_prod)
    ni = nj - lroots_prod
    ndim = nj[-1]
    for opt in range (2):
        ham = op[opt].ham (las, h1, h2, ci_fr, nelec)[0]
        hket_fr_pabq = op[opt].contract_ham_ci (las, h1, h2, ci_fr, nelec, ci_fr, nelec)
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
    dens, fbf_sf, fbf_ch = lsi.make_fbfdm ()
    for i,s in itertools.product (range (lsi.nfrags), range(2)):
        if fbf_sf[i][s] is None: continue
        ddens = np.trace (fbf_sf[i][s])
        ks.assertTrue (ddens>=0)
        ks.assertTrue (dens[i]>=0)
        dens[i] += ddens
    for i,a,s in itertools.product (range (lsi.nfrags), range (lsi.nfrags), range(4)):
        if fbf_ch[i][a][s] is None: continue
        ddens = np.trace (np.trace (fbf_ch[i][a][s]))
        ks.assertTrue (ddens>=0)
        dens[i] += ddens
        dens[a] += ddens
    for i in range (lsi.nfrags):
        ks.assertAlmostEqual (dens[i],1.0,5)

