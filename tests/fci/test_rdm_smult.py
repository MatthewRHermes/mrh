import numpy as np
from pyscf.csf_fci.csfstring import CSFTransformer
import itertools, functools
from mrh.my_pyscf.fci import rdm as mrh_rdm
from pyscf.fci.direct_spin1 import trans_rdm1s, trans_rdm12s

def trans_rdm2s (cibra, ciket, norb, nelec):
    return trans_rdm12s (cibra, ciket, norb, nelec)[1]

def trans_rdm3ha_des (cibra, ciket, norb, nelec):
    return mrh_rdm.trans_rdm13ha_des (cibra, ciket, norb, nelec)[1]

def trans_rdm3hb_des (cibra, ciket, norb, nelec):
    return mrh_rdm.trans_rdm13hb_des (cibra, ciket, norb, nelec)[1]

max_d2s = {'h': 1,
           'hh': 2,
           'dm1': 2,
           'sm': 2,
           'phh': 3,
           'dm2': 4}

spin_op_cases = {'h': [-1,1],
                 'hh': [-2,0,2],
                 'dm1': 1,
                 'sm': 1,
                 'phh': [-1,1],
                 'dm2': 1}

make_dm = {('h',0): mrh_rdm.trans_rdm1ha_des,
           ('h',1): mrh_rdm.trans_rdm1hb_des,
           ('hh',0): functools.partial (mrh_rdm.trans_hhdm, spin=0),
           ('hh',1): functools.partial (mrh_rdm.trans_hhdm, spin=1),
           ('hh',2): functools.partial (mrh_rdm.trans_hhdm, spin=2),
           ('dm1',0): trans_rdm1s,
           ('sm',0): mrh_rdm.trans_sfddm1,
           ('phh',0): trans_rdm3ha_des,
           ('phh',1): trans_rdm3hb_des,
           ('dm2',0): trans_rdm2s}

def smult_loop (ks, dm):
    for smult_ket in range (1, 4+max_d2s[dm], 2):
        # 4 = smult off-by-1 + range off-by-1 + an additional step of 2 above the max
        par = max_d2s[dm] % 2
        max_d2s_p1 = max_d2s[dm] + 1 # test zeros
        min_smult_bra = max (1+par, smult_ket-max_d2s_p1)
        max_smult_bra = smult_ket+max_d2s_p1
        for smult_bra in range (min_smult_bra, max_smult_bra, 2):
            norb_nelec_loop (ks, dm, smult_bra, smult_ket)
            if par > 0:
                norb_nelec_loop (ks, dm, smult_ket, smult_bra)

def norb_nelec_loop (ks, dm, smult_bra, smult_ket):
    min_n = max (smult_bra, smult_ket) - 1
    for norb in (min_n, min_n+1, min_n+2):
        max_n = min (norb, min_n+2):
        for nelec in range (min_n, max_n+1, 2):
            spin_loop (ks, dm, smult_bra, smult_ket, norb, nelec)


def spin_loop (ks, dm, smult_bra, smult_ket, norb, nelec):
    ci_bra = get_civecs (norb, nelec, smult_bra)
    ci_ket = get_civecs (norb, nelec, smult_ket)
    for spin_op, dspin_op in enumerate (spin_op_cases[dm]):
        max_spin_ket = min (smult_ket, smult_bra-dspin_op)-1
        spin_ket_range = range (-max_spin_ket,max_spin_ket+1,2)
        dm_maker = make_dm[(dm,spin_op)]
        tdms = {spin_ket: make_tdms (dm_maker, ci_bra, ci_ket, norb, nelec, spin_ket)
                for spin_ket in spin_ket_range}
        for m1, m2 in itertools.product (spin_ket_range, repeat=2):
            for i in range (3):
                ref = tdms[m2][i]
                test = mdown (mup (tdms[m1][i],m1), m2)
                ks.assertAlmostEqual (lib.fp (test), lib.fp (ref), 8)

civec_cache = {}
def get_civecs (norb, nelec, smult):
    assert (smult >= 1)
    assert (smult-1 <= 2*nelec)
    assert (2*norb >= nelec)
    if (norb,nelec,smult) in civec_cache.keys ():
        return civec_cache[(norb,nelec,smult)]
    spin = smult-1
    csfvec = None
    civecs = {}
    for spin in range (smult-1, -smult, -2):
        neleca = (nelec + spin) // 2
        nelecb = (nelec - spin) // 2
        t = CSFTransformer (norb, neleca, nelecb, smult)
        if csfvec is None:
            csfvec = 2*np.random.rand (4,t.ncsf) - 1
            csfvec /= linalg.norm (csfvec, axis=1)
        civecs[spin] = t.vec_csf2det (csfvec)
    civec_cache[(norb,nelec,smult)] = civecs
    return civecs

def make_tdms (dm, ci_bra, ci_ket, norb, nelec, spin_ket):
    neleca = (nelec + spin_ket) // 2
    nelecb = (nelec - spin_ket) // 2
    nelec = (neleca,nelecb)
    tdm_list = [dm_maker (ci_bra[spin_ket][i], ci_ket[spin_ket][i], norb, nelec)
                for i in range (4)]
    tdm_shape = tdm_list[0].shape
    third_shape = (2,2) + tdm_shape
    tdms = [tdm_list[0],
            np.stack (tdm_list[:2], axis=0),
            np.stack (tdm_list, axis=0).reshape (*third_shape)]
    return tdms


