import unittest
import hashlib
import sys, os
import numpy as np
from scipy import linalg
from pyscf.csf_fci.csfstring import CSFTransformer
import itertools, functools
from mrh.my_pyscf.fci import rdm as mrh_rdm
from mrh.my_pyscf.fci import rdm_smult, spin_op
from pyscf.fci import direct_spin1
from pyscf import lib

def trans_rdm1s (cibra, ciket, norb, nelec):
    return np.stack (direct_spin1.trans_rdm1s (cibra, ciket, norb, nelec), axis=0)

def trans_rdm2s (cibra, ciket, norb, nelec):
    return np.stack (direct_spin1.trans_rdm12s (cibra, ciket, norb, nelec)[1], axis=0)

def trans_rdm3ha_des (cibra, ciket, norb, nelec):
    return np.stack (mrh_rdm.trans_rdm13ha_des (cibra, ciket, norb, nelec)[1], axis=0)

def trans_rdm3hb_des (cibra, ciket, norb, nelec):
    return np.stack (mrh_rdm.trans_rdm13hb_des (cibra, ciket, norb, nelec)[1], axis=0)


def setUpModule ():
    global max_d2s, spin_op_cases, make_dm, scale_map, scale_proc, dnelec
    max_d2s = {'h': 1,
               'hh': 2,
               'dm1': 2,
               'sm': 2,
               'phh': 3,
               'dm2': 4}
    
    spin_op_cases = {'h': [-1,1],
                     'hh': [-2,0,2],
                     'dm1': [0,],
                     'sm': [-2,],
                     'phh': [-1,1],
                     'dm2': [0,]}
    
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

    scale_map = {'phh': 'h',
                 'dm1': 'dm',
                 'dm2': 'dm'}

    scale_proc = {('phh',0): lambda x: x.sum (-4),
                  ('phh',1): lambda x: x.sum (-4),
                  ('dm1',0): lambda x: x.sum (-3),
                  ('dm2',0): lambda x: x.sum (-5),
                  ('hh',1): lambda x: x + x.T}

    dnelec = {'h': -1,
              'hh': -2,
              'phh': -1}

def tearDownModule ():
    global max_d2s, spin_op_cases, make_dm, scale_map, scale_proc, dnelec
    del max_d2s, spin_op_cases, make_dm, scale_map, scale_proc, dnelec

def smult_loop (ks, dm):
    for smult_ket in range (1, 4+max_d2s[dm]):
        # 4 = smult off-by-1 + range off-by-1 + an additional step of 2 above the max
        par_bra = (smult_ket + max_d2s[dm]) % 2
        max_d2s_p2 = max_d2s[dm] + 2 # test zeros
        min_smult_bra = max (2-par_bra, smult_ket-max_d2s_p2)
        max_smult_bra = smult_ket+max_d2s_p2
        for smult_bra in range (min_smult_bra, max_smult_bra, 2):
            sublbls = {'smult_bra': smult_bra,
                       'smult_ket': smult_ket}
            norb_nelec_loop (ks, dm, sublbls, smult_bra, smult_ket)

def norb_nelec_loop (ks, dm, sublbls, smult_bra, smult_ket):
    dnelec = {'h': -1, 'hh': -2, 'phh': -1}.get (dm, 0)
    min_somo = max (smult_bra-dnelec, smult_ket) - 1
    min_domo = (0,2)[min_somo==0]
    min_norb = min_somo + min_domo
    if dm == 'sm': min_norb += 1
    min_nelec = 2*min_domo + min_somo
    for extra_domo in (0,1):
        nelec = min_nelec + 2*extra_domo
        for extra_umo in (0,1):
            norb = min_norb + extra_domo + extra_umo
            sublbls1 = {'norb': norb, 'nelec': nelec}
            sublbls1.update (sublbls)
            spin_loop (ks, dm, sublbls1, smult_bra, smult_ket, norb, nelec)


def spin_loop (ks, dm, sublbls, smult_bra, smult_ket, norb, nelec):
    nelec_ket = nelec
    nelec_bra = nelec + dnelec.get (dm, 0)
    ci_bra = get_civecs (norb, nelec_bra, smult_bra)
    ci_ket = get_civecs (norb, nelec_ket, smult_ket)
    for spin_op, dspin_op in enumerate (spin_op_cases[dm]):
        min_spin_ket = max (-smult_ket, -smult_bra-dspin_op)+1
        max_spin_ket = min (smult_ket, smult_bra-dspin_op)-1
        if min_spin_ket > max_spin_ket: continue
        if dspin_op == 0:
            assert (abs (min_spin_ket) == abs (max_spin_ket))
        spin_ket_range = range (min_spin_ket,max_spin_ket+1,2)
        dm_maker = make_dm[(dm,spin_op)]
        tdms = {spin_ket: make_tdms (dm, dm_maker, ci_bra, spin_op, ci_ket, norb, nelec,
                                     spin_ket, None, None)
                for spin_ket in spin_ket_range}
        tdms_highm = {spin_ket: make_tdms (dm, dm_maker, ci_bra, spin_op, ci_ket, norb, nelec,
                                           spin_ket, smult_bra, smult_ket)
                      for spin_ket in spin_ket_range}
        sublbls1 = {'spin_op': spin_op}
        sublbls1.update (sublbls)
        dim_loop (ks, dm, sublbls1, smult_bra, spin_op, smult_ket, tdms, tdms_highm)

def dim_loop (ks, dm, sublbls, smult_bra, spin_op, smult_ket, tdms, tdms_highm):
    for i in range (3):
        mytdms = {key: val[i] for key, val in tdms.items ()}
        mytdms_highm = {key: val[i] for key, val in tdms_highm.items ()}
        if i==0:
            case_scale (ks, dm, sublbls, smult_bra, spin_op, smult_ket, mytdms)
        sublbls1 = {'dim': i}
        sublbls1.update (sublbls)
        stored = case_mup (ks, dm, sublbls1, smult_bra, spin_op, smult_ket, mytdms_highm)
        case_mdown (ks, dm, sublbls1, smult_bra, spin_op, smult_ket, mytdms, stored)
        #break

def _spin_sum_dm2 (dm2):
    idx = np.arange (dm2.ndim-1, dtype=int)
    idx[-4:] = [idx[-3],idx[-4],idx[-1],idx[-2]]
    return (dm2[...,0,:,:,:,:]
            + dm2[...,1,:,:,:,:]
            + dm2[...,1,:,:,:,:].transpose (*idx),
            + dm2[...,2,:,:,:,:])

def get_scale_fns (dm, smult_bra, spin_op, smult_ket):
    proc = scale_proc.get ((dm,spin_op), lambda x:x)
    fn0 = 'scale_' + scale_map.get (dm, dm)
    fn0 = getattr (rdm_smult, fn0)
    if len (spin_op_cases[dm]) > 1:
        def fn1 (spin_ket):
            return fn0 (smult_bra, spin_op, smult_ket, spin_ket)
    else:
        def fn1 (spin_ket):
            return fn0 (smult_bra, smult_ket, spin_ket)
    return fn1, proc

def case_scale (ks, dm, sublbls, smult_bra, spin_op, smult_ket, tdms):
    fn, proc = get_scale_fns (dm, smult_bra, spin_op, smult_ket)
    spins_ket = list (tdms.keys ())
    scales = np.asarray ([fn (spin_ket) for spin_ket in spins_ket])
    idx_ref = np.argmax (np.abs (scales))
    ref = proc (tdms[spins_ket[idx_ref]]) / (scales[idx_ref] + np.finfo (float).tiny)
    for spin_ket, scale in zip (spins_ket, scales):
        test = proc (tdms[spin_ket])
        sublbls1 = {'spin_ket': spin_ket, 'ptr_spin': spins_ket[idx_ref]}
        sublbls1.update (sublbls)
        with ks.subTest ('scale', **sublbls1):
            ks.assertAlmostEqual (lib.fp (test), lib.fp (ref*scale), 8)


def get_transpose_fn (dm, fn, smult_bra, spin_op, smult_ket):
    fn0 = getattr (rdm_smult, fn+'_'+dm)
    if len (spin_op_cases[dm]) > 1:
        def fn1 (mat, spin_ket):
            return fn0 (mat, smult_bra, spin_op, smult_ket, spin_ket)
    else:
        def fn1 (mat, spin_ket):
            return fn0 (mat, smult_bra, smult_ket, spin_ket)
    return fn1

def case_mup (ks, dm, sublbls, smult_bra, spin_op, smult_ket, tdms):
    spins_ket = list (tdms.keys ())
    ref = tdms[spins_ket[0]]
    for spin_ket in spins_ket[1:]:
        test = tdms[spin_ket]
        sublbls1 = {'spin_ket': spin_ket}
        sublbls1.update (sublbls)
        with ks.subTest ('mup', **sublbls1):
            ks.assertAlmostEqual (lib.fp (test), lib.fp (ref), 8)
    spin_ket = np.amax (spins_ket)
    return ref

def case_mdown (ks, dm, sublbls, smult_bra, spin_op, smult_ket, tdms, stored):
    spins_ket = list (tdms.keys ())
    mdown = get_transpose_fn (dm, 'mdown', smult_bra, spin_op, smult_ket)
    for spin_ket in spins_ket:
        ref = tdms[spin_ket]
        sublbls1 = {'spin_ket': spin_ket}
        sublbls1.update (sublbls)
        with ks.subTest ('down', **sublbls1):
            test = mdown (stored, spin_ket)
            ks.assertAlmostEqual (lib.fp (test), lib.fp (ref), 8)
    return ref

civec_cache = {}
def get_civecs (norb, nelec, smult):
    assert (smult >= 1)
    assert (smult-1 <= 2*nelec)
    assert (2*norb >= nelec)
    assert (nelec >= 0)
    if (norb,nelec,smult) in civec_cache.keys ():
        return civec_cache[(norb,nelec,smult)]
    if nelec == 0:
        civecs = {0: np.zeros ((4,1,1), dtype=float)}
        civec_cache[(norb,nelec,smult)] = civecs
        return civecs
    spin = smult-1
    csfvec = None
    civecs = {}
    civec_up = None
    for spin in range (smult-1, -smult, -2):
        neleca = (nelec + spin) // 2
        nelecb = (nelec - spin) // 2
        if civec_up is None:
            t = CSFTransformer (norb, neleca, nelecb, smult)
        #if csfvec is None:
            csfvec = 2*np.random.rand (4,t.ncsf) - 1
            csfvec /= linalg.norm (csfvec, axis=1)[:,None]
            civec_up = t.vec_csf2det (csfvec).reshape (-1,t.ndeta,t.ndetb)
        civecs[spin] = spin_op.mdown (civec_up, norb, (neleca,nelecb), smult)
        #civecs[spin] = t.vec_csf2det (csfvec).reshape (-1,t.ndeta,t.ndetb)
    civec_cache[(norb,nelec,smult)] = civecs
    return civecs

def make_tdms (dm, dm_maker, ci_bra, spin_op, ci_ket, norb, nelec, spin_ket, smult_bra, smult_ket):
    ci_bra, ci_ket, nelec = _get_cibra_ciket_nelec (dm, ci_bra, spin_op, ci_ket, norb, nelec,
                                                    spin_ket, smult_bra, smult_ket)
    tdm_list = [dm_maker (ci_bra[i], ci_ket[i], norb, nelec)
                for i in range (4)]
    tdm_shape = tdm_list[0].shape
    third_shape = (2,2) + tdm_shape
    tdms = [tdm_list[0],
            np.stack (tdm_list[:2], axis=0),
            np.stack (tdm_list, axis=0).reshape (*third_shape)]
    return tdms

def _get_cibra_ciket_nelec (dm, ci_bra, spin_op, ci_ket, norb, nelec, spin_ket, smult_bra, smult_ket):
    neleca = (nelec + spin_ket) // 2
    nelecb = (nelec - spin_ket) // 2
    nelec = (neleca,nelecb)
    spin_bra = spin_ket + spin_op_cases[dm][spin_op]
    ci_bra = ci_bra[spin_bra]
    ci_ket = ci_ket[spin_ket]
    fn = getattr (rdm_smult, 'get_highm_civecs_{}'.format (scale_map.get (dm, dm)))
    if len (spin_op_cases[dm]) > 1:
        return fn (ci_bra, ci_ket, norb, nelec, spin_op, smult_bra=smult_bra, smult_ket=smult_ket)
    else:
        return fn (ci_bra, ci_ket, norb, nelec, smult_bra=smult_bra, smult_ket=smult_ket)

class KnownValues(unittest.TestCase):

    def test_sha256 (self):
        # Check if the automatically-generated code was ever edited
        fname = os.path.abspath (rdm_smult.__file__)
        with open (fname, 'rb') as f:
            filebytes = f.read ()
        test = hashlib.sha256 (filebytes).hexdigest ()
        ref = 'bbc95562fd76553d6d1e3fc316f1da32371dc53a739a7875a74ed56f03c13c9c'
        self.assertEqual (test, ref)

    #@unittest.skip ('debugging')
    def test_h (self):
        smult_loop (self, 'h')

    #@unittest.skip ('debugging')
    def test_hh (self):
        smult_loop (self, 'hh')

    #@unittest.skip ('debugging')
    def test_sm (self):
        smult_loop (self, 'sm')

    #@unittest.skip ('debugging')
    def test_dm1 (self):
        smult_loop (self, 'dm1')

    #@unittest.skip ('debugging')
    def test_phh (self):
        smult_loop (self, 'phh')

    #@unittest.skip ('debugging')
    def test_dm2 (self):
        smult_loop (self, 'dm2')

if __name__ == '__main__':
    print ("Full tests for rdm_smult")
    unittest.main()



