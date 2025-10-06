import copy
from tqdm import tqdm
import numpy as np
import sympy
from mrh.my_sympy.spin import spin_1h
from sympy import S, Rational, symbols, Matrix, Poly, apart, powsimp, cancel
from sympy.utilities.lambdify import lambdastr
import itertools
from mrh.my_sympy.spin.lassi_tdms_spins.glob import *
from mrh.my_sympy.spin.lassi_tdms_spins.operators import CrVector, AnVector, CrAnOperator, OpSum
from mrh.my_sympy.spin.lassi_tdms_spins.expressions import TDMExpression, combine_TDMSystem, TDMSystem, TDMScaleArray
from mrh.my_sympy.spin.lassi_tdms_spins.documentation import latex_header, _file_docstring

def solve_pure_destruction (d2s_bra, anops, d2s_ket, d2m_ket):
    return solve_pure_creation (d2s_bra, anops, d2s_ket, d2m_ket, H=True)

def solve_pure_creation (d2s_bra, crops, d2s_ket, d2m_ket, H=False):
    s_bra = s + Rational (d2s_bra, 2)
    s_ket = s + Rational (d2s_ket, 2)
    m_ket = m + Rational (d2m_ket, 2)
    if H:
        lhs = AnVector (s_bra, crops, s_ket, m_ket)
    else:
        lhs = CrVector (s_bra, crops, s_ket, m_ket)
    return lhs.solve ()

def solve_density (d2s_bra, crops, anops, d2s_ket, d2m_ket):
    s_bra = s + Rational (d2s_bra, 2)
    s_ket = s + Rational (d2s_ket, 2)
    m_ket = m + Rational (d2m_ket, 2)
    lhs = CrAnOperator (s_bra, crops, anops, s_ket, m_ket)
    return lhs.solve ()

def get_eqn_dict ():
    print ("Building equation dictionary...", flush=True)
    with tqdm(total=93) as pbar:
        #print ("============= All creation/all destruction =============")
        a = []
        #print ("------- Alpha only -------")
        a.append (TDMSystem ([solve_pure_destruction (-1, [0,], 0, 0)]))
        pbar.update (1)
        a.append (TDMSystem ([solve_pure_destruction (1, [0,], 0, 0)]))
        pbar.update (1)
        a.append (TDMSystem ([solve_pure_destruction (-2, [0,0], 0, 0)]))
        pbar.update (1)
        a.append (TDMSystem ([solve_pure_destruction (0, [0,0], 0, 0)]))
        pbar.update (1)
        a.append (TDMSystem ([solve_pure_destruction (2, [0,0], 0, 0)]))
        pbar.update (1)
        a = [e.subs_mket_to_m ().subs_sket_to_s () for e in a]
        #for expr in a: print (expr)
        b = []
        #print ("\n------- Beta only -------")
        b.append (TDMSystem ([solve_pure_destruction (1, [1,], 0, 0)]))
        pbar.update (1)
        b.append (TDMSystem ([solve_pure_destruction (-1, [1,], 0, 0)]))
        pbar.update (1)
        b.append (TDMSystem ([solve_pure_destruction (2, [1,1], 0, 0)]))
        pbar.update (1)
        b.append (TDMSystem ([solve_pure_destruction (0, [1,1], 0, 0)]))
        pbar.update (1)
        b.append (TDMSystem ([solve_pure_destruction (-2, [1,1], 0, 0)]))
        pbar.update (1)
        b = [e.subs_mket_to_m ().subs_sket_to_s () for e in b]
        #for expr in b: print (expr)
        ab = []
        #print ("\n------- Mixed -------")
        ab.append (TDMSystem ([solve_pure_destruction (-2, [1,0], 0, 0)]))
        pbar.update (1)
        ab.append (TDMSystem ([solve_pure_destruction (0, [1,0], 0, 0),
                               solve_pure_destruction (0, [0,1], 0, 0)]))
        pbar.update (4)
        ab.append (TDMSystem ([solve_pure_destruction (2, [1,0], 0, 0)]))
        pbar.update (1)
        ab = [e.subs_mket_to_m ().subs_sket_to_s () for e in ab]
        #for expr in ab: print (expr)
        gamma1 = []
        #print ("\n\n============= One-body density =============")
        gamma1.append (TDMSystem ([solve_density (0, [0,], [0,], 0, 0),
                                   solve_density (0, [1,], [1,], 0, 0)]))
        pbar.update (4)
        gamma1.append (TDMSystem ([solve_density (-2, [0,], [0,], 0, 0)]))
        pbar.update (1)
        gamma1.append (TDMSystem ([solve_density (-2, [1,], [1,], 0, 0)]))
        pbar.update (1)
        gamma1.append (TDMSystem ([solve_density (-2, [1,], [0,], 0, 0)]))
        pbar.update (1)
        gamma1.append (TDMSystem ([solve_density (0, [1,], [0,], 0, 0)]))
        pbar.update (1)
        gamma1.append (TDMSystem ([solve_density (2, [1,], [0,], 0, 0)]))
        pbar.update (1)
        gamma1 = [e.subs_mket_to_m ().subs_sket_to_s () for e in gamma1]
        #for expr in gamma1: print (expr)
        gamma3h = []
        #print ("\n\n============= Three-half-particle operators =============")
        gamma3h.append (TDMSystem ([solve_density (-2, [0,], [0,0], 1, 1)]))
        pbar.update (1)
        gamma3h.append (TDMSystem ([solve_density (-2, [1,], [1,0], 1, 1)]))
        pbar.update (1)
        gamma3h.append (TDMSystem ([solve_density (-2, [0,], [0,1], 1, 1)]))
        pbar.update (1)
        gamma3h.append (TDMSystem ([solve_density (-2, [1,], [1,1], 1, 1)]))
        pbar.update (1)
        gamma3h.append (TDMSystem ([solve_density (3, [0,], [0,0], 0, 0)]))
        pbar.update (1)
        gamma3h.append (TDMSystem ([solve_density (3, [1,], [1,0], 0, 0)]))
        pbar.update (1)
        gamma3h.append (TDMSystem ([solve_density (3, [0,], [0,1], 0, 0)]))
        pbar.update (1)
        gamma3h.append (TDMSystem ([solve_density (3, [1,], [1,1], 0, 0)]))
        pbar.update (1)
        gamma3h.append (TDMSystem ([solve_density (0, [0,], [0,0], 1, 1),
                                    solve_density (0, [1,], [1,0], 1, 1),
                                    solve_density (0, [1,], [0,1], 1, 1)]))
        pbar.update (9)
        gamma3h.append (TDMSystem ([solve_density (0, [0,], [0,0], -1, -1),
                                    solve_density (0, [1,], [1,0], -1, -1),
                                    solve_density (0, [1,], [0,1], -1, -1)]))
        gamma3h[-1] = gamma3h[-1].subs_s(s+Rational(1,2))
        gamma3h[-1].simplify_()
        pbar.update (9)
        gamma3h.append (TDMSystem ([solve_density (0, [0,], [0,1], 1, 1),
                                    solve_density (0, [0,], [1,0], 1, 1),
                                    solve_density (0, [1,], [1,1], 1, 1)]))
        pbar.update (9)
        gamma3h.append (TDMSystem ([solve_density (0, [0,], [0,1], -1, 1),
                                    solve_density (0, [0,], [1,0], -1, 1),
                                    solve_density (0, [1,], [1,1], -1, 1)]))
        gamma3h[-1] = gamma3h[-1].subs_s(s+Rational(1,2))
        gamma3h[-1].simplify_()
        pbar.update (9)
        #gamma3h = [e.subs_mket_to_m () for e in gamma3h]
        #gamma3h = [e.subs_sket_to_s () for e in gamma3h]
        #for expr in gamma3h: print (expr)
        gamma2 = []
        #print ("\n\n============= Two-body density =============")
        gamma2.append (TDMSystem ([solve_density (0, [0,0], [0,0], 4, 0)]))
        pbar.update (1)
        gamma2.append (TDMSystem ([solve_density (0, [0,1], [1,0], 4, 0)]))
        pbar.update (1)
        gamma2.append (TDMSystem ([solve_density (0, [1,1], [1,1], 4, 0)]))
        pbar.update (1)
        gamma2.append (TDMSystem ([solve_density (0, [0,0], [0,0], 2, 0),
                                   solve_density (0, [0,1], [1,0], 2, 0),
                                   #solve_density (2, [1,0], [0,1], 0, 0),
                                   solve_density (0, [1,0], [1,0], 2, 0),
                                   solve_density (0, [1,1], [1,1], 2, 0)]))
        pbar.update (16)
        gamma2.append (TDMSystem ([solve_density (0, [0,0], [0,0], 0, 0),
                                   solve_density (0, [0,1], [1,0], 0, 0),
                                   solve_density (0, [1,0], [0,1], 0, 0),
                                   solve_density (0, [1,1], [1,1], 0, 0)],
                                  _try_inverse=False))
        #gamma2[-2].exprs.append (gamma2[-2].exprs[1].transpose ((0,1,3,2)))
        #gamma2[-2].exprs.append (gamma2[-2].exprs[2].transpose ((0,1,3,2)))
        #gamma2[-2]._init_from_exprs (gamma2[-2].exprs)
        gamma2[-1].exprs.append (gamma2[-1].exprs[1].transpose ((0,1,3,2)))
        gamma2[-1].exprs.append (gamma2[-1].exprs[2].transpose ((0,1,3,2)))
        exprs = gamma2[-1].exprs
        #exprs = [exprs[0],exprs[3],exprs[1]+exprs[2],exprs[4]+exprs[5],exprs[1]-exprs[2],exprs[4]-exprs[5]]
        gamma2[-1]._init_from_exprs (exprs)
        gamma2[-1].simplify_cols_()
        #gamma2 = [e.subs_mket_to_m () for e in gamma2]
        #gamma2 = [e.subs_sket_to_s () for e in gamma2]
        pbar.update (5)


    read_exprs = a + b + ab + gamma1 + gamma3h + gamma2

    lbls = ['ha_d', 'ha_u', 'hb_d', 'hb_u', 'hh_d', 'hh_0', 'hh_u',
                   'sm',
                   'phh_a_3d', 'phh_b_3d', 'phh_a_3u', 'phh_b_3u',
                   'dm_2', 'dm_1', 'dm_0']
    subsec = []
    subsec.append ([read_exprs[i] for i in (0,27)]) # ha_d
    subsec.append ([read_exprs[i] for i in (1,28)]) # ha_u
    subsec.append ([read_exprs[i] for i in (6,29)]) # hb_d
    subsec.append ([read_exprs[i] for i in (5,30)]) # hb_u
    subsec.append ([read_exprs[i] for i in (2,10,9)]) # hh_d
    subsec.append ([read_exprs[i] for i in (3,11,8)]) # hh_0
    subsec.append ([read_exprs[i] for i in (4,12,7)]) # hh_u
    subsec.append ([read_exprs[i] for i in (16,17,18)]) # sm
    subsec.append ([combine_TDMSystem ([read_exprs[i] for i in (19,20)])]) # phh_a_3d
    subsec.append ([combine_TDMSystem ([read_exprs[i] for i in (21,22)])]) # phh_b_3d
    subsec.append ([combine_TDMSystem ([read_exprs[i] for i in (23,24)])]) # phh_a_3u
    subsec.append ([combine_TDMSystem ([read_exprs[i] for i in (25,26)])]) # phh_b_3u
    subsec.append ([combine_TDMSystem ([read_exprs[i] for i in (31,32,33)])]) # dm_2
    subsec.append ([combine_TDMSystem ([read_exprs[i] for i in (14,15)]),
                    read_exprs[34]]) # dm_1
    subsec.append ([read_exprs[i] for i in (13,35)]) # dm_0

    return {key: val for key, val in zip (lbls, subsec)}

def standardize_m_s (eqn_dict):
    eqn_dict1 = {}
    for lbl, sector in eqn_dict.items ():
        sector1 = []
        for tdmsystem in sector:
            new_s = (2*s) - tdmsystem.exprs[0].lhs.get_s_ket ()
            new_m = (2*m) - tdmsystem.exprs[0].lhs.get_m_ket ()
            tdmsystem1 = tdmsystem.subs_m (new_m).subs_s (new_s)
            sector1.append (tdmsystem1)
        eqn_dict1[lbl] = sector1
    return eqn_dict1

def get_scale_constants (eqn_dict):
    scale = {}
    scale['h'] = TDMScaleArray ('h',
        [[eqn_dict['phh_a_3d'], eqn_dict['phh_b_3d']],
         [eqn_dict['ha_d'], eqn_dict['hb_d']],
         [eqn_dict['ha_u'], eqn_dict['hb_u']],
         [eqn_dict['phh_a_3u'], eqn_dict['phh_b_3u']]]
    )
    scale['hh'] = TDMScaleArray ('hh', [[[el,] for el in eqn_dict[key]]
                                        for key in ('hh_d', 'hh_0', 'hh_u')])
    scale['sm'] = TDMScaleArray ('sm', [[[el,],] for el in eqn_dict['sm']])
    scale['dm'] = TDMScaleArray ('dm', [[eqn_dict['dm_2']],
                                        [eqn_dict['dm_1']],
                                        [eqn_dict['dm_0']]])
    return scale

def invert_eqn_dict (eqn_dict):
    inv_eqn_dict = {}
    barlen = sum ([len (sector) for sector in eqn_dict.values ()])
    print ("Inverting equation dictionary...")
    with tqdm(total=barlen) as pbar:
        for lbl, sector in eqn_dict.items ():
            sectorI = []
            for tdmsystem in sector:
                new_s = (2*s) - tdmsystem.exprs[0].lhs.get_s_ket ()
                new_m = (2*m) - tdmsystem.exprs[0].lhs.get_m_ket ()
                tdmsystemI = tdmsystem.inv ().subs_m (new_m).subs_s (new_s)
                pbar.update (1)
                sectorI.append (tdmsystemI)
            inv_eqn_dict[lbl] = sectorI
    return inv_eqn_dict

def invert_transpose_eqns (scale):
    cnt = sum ([scalearray.count_transpose_eqns () for scalearray in scale.values ()])
    transpose_eqns = {}
    print ("Inverting transpose equations...")
    with tqdm(total=cnt) as pbar:
        for lbl, scalearray in scale.items ():
            my_transpose_eqns = scalearray.get_transpose_eqns ()
            if len (my_transpose_eqns) > 0:
                transpose_eqns[lbl] = my_transpose_eqns
                pbar.update (len (my_transpose_eqns))
    return dm2_hacks (transpose_eqns)

def dm2_hacks (transpose_eqns):
    mulliken = {'p': 'p',
                'q': 'r',
                'r': 's',
                's': 'q'}
    for key, (forward, reverse) in transpose_eqns['dm'].items ():
        if key[2] == 4:
            forward.subs_labels_(mulliken)
            forward.insert_transpose_(1,2,(1,0,3,2))
            forward.normal_order_labels_(spin_priority=False,
                                         keep_particles_together=True)
            reverse.subs_labels_(mulliken)
            reverse.insert_transpose_(1,2,(1,0,3,2))
            reverse.normal_order_labels_(spin_priority=False,
                                         keep_particles_together=True)
    mulliken = {'p': 'q',
                'q': 'r',
                'r': 'p'}
    for key, (forward, reverse) in transpose_eqns['h'].items ():
        if key[2] == 3:
            forward.subs_labels_(mulliken)
            reverse.subs_labels_(mulliken)
    return transpose_eqns

if __name__=='__main__':
    import sys, hashlib
    from pyscf.lib.misc import repo_info
    from datetime import datetime
    mrh_path = os.path.abspath (os.path.join (topdir, '..', '..', '..'))
    mrh_info = repo_info (mrh_path)
    file_docstring = "'''" + _file_docstring.format (
        executable=sys.executable,
        filepath=os.path.abspath(__file__),
        mrh_path=repo_info(mrh_path)['path'],
        mrh_git=repo_info(mrh_path)['git'],
        datetime=datetime.now ()
    ) + "'''\n"
    eqn_dict = get_eqn_dict ()
    #inv_eqn_dict = invert_eqn_dict (eqn_dict)
    eqn_dict = standardize_m_s (eqn_dict)
    scale = get_scale_constants (eqn_dict)
    transpose_eqns = invert_transpose_eqns (scale)
    fbase = os.path.splitext (os.path.basename (__file__))[0]
    fname_tex = fbase + '.tex'
    fname_py = fbase + '.generated.py'
    fname_manual_py = os.path.join (topdir, '_manual_code.py')
    with open (fname_tex, 'w') as f:
        f.write (latex_header)
        f.write ('\\section{TDM scaling constants}\n\n')
        print ("=================== TDM scaling constants ===================")
        for lbl, scalearray in scale.items ():
            print (scalearray)
            f.write (scalearray.latex ())
        f.write ('\\section{TDM transpose equations}\n\n')
        print ("=================== TDM transpose equations ===================")
        for lbl, my_transpose_eqns in transpose_eqns.items ():
            print ("------------------ " + lbl + " ------------------")
            lbl_latex = lbl.replace ('_', '\\_')
            for key, (read_eq, write_eq) in my_transpose_eqns.items ():
                print ("Read " + str(key) + ":")
                print (read_eq)
                for key2, val in read_eq.get_abs_m_eq_s_cases ().items ():
                    print ("Read special case m = " + str (key2) + ":")
                    print (val)
                print ("Write " + str(key) + ":")
                print (write_eq)
                f.write ('{}, {} read:\n'.format (lbl_latex, key))
                f.write (read_eq.latex () + '\n\n')
                f.write ('{}, {} write:\n'.format (lbl_latex, key))
                f.write (write_eq.latex () + '\n\n')
        f.write ('\n\n\\end{document}')
    with open (fname_manual_py, 'r') as f:
        manual_code = f.read ()
    with open (fname_py, 'w') as f:
        f.write (file_docstring)
        f.write (manual_code)
        for scalearray in scale.values ():
            f.write (scalearray.get_highm_civecs_code ())
            f.write (scalearray.get_scale_code ())
        f.write (scale['h'].get_mdown_code (1, 'h', transpose_eqns=transpose_eqns['h']))
        f.write (scale['h'].get_mdown_code (3, 'phh', transpose_eqns=transpose_eqns['h']))
        f.write (scale['hh'].get_mdown_code (2, 'hh', transpose_eqns=transpose_eqns['hh']))
        f.write (scale['sm'].get_mdown_code (2, 'sm'))
        f.write (scale['dm'].get_mdown_code (2, 'dm1', transpose_eqns=transpose_eqns['dm']))
        f.write (scale['dm'].get_mdown_code (4, 'dm2', transpose_eqns=transpose_eqns['dm']))
    with open (fname_py, 'rb') as f:
        filebytes = f.read ()
    print ("The SHA-256 hash of main.generated.py is", hashlib.sha256 (filebytes).hexdigest ())


