#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import copy
import unittest
import numpy as np
from pyscf import gto, scf, mcscf, df
from pyscf.lib import chkfile
from pyscf.mcscf import avas
from pyscf.data import nist
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.mcscf import lasscf_async
from mrh.tests.lasscf.c2h4n4_struct import structure as struct
from mrh.my_pyscf import lassi
topdir = os.path.abspath (os.path.join (__file__, '..'))

au2cm = nist.HARTREE2J / nist.PLANCK / nist.LIGHT_SPEED_SI * 1e-2
def yamaguchi (e_roots, s2, nelec=6):
    '''The Yamaguchi formula for nelec unpaired electrons'''
    hs = (nelec/2.0) * ((nelec/2.0)+1.0)
    ls = hs - np.floor (hs)
    idx = np.argsort (e_roots)
    e_roots = e_roots[idx]
    s2 = s2[idx]
    idx_hs = (np.around (s2) == np.around (hs))
    assert (np.count_nonzero (idx_hs)), 'high-spin ground state not found'
    idx_hs = np.where (idx_hs)[0][0]
    e_hs = e_roots[idx_hs]
    idx_ls = (np.around (s2) == np.around (ls))
    assert (np.count_nonzero (idx_ls)), 'low-spin ground state not found'
    idx_ls = np.where (idx_ls)[0][0]
    e_ls = e_roots[idx_ls]
    j = (e_ls - e_hs) / (hs - ls)
    return j*au2cm

def setUpModule ():
    pass

def tearDownModule():
    pass

class KnownValues(unittest.TestCase):

    #def test_c2h4n4_3frag (self):
    #    # str
    #    mol = struct (2.0, 2.0, '6-31g', symmetry=False)
    #    mol.output = '/dev/null'
    #    mol.verbose = 0
    #    mol.spin = 8
    #    mol.build ()
    #    mf = scf.RHF (mol).run () 
    #    las = LASSCF (mf, (4,2,4), ((2,2),(1,1),(2,2)), spin_sub=(1,1,1))
    #    mo_coeff = las.localize_init_guess ([[0,1,2],[3,4,5,6],[7,8,9]])
    #    las.kernel (mo_coeff)
    #    lsi = lassi.LASSIS (las).run ()
    #    e_str = lsi.e_roots[0]
    #    with self.subTest ('str converged'):
    #        self.assertTrue (lsi.converged)
    #    # equil
    #    mol = struct (0.0, 0.0, '6-31g', symmetry=False)
    #    mol.spin = 0
    #    mol.verbose = 0
    #    mol.output = '/dev/null'
    #    mol.build ()
    #    mf = scf.RHF (mol).run ()
    #    las = LASSCF (mf, (4,2,4), ((2,2),(1,1),(2,2)), spin_sub=(1,1,1))
    #    mo_coeff = las.sort_mo ([7,8,16,18,22,23,24,26,33,34])
    #    mo_coeff = las.localize_init_guess ([[0,1,2],[3,4,5,6],[7,8,9]], mo_coeff=mo_coeff)
    #    las.kernel (mo_coeff)
    #    lsi = lassi.LASSIS (las).run ()
    #    e_equil = lsi.e_roots[0]
    #    with self.subTest ('equil converged'):
    #        self.assertTrue (lsi.converged)
    #    # test
    #    de = 1000 * (e_str - e_equil)
    #    self.assertAlmostEqual (de, 208.27109298022606, 1)

    #def test_c2h4n4_3frag_davidson (self):
    #    # str
    #    mol = struct (2.0, 2.0, '6-31g', symmetry=False)
    #    mol.output = '/dev/null'
    #    mol.verbose = 0
    #    mol.spin = 8
    #    mol.build ()
    #    mf = scf.RHF (mol).run () 
    #    las = LASSCF (mf, (4,2,4), ((2,2),(1,1),(2,2)), spin_sub=(1,1,1))
    #    mo_coeff = las.localize_init_guess ([[0,1,2],[3,4,5,6],[7,8,9]])
    #    las.kernel (mo_coeff)
    #    lsi = lassi.LASSIS (las).run (davidson_only=True)
    #    e_str = lsi.e_roots[0]
    #    with self.subTest ('str converged'):
    #        self.assertTrue (lsi.converged)
    #    # equil
    #    mol = struct (0.0, 0.0, '6-31g', symmetry=False)
    #    mol.spin = 0
    #    mol.verbose = 0
    #    mol.output = '/dev/null'
    #    mol.build ()
    #    mf = scf.RHF (mol).run ()
    #    las = LASSCF (mf, (4,2,4), ((2,2),(1,1),(2,2)), spin_sub=(1,1,1))
    #    mo_coeff = las.sort_mo ([7,8,16,18,22,23,24,26,33,34])
    #    mo_coeff = las.localize_init_guess ([[0,1,2],[3,4,5,6],[7,8,9]], mo_coeff=mo_coeff)
    #    las.kernel (mo_coeff)
    #    lsi = lassi.LASSIS (las).run (davidson_only=True)
    #    e_equil = lsi.e_roots[0]
    #    with self.subTest ('equil converged'):
    #        self.assertTrue (lsi.converged)
    #    # test
    #    de = 1000 * (e_str - e_equil)
    #    self.assertAlmostEqual (de, 208.27109298022606, 1)

    #def test_c2h4n4_2frag (self):
    #    # str
    #    mol = struct (2.0, 2.0, '6-31g', symmetry=False)
    #    mol.output = '/dev/null'
    #    mol.verbose = 0
    #    mol.spin = 8
    #    mol.build ()
    #    mf = scf.RHF (mol).run () 
    #    las = lasscf_async.LASSCF (mf, (5,5), ((3,2),(2,3)))
    #    las = las.state_average ([0.5,0.5],
    #        spins=[[1,-1],[-1,1]],
    #        smults=[[2,2],[2,2]],
    #        charges=[[0,0],[0,0]],
    #        wfnsyms=[[0,0],[0,0]])
    #    mo = las.set_fragments_((list (range (5)), list (range (5,10))))
    #    las.kernel (mo)
    #    mo_coeff = las.mo_coeff
    #    las = lasscf_async.LASSCF (mf, (5,5), ((3,2),(2,3)))
    #    las.lasci_(mo_coeff)
    #    lsi = lassi.LASSIS (las).run ()
    #    e_str = lsi.e_roots[0]
    #    with self.subTest ('str converged'):
    #        self.assertTrue (lsi.converged)
    #    # equil
    #    mol = struct (0.0, 0.0, '6-31g', symmetry=False)
    #    mol.spin = 0
    #    mol.verbose = 0
    #    mol.output = '/dev/null'
    #    mol.build ()
    #    mf = scf.RHF (mol).run ()
    #    las = lasscf_async.LASSCF (mf, (5,5), ((3,2),(2,3)))
    #    las = las.state_average ([0.5,0.5],
    #        spins=[[1,-1],[-1,1]],
    #        smults=[[2,2],[2,2]],
    #        charges=[[0,0],[0,0]],
    #        wfnsyms=[[0,0],[0,0]])
    #    mo = las.sort_mo ([7,8,16,18,22,23,24,26,33,34])
    #    mo = las.set_fragments_((list (range (5)), list (range (5,10))), mo)
    #    las.kernel (mo)
    #    mo_coeff = las.mo_coeff
    #    las = lasscf_async.LASSCF (mf, (5,5), ((3,2),(2,3)))
    #    las.lasci_(mo_coeff)
    #    lsi = lassi.LASSIS (las).run ()
    #    e_equil = lsi.e_roots[0]
    #    with self.subTest ('equil converged'):
    #        self.assertTrue (lsi.converged)
    #    # test
    #    de = 1000 * (e_str - e_equil)
    #    self.assertAlmostEqual (de, 190.6731766549683, 1)

    #def test_c2h4n4_2frag_davidson (self):
    #    # str
    #    mol = struct (2.0, 2.0, '6-31g', symmetry=False)
    #    mol.output = '/dev/null'
    #    mol.verbose = 0
    #    mol.spin = 8
    #    mol.build ()
    #    mf = scf.RHF (mol).run () 
    #    las = lasscf_async.LASSCF (mf, (5,5), ((3,2),(2,3)))
    #    las = las.state_average ([0.5,0.5],
    #        spins=[[1,-1],[-1,1]],
    #        smults=[[2,2],[2,2]],
    #        charges=[[0,0],[0,0]],
    #        wfnsyms=[[0,0],[0,0]])
    #    mo = las.set_fragments_((list (range (5)), list (range (5,10))))
    #    las.kernel (mo)
    #    mo_coeff = las.mo_coeff
    #    las = lasscf_async.LASSCF (mf, (5,5), ((3,2),(2,3)))
    #    las.lasci_(mo_coeff)
    #    lsi = lassi.LASSIS (las).run (davidson_only=True)
    #    e_str = lsi.e_roots[0]
    #    with self.subTest ('str converged'):
    #        self.assertTrue (lsi.converged)
    #    # equil
    #    mol = struct (0.0, 0.0, '6-31g', symmetry=False)
    #    mol.spin = 0
    #    mol.verbose = 0
    #    mol.output = '/dev/null'
    #    mol.build ()
    #    mf = scf.RHF (mol).run ()
    #    las = lasscf_async.LASSCF (mf, (5,5), ((3,2),(2,3)))
    #    las = las.state_average ([0.5,0.5],
    #        spins=[[1,-1],[-1,1]],
    #        smults=[[2,2],[2,2]],
    #        charges=[[0,0],[0,0]],
    #        wfnsyms=[[0,0],[0,0]])
    #    mo = las.sort_mo ([7,8,16,18,22,23,24,26,33,34])
    #    mo = las.set_fragments_((list (range (5)), list (range (5,10))), mo)
    #    las.kernel (mo)
    #    mo_coeff = las.mo_coeff
    #    las = lasscf_async.LASSCF (mf, (5,5), ((3,2),(2,3)))
    #    las.lasci_(mo_coeff)
    #    lsi = lassi.LASSIS (las).run (davidson_only=True)
    #    e_equil = lsi.e_roots[0]
    #    with self.subTest ('equil converged'):
    #        self.assertTrue (lsi.converged)
    #    # test
    #    de = 1000 * (e_str - e_equil)
    #    self.assertAlmostEqual (de, 190.6731766549683, 1)

    #def test_kremer_cr2_model (self):
    #    xyz='''Cr    -1.320780000000   0.000050000000  -0.000070000000
    #    Cr     1.320770000000   0.000050000000  -0.000070000000
    #    O      0.000000000000  -0.165830000000   1.454680000000
    #    O      0.000000000000   1.342770000000  -0.583720000000
    #    O      0.000000000000  -1.176830000000  -0.871010000000
    #    H      0.000020000000   0.501280000000   2.159930000000
    #    H      0.000560000000   1.618690000000  -1.514480000000
    #    H     -0.000440000000  -2.120790000000  -0.644130000000
    #    N     -2.649800000000  -1.445690000000   0.711420000000
    #    H     -2.186960000000  -2.181980000000   1.244400000000
    #    H     -3.053960000000  -1.844200000000  -0.136070000000
    #    H     -3.367270000000  -1.005120000000   1.287210000000
    #    N     -2.649800000000   1.339020000000   0.896300000000
    #    N     -2.649800000000   0.106770000000  -1.607770000000
    #    H     -3.367270000000  -0.612160000000  -1.514110000000
    #    H     -3.053960000000   0.804320000000   1.665160000000
    #    N      2.649800000000  -1.445680000000   0.711420000000
    #    N      2.649790000000   1.339030000000   0.896300000000
    #    N      2.649800000000   0.106780000000  -1.607770000000
    #    H     -2.186970000000   2.168730000000   1.267450000000
    #    H     -3.367270000000   1.617370000000   0.226860000000
    #    H     -2.186960000000   0.013340000000  -2.511900000000
    #    H     -3.053970000000   1.039980000000  -1.529140000000
    #    H      2.186960000000  -2.181970000000   1.244400000000
    #    H      3.053960000000  -1.844190000000  -0.136080000000
    #    H      3.367270000000  -1.005100000000   1.287200000000
    #    H      2.186950000000   2.168740000000   1.267450000000
    #    H      3.053960000000   0.804330000000   1.665160000000
    #    H      3.367260000000   1.617380000000   0.226850000000
    #    H      2.186960000000   0.013350000000  -2.511900000000
    #    H      3.053960000000   1.039990000000  -1.529140000000
    #    H      3.367270000000  -0.612150000000  -1.514110000000'''
    #    basis = {'C': 'sto-3g','H': 'sto-3g','O': 'sto-3g','N': 'sto-3g','Cr': 'cc-pvdz'}
    #    mol = gto.M (atom=xyz, spin=6, charge=3, basis=basis,
    #               verbose=0, output='/dev/null')
    #    mf = scf.ROHF(mol)
    #    mf.chkfile = 'test_lassis_targets_slow.kremer_cr2_model.chk'
    #    mf.kernel ()
    #    las = LASSCF(mf,(6,6),((3,0),(0,3)),spin_sub=(4,4))
    #    try:
    #        mo_coeff = chkfile.load (las.chkfile, 'las')['mo_coeff']
    #    except (OSError, TypeError, KeyError) as e:
    #        ncas_avas, nelecas_avas, mo_coeff = avas.kernel (mf, ['Cr 3d', 'Cr 4d'], minao=mol.basis)
    #        mc_avas = mcscf.CASCI (mf, ncas_avas, nelecas_avas)
    #        mo_list = mc_avas.ncore + np.array ([5,6,7,8,9,10,15,16,17,18,19,20])
    #        mo_coeff = las.sort_mo (mo_list, mo_coeff)
    #        mo_coeff = las.localize_init_guess (([0],[1]), mo_coeff)
    #    las = lassi.spaces.spin_shuffle (las) # generate direct-exchange states
    #    las.weights = [1.0/las.nroots,]*las.nroots # set equal weights
    #    nroots_ref = las.nroots
    #    las.kernel (mo_coeff) # optimize orbitals
    #    mo_coeff = las.mo_coeff
    #    las = LASSCF(mf,(6,6),((3,0),(0,3)),spin_sub=(4,4))
    #    las.lasci_(mo_coeff)
    #    for dson in (False, True):
    #        lsi = lassi.LASSIS (las).run (davidson_only=dson, nroots_si=4)
    #        with self.subTest('convergence', davidson_only=dson):
    #            self.assertTrue (lsi.converged)
    #        with self.subTest(davidson_only=dson):
    #            self.assertAlmostEqual (yamaguchi (lsi.e_roots, lsi.s2, 6), -12.406510069940726, 2)

    def test_alfefe (self):
        xyz='''O -2.2201982441 0.3991903003 1.6944716989
        O -1.6855532446 -1.7823063217 1.4313995072
        C -2.2685178651 -0.8319550379 1.983951274
        H -2.9133420167 -1.0767285885 2.8437868053
        O 1.3882880809 2.0795561562 -1.3470856753
        O -0.7599088595 2.5809236347 -0.849227704
        C 0.3465674686 2.753895325 -1.438835699
        H 0.3723796145 3.6176996598 -2.1230911503
        O 1.0469609081 -2.6425342 0.9613246722
        O -0.3991903003 2.2201982441 1.6944716989
        O 1.7823063217 1.6855532446 1.4313995072
        C 0.8319550379 2.2685178651 1.983951274
        H 1.0767285885 2.9133420167 2.8437868053
        O -2.0795561562 -1.3882880809 -1.3470856753
        O -2.5809236347 0.7599088595 -0.849227704
        C -2.753895325 -0.3465674686 -1.438835699
        H -3.6176996598 -0.3723796145 -2.1230911503
        Fe -0.3798741893 -1.7926033629 -0.2003377303
        O 0.6422231803 -2.237796553 -1.8929145176
        Fe 1.7926033629 0.3798741893 -0.2003377303
        O 2.237796553 -0.6422231803 -1.8929145176
        O 2.6425342 -1.0469609081 0.9613246722
        Al -1.2362716421 1.2362716421 0.350650148
        O -0.0449501954 0.0449501954 0.0127621413
        C 1.645528685 -1.645528685 -2.3571124654
        H 2.0569062984 -2.0569062984 -3.2945712916
        C 2.1609242811 -2.1609242811 1.2775841868
        H 2.7960805433 -2.7960805433 1.9182443024'''
        basis = {'C': 'sto-3g','H': 'sto-3g','O': 'sto-3g','Al': 'cc-pvdz','Fe': 'cc-pvdz'}
        mol = gto.M (atom=xyz, spin=9, charge=0, basis=basis, max_memory=10000, verbose=4,
                     output='debug_lassis_targets_slow.alfefe.log')
        mf = scf.ROHF(mol)
        mf.init_guess='chk'
        mf.chkfile='test_lassis_targets_slow.alfefe.chk'
        mf = mf.density_fit(auxbasis = df.aug_etb (mol))
        mf.max_cycle=1
        mf.kernel()
        las = LASSCF (mf, (5,5), ((5,1),(5,0)), spin_sub=(5,6))
        try:
            las.load_chk_()
            las.lasci ()
            las.kernel ()
        except (OSError, TypeError, KeyError) as e:
            raise (e)
            ncas, nelecas, mo_coeff = avas.kernel (mf, ['Fe 3d'], openshell_option=3)
            mo_coeff = las.localize_init_guess (([17],[19]), mo_coeff)
            las.kernel (mo_coeff)
        with self.subTest ('LASSCF convergence'):
            self.assertTrue (las.converged)
        with self.subTest ('same LASSCF result'):
            self.assertAlmostEqual (las.e_tot, -3955.98148841553, 6)
        mf.chkfile = None # prevent the spins flips down there from messing things up
        las2 = LASSCF (mf, (5,5), ((1,5),(5,0)), spin_sub=(5,6))
        las2.lasci_(las.mo_coeff)
        for dson in (False, True):
            lsi = lassi.LASSIS (las2).run (davidson_only=dson, nroots_si=6)
            print (lsi.e_roots)
            print (lsi.s2)
            with self.subTest('LASSI convergence', davidson_only=dson):
                self.assertTrue (lsi.converged)
            with self.subTest(davidson_only=dson):
                self.assertAlmostEqual (yamaguchi (lsi.e_roots, lsi.s2, 9), -4.885066730567389, 2)


if __name__ == "__main__":
    print("Full Tests for SA-LASSI of c2h4n4 molecule")
    unittest.main()

