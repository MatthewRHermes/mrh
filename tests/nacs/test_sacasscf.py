import numpy as np
from pyscf import gto, scf, df, mcscf, lib
from mrh.my_pyscf.grad.sacasscf_nacs import NonAdiabaticCouplings
import unittest


def diatomic(atom1, atom2, r, basis, ncas, nelecas, nstates,
             charge=None, spin=None, symmetry=False, cas_irrep=None):
    xyz = '{:s} 0.0 0.0 0.0; {:s} {:.3f} 0.0 0.0'.format(atom1, atom2, r)
    mol = gto.M(atom=xyz, basis=basis, charge=charge, spin=spin,
                symmetry=symmetry, verbose=0, output='/dev/null')
    #mol = gto.M (atom=xyz, basis=basis, charge=charge, spin=spin, symmetry=symmetry, verbose=3)

    mf = scf.RHF(mol)

    mc = mcscf.CASSCF(mf.run(), ncas, nelecas).set(natorb=True)

    if spin is not None:
        s = spin*0.5

    else:
        s = (mol.nelectron % 2)*0.5

    mc.fix_spin_(ss=s*(s+1), shift=1)
    mc = mc.state_average([1.0/float(nstates), ]*nstates)
    mc.conv_tol = mc.conv_tol_diabatize = 1e-12
    mo = None

    if symmetry and (cas_irrep is not None):
        mo = mc.sort_mo_by_irrep(cas_irrep)

    mc.kernel(mo)

    return NonAdiabaticCouplings(mc)


def tearDownModule():
    global diatomic
    del diatomic


class KnownValues(unittest.TestCase):

    def test_nac_lih_sa2casscf22_sto3g(self):
        # z_orb:    yes
        # z_ci:     yes
        # z_is:     yes
        mc_grad = diatomic('Li', 'H', 1.5, 'STO-3G', 2, 2, 2)

        # OpenMolcas v22.02
        de_ref = np.array([[1.83701578323929E-01, -6.91459741744125E-02],
                           [9.14840490109073E-02, -9.14840490109074E-02]])
        for i in range(2):
            with self.subTest(use_etfs=bool(i)):
                de = mc_grad.kernel(state=(0, 1), use_etfs=bool(i))[:, 0]
                de *= np.sign(de[0]) * np.sign(de_ref[i, 0])
                # TODO: somehow confirm sign convention
                self.assertAlmostEqual(de[0], de_ref[i, 0], 5)
                self.assertAlmostEqual(de[1], de_ref[i, 1], 5)

    def test_grad_h2_sa3casscf22_sto3g(self):
        # z_orb:    no
        # z_ci:     no
        # z_is:     no
        mc_grad = diatomic('H', 'H', 1.3, 'STO-3G', 2, 2, 3)

        # OpenMolcas v22.02
        de_ref = np.array([[2.24611972496342E-01, 2.24611972496342E-01],
                           [-3.91518173397213E-18, 3.91518173397213E-18]])
        for i in range(2):
            with self.subTest(use_etfs=bool(i)):
                de = mc_grad.kernel(state=(0, 1), use_etfs=bool(i))[:, 0]
                de *= np.sign(de[0]) * np.sign(de_ref[i, 0])
                # TODO: somehow confirm sign convention
                self.assertAlmostEqual(de[0], de_ref[i, 0], 5)
                self.assertAlmostEqual(de[1], de_ref[i, 1], 5)

    def test_grad_h2_sa2casscf22_sto3g(self):
        # z_orb:    no
        # z_ci:     yes
        # z_is:     no
        mc_grad = diatomic('H', 'H', 1.3, 'STO-3G', 2, 2, 2)

        # OpenMolcas v22.02
        de_ref = np.array([[2.24611972496342E-01, 2.24611972496342E-01],
                           [2.39916167049495E-18, -2.39916167049495E-18]])
        for i in range(2):
            with self.subTest(use_etfs=bool(i)):
                de = mc_grad.kernel(state=(0, 1), use_etfs=bool(i))[:, 0]
                de *= np.sign(de[0]) * np.sign(de_ref[i, 0])
                # TODO: somehow confirm sign convention
                self.assertAlmostEqual(de[0], de_ref[i, 0], 5)
                self.assertAlmostEqual(de[1], de_ref[i, 1], 5)

    def test_grad_h2_sa3casscf22_631g(self):
        # z_orb:    yes
        # z_ci:     no
        # z_is:     no
        mc_grad = diatomic('H', 'H', 1.3, '6-31G', 2, 2, 3)

        # OpenMolcas v22.02
        de_ref = np.array([[-2.61261687627056E-01, -2.61264418066149E-01],
                           [1.36521954655411E-06, -1.36521954652113E-06]])

        for i in range(2):
            with self.subTest(use_etfs=bool(i)):
                de = mc_grad.kernel(state=(0, 1), use_etfs=bool(i))[:, 0]
                de *= np.sign(de[0]) * np.sign(de_ref[i, 0])
                # TODO: somehow confirm sign convention
                self.assertAlmostEqual(de[0], de_ref[i, 0], 5)
                self.assertAlmostEqual(de[1], de_ref[i, 1], 5)

    def test_grad_h2_cms2ftlda22_631g(self):
        # z_orb:    yes
        # z_ci:     yes
        # z_is:     no
        mc_grad = diatomic('H', 'H', 1.3, '6-31G', 2, 2, 2)

        # OpenMolcas v22.02
        de_ref = np.array([[2.63335711494442E-01, 2.63335711494442E-01],
                           [-9.19189316184818E-17, 9.19189316184818E-17]])

        for i in range(2):
            with self.subTest(use_etfs=bool(i)):
                de = mc_grad.kernel(state=(0, 1), use_etfs=bool(i))[:, 0]
                de *= np.sign(de[0]) * np.sign(de_ref[i, 0])
                # TODO: somehow confirm sign convention
                self.assertAlmostEqual(de[0], de_ref[i, 0], 5)
                self.assertAlmostEqual(de[1], de_ref[i, 1], 5)

    # def test_grad_lih_cms2ftlda44_sto3g (self):
    #    # z_orb:    no
    #    # z_ci:     yes
    #    # z_is:     yes
    #    mc_grad = diatomic ('Li', 'H', 1.8, 'ftLDA,VWN3', 'STO-3G', 4, 4, 2, symmetry=True, cas_irrep={'A1': 4})
    #    de_ref = [0.0659740768, -0.005995224082]
    #    # Numerical from this software
    #    for i in range (2):
    #     with self.subTest (state=i):
    #        de = mc_grad.kernel (state=i) [1,0] / BOHR
    #        self.assertAlmostEqual (de, de_ref[i], 6)

    def test_grad_lih_cms3ftlda22_sto3g(self):
        # z_orb:    yes
        # z_ci:     no
        # z_is:     yes
        mc_grad = diatomic('Li', 'H', 2.5, 'STO-3G', 2, 2, 3)

        # OpenMolcas v22.02
        de_ref = np.array([[2.68017988841929E-01, -6.48495477749689E-02],
                           [1.24872843191788E-01, -1.24872843191788E-01]])

        for i in range(2):
            with self.subTest(use_etfs=bool(i)):
                de = mc_grad.kernel(state=(0, 1), use_etfs=bool(i))[:, 0]
                de *= np.sign(de[0]) * np.sign(de_ref[i, 0])
                # TODO: somehow confirm sign convention
                self.assertAlmostEqual(de[0], de_ref[i, 0], 5)
                self.assertAlmostEqual(de[1], de_ref[i, 1], 5)

    # Hennefarth: I don't believe we need this test since it is the same as the first one, but with density fitting.
    # def test_grad_lih_cms2ftlda22_sto3g_df (self):
    #    # z_orb:    yes
    #    # z_ci:     yes
    #    # z_is:     yes
    #    mc_grad = diatomic ('Li', 'H', 2.5, 'ftLDA,VWN3', 'STO-3G', 2, 2, 2, density_fit=True)
    #    de_ref = [0.1074553399, 0.03956955205]
    #    # Numerical from this software
    #    for i in range (2):
    #     with self.subTest (state=i):
    #        de = mc_grad.kernel (state=i) [1,0] / BOHR
    #        self.assertAlmostEqual (de, de_ref[i], 6)


if __name__ == "__main__":
    print("Full Tests for SA-CASSCF non-adiabatic couplings of diatomic molecules")
    unittest.main()
