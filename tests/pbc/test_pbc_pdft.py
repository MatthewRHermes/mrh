import unittest
import numpy as np

from pyscf.pbc import gto, scf, dft

from pyscf.csf_fci import csf_solver as mol_csf_solver
from mrh.my_pyscf.pbc.fci import csf_solver
from mrh.my_pyscf.pbc.mcpdft._dms import dm2_cumulant_complex
from mrh.my_pyscf.pbc.mcscf import avas

# Author: Bhavnesh Jangid

'''
Test cases for the PBC PDFT module.
Test-1: Gamma point MC-PDFT
Test-2: k-MCPDFT in the limit of single determinant should match k-DFT results.
Test-3: k-MC-PDFT results should match the supercell MC-PDFT results for
    the same system computed with a supercell of the same size as the k-mesh. Provided
    the grids and other details are the same.
Test-4: The complex cumulant should be covariant under orbital rotations.
'''

my_grids = {'prune': False,
            'radii_adjust': None,
            'level': 6,}


def build_cell():
    cell = gto.Cell()
    cell.a=np.diag([2.5, 17.500000000, 17.500000000])
    cell.atom='''
    H 0 0 0
    H 0.74 0 0
    '''
    cell.basis = '631G'
    cell.verbose = 0
    cell.precision = 1e-14
    cell.output = '/dev/null'
    cell.build()
    return cell

class KnownValues(unittest.TestCase):
    def test_kmcpdft_cumulant(self):
        rng = np.random.default_rng(12)
        norb = 4

        x = rng.normal(size=(norb, norb))
        x = x + 1j * rng.normal(size=x.shape)
        u = np.linalg.qr(x)[0]

        dm1s = rng.normal(size=(2, norb, norb))
        dm1s = dm1s + 1j * rng.normal(size=dm1s.shape)
        dm1s = dm1s + dm1s.swapaxes(-1, -2).conj()

        dm2 = rng.normal(size=(norb,) * 4)
        dm2 = dm2 + 1j * rng.normal(size=dm2.shape)

        dm1s_rot = np.einsum(
            "pi,spq,qj->sij", u, dm1s, u.conj(), optimize=True
        )
        dm2_rot = np.einsum(
            "pi,qj,pqrs,rk,sl->ijkl",
            u.conj(), u, dm2, u.conj(), u, optimize=True,
        )

        cumulant = dm2_cumulant_complex(dm2, dm1s)
        cumulant_rot = dm2_cumulant_complex(dm2_rot, dm1s_rot)
        cumulant_ref = np.einsum(
            "pi,qj,pqrs,rk,sl->ijkl",
            u.conj(), u, cumulant, u.conj(), u, optimize=True,
        )

        np.testing.assert_allclose(
            cumulant_rot, cumulant_ref, atol=1e-12, rtol=1e-12
        )

    def test_mcpdft_gamma_point(self):
        from mrh.my_pyscf.pbc import mcpdft
        cell = gto.M(a = np.eye(3)*5,
        atom = '''
            H         -6.37665        2.20769        3.00000
            H         -5.81119        2.63374        3.00000
        ''',
        basis = '6-31g',
        verbose = 1, max_memory=10000)
        cell.output = '/dev/null'
        cell.build()

        mf = scf.RHF(cell).density_fit() # GDF
        mf.exxdiv = None
        mf.kernel()

        ks = dft.RKS(cell).density_fit()
        ks.xc = 'pbe'
        ks.exxdiv = None
        ks.max_cycle = 0
        emf = ks.kernel(mf.make_rdm1())

        mc = mcpdft.CASCI(mf,'tPBE', 1, 2)
        ecasci = mc.kernel(mf.mo_coeff)[0]

        self.assertAlmostEqual (emf, ecasci, 7)

        with self.assertRaisesRegex(NotImplementedError, "periodic HF"):
            mcpdft.CASCI(ks, 'tPBE', 1, 2)

    def test_kmcpdft_limit_to_single_determinant(self):
        from mrh.my_pyscf.pbc import mcpdft

        cell = build_cell()

        kmesh = [3, 1, 1]
        kpts = cell.make_kpts(kmesh, wrap_around=True)

        kmf = scf.KRHF(cell, kpts=kpts).density_fit()
        kmf.max_cycle = 100
        kmf.exxdiv = None
        kmf.conv_tol = 1e-10
        kmf.kernel()

        e_khf = kmf.e_tot

        def compute_dft_energy(kmf, kpts, xc='pbe'):
            kdft = dft.KRKS(cell, kpts=kpts).density_fit() # GDF
            kdft.xc = xc
            kdft.exxdiv = None
            kdft.max_cycle = 0
            kdft.kernel(kmf.make_rdm1())
            return kdft.e_tot

        e_klda = compute_dft_energy(kmf, kpts, xc='lda')
        e_kpbe = compute_dft_energy(kmf, kpts, xc='pbe')
        e_km06l = compute_dft_energy(kmf, kpts, xc='m06l')

        mo_coeff = avas.kernel(kmf, ['H 1s'], minao=cell.basis)[2]

        if np.prod(kmesh) == 1:
            # For single kpts making the mo_coeff compatible with kmc kernel
            mo_coeff = mo_coeff[None, :, :].astype(np.complex128)

        ncas = 1
        nelec = (1, 1)

        kmc = mcpdft.KCASSCF(kmf, 'tPBE', ncas, nelec)
        kmc.kpts = kpts
        kmc.kmesh = kmesh
        kmc.fcisolver = csf_solver(cell, smult=abs(nelec[1] - nelec[0]) + 1)
        kmc.max_cycle_macro = 50
        kmc.kernel(mo_coeff)

        assert kmc.converged, "k-MC-SCF did not converge"

        e_kmcscf = kmc.e_mcscf
        mo_coeff = kmc.mo_coeff.copy()

        def compute_kmcpdft_energy(kmf, kpts, kmesh, mo_coeff, ot='tPBE',
                                   ncas=1, nelec=(1, 1)):
            kmc = mcpdft.KCASCI(kmf, ot, ncas, nelec)
            kmc.kpts = kpts
            kmc.kmesh = kmesh
            kmc.fcisolver = csf_solver(cell, smult=abs(nelec[1] - nelec[0]) + 1)
            kmc.kernel(mo_coeff.copy())
            return kmc.e_tot

        e_ktlda = compute_kmcpdft_energy(kmf, kpts, kmesh, mo_coeff, ot='tLDA',
                                         ncas=ncas, nelec=nelec)
        e_ktpbe = compute_kmcpdft_energy(kmf, kpts, kmesh, mo_coeff, ot='tPBE',
                                         ncas=ncas, nelec=nelec)
        e_ktm06l = compute_kmcpdft_energy(kmf, kpts, kmesh, mo_coeff, ot='tM06L',
                                          ncas=ncas, nelec=nelec)

        self.assertAlmostEqual(e_khf, e_kmcscf, 7)
        self.assertAlmostEqual(e_klda, e_ktlda, 7)
        self.assertAlmostEqual(e_kpbe, e_ktpbe, 7)
        self.assertAlmostEqual(e_km06l, e_ktm06l, 7)

    def test_kmcpdft_with_supercell(self):
        from mrh.my_pyscf.pbc import mcpdft

        cell = build_cell()

        kmesh = [3, 1, 1]
        kpts = cell.make_kpts(kmesh, wrap_around=True)

        kmf = scf.KRHF(cell, kpts=kpts).density_fit()
        kmf.max_cycle = 100
        kmf.exxdiv = None
        kmf.conv_tol = 1e-10
        kmf.kernel()

        e_khf = kmf.e_tot

        mo_coeff = avas.kernel(kmf, ['H 1s'], minao=cell.basis)[2]

        if np.prod(kmesh) == 1:
            # For single kpts making the mo_coeff compatible with kmc kernel
            mo_coeff = mo_coeff[None, :, :].astype(np.complex128)

        ncas = 2
        nelec = (1, 1)

        kmc = mcpdft.KCASSCF(kmf, 'tPBE', ncas, nelec,
                             grids_attr=my_grids)
        kmc.kpts = kpts
        kmc.kmesh = kmesh
        kmc.fcisolver = csf_solver(cell, smult=abs(nelec[1] - nelec[0]) + 1)
        kmc.max_cycle_macro = 50
        kmc.conv_tol = 1e-10
        kmc.kernel(mo_coeff)

        assert kmc.converged, "k-MC-SCF did not converge"

        e_kmcscf = kmc.e_mcscf
        mo_coeff = kmc.mo_coeff.copy()

        def compute_kmcpdft_energy(kmf, kpts, kmesh, mo_coeff, ot='tPBE',
                                   ncas=2, nelec=(1, 1)):
            kmc = mcpdft.KCASCI(kmf, ot, ncas, nelec,
                                grids_attr=my_grids)
            kmc.kpts = kpts
            kmc.kmesh = kmesh
            kmc.fcisolver = csf_solver(cell, smult=abs(nelec[1] - nelec[0]) + 1)
            kmc.kernel(mo_coeff.copy())
            return kmc.e_tot

        e_ktlda = compute_kmcpdft_energy(kmf, kpts, kmesh, mo_coeff, ot='tLDA',
                                         ncas=ncas, nelec=nelec)
        e_ktpbe = compute_kmcpdft_energy(kmf, kpts, kmesh, mo_coeff, ot='tPBE',
                                         ncas=ncas, nelec=nelec)
        e_ktm06l = compute_kmcpdft_energy(kmf, kpts, kmesh, mo_coeff, ot='tM06L',
                                          ncas=ncas, nelec=nelec)
        e_kpbe0 = compute_kmcpdft_energy(kmf, kpts, kmesh, mo_coeff, ot='tPBE0',
                                         ncas=ncas, nelec=nelec)

        from pyscf.pbc import tools
        scell = tools.super_cell(cell, kmesh)

        mf = scf.RHF(scell).density_fit()
        mf.max_cycle = 100
        mf.exxdiv = None
        mf.conv_tol = 1e-10
        mf.kernel()

        e_hf = mf.e_tot

        assert mf.converged, "SCF did not converge for the supercell"

        nkpts = np.prod(kmesh)
        ncas = ncas * nkpts
        nelec = (nelec[0] * nkpts, nelec[1] * nkpts)

        mo_coeff = avas.kernel(mf, ['H 1s'], minao=mf.cell.basis)[2]

        mc = mcpdft.CASSCF(mf, 'tPBE', ncas, nelec,
                           grids_attr=my_grids)
        mc.max_cycle_macro = 50
        mc.conv_tol = 1e-10
        mc.fcisolver = mol_csf_solver(scell, smult=1)
        mc.kernel(mo_coeff)

        assert mc.converged, "MC-SCF did not converge"

        e_mcscf = mc.e_mcscf
        mo_coeff = mc.mo_coeff.copy()

        def compute_mcpdft_energy(mf, mo_coeff, ot='tPBE',
                                  ncas=2, nelec=(1, 1)):
            mc = mcpdft.CASCI(mf, ot, ncas, nelec,
                              grids_attr=my_grids)
            mc.fcisolver = mol_csf_solver(scell, smult=1)
            mc.kernel(mo_coeff.copy())
            return mc.e_tot

        e_tlda = compute_mcpdft_energy(mf, mo_coeff, ot='tLDA',
                                       ncas=ncas, nelec=nelec)
        e_tpbe = compute_mcpdft_energy(mf, mo_coeff, ot='tPBE',
                                       ncas=ncas, nelec=nelec)
        e_tm06l = compute_mcpdft_energy(mf, mo_coeff, ot='tM06L',
                                        ncas=ncas, nelec=nelec)
        e_tpbe0 = compute_mcpdft_energy(mf, mo_coeff, ot='tPBE0',
                                        ncas=ncas, nelec=nelec)

        self.assertAlmostEqual(e_khf.real, e_hf/nkpts, 7)
        self.assertAlmostEqual(e_kmcscf.real, e_mcscf/nkpts, 7)

        # k-MC-PDFT have slightly lower agreement with supercell MC-PDFT
        # due to possible grid differences.
        self.assertAlmostEqual(e_ktlda.real, e_tlda/nkpts, 6)
        self.assertAlmostEqual(e_ktpbe.real, e_tpbe/nkpts, 6)
        self.assertAlmostEqual(e_ktm06l.real, e_tm06l/nkpts, 6)
        self.assertAlmostEqual(e_kpbe0.real, e_tpbe0/nkpts, 6)

if __name__ == "__main__":
    print("Full Tests for PBC-PDFT (k-MC-PDFT)")
    unittest.main()
