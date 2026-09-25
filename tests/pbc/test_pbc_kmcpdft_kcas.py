"""Integration and reference-energy tests for periodic k-MC-PDFT."""

import unittest
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
from pyscf.lib import logger
from pyscf.pbc import gto, scf

from mrh.my_pyscf.pbc import mcscf, mcpdft as pbc_mcpdft
from mrh.my_pyscf.pbc.mcpdft import _dms as pbc_dms
from mrh.my_pyscf.pbc.mcpdft import kmcpdft


THREE_K_MCPDFT_NEUTRAL_REFERENCE_ENERGIES = (
    -1.247116900112817,
    -1.038898418365533,
    -1.038898418365532,
)
THREE_K_MCPDFT_CHARGED_REFERENCE_ENERGIES = {
    1: (
        -0.990750161948622,
        -1.023337708981894,
        -1.023337708981894,
    ),
    -1: (
        -0.909551446536919,
        -1.075866509710293,
        -1.075866509710295,
    ),
}


def _make_reference_periodic_h2():
    """Build the three-k-point H2 cell used for fixed energy references."""
    intra_h, inter_h, vacuum = 0.74, 1.5, 17.5
    cell = gto.Cell()
    cell.a = np.diag([intra_h + inter_h, intra_h + inter_h, vacuum])
    cell.atom = [
        ["H", (0.0, 0.0, vacuum / 2.0)],
        ["H", (intra_h, 0.0, vacuum / 2.0)],
    ]
    cell.basis = "STO-6G"
    cell.unit = "Angstrom"
    cell.ke_cutoff = 100
    cell.precision = 1e-10
    cell.verbose = 0
    cell.build()
    return cell


class KCASPDFTReferenceEnergyTests(unittest.TestCase):
    """Compare H2 k-MC-PDFT energies with fixed kCASCI-sector references.

    Each value is the per-cell tPBE total energy evaluated on a neutral or
    charged kCASCI wavefunction. The references were generated with git commit
    45786a0a8e6c410dc2168058e1ed037ddb6d9b74 using CAS(2,2), grid level 1,
    and a three-point [3, 1, 1] k-mesh.
    """

    @classmethod
    def setUpClass(cls):
        cls.cell = _make_reference_periodic_h2()
        cls.kmesh = [3, 1, 1]
        cls.kpts = cls.cell.make_kpts(cls.kmesh, wrap_around=True)
        cls.kmf = scf.KRHF(cls.cell, kpts=cls.kpts).density_fit(
            auxbasis="def2-svp-jkfit",
        )
        cls.kmf.max_cycle = 1000
        cls.kmf.exxdiv = None
        cls.kmf.conv_tol = 1e-10
        cls.kmf.verbose = 0
        cls.kmf.kernel()
        if not cls.kmf.converged:
            raise RuntimeError("Three-k-point periodic H2 KRHF did not converge")
        cls.mo_coeff = np.asarray(cls.kmf.mo_coeff)

    def make_pdft(self, target_k=None, charge=None):
        mc = pbc_mcpdft.KCASCI(
            self.kmf, "tPBE", 2, 2, ncore=0,
            target_k=target_k, charge=charge,
            grids_attr={"level": 1},
        )
        mc.kmesh = self.kmesh
        mc.kpts = self.kpts
        mc.verbose = 0
        mc.fcisolver.verbose = 0
        mc.canonicalization = False
        return mc

    def test_neutral_energies_for_all_target_k(self):
        for target_k, reference in enumerate(
                THREE_K_MCPDFT_NEUTRAL_REFERENCE_ENERGIES):
            with self.subTest(target_k=target_k):
                mc = self.make_pdft(target_k=target_k)
                mc.kernel(self.mo_coeff)
                self.assertTrue(np.isrealobj(mc.e_tot))
                self.assertAlmostEqual(mc.e_tot, reference, places=7)

    def test_charged_energies_for_all_target_k(self):
        for charge, references in (
                THREE_K_MCPDFT_CHARGED_REFERENCE_ENERGIES.items()):
            with self.subTest(charge=charge):
                mc = self.make_pdft(charge=charge)
                mc.kernel(self.mo_coeff)
                self.assertEqual(np.asarray(mc.e_tot).shape, (3,))
                self.assertEqual(
                    [result["target_k"]
                     for result in mc.charged_pdft_results],
                    [0, 1, 2],
                )
                self.assertTrue(np.isrealobj(mc.e_tot))
                for actual, reference in zip(mc.e_tot, references):
                    self.assertAlmostEqual(actual, reference, places=7)


@lru_cache(maxsize=1)
def build_periodic_h2():
    cell = gto.Cell()
    cell.a = np.diag([2.24, 2.24, 12.0])
    cell.atom = [
        ["H", (0.0, 0.0, 6.0)],
        ["H", (0.74, 0.0, 6.0)],
    ]
    cell.basis = "sto-6g"
    cell.unit = "Angstrom"
    cell.precision = 1e-9
    cell.verbose = 0
    cell.build()

    kmesh = (2, 1, 1)
    kpts = cell.make_kpts(kmesh, wrap_around=True)
    kmf = scf.KRHF(cell, kpts=kpts).density_fit()
    kmf.exxdiv = None
    kmf.conv_tol = 1e-9
    kmf.verbose = 0
    kmf.kernel()
    if not kmf.converged:
        raise RuntimeError("Periodic H2 KRHF did not converge")
    return cell, kmf, kmesh, np.asarray(kmf.mo_coeff)


class KCASPDFTWavefunctionEnergyTests(unittest.TestCase):

    def test_full_hybrid_reconstructs_kcasci_energy(self):
        _, kmf, kmesh, mo_coeff = build_periodic_h2()
        kpts = kmf.kpts

        numint = SimpleNamespace(
            rsh_and_hybrid_coeff=lambda otxc, spin:
                (0.0, 0.0, (1.0, 1.0)),
        )
        ot = SimpleNamespace(_numint=numint, otxc="full-MC")
        for target_k in range(len(kpts)):
            with self.subTest(target_k=target_k):
                mc = mcscf.KCASCI(
                    kmf, 2, 2, ncore=0, target_k=target_k,
                )
                mc.kmesh = kmesh
                mc.verbose = 0
                mc.fcisolver.verbose = 0
                mc.canonicalization = False
                energy_kcas = mc.kernel(mo_coeff)[0]

                casdm1s = pbc_dms.make_one_casdm1s_kcas(
                    mc, mc.ci,
                )
                casdm2 = pbc_dms.make_one_casdm2_kcas(
                    mc, mc.ci,
                )
                energy_reconstructed = kmcpdft.energy_mcwfn(
                    mc, ot=ot, casdm1s=casdm1s, casdm2=casdm2,
                    verbose=logger.QUIET, rdm_representation="bloch",
                )

                np.testing.assert_allclose(
                    energy_reconstructed, energy_kcas,
                    atol=1e-9, rtol=1e-9,
                )

    def test_full_hybrid_reconstructs_charged_kcasci_sectors(self):
        _, kmf, kmesh, mo_coeff = build_periodic_h2()
        numint = SimpleNamespace(
            rsh_and_hybrid_coeff=lambda otxc, spin:
                (0.0, 0.0, (1.0, 1.0)),
        )
        ot = SimpleNamespace(_numint=numint, otxc="full-MC")

        for charge in (1, -1):
            with self.subTest(charge=charge):
                mc = mcscf.KCASCI(
                    kmf, 2, 2, ncore=0, charge=charge,
                )
                mc.kmesh = kmesh
                mc.verbose = 0
                mc.fcisolver.verbose = 0
                mc.kernel(mo_coeff)

                for result in mc.charged_results:
                    target_k = result["target_k"]
                    casdm1s = \
                        kmcpdft.make_one_casdm1s_charged_kcas(
                            mc, target_k=target_k,
                        )
                    casdm2 = \
                        kmcpdft.make_one_casdm2_charged_kcas(
                            mc, target_k=target_k,
                        )
                    energy_reconstructed = kmcpdft.energy_mcwfn(
                        mc, ot=ot, casdm1s=casdm1s, casdm2=casdm2,
                        verbose=logger.QUIET, rdm_representation="bloch",
                    )
                    np.testing.assert_allclose(
                        energy_reconstructed, result["e_tot"],
                        atol=1e-9, rtol=1e-9,
                    )


class KCASPDFTEndToEndTests(unittest.TestCase):

    grids_attr = {"level": 1}

    def make_pdft(self, ncas, target_k=None, charge=None,
                  charged_spin=None):
        _, kmf, kmesh, _ = build_periodic_h2()
        kwargs = {
            "ncore": 0,
            "grids_attr": self.grids_attr,
        }
        if target_k is not None:
            kwargs["target_k"] = target_k
        if charge is not None:
            kwargs["charge"] = charge
        if charged_spin is not None:
            kwargs["charged_spin"] = charged_spin
        mc = pbc_mcpdft.KCASCI(kmf, "tPBE", ncas, 2, **kwargs)
        mc.kpts = kmf.kpts
        mc.kmesh = kmesh
        mc.verbose = 0
        mc.fcisolver.verbose = 0
        mc.canonicalization = False
        return mc

    def test_target_k_zero_matches_conventional_kcasci_pdft(self):
        _, _, _, mo_coeff = build_periodic_h2()
        conventional = self.make_pdft(ncas=2)
        momentum = self.make_pdft(
            ncas=2, target_k=0,
        )

        conventional.kernel(mo_coeff)
        momentum.kernel(mo_coeff)

        self.assertFalse(conventional.momentum_resolved)
        self.assertTrue(momentum.momentum_resolved)
        self.assertEqual(momentum.target_k, 0)
        self.assertTrue(np.isrealobj(conventional.e_tot))
        self.assertTrue(np.isrealobj(momentum.e_tot))
        np.testing.assert_allclose(
            momentum.e_mcscf, conventional.e_mcscf,
            atol=1e-9, rtol=1e-9,
        )
        np.testing.assert_allclose(
            momentum.e_ot, conventional.e_ot,
            atol=1e-8, rtol=1e-8,
        )
        np.testing.assert_allclose(
            momentum.e_tot, conventional.e_tot,
            atol=1e-8, rtol=1e-8,
        )

    def test_nonzero_target_matches_existing_kcasci_route(self):
        _, kmf, kmesh, mo_coeff = build_periodic_h2()
        direct = self.make_pdft(
            ncas=2, target_k=1,
        )
        direct.kernel(mo_coeff)

        kcas = mcscf.KCASCI(kmf, 2, 2, ncore=0, target_k=1)
        kcas.kmesh = kmesh
        kcas.verbose = 0
        kcas.fcisolver.verbose = 0
        kcas.canonicalization = False
        kcas.kernel(mo_coeff)
        wrapped = pbc_mcpdft.KCASCI(
            kcas, "tPBE", 2, 2, ncore=0,
            grids_attr=self.grids_attr,
        )
        wrapped.verbose = 0
        wrapped.compute_pdft_energy_()

        self.assertEqual(direct.target_k, 1)
        self.assertEqual(wrapped.target_k, 1)
        self.assertEqual(wrapped.fcisolver.target_k, 1)
        np.testing.assert_allclose(
            direct.e_mcscf, wrapped.e_mcscf,
            atol=1e-9, rtol=1e-9,
        )
        np.testing.assert_allclose(
            direct.e_tot, wrapped.e_tot,
            atol=1e-8, rtol=1e-8,
        )

    def test_single_determinant_matches_conventional_pdft(self):
        _, kmf, _, mo_coeff = build_periodic_h2()
        conventional = self.make_pdft(ncas=1)
        momentum = self.make_pdft(
            ncas=1, target_k=0,
        )

        conventional.kernel(mo_coeff)
        momentum.kernel(mo_coeff)

        self.assertEqual(np.size(momentum.ci), 1)
        np.testing.assert_allclose(
            momentum.e_mcscf, kmf.e_tot,
            atol=1e-9, rtol=1e-9,
        )
        np.testing.assert_allclose(
            momentum.e_tot, conventional.e_tot,
            atol=1e-8, rtol=1e-8,
        )

    def test_charged_hole_and_particle_sector_sweeps(self):
        _, _, _, mo_coeff = build_periodic_h2()
        neutral = self.make_pdft(
            ncas=2, target_k=0,
        )
        hole = self.make_pdft(
            ncas=2, charge=1,
        )
        particle = self.make_pdft(
            ncas=2, charge=-1,
        )

        neutral.kernel(mo_coeff)
        hole.kernel(mo_coeff)
        particle.kernel(mo_coeff)

        self.assertEqual(hole.charged_nelecastot, (2, 1))
        self.assertEqual(particle.charged_nelecastot, (3, 2))
        self.assertEqual(
            [result["target_k"] for result in hole.charged_pdft_results],
            [0, 1],
        )
        self.assertEqual(
            [result["target_k"]
             for result in particle.charged_pdft_results],
            [0, 1],
        )
        np.testing.assert_allclose(
            hole.e_mcscf,
            [result["e_tot"] for result in hole.charged_results],
        )
        np.testing.assert_allclose(
            particle.e_mcscf,
            [result["e_tot"] for result in particle.charged_results],
        )

        hole_bands = hole.band_energies(neutral.e_tot)
        particle_bands = particle.band_energies(neutral.e_tot)
        for band, result in zip(hole_bands, hole.charged_pdft_results):
            np.testing.assert_allclose(
                band["energy"],
                hole.nkpts * (neutral.e_tot - result["e_tot"]),
            )
        for band, result in zip(
                particle_bands, particle.charged_pdft_results):
            np.testing.assert_allclose(
                band["energy"],
                particle.nkpts * (result["e_tot"] - neutral.e_tot),
            )

    def test_charged_explicit_sector_returns_scalar_energy(self):
        _, _, _, mo_coeff = build_periodic_h2()
        hole = self.make_pdft(
            ncas=2, charge=1, target_k=1,
        )

        hole.kernel(mo_coeff)

        self.assertEqual(hole.target_k, 1)
        self.assertEqual(len(hole.charged_results), 1)
        self.assertEqual(len(hole.charged_pdft_results), 1)
        self.assertEqual(hole.charged_pdft_results[0]["target_k"], 1)
        self.assertEqual(np.ndim(hole.e_tot), 0)
        self.assertEqual(np.ndim(hole.e_ot), 0)
        self.assertTrue(np.isrealobj(hole.e_tot))


if __name__ == "__main__":
    unittest.main()
