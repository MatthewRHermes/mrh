#!/usr/bin/env python

import unittest

import numpy as np
from scipy import linalg

from pyscf import lib
from pyscf.pbc import gto, scf

from mrh.my_pyscf.pbc import mcscf
from mrh.my_pyscf.pbc.mcscf import avas
from mrh.my_pyscf.pbc.mcscf import klasscf  # noqa: F401


def _copy_ci(ci):
    return [[np.array(c, copy=True) for c in roots] for roots in ci]


def _displace_ci(ci, direction, step):
    displaced = []
    for ci0_r, direction_r in zip(ci, direction):
        displaced_r = []
        for c0, d in zip(ci0_r, direction_r):
            c1 = c0 + step * d
            c1 /= np.linalg.norm(c1)
            displaced_r.append(c1)
        displaced.append(displaced_r)
    return displaced


def _make_ci_direction(ugg, seed=17):
    rng = np.random.default_rng(seed)
    trial = (
        rng.standard_normal(ugg.nvar_ci)
        + 1j * rng.standard_normal(ugg.nvar_ci)
    )
    direction = ugg.unpack_ci(trial)
    for ci0_r, direction_r in zip(ugg.ci, direction):
        for c0, d in zip(ci0_r, direction_r):
            d -= np.vdot(c0, d) * c0
    packed = ugg.pack_ci(direction)
    packed /= np.linalg.norm(packed)
    return packed, ugg.unpack_ci(packed)


def _build_reference(lattice, kmesh, atom, basis, active_labels):
    cell = gto.Cell()
    cell.a = np.diag(lattice)
    cell.atom = atom
    cell.basis = basis
    cell.unit = "Angstrom"
    cell.precision = 1e-12
    cell.ke_cutoff = 20
    cell.verbose = lib.logger.QUIET
    cell.build()

    kpts = cell.make_kpts(kmesh, wrap_around=True)
    kmf = scf.KRHF(cell, kpts=kpts).density_fit()
    kmf.exxdiv = None
    kmf.max_cycle = 0
    kmf.kernel()

    mo_coeff = avas.kernel(kmf, active_labels, minao=cell.basis)[2]
    klas = mcscf.KLASCI(kmf, 2, (1, 1), kmesh=kmesh)
    klas.conv_tol_grad = 1e-8
    klas.conv_tol_self = 1e-10
    mo_coeff = klas.localize_init_guess(
        active_labels, mo_coeff=mo_coeff,
    )
    klas.kernel(mo_coeff)

    ci = _copy_ci(klas.ci)
    ugg = klas.get_ugg(mo_coeff=mo_coeff, ci=ci)
    hop = klas.get_hop(mo_coeff=mo_coeff, ci=ci, ugg=ugg)
    hop.level_shift = 0.0
    return klas, np.asarray(mo_coeff), ci, ugg, hop


def _make_orbital_direction(klas, blocks, seed):
    ncore = klas.ncore
    nocc = ncore + klas.ncas
    nmo = klas.mo_coeff.shape[-1]
    spaces = {
        "core": slice(0, ncore),
        "active": slice(ncore, nocc),
        "virtual": slice(nocc, nmo),
    }
    rng = np.random.default_rng(seed)
    kappa = np.zeros((klas.nkpts, nmo, nmo), dtype=np.complex128)
    for k in range(klas.nkpts):
        for block in blocks:
            left_name, right_name = block.split("-")
            left = spaces[left_name]
            right = spaces[right_name]
            shape = (
                len(range(*right.indices(nmo))),
                len(range(*left.indices(nmo))),
            )
            values = (
                rng.standard_normal(shape)
                + 1j * rng.standard_normal(shape)
            )
            kappa[k, right, left] = values
            kappa[k, left, right] = -values.conj().T
    kappa /= np.linalg.norm(kappa)
    return kappa


def _rotate_mos(mo_coeff, kappa, step):
    return np.asarray([
        mo @ linalg.expm(step * generator)
        for mo, generator in zip(mo_coeff, kappa)
    ])


class KnownValuesKLASSCFHessianFiniteDifference(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        klas, mo_coeff, ci, ugg, hop = _build_reference(
            (4.0, 10.0, 10.0), (2, 1, 1),
            "H 0 0 0; H 1.5 0 0", "6-31g", ["H 1s"],
        )

        cls.klas = klas
        cls.mo_coeff = np.asarray(mo_coeff)
        cls.ci = ci
        cls.ugg = ugg
        cls.hop = hop
        cls._dimensional_references = {}

    @classmethod
    def _dimensional_reference(cls, name, lattice, kmesh):
        if name not in cls._dimensional_references:
            cls._dimensional_references[name] = _build_reference(
                lattice, kmesh,
                "Li 0 0 0; H 1.6 0 0", "sto-3g",
                ["Li 2s", "H 1s"],
            )
        return cls._dimensional_references[name]

    def _packed_ci_gradient(self, ci, h1eff=None):
        gradient = self.klas.get_grad_ci(
            mo_coeff=self.mo_coeff,
            ci=ci,
            h1eff=h1eff,
            h2eff=self.hop.eri_cas,
        )
        return self.ugg.pack_ci(gradient)

    def _h1eff(self, ci):
        casdm1frs = self.klas.states_make_casdm1s_sub(
            ci=ci,
            ncas_sub=self.klas.ncas_sub,
            nelecas_sub=self.klas.nelecas_sub,
        )
        casdm1s_sub = self.klas.make_casdm1s_sub(
            ci=ci, casdm1frs=casdm1frs,
        )
        return self.klas.h1e_for_las(
            mo_coeff=self.mo_coeff,
            ci=ci,
            ncas_sub=self.klas.ncas_sub,
            nelecas_sub=self.klas.nelecas_sub,
            casdm1s_sub=casdm1s_sub,
            casdm1frs=casdm1frs,
            eri_cas=self.hop.eri_cas,
        )

    def _finite_difference(self, direction, h1eff=None, step=1e-5):
        ci_plus = _displace_ci(self.ci, direction, step)
        ci_minus = _displace_ci(self.ci, direction, -step)
        return (
            self._packed_ci_gradient(ci_plus, h1eff=h1eff)
            - self._packed_ci_gradient(ci_minus, h1eff=h1eff)
        ) / (2.0 * step)

    def test_ci_hessian_blocks_match_finite_difference(self):
        _, direction = _make_ci_direction(self.ugg)
        tdm1rs = self.hop.make_tdm1s_sub(direction)
        h1eff_response = self.hop.get_h1eff_response(tdm1rs)
        diagonal = self.ugg.pack_ci(
            self.hop.ci_response_diag(direction),
        )
        offdiagonal = self.ugg.pack_ci(
            self.hop.ci_response_offdiag(h1eff_response),
        )

        finite_diagonal = self._finite_difference(
            direction, h1eff=self.hop.h1frs,
        )
        finite_total = self._finite_difference(direction)
        finite_offdiagonal = finite_total - finite_diagonal

        step = 1e-5
        ci_plus = _displace_ci(self.ci, direction, step)
        ci_minus = _displace_ci(self.ci, direction, -step)
        h1eff_plus = self._h1eff(ci_plus)
        h1eff_minus = self._h1eff(ci_minus)
        finite_h1eff = [
            (plus - minus) / (2.0 * step)
            for plus, minus in zip(h1eff_plus, h1eff_minus)
        ]
        dm1frs_plus = self.klas.states_make_casdm1s_sub(ci=ci_plus)
        dm1frs_minus = self.klas.states_make_casdm1s_sub(ci=ci_minus)
        finite_tdm1rs = np.zeros_like(tdm1rs)
        offsets = np.cumsum(np.concatenate(([0], self.klas.ncas_sub)))
        for ifrag, (i, j) in enumerate(zip(offsets[:-1], offsets[1:])):
            finite_tdm1rs[:, :, i:j, i:j] = (
                dm1frs_plus[ifrag] - dm1frs_minus[ifrag]
            ) / (2.0 * step)

        np.testing.assert_allclose(
            tdm1rs, finite_tdm1rs, atol=2e-8, rtol=2e-8,
        )
        for analytic, finite in zip(h1eff_response, finite_h1eff):
            np.testing.assert_allclose(
                analytic, finite, atol=2e-8, rtol=2e-8,
            )
        np.testing.assert_allclose(
            diagonal, finite_diagonal, atol=2e-7, rtol=2e-7,
        )
        np.testing.assert_allclose(
            offdiagonal, finite_offdiagonal, atol=2e-7, rtol=2e-7,
        )

    def test_dimensional_ci_hops_match_finite_difference(self):
        cases = {
            "1D": ((4.0, 10.0, 10.0), (2, 1, 1)),
            "2D": ((4.0, 4.0, 10.0), (2, 2, 1)),
            "3D": ((4.0, 4.0, 4.0), (2, 2, 2)),
        }
        for index, (name, (lattice, kmesh)) in enumerate(cases.items()):
            with self.subTest(dimension=name):
                klas, mo_coeff, ci, ugg, hop = (
                    self._dimensional_reference(name, lattice, kmesh)
                )
                packed_direction, direction = _make_ci_direction(
                    ugg, seed=31 + index,
                )
                trial = np.zeros(ugg.nvar_tot, dtype=np.complex128)
                trial[ugg.nvar_orb:] = packed_direction
                analytic = hop.matvec(trial)[ugg.nvar_orb:]

                step = 1e-5
                ci_plus = _displace_ci(ci, direction, step)
                ci_minus = _displace_ci(ci, direction, -step)
                gradient_plus = ugg.pack_ci(klas.get_grad_ci(
                    mo_coeff=mo_coeff, ci=ci_plus,
                    h2eff=hop.eri_cas,
                ))
                gradient_minus = ugg.pack_ci(klas.get_grad_ci(
                    mo_coeff=mo_coeff, ci=ci_minus,
                    h2eff=hop.eri_cas,
                ))
                finite = (gradient_plus - gradient_minus) / (2.0 * step)
                relative_error = np.linalg.norm(analytic - finite) / max(
                    np.linalg.norm(analytic), 1e-14,
                )
                self.assertLess(relative_error, 2e-7)

    def test_dimensional_orbital_hops_match_finite_difference(self):
        cases = {
            "1D": (
                (4.0, 10.0, 10.0), (2, 1, 1),
                ("core-active",), 41,
            ),
            "2D": (
                (4.0, 4.0, 10.0), (2, 2, 1),
                ("active-virtual",), 43,
            ),
            "3D": (
                (4.0, 4.0, 4.0), (2, 2, 2),
                ("core-active", "core-virtual", "active-virtual"), 47,
            ),
        }
        for name, (lattice, kmesh, blocks, seed) in cases.items():
            with self.subTest(dimension=name):
                klas, mo_coeff, ci, ugg, hop = (
                    self._dimensional_reference(name, lattice, kmesh)
                )
                kappa = _make_orbital_direction(klas, blocks, seed)
                trial = np.zeros(ugg.nvar_tot, dtype=np.complex128)
                trial[:ugg.nvar_orb] = ugg.pack_orb(kappa)
                analytic = 2.0 * hop.matvec(trial)[:ugg.nvar_orb]

                connection = np.asarray([
                    (fock @ generator - generator @ fock) / 2.0
                    for fock, generator in zip(hop.fock1, kappa)
                ])
                connection -= connection.conj().transpose(0, 2, 1)

                step = 2.5e-3
                gradient_plus = ugg.pack_orb(klas.get_grad_orb(
                    mo_coeff_kpts=_rotate_mos(
                        mo_coeff, kappa, step,
                    ),
                    ci=ci,
                ))
                gradient_minus = ugg.pack_orb(klas.get_grad_orb(
                    mo_coeff_kpts=_rotate_mos(
                        mo_coeff, kappa, -step,
                    ),
                    ci=ci,
                ))
                finite = (
                    (gradient_plus - gradient_minus) / (2.0 * step)
                    - ugg.pack_orb(connection)
                )
                relative_error = np.linalg.norm(analytic - finite) / max(
                    np.linalg.norm(analytic), 1e-14,
                )
                self.assertLess(relative_error, 2e-5)

    def test_real_get_hop_cross_blocks_match_finite_differences(self):
        # Test both reciprocal derivatives explicitly.  A plain complex-vdot
        # comparison is not the relevant metric because orbital responses use
        # the molecular half-generator packing convention while CI responses
        # do not.
        self.assertIsInstance(
            self.hop, klasscf.KLASSCF_HessianOperator,
        )
        orbital_trial = np.zeros(self.ugg.nvar_tot, dtype=np.complex128)
        kappa = _make_orbital_direction(
            self.klas, ("active-virtual",), 59,
        )
        orbital_trial[:self.ugg.nvar_orb] = self.ugg.pack_orb(kappa)
        packed_ci, _ = _make_ci_direction(self.ugg, seed=61)
        ci_trial = np.zeros(self.ugg.nvar_tot, dtype=np.complex128)
        ci_trial[self.ugg.nvar_orb:] = packed_ci

        hop_orbital = self.hop.matvec(orbital_trial)
        hop_ci = self.hop.matvec(ci_trial)

        _, ci_direction = _make_ci_direction(self.ugg, seed=61)
        step = 1e-5
        ci_gradient_plus = self.ugg.pack_ci(self.klas.get_grad_ci(
            mo_coeff=_rotate_mos(self.mo_coeff, kappa, step),
            ci=self.ci,
        ))
        ci_gradient_minus = self.ugg.pack_ci(self.klas.get_grad_ci(
            mo_coeff=_rotate_mos(self.mo_coeff, kappa, -step),
            ci=self.ci,
        ))
        finite_ci_orbital = (
            ci_gradient_plus - ci_gradient_minus
        ) / (2.0 * step)

        ci_plus = _displace_ci(self.ci, ci_direction, step)
        ci_minus = _displace_ci(self.ci, ci_direction, -step)
        orbital_gradient_plus = self.ugg.pack_orb(
            self.klas.get_grad_orb(
                mo_coeff_kpts=self.mo_coeff, ci=ci_plus,
            )
        )
        orbital_gradient_minus = self.ugg.pack_orb(
            self.klas.get_grad_orb(
                mo_coeff_kpts=self.mo_coeff, ci=ci_minus,
            )
        )
        finite_orbital_ci = (
            orbital_gradient_plus - orbital_gradient_minus
        ) / (2.0 * step)

        ci_orbital_error = np.linalg.norm(
            hop_orbital[self.ugg.nvar_orb:] - finite_ci_orbital
        )
        orbital_ci_error = np.linalg.norm(
            hop_ci[:self.ugg.nvar_orb] - finite_orbital_ci
        )
        self.assertLess(ci_orbital_error, 2e-7)
        self.assertLess(orbital_ci_error, 2e-7)

    def test_translation_symmetric_get_hop_matches_ci_finite_difference(self):
        trans_klas = mcscf.KLASCI(
            self.klas._scf, 2, (1, 1), kmesh=(2, 1, 1),
            trans_sym=True, ref_cell=1,
        )
        trans_klas.conv_tol_grad = 1e-8
        trans_klas.conv_tol_self = 1e-10
        trans_klas.kernel(np.array(self.mo_coeff, copy=True))
        ci = _copy_ci(trans_klas.ci)
        ugg = trans_klas.get_ugg(mo_coeff=self.mo_coeff, ci=ci)
        hop = trans_klas.get_hop(
            mo_coeff=self.mo_coeff, ci=ci, ugg=ugg,
        )
        hop.level_shift = 0.0
        self.assertIsInstance(
            hop, klasscf.KLASSCF_TransSymmHessianOperator,
        )

        _, arbitrary_direction = _make_ci_direction(ugg, seed=67)
        direction = hop._unpack_cif(hop._pack_ci(arbitrary_direction))
        packed_direction = ugg.pack_ci(direction)
        packed_direction /= np.linalg.norm(packed_direction)
        direction = ugg.unpack_ci(packed_direction)

        trial = np.zeros(ugg.nvar_tot, dtype=np.complex128)
        trial[ugg.nvar_orb:] = packed_direction
        analytic = hop.matvec(trial)[ugg.nvar_orb:]
        hop._pack_ci(
            ugg.unpack_ci(analytic), validate=True, tol=2e-8,
        )

        step = 1e-5
        ci_plus = _displace_ci(ci, direction, step)
        ci_minus = _displace_ci(ci, direction, -step)
        finite = (
            ugg.pack_ci(trans_klas.get_grad_ci(
                mo_coeff=self.mo_coeff, ci=ci_plus,
                h2eff=hop.eri_cas,
            ))
            - ugg.pack_ci(trans_klas.get_grad_ci(
                mo_coeff=self.mo_coeff, ci=ci_minus,
                h2eff=hop.eri_cas,
            ))
        ) / (2.0 * step)
        np.testing.assert_allclose(
            analytic, finite, atol=2e-7, rtol=2e-7,
        )

    def test_real_preconditioner_uses_complete_finite_diagonal(self):
        preconditioner = self.hop.get_prec()
        self.assertEqual(
            preconditioner.shape,
            (self.ugg.nvar_tot, self.ugg.nvar_tot),
        )
        self.assertEqual(preconditioner.Hdiag.shape, (self.ugg.nvar_tot,))
        self.assertFalse(np.any(np.isnan(preconditioner.Hdiag)))
        self.assertFalse(np.any(np.abs(preconditioner.Hdiag) < 1e-8))

        rng = np.random.default_rng(71)
        trial = (
            rng.standard_normal(self.ugg.nvar_tot)
            + 1j * rng.standard_normal(self.ugg.nvar_tot)
        )
        actual = preconditioner.matvec(trial)
        expected = trial / preconditioner.Hdiag
        np.testing.assert_allclose(actual, expected)

    def test_frozen_orbital_path_builds_and_applies_real_hop(self):
        frozen = self.klas.ncore
        original_frozen = getattr(self.klas, "frozen", None)
        try:
            self.klas.frozen = [frozen]
            ugg = self.klas.get_ugg(
                mo_coeff=self.mo_coeff, ci=self.ci,
            )
            self.assertLess(ugg.nvar_orb, self.ugg.nvar_orb)
            rng = np.random.default_rng(73)
            packed_orbital = (
                rng.standard_normal(ugg.nvar_orb)
                + 1j * rng.standard_normal(ugg.nvar_orb)
            )
            kappa = ugg.unpack_orb(packed_orbital)
            np.testing.assert_array_equal(kappa[:, frozen, :], 0.0)
            np.testing.assert_array_equal(kappa[:, :, frozen], 0.0)

            hop = self.klas.get_hop(
                mo_coeff=self.mo_coeff, ci=self.ci, ugg=ugg,
            )
            trial = np.zeros(ugg.nvar_tot, dtype=np.complex128)
            trial[:ugg.nvar_orb] = packed_orbital
            response = hop.matvec(trial)
            self.assertEqual(response.shape, trial.shape)
            self.assertTrue(np.all(np.isfinite(response)))
        finally:
            self.klas.frozen = original_frozen


if __name__ == "__main__":
    unittest.main()
