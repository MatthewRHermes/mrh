import numpy as np
from unittest import mock
from pyscf import fci
from pyscf import lib
from scipy import linalg

from mrh.exploratory.luscc import LUSCC
from mrh.exploratory.luscc.excitations import apply_operator_string_fci
from mrh.my_pyscf.lassi import op_o0, op_o1
from mrh.my_pyscf.lassi.lassi import ham_2q


def _apply_alpha_excitation(ci, norb, nelec, creation, annihilation):
    result, result_nelec = apply_operator_string_fci(
        ci,
        norb,
        nelec,
        ops=(("ann", annihilation, "alpha"),
             ("cre", creation, "alpha")),
    )
    assert result_nelec == nelec
    assert result is not None
    return result / np.linalg.norm(result)


def _full_ci_model_space_energy(las, creation, annihilation):
    """Diagonalize LAS, T|LAS>, and T-dagger|LAS> in the full-CAS basis."""
    ci_products, electron_numbers = op_o0.ci_outer_product(
        las.ci, las.ncas_sub, las.get_nelec_frs())
    assert len(ci_products) == 1
    reference = ci_products[0] / np.linalg.norm(ci_products[0])
    nelec = tuple(electron_numbers[0])
    states = [
        reference,
        _apply_alpha_excitation(
            reference, las.ncas, nelec, creation, annihilation),
        _apply_alpha_excitation(
            reference, las.ncas, nelec, annihilation, creation),
    ]

    h0, h1, h2 = ham_2q(las, las.mo_coeff)
    fci_solver = fci.solver(las.mol)
    h2eff = fci_solver.absorb_h1e(
        h1, h2, las.ncas, nelec, fac=0.5)
    h_states = [
        fci_solver.contract_2e(h2eff, state, las.ncas, nelec)
        for state in states
    ]
    overlap = np.asarray([
        [np.vdot(bra, ket) for ket in states]
        for bra in states
    ])
    hamiltonian = np.asarray([
        [np.vdot(bra, h_ket) + h0 * overlap[i, j]
         for j, h_ket in enumerate(h_states)]
        for i, bra in enumerate(states)
    ])
    return linalg.eigh(hamiltonian, overlap, eigvals_only=True)[0]


def test_h4_luscc_energy(h4_las):
    # One fixed alpha charge-transfer generator and its de-excitation produce
    # a small, deterministic end-to-end LUSCC model space.
    energy, _ = LUSCC(
        h4_las,
        a_idxs=[np.array([2])],
        i_idxs=[np.array([0])],
    ).kernel()
    reference_energy = _full_ci_model_space_energy(
        h4_las, creation=2, annihilation=0)
    np.testing.assert_allclose(energy[0], reference_energy, atol=1e-10)
    assert energy[0] <= h4_las.e_tot + 1e-10


def test_h4_luscc_exact_singlet(h4_las):
    with mock.patch.object(lib, "davidson1", wraps=lib.davidson1) as davidson:
        energy, coeff, s2, residual = LUSCC(
            h4_las, a_idxs=[], i_idxs=[], target_spin=0).kernel()
    davidson.assert_called_once()
    assert coeff.shape == (1, 1)
    np.testing.assert_allclose(energy[0], h4_las.e_tot, atol=1e-9)
    np.testing.assert_allclose(s2[0], 0.0, atol=1e-10)
    np.testing.assert_allclose(residual[0], 0.0, atol=1e-10)


def test_exact_spin_reports_absent_target(h4_las):
    with np.testing.assert_raises_regex(ValueError, "No exact total-spin S=1"):
        LUSCC(h4_las, a_idxs=[], i_idxs=[], target_spin=1).kernel()


def test_shared_factors_preserve_prepared_states(h4_lassis):
    from mrh.exploratory.luscc import solver
    args = dict(a_idxs=[[2], [3], [6], [7]], i_idxs=[[0], [0], [4], [4]], top_m=4)
    originals = [[ci.copy() for ci in root] for root in h4_lassis.ci]
    with mock.patch.object(solver, 'apply_operator_string_fci',
                           wraps=solver.apply_operator_string_fci) as apply:
        copied = LUSCC(h4_lassis, share_spectator_ci=False, **args).prepare_states_()
        uncached_calls = apply.call_count
        apply.reset_mock()
        shared = LUSCC(h4_lassis, **args).prepare_states_()
        assert apply.call_count < uncached_calls
    np.testing.assert_array_equal(shared.get_nelec_frs(), copied.get_nelec_frs())
    assert shared.nroots == copied.nroots
    for left, right in zip(shared.ci, copied.ci):
        for a, b in zip(left, right):
            np.testing.assert_array_equal(a, b)
    for current, original in zip(h4_lassis.ci, originals):
        for a, b in zip(current, original):
            np.testing.assert_array_equal(a, b)


def test_direct_overlap_and_projected_hamiltonian(h4_lassis):
    from mrh.exploratory.luscc.spin import residual_gram, exact_spin_basis
    luscc = LUSCC(h4_lassis, [[2], [6]], [[0], [4]], top_m=4,
                  target_spin=0).prepare_states_()
    h0, h1, h2 = luscc.ham_2q()
    hop, s2op, mop, hdiag, get_overlap = op_o1.gen_contract_op_si_hdiag(
        luscc, h1, h2, luscc.ci, luscc.get_nelec_frs(), smult_fr=None,
        disc_fr=luscc.get_disc_fr())
    overlap = get_overlap()
    np.testing.assert_allclose(overlap, mop(np.eye(hop.shape[0])), atol=1e-12, rtol=0)
    projector, _ = exact_spin_basis(overlap, residual_gram(luscc, 0), 0)
    # Independent dense diagonalization checks the constrained root.
    hprojected = projector.conj().T @ hop(projector)
    expected = linalg.eigh(hprojected, eigvals_only=True)[0]
    conv, energy, coeff, s2 = luscc.sisolver.kernel_projected(
        hop, s2op, projector, nroots=1)
    assert conv
    np.testing.assert_allclose(energy[0], expected, atol=1e-9, rtol=0)
    np.testing.assert_allclose(s2, 0., atol=1e-10, rtol=0)
    np.testing.assert_allclose(coeff.conj().T @ overlap @ coeff, 1., atol=1e-10)
    targeted = LUSCC(h4_lassis, [[2], [6]], [[0], [4]], top_m=4, target_spin=0)
    energy, coeff, s2, _ = targeted.kernel()
    np.testing.assert_allclose(energy[0], h0 + expected, atol=1e-9, rtol=0)
    np.testing.assert_allclose(s2, 0., atol=1e-10, rtol=0)
    # Check the physical residual directly, without square-root amplification
    # of roundoff in c^H K c for a null vector.
    products, nelecs = op_o0.ci_outer_product(
        targeted.ci, targeted.ncas_sub, targeted.get_nelec_frs())
    wavefunction = np.einsum('i,iab->ab', coeff[:, 0], np.stack(products))
    from pyscf.fci.spin_op import contract_ss
    np.testing.assert_allclose(contract_ss(wavefunction, 4, tuple(nelecs[0])),
                               0., atol=1e-10, rtol=0)


def test_exact_spin_records_stage_timings(h4_las):
    luscc = LUSCC(h4_las, [], [], target_spin=0)
    luscc.kernel()
    assert set(luscc.timings) == {'prepare_states', 'integrals', 'operators',
                                  'overlap', 'residual', 'spin_basis',
                                  'projected_davidson', 'total'}
    assert all(seconds >= 0 for seconds in luscc.timings.values())


def test_operator_assembly_matches_matrix_free_and_respects_memory(h4_lassis):
    luscc = LUSCC(h4_lassis, [[2], [6]], [[0], [4]], top_m=4).prepare_states_()
    _, h1, h2 = luscc.ham_2q()
    with mock.patch.object(lib, 'current_memory', return_value=(0., 0.)):
        luscc.max_memory = 4000
        dense = luscc._get_exact_spin_operators(h1, h2)
        assert luscc.exact_spin_operator_mode == 'dense'
        luscc.operator_mode = 'matrix_free'
        forced = luscc._get_exact_spin_operators(h1, h2)
        assert luscc.exact_spin_operator_mode == 'matrix_free'
        # Test dispatch under pressure without allocating below the memory
        # needed even for the fallback's interaction list.
        luscc.operator_mode = 'auto'
        luscc.max_memory = 0
        with mock.patch.object(op_o1, 'gen_contract_op_si_hdiag',
                               return_value=forced) as fallback:
            sparse = luscc._get_exact_spin_operators(h1, h2)
            fallback.assert_called_once()
        assert luscc.exact_spin_operator_mode == 'matrix_free'
    rng = np.random.default_rng(101)
    trial = rng.normal(size=(dense[0].shape[0], 2))
    for a, b, c in zip(dense[:3], sparse[:3], forced[:3]):
        np.testing.assert_allclose(a(trial), b(trial), atol=1e-11, rtol=0)
        np.testing.assert_allclose(a(trial), c(trial), atol=1e-11, rtol=0)
    np.testing.assert_allclose(dense[4](), sparse[4](), atol=1e-12, rtol=0)
