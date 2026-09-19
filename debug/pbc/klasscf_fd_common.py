"""Shared fixtures, reporting, and convergence assertions for k-LASSCF checks.

Forward differences must show order one over the fixed step-halving window,
while centered differences have order two. Steps stop before roundoff dominates;
a finite-precision calculation cannot demonstrate a literal zero-step limit.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy import linalg
from pyscf import lib
from pyscf.pbc import gto, scf

from mrh.my_pyscf.pbc import mcscf
from mrh.my_pyscf.pbc.mcscf import avas
from mrh.my_pyscf.pbc.mcscf import klasscf  # Register gradient/Hessian methods.
from mrh.my_pyscf.pbc.mcscf.productstate import ImpureProductStateFCISolver

# Author: Bhavnesh Jangid

FD_STEPS = 1e-2 / 2.0 ** np.arange(4)
CASES = {
    "1D": dict(lattice=(4.0, 10.0, 10.0), kmesh=(2, 1, 1),
               rotation_blocks=("core-active",), seed=17),
    "2D": dict(lattice=(4.0, 4.0, 10.0), kmesh=(2, 2, 1),
               rotation_blocks=("active-virtual",), seed=23),
    "3D": dict(lattice=(4.0, 4.0, 4.0), kmesh=(2, 2, 2),
               rotation_blocks=("core-active", "core-virtual", "active-virtual"),
               seed=31),
}


def nonzero_norm(vector, label):
    """Reject a zero/nonfinite denominator instead of masking it with a floor."""
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-14:
        raise AssertionError(f"{label} norm must exceed 1e-14; got {norm}")
    return norm


def build_reference(lattice, kmesh):
    """Build the fixed-CI periodic LiH reference."""
    cell = gto.Cell()
    cell.a = np.diag(lattice)
    cell.atom = "Li 0 0 0; H 1.6 0 0"
    cell.basis = "sto-3g"
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

    active_labels = ["Li 2s", "H 1s"]
    mo_coeff = avas.kernel(kmf, active_labels, minao=cell.basis)[2]
    klas = mcscf.KLASCI(kmf, 2, (1, 1), kmesh=kmesh)
    klas.conv_tol_grad = 1e-10
    klas.conv_tol_self = 1e-12
    mo_ref = klas.localize_init_guess(active_labels, mo_coeff=mo_coeff)
    klas.kernel(mo_ref)
    return klas, np.asarray(mo_ref)


def copy_ci(ci):
    return [[np.array(c, copy=True) for c in roots] for roots in ci]


def make_direction(klas, rotation_blocks, seed, mo_coeff=None):
    """Build an orbital direction, using independent coordinates for AA."""
    nkpts, nmo = klas.nkpts, klas.mo_coeff.shape[-1]
    ncore = klas.ncore
    nocc = ncore + klas.ncas
    spaces = {
        "core": slice(0, ncore),
        "active": slice(ncore, nocc),
        "virtual": slice(nocc, nmo),
    }
    dimensions = {
        "core": ncore,
        "active": klas.ncas,
        "virtual": nmo - nocc,
    }

    rng = np.random.default_rng(seed)
    kappa = np.zeros((nkpts, nmo, nmo), dtype=np.complex128)
    if "active-active" in rotation_blocks:
        ugg = klas.get_ugg(mo_coeff=mo_coeff)
        rotation_map = ugg.active_active_map
        coordinates = (rng.standard_normal(rotation_map.nvar)
                       + 1j * rng.standard_normal(rotation_map.nvar))
        nonzero_norm(coordinates, "active-active coordinates")
        kappa[:, ncore:nocc, ncore:nocc] = rotation_map.unpack(coordinates)
    for k in range(nkpts):
        for block_name in rotation_blocks:
            if block_name == "active-active":
                continue
            left_name, right_name = block_name.split("-")
            nleft = dimensions[left_name]
            nright = dimensions[right_name]
            if nleft == 0 or nright == 0:
                raise RuntimeError(f"empty orbital space in {block_name}")
            block = (
                rng.standard_normal((nright, nleft))
                + 1j * rng.standard_normal((nright, nleft))
            )
            left = spaces[left_name]
            right = spaces[right_name]
            kappa[k, right, left] = block
            kappa[k, left, right] = -block.conj().T

    kappa /= nonzero_norm(kappa, "orbital direction")
    return kappa


def orbital_parser(description):
    """Select orbital sectors explicitly for standalone convergence checks."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--rotation-blocks", nargs="+",
        choices=("core-active", "core-virtual", "active-virtual", "active-active"),
        help="override the case's orbital sectors (e.g. active-active)",
    )
    return parser


def rotate_mos(mo_ref, kappa, step):
    """Apply ``exp(step * kappa)`` to the MOs at every k-point."""
    return np.asarray([
        mo_k @ linalg.expm(step * kappa_k)
        for mo_k, kappa_k in zip(mo_ref, kappa)
    ])


def fixed_ci_energy(klas, mo_coeff, ci):
    """Evaluate the per-cell k-LAS energy without relaxing CI vectors."""
    h1eff, ecore = klas.h1e_for_cas(
        mo_coeff=mo_coeff, ncas=klas.ncas, ncore=klas.ncore,
    )
    h2eff = klas.get_h2cas(mo_coeff)
    solver = make_product_solver(klas)
    energy = solver.energy_elec(
        h1eff, h2eff, [roots[0] for roots in ci],
        klas.ncas_sub, klas.nelecas_sub, ecore=ecore,
    ) / klas.nkpts
    if abs(np.imag(energy)) > 1e-9:
        raise AssertionError(f"energy has imaginary part {np.imag(energy)}")
    return float(np.real(energy))


def make_product_solver(klas):
    """Single-root product solver used for the independent energy reference."""
    fcisolvers = [box.fcisolvers[0] for box in klas.fciboxes]
    return ImpureProductStateFCISolver(
        fcisolvers, lweights=[[1.0] for _ in fcisolvers],
        stdout=klas.stdout, verbose=lib.logger.QUIET,
    )


def displace_ci(ci, direction, step):
    """Follow a normalized CI path with the supplied tangent at step zero."""
    displaced = copy_ci(ci)
    for roots, directions in zip(displaced, direction):
        for c, d in zip(roots, directions):
            c += step * d
            c /= nonzero_norm(c, "displaced CI")
    return displaced


def make_ci_direction(ugg, seed):
    """Make a unit CSF direction orthogonal to every reference CI root."""
    rng = np.random.default_rng(seed)
    packed = rng.standard_normal(ugg.nvar_ci) + 1j * rng.standard_normal(ugg.nvar_ci)
    direction = ugg.unpack_ci(packed)
    for roots, directions in zip(ugg.ci, direction):
        for c, d in zip(roots, directions):
            d -= np.vdot(c, d) * c
    packed = ugg.pack_ci(direction)
    packed /= nonzero_norm(packed, "CI tangent")
    return packed, ugg.unpack_ci(packed)


def convergence_result(analytic, differences, steps):
    """Measure relative errors and orders for every adjacent pair of steps."""
    steps = np.asarray(steps, dtype=float)
    if (steps.ndim != 1 or steps.size < 3 or not np.all(np.isfinite(steps))
            or np.any(steps <= 0) or np.any(np.diff(steps) >= 0)):
        raise ValueError("provide at least three finite, positive, decreasing steps")
    analytic_norm = nonzero_norm(analytic, "analytic derivative/response")
    errors = np.asarray([
        np.linalg.norm(np.asarray(finite) - analytic) / analytic_norm
        for finite in differences
    ])
    if errors.shape != steps.shape:
        raise ValueError("one finite difference is required for each step")
    if not np.all(np.isfinite(errors)) or np.any(errors <= 0):
        raise AssertionError(
            f"convergence orders need finite, positive errors: {errors}"
        )
    orders = np.log(errors[:-1] / errors[1:]) / np.log(steps[:-1] / steps[1:])
    return dict(steps=steps, errors=errors, orders=orders, analytic_norm=analytic_norm)


def assert_convergence(result, expected_order=1.0, order_tolerance=0.15):
    """Require the expected order at every halving, not just a fitted intercept."""
    orders = result["orders"]
    if (not np.all(np.isfinite(orders))
            or np.any(np.abs(orders - expected_order) > order_tolerance)):
        raise AssertionError(
            f"expected order {expected_order} +/- {order_tolerance}; "
            f"observed {orders}; steps={result['steps']}; "
            f"relative errors={result['errors']}"
        )


def plot_convergence(results, path, description, expected_order=1.0,
                     order_tolerance=0.15):
    """Save relative-error and observed-order plots without opening a GUI."""
    from textwrap import fill
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(10, 4.5), layout="constrained")
    FigureCanvasAgg(figure)
    error_axis, order_axis = figure.subplots(1, 2)
    for index, (name, result) in enumerate(results.items()):
        steps, errors = result["steps"], result["errors"]
        line, = error_axis.loglog(steps, errors, "o-", label=name)
        error_axis.loglog(
            steps, errors[0] * (steps / steps[0]) ** expected_order,
            "--", color=line.get_color(), alpha=0.5,
            label=f"Expected slope {expected_order:g}" if index == 0 else None,
        )
        order_axis.semilogx(
            steps[1:], result["orders"], "o-", color=line.get_color(), label=name,
        )
    order_axis.axhspan(
        expected_order - order_tolerance, expected_order + order_tolerance,
        color="gray", alpha=0.15,
        label=f"Allowed: {expected_order:g} ± {order_tolerance:g}",
    )
    order_axis.axhline(expected_order, color="gray", linestyle="--", linewidth=1)
    error_axis.set(xlabel="Step size", ylabel="Relative disagreement")
    order_axis.set(xlabel="Smaller step in each pair", ylabel="Observed order")
    for axis in (error_axis, order_axis):
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize="small")
    figure.suptitle(fill(description, width=80))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)


def run_checks(evaluate, description, expected_order=1.0, parser=None,
               default_steps=FD_STEPS):
    """Run checks and save plots in the current directory without opening a GUI."""
    if parser is None:
        parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--case", choices=[*CASES, "all"], default="all")
    parser.add_argument("--steps", nargs="+", type=float, default=default_steps)
    parser.add_argument(
        "--plot", type=Path, default=Path(f"{Path(parser.prog).stem}.png"),
        help="plot filename (default: <script-name>.png in the current directory)",
    )
    args = parser.parse_args()
    # Forward script-specific options, such as the orbital rotation sectors.
    evaluate_kwargs = {
        key: value for key, value in vars(args).items()
        if key not in ("case", "steps", "plot")
    }
    names = CASES if args.case == "all" else [args.case]
    results = {}
    try:
        for name in names:
            config = CASES[name]
            klas, mo_coeff = build_reference(config["lattice"], config["kmesh"])
            result = evaluate(
                klas, mo_coeff, config, np.asarray(args.steps), **evaluate_kwargs,
            )
            results[name] = result
            print(f"{name}: analytic norm = {result['analytic_norm']:.8e}", flush=True)
            print("       step    relative error    observed order", flush=True)
            for index, (step, error) in enumerate(zip(result["steps"], result["errors"])):
                order = "--" if index == 0 else f"{result['orders'][index - 1]:.5f}"
                print(f"{step:11.4e}    {error:12.5e}    {order}", flush=True)
            assert_convergence(result, expected_order=expected_order)
    finally:
        if args.plot and results:
            plot_convergence(results, args.plot, description, expected_order)
