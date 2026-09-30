"""Reproducible, per-stage LAS-LUSCC timing and numerical comparison.

Run with --output pointing to a NEW directory. For an unchanged-code baseline,
pass --baseline-dir containing spin-baseline.py, solver-baseline.py, and
sisolver-baseline.py saved before editing. No existing output is overwritten.
"""

import argparse
import cProfile
import importlib.util
import json
from functools import partial
from itertools import product
from pathlib import Path
import pstats
from time import perf_counter
from unittest import mock

import numpy as np
from scipy.sparse.linalg import aslinearoperator
from pyscf import gto, scf
from pyscf.lib import chkfile

from mrh.exploratory.luscc import LUSCC
from mrh.exploratory.luscc import spin as spin_module
from mrh.my_pyscf.lassi import LASSI, LASSIS, op_o1
from mrh.my_pyscf.lassi.chkfile import dump_lsi, KEYS_RESULTS_LASSI
from mrh.my_pyscf.lassi.sisolver import kernel_projected_davidson
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.fci.spin_op import mup, mdown


def load_baseline(directory, name, package):
    spec = importlib.util.spec_from_file_location(
        package + '._benchmark_' + name, directory / (name + '-baseline.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fe4_reference(las, checkpoint, spin, output, integrals=None):
    """110 M=0 products of the saved (S=2,5/2,2,5/2) local multiplets."""
    las.load_chk(str(checkpoint))
    nelec = las.get_nelec_frs()[:, 0]
    smults = np.asarray([5, 6, 5, 6])
    rotations = []
    for f, (norb, ne, smult) in enumerate(zip(las.ncas_sub, nelec, smults)):
        high = mup(las.ci[f][0], norb, tuple(ne), smult)
        images = {}
        for m in range(1-int(smult), int(smult), 2):
            nel = ((int(sum(ne))+m)//2, (int(sum(ne))-m)//2)
            images[m] = mdown(high, norb, nel, smult)
        rotations.append(images)
    spins = np.asarray([ms for ms in product(*(list(r) for r in rotations))
                        if sum(ms) == 0])
    weights = np.zeros(len(spins))
    weights[0] = 1
    ref_las = las.state_average(weights=weights,
        charges=np.tile(las.ncas_sub-nelec.sum(axis=1), (len(spins), 1)),
        spins=spins, smults=np.tile(smults, (len(spins), 1)), assert_no_dupes=False)
    ref_las.ci = [[rotations[f][ms[f]] for ms in spins] for f in range(4)]
    ref = LASSI(ref_las)
    if integrals:
        saved = np.load(integrals)
        h0, h1, h2 = (saved[k] for k in ('h0', 'h1', 'h2'))
    else:
        h0, h1, h2 = ref.ham_2q()
    np.savez(output / 'reference-integrals.npz', h0=h0, h1=h1, h2=h2)
    hop, s2op, _, _, get_overlap = op_o1.gen_contract_op_si_hdiag(
        ref, h1, h2, ref.ci, ref.get_nelec_frs(), smult_fr=None,
        disc_fr=ref.get_disc_fr())
    projector, _ = spin_module.exact_spin_basis(get_overlap(),
        spin_module.residual_gram(ref, spin), spin)
    hcon = projector.T @ hop(projector)
    energy, vectors = np.linalg.eigh((hcon+hcon.T)/2)
    ref.si = projector @ vectors[:, :1]
    ref.e_roots = energy[:1] + h0
    ref.s2 = np.asarray([np.vdot(ref.si[:, 0], s2op(ref.si[:, 0])).real])
    ref.converged = True
    return ref


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-dir', type=Path)
    parser.add_argument('--atoms', type=int, choices=(4, 8), default=4)
    parser.add_argument('--generators', type=int, default=8)
    parser.add_argument('--top-m', type=int, default=4)
    parser.add_argument('--spin', type=float, default=0)
    parser.add_argument('--max-memory', type=float, default=4000,
                        help='PySCF memory budget in MB (default: 4000)')
    parser.add_argument('--reference', choices=('las', 'lassis'), default='lassis')
    parser.add_argument('--compare', type=Path)
    parser.add_argument('--checkpoint', type=Path,
                        help='Fe4 paper checkpoint: use its 110 M=0 spin products')
    parser.add_argument('--no-profile', action='store_true')
    parser.add_argument('--reuse-reference', type=Path,
                        help='Reuse a benchmark reference with a different number of generators')
    parser.add_argument('--integrals', type=Path,
                        help='Reuse h0/h1/h2 for the SAME checkpoint orbitals')
    parser.add_argument('--dense-operators', action='store_true',
                        help='Diagnostic: use LASSI dense H/S2 assembly')
    parser.add_argument('--matrix-free', action='store_true',
                        help='Force the optimized matrix-free fallback')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    solver_class, spin, projected = LUSCC, spin_module, kernel_projected_davidson
    if args.baseline_dir:
        solver_class = load_baseline(args.baseline_dir, 'solver',
                                     'mrh.exploratory.luscc').LUSCC
        spin = load_baseline(args.baseline_dir, 'spin', 'mrh.exploratory.luscc')
        projected = load_baseline(args.baseline_dir, 'sisolver',
                                  'mrh.my_pyscf.lassi').kernel_projected_davidson
    timings = {}
    saved_reference = args.compare or args.reuse_reference

    def timed(name, function):
        start = perf_counter()
        result = function()
        timings[name] = perf_counter() - start
        print(name, timings[name], flush=True)
        return result

    def reference():
        nfrag = args.atoms // 2
        if args.checkpoint:
            mol = chkfile.load_mol(str(args.checkpoint))
            mol.verbose = 0
            mol.stdout = (args.output / 'calculation.log').open('x')
            mol.max_memory = args.max_memory
            mf = scf.ROHF(mol).density_fit()
            las = LASSCF(mf, (5,)*4, (6, 5, 6, 5), spin_sub=(5, 6, 5, 6))
        else:
            mol = gto.M(atom=[('H', (3 * (i // 2) + i % 2, 0, 0))
                          for i in range(args.atoms)], basis='sto-3g',
                    verbose=0, output=str(args.output / 'calculation.log'))
            mf = scf.RHF(mol)
            las = LASSCF(mf, (2,) * nfrag, (2,) * nfrag, spin_sub=(1,) * nfrag)
        if saved_reference:
            ref = LASSI(las) if args.reference == 'lassis' else las
            # LASSIS may have distinct CI spaces with identical spin/charge
            # metadata. The generic loader otherwise rejects those spaces.
            with mock.patch.object(las, 'state_average',
                                   partial(las.state_average, assert_no_dupes=False)):
                ref.load_chk(str(saved_reference / 'reference.chk'))
            return ref
        if args.checkpoint:
            return fe4_reference(las, args.checkpoint, args.spin, args.output, args.integrals)
        mf.kernel()
        guess = las.localize_init_guess(tuple((2*i, 2*i+1) for i in range(nfrag)),
                                       mf.mo_coeff)
        las.kernel(guess)
        return LASSIS(las).run() if args.reference == 'lassis' else las

    ref = timed('reference', reference)
    if args.reference == 'lassis':
        dump_lsi(ref, chkfile=str(args.output / 'reference.chk'),
                 keys_results=[k for k in KEYS_RESULTS_LASSI
                               if getattr(ref, k) is not None])
    else:
        ref.dump_chk(chkfile=str(args.output / 'reference.chk'))
    # Fixed ordered generators, independent of gradient rankings or timings.
    orbs_per_frag = 5 if args.checkpoint else 2
    norb = int(ref.ncas)
    pairs = [(a + s*norb, i + s*norb)
             for i in range(norb) for a in range(i+1, norb)
             if a//orbs_per_frag != i//orbs_per_frag for s in range(2)][:args.generators]
    luscc = solver_class(ref, [[a] for a, i in pairs], [[i] for a, i in pairs],
                         top_m=args.top_m, target_spin=args.spin)
    if args.matrix_free and not args.baseline_dir:
        luscc.operator_mode = 'matrix_free'
    profiler = cProfile.Profile()
    if not args.no_profile:
        profiler.enable()
    start = perf_counter()
    timed('prepare_states', luscc.prepare_states_)
    if saved_reference or args.checkpoint:
        saved = np.load(saved_reference / 'integrals.npz' if saved_reference else
                        args.output / 'reference-integrals.npz')
        h0, h1, h2 = (saved[k] for k in ('h0', 'h1', 'h2'))
        timings['integrals'] = 0.0
    else:
        h0, h1, h2 = timed('integrals', luscc.ham_2q)
    np.savez(args.output / 'integrals.npz', h0=h0, h1=h1, h2=h2)
    if args.dense_operators:
        ham, s2mat, ovlp, _ = timed('operators', lambda:
            op_o1.ham(luscc, h1, h2, luscc.ci, luscc.get_nelec_frs(),
                      smult_fr=None, disc_fr=luscc.get_disc_fr()))
        hop, s2op, mop = map(aslinearoperator, (ham, s2mat, ovlp))
        get_overlap = lambda: ovlp
    elif args.baseline_dir:
        hop, s2op, mop, _, get_overlap = timed('operators', lambda:
            op_o1.gen_contract_op_si_hdiag(luscc, h1, h2, luscc.ci,
                luscc.get_nelec_frs(), smult_fr=None, disc_fr=luscc.get_disc_fr()))
    else:
        hop, s2op, mop, _, get_overlap = timed('operators', lambda:
            luscc._get_exact_spin_operators(h1, h2))
    overlap = timed('overlap', lambda: np.asarray(mop(np.eye(hop.shape[0])))
                    if args.baseline_dir else get_overlap())
    residual = timed('residual', lambda: spin.residual_gram(luscc, args.spin))
    projector, keig = timed('spin_basis', lambda:
        spin.exact_spin_basis(overlap, residual, args.spin,
                              lin_dep_tol=luscc.lin_dep_tol, spin_tol=luscc.spin_tol))
    converged, energy, coeff, s2 = timed('projected_davidson', lambda:
        projected(luscc.sisolver, hop, s2op, projector, nroots=1))
    timings['total'] = perf_counter() - start
    if not args.no_profile:
        profiler.disable()
        profiler.dump_stats(str(args.output / 'profile.pstats'))
        with (args.output / 'profile.txt').open('x') as stream:
            pstats.Stats(profiler, stream=stream).sort_stats('cumulative').print_stats(45)
    norm = np.sqrt(np.maximum(0., np.real(np.einsum('ip,ij,jp->p',
                                         coeff.conj(), residual, coeff))))
    energy = energy + h0
    np.savez(args.output / 'matrices.npz', overlap=overlap, residual=residual,
             energy=energy, coeff=coeff, s2=s2, residual_norm=norm,
             spin_eigenvalues=keig)
    result = dict(timings=timings, nstates=int(hop.shape[0]),
                  spin_dimension=int(projector.shape[1]), converged=bool(converged),
                  energy=energy.tolist(), s2=s2.tolist(), residual_norm=norm.tolist(),
                  atoms=int(ref.mol.natm), generators=pairs, top_m=args.top_m,
                  spin=args.spin, reference=args.reference,
                  checkpoint=str(args.checkpoint), profiled=not args.no_profile,
                  dense_operators=args.dense_operators,
                  operator_mode=getattr(luscc, 'exact_spin_operator_mode', 'matrix_free'),
                  max_memory_mb=float(luscc.max_memory),
                  baseline_dir=str(args.baseline_dir))
    if args.dense_operators or getattr(luscc, 'exact_spin_operator_mode', None) == 'dense':
        hp = projector.conj().T @ hop(projector)
        result['dense_projected_energies'] = (np.linalg.eigvalsh((hp+hp.conj().T)/2)+h0).tolist()
    if args.compare:
        old = np.load(args.compare / 'matrices.npz')
        differences = {}
        for name, new, tol in [('overlap', overlap, 1e-10),
                               ('residual', residual, 1e-10),
                               ('energy', energy, 1e-9), ('s2', s2, 1e-10)]:
            differences[name] = float(np.max(np.abs(new - old[name])))
            np.testing.assert_allclose(new, old[name], atol=tol, rtol=0)
        result['max_absolute_differences'] = differences
    with (args.output / 'result.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))
    assert converged
    np.testing.assert_allclose(s2, args.spin*(args.spin+1), atol=1e-10, rtol=0)
    ref.mol.stdout.close()


if __name__ == '__main__':
    main()
