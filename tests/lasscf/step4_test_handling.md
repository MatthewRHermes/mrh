This is my summary of what the mrh LASSCF unit tests are for and my assessment of what to do with
them in the step 4 migration to PySCF-Forge. Additions from the 2026-11-10 review with Claude are
folded in below.

# Destinations in the Forge worktree

- LAS tests and their data/helpers (`*_struct.py`, `*.xyz`, `*.gjf`, `*.dat`):
  `pyscf/mcscf/las/test/`
- LASSCF nuclear gradients: `pyscf/grad/test/test_grad_lasscf.py`
- LAS-PDFT: `pyscf/mcpdft/test/test_laspdft.py`

Forge CI runs `pytest -k 'not _slow'`. `-k` deselects *after* pytest imports each module, so any
SCF/LASSCF run at module level executes on every CI pass even if every test in the file is
deselected. All setup in copied tests goes in `setUpModule`. Logs go to `/dev/null`, never to the
working directory.

# Existing mrh tests

## `tests/lasscf`

- `test_utilities.py`, `test_old_*.py`, `test_lasscf_async.py`, `test_laspscf_mod.py`,
  `test_lasscf_async_crunch.py`: They test features not transferred; omit.
- `test_h4_lasscf.py`, `test_h2minus.py`: these are currently comparisons between synchronous and
  asynchronous kernels. Only the synchronous kernel was transferred, so copy the `syn` parts as
  tests against fixed references of `las_syn.e_states`. Generate the references from `syn` at its
  own convergence tolerance; don't inherit `places=4` from the old async comparison in h4.
- `test_rohf_ref.py`: Supposed to make sure that `localize_init_guess` doesn't deoptimize a
  converged ROHF calculation, so that you can always start from one. Copy; remove the (meaningless,
  unused) `lasscf_async` import. The two `assertNotEqual (..., mf.e_tot)` checks encode the missing
  aufbau CI guess and would fail the day it's implemented: in Forge they become `assertAlmostEqual`
  in separate methods marked `@unittest.expectedFailure`. `test_smallbasis*` are assertion-free smoke
  tests for mrh issue #187; say so in a comment.
- `test_lasscf_rdm123s.py`, `test_lasscf_3rdm.py`: `3rdm` is a strict subset of `rdm123s` (same H8
  system, same four `spin_sub` cases, same `make_casdm3` assertion), so copy only `rdm123s`.
  Also, both call `LASSCF (mf, nelesub, norbsub)` with the arguments swapped (harmless only because
  both are `(2,2,2,2)`), and `test_spin1` duplicates `trace_down_test` inline. Clean up in step 5.
- `test_c2h4n4_symm_slow.py`, `test_c2h4n4_slow.py`, `test_chkfile.py`: Self-evident purpose; copy.
  (`test_chkfile.py` becomes `test_lasscf_chkfile.py` to avoid a generic basename.)
- `test_me2n2.py`, `test_lasscf_sync.py`, `test_lasscf_rdm.py`, `test_me2n2_sa.py`,
  `test_sa_lasscf.py`: These all contain checks of gradients, Hessians, and preconditioner
  operators. Copy; might need renaming or reorganization in step 5. The `test_lasci_{mo,ci0,ci1}.dat`
  files are used only by `test_lasscf_sync.py`; in Forge they are `lasscf_sync_{mo,ci0,ci1}.dat`.
- `test_lasci.py`: not a plain LASCI test. It tests `las.lasci (lroots=...)` (several local roots per
  fragment per state), which is the only route in the migrated code into
  `ImpureProductStateFCISolver` (`lasci.py`, the `lasci` kernel). The reference is SA-LASSCF with
  every orbital frozen (= SA-LASCI); it checks ground-state `e_states`, the produced root counts, and
  `e_lexc >= 0`, starting both from the reference ("sanity") and from ground-state LASSCF
  ("convergence"), with and without Cs symmetry. It is the only test of `productstate.py` that can
  move (the direct ones are in `lassi/` and `kLASSCF/`). Copy as `test_lasci_lroots.py`.

## Outside `tests/lasscf`

- `tests/lasscf_nucgrad/test_grad_lasscf.py`: copy.
- `tests/mcpdft/test_laspdft.py`: copy the sync half; drop the `lasscf_async` and `my_dmet` imports.
  Forge has no LAS-PDFT constructor, so build it with `laspdft.get_mcpdft_child_class`.
- `tests/mcpdft/test_edalaspdft.py`, `tests/mcpdft/test_lassipdft.py`: need LASSI; can't move.

# Coverage gaps: new Forge tests

Status as of 2026-11-10. "Known failure" means the test is written as the behavior should be and
marked `@unittest.expectedFailure`, with a TODO; all of these bugs were already in mrh before the
migration (checked against `~/gits/mrh` at `dfb3c5b7`), none are step-3 regressions.

1. LASCI end-to-end -- `pyscf/mcscf/las/test/test_lasci_kernel.py`. Two H2 50 A apart (LASCI =
   CASCI in the same orbitals, to 1e-8) for no-symmetry, DF, and `LASCI (mol, ...)`; plus LASCI
   above CASCI when the fragments interact.
   - **Known failure:** `LASCI` on any symmetric `mf`. `LASCISymm.kernel` passes `casdm0_fr` to
     `LASCINoSymm.kernel`, which doesn't take it (`TypeError`).
   - Note: `LASCI (mol, ...)` doesn't run the SCF, so `localize_init_guess` needs `mo_coeff`.
2. Nuclear gradients -- `pyscf/grad/test/test_grad_lasscf_numerical.py`. Two-fragment H4/STO-3G
   against central finite differences: single-state and state-averaged agree to 1e-6.
   - **Known failure:** DF. `grad/lasscf.py` doesn't raise for DF LASSCF; it returns a gradient
     ~1e-5 off finite differences (and not equal to the conventional-integral gradient either).
     Suggest raising `NotImplementedError` until implemented.
3. lassqd guards -- `pyscf/mcscf/las/test/test_lasscf_rdm_lassqd.py`: `lasscf_rdm.LASSCF` factory
   (mol, DF -> `_DFLASCI`); capture (h0 = 0, h1s/h2 shapes, fragment 1's Hamiltonian independent
   of fragment 0's result); replay (each fragment solved once, `converged` False because the
   RDM-jk loop reports not converged after zero cycles, one downhill orbital step with a pinned
   energy, canonicalized RDMs consistent with the returned orbitals, `dump_chk` writes); LAS-PDFT
   with monkeypatched `make_one_casdm1s`/`make_one_casdm2` reproduces the CI-based LAS-PDFT.
4. LAS-PDFT beyond one state -- added to `pyscf/mcpdft/test/test_laspdft.py`: one-fragment SA
   LAS-PDFT equals core SA MC-PDFT state by state. `get_energy_decomposition` raises
   `NotImplementedError` for LAS-PDFT (so the original "test it on plain LAS" item is moot; the
   test asserts the raise).
   - Interface difference from core MC-PDFT: SA LAS-PDFT puts the per-state energies in `e_tot`
     (core: weighted average).
   - **Known failure:** the `e_states` attribute stays `[0, 0]` after `kernel`;
     `compute_pdft_energy_ ()` returns the right per-state energies.
5. Partial `frozen` -- `pyscf/mcscf/las/test/test_lasscf_frozen.py` (one fragment, LASSCF vs
   CASSCF with the same `frozen`, canonicalization off).
   - **Known failure, both cases:** every macrocycle of `lasscf_sync_o0.kernel` rediagonalizes the
     inactive and virtual blocks (`_eig_inactive_virtual`) without regard to `frozen`, so frozen
     orbitals mix with unfrozen ones and the indices drift. Frozen 1s cores: energy ~1e-7 below
     CASSCF; frozen virtuals: ~1e-3 below. CASSCF leaves the frozen columns exactly unchanged.
6. High-spin RDMs -- class `HighSpin` in `test_lasscf_rdm123s.py`: fragments `(2,0),(1,1),(2,0),(1,1)`
   (N_alpha = 6, N_beta = 2), with trace-down factors from the actual counts. aa/bb/aaa/bbb traces
   pass.
   - **Known failure:** the mixed-spin blocks (ab, aab, abb) of `make_casdm2s`/`make_casdm3s` are
     wrong when fragments have different spin polarizations (`sum dm2ab_iijj` = 16, not
     N_alpha N_beta = 12). In `LASCINoSymm.make_casdm2s` the inter-fragment Coulomb ab block (J,I)
     is filled by transposing block (I,J) ("ab = ba"), giving dm1a_I dm1b_J where dm1a_J dm1b_I is
     needed. Spin-summed `make_casdm2`/`make_casdm3` are correct, and uniform-M cases are fine.
     Nothing in Forge calls these internally; in mrh, `lassi/lassi.py`, `mcpdft/lassipdft.py` and
     `lasscf_async/crunch.py` mention the names (not yet checked whether they use these methods).
   - The original `trace_down_test` would also be wrong for M != 0 (it traces `dm2ab` over its
     beta pair and compares with `dm1b`).
7. Not done (lower priority): `localize_init_guess` with `frags_by_AOs` and `lo_coeff`;
   `state_average_n_mix` in `addons.py`.

Sparse DF and `liblas` are exercised whenever the DF tests run (c2h4n4, me2n2); nothing extra needed.

`test_lasscf_rdm123s.py` was split: the spin-separated 3-RDM comparison against FCI is in `_slow`
methods because PySCF core's `fci.direct_spin1.make_rdm123s` takes ~50 s per case here; the
LASSCF kernel and all LAS RDM builders take ~2 s.

# Other findings

- Single-state LASSCF of two-H2 systems in 6-31G converges erratically: at the
  `test_lasscf_rdm.py` geometry the energy rises ~27 mEh above the first macrocycle, and runs
  whose first macrocycle differs at 1e-11 (mrh vs Forge, or run to run) settle on different
  stationary points; at a 1.9 A separation it doesn't converge in 50 cycles. STO-3G is fine.
  Not a migration regression.

For `_slow` decisions: the whole fast mrh `tests/lasscf` battery (including old_ and async tests)
took 372 s in the step-3 run; `test_c2h4n4_slow.py` took 40 s and `test_c2h4n4_symm_slow.py` 20 s.
