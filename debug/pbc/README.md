# k-LASSCF checks

For periodic LiH with a bond length of 1.6 Å, we use the STO-3G basis and two
active electrons in two active orbitals per cell. We do these tests in 1D, 2D,
and 3D:

- Orbital gradients: compare with forward and centered energy differences.
- CI gradients: compare with forward energy differences.
- Orbital Hessian-vector products: compare with forward orbital-gradient differences.
- CI Hessian-vector products: compare with forward CI-gradient differences.
- Active–active orbital rotations: check both gradients and Hessian-vector products.
- Automatic fragment Hamiltonians: compare with direct Wannier projection.

We keep CI vectors fixed for orbital changes and orbitals fixed for CI changes.
Relative errors use the numerical energy or gradient change as the denominator.
The forward-difference errors should decrease linearly with the step size;
centered energy-difference errors should decrease quadratically.
We fit one log-log slope using all scan points except the first four and require it above 0.8.
