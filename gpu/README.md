# GPU-enabled LASSCF

The following is a short summary documenting how to run LASSCF calculations using the `mrh` code accelerated by `gpu4mrh`, which supports multiple backends targeting GPUs from different vendors. Similar calculations like HF, DFT and CASSCF ran with density fitting can also be accelerated transparently. For HF and DFT, only the construction of J and K matrices is accelerated. For CASSCF, JK and AO2MO kernels are accelerated. 

## Compiling gpu-enabled LASSCF

Examples for compiling the full software stack PySCF + mrh + gpu4mrh on a handful of HPC systems is available in the `machine` directory.

## Running gpu-enabled LASSCF calculations

An example input-deck is provided in the following directory: `mrh/examples/gpu/polymer_async`.

The following is a partial example code focusing on those lines key for running a LASSCF calculation.

```bash
from pyscf import gto, scf
mol = gto.M(atom'geom.xyz', basis=basis)
mf = scf.ROHF(mol).density_fit().run()

from mrh.my_pyscf.mcscf.lasscf_async import LASSCF
las = LASSCF(mf, (no1, no2), (ne1, ne2))
lo = las.set_fragments_((atom_list1, atom_list2), mf.m_coeff)

las.kernel(lo)
```

The same sample can be modified as below to enable GPU-accelerated calculations.

```bash
from mrh.my_pyscf.gpu import libgpu
import pyscf
from gpu4mrh import patch_pyscf
from pyscf import gto, scf, lib

gpu = libgpu.init()
lib.param.use_gpu = gpu
libgpu.set_verbose_(gpu, 1)

mol = gto.M(atom'geom.xyz', basis=basis)
mf = scf.ROHF(mol).density_fit().run()

from mrh.my_pyscf.mcscf.lasscf_async import LASSCF
las = LASSCF(mf, (no1, no2), (ne1, ne2))
lo = las.set_fragments_((atom_list1, atom_list2), mf.m_coeff)

las.kernel(lo)
libgpu.destroy_device(gpu)

```

Key modifications to a "normal" LASSCF input file are as follows.
- `from gpu4mrh import patch_pyscf` : enable monkey patching for a select number of PySCF source files, such as updating the Molecule object to track the new `use_gpu` variable.
- `from mrh.my_pyscf.gpu import libgpu` : enable access to the libgpu interface 
- `gpu = libgpu.init()` : initialize the gpu library and return a handle. This gpu handle is to be passed to a small number of functions (and likely smaller in the future).
- 'lib.param.use_gpu = gpu' : sets the global gpu handle ensuring all supported features use the same device
- `libgpu.set_verbose_(gpu, 1)` : (optional) enables outputting additional information on CPU affinity, devices used, timing summaries, and memory statistics for ERI blocks. This function needs to be called immediately after `libgpu_init()` for timing summaries to be complete. 

- `libgpu.destroy_device(gpu)` : always good to clean up after ourselves and prevent out-of-memory issues in more complex workflows. Also prints additional information if requested via `libgpu.set_verbose_(gpu, 1)`.

## Assigning a device per calculation

`lib.param.use_gpu` is a single process-global slot, and only the user script writes it. A script that runs more than one calculation allocates a device per calculation, so a second `libgpu.init()` overwrites that global.

To keep an already-built calculation from being retargeted, a Molecule records its device when it is constructed, and the patched kernels read that record:

- `gpu = libgpu.init()` then `lib.param.use_gpu = gpu`, before building anything, pins every Molecule created afterwards to that device.
- `mol = gto.M(use_gpu=gpu, atom=...)` assigns a device explicitly and overrides the global for that Molecule.
- `las = LASSCF(mf, list((2,)*nfrags), list((2,)*nfrags), use_gpu=gpu)` does the same for the LASSCF object, which passes it to the impurity calculations it creates.

A Molecule records whichever of those two applies, so the handle is fixed for the life of the object. Only a recorded handle counts, so a Molecule built without one simply follows the global like any other caller. This is why the assignment order matters — set the global, or pass `use_gpu=`, *before* constructing the Molecule.

The patched kernels resolve a device in this order: the object's own `use_gpu`, then the `mol` it carries, then a context-scoped device, then `lib.param.use_gpu`. Existing scripts that set the global before building continue to work unchanged.

### The FCI kernels

The FCI RDM/TDM kernels (`gpu4mrh/fci/rdm.py`, `gpu4mrh/fci/direct_spin1.py`, `gpu4mrh/fci/rdm_loops.py`) are entered with plain arrays and no object reference, so they have nothing to read a recorded handle from. They resolve through `mrh.my_pyscf.gpu.context.current_device()`, which consults a context variable before the global. Code that knows its device wraps the call:

```python
from mrh.my_pyscf.gpu.context import gpu_scope

with gpu_scope (self.use_gpu):
    casdm1, casdm2 = mc.fcisolver.make_rdm12 (ci, ncas, nelecas)
```

`gpu_scope(None)` is a no-op, so CPU-only code needs no special-casing, and the scope unwinds on exception and may be nested. `mrh/my_pyscf/gpu/context.py` imports nothing from `gpu4mrh` and loads no shared library, so it is safe to import from a CPU-only install.

Most LASSCF density matrices never reach these kernels: `make_rdm1`/`make_rdm12` go through `self.fciboxes`, whose `FCIBox.states_make_rdm1s` slices the CI vector directly in numpy. The GPU-only paths are the 3-RDMs, the CASSCF gradient, and the lassi operator/TDM routines.

Two lassi sites still follow the global because their device cannot be reached: `my_pyscf/lassi/op_o0.py` `_make_rdm3s_spinless_pair` is a module-level function with no object, and `my_pyscf/lassi/op_o1/frag.py` `_trans_rdm12s_loop` is a method whose class holds no Molecule. Threading a handle to those is left as future work; a script that uses them should set `lib.param.use_gpu` itself.

Note that `use_gpu` is not JSON-serializable, so `mol.dumps` drops it. A checkpoint written by `my_pyscf/mcscf/chkfile.py` therefore loses the recorded handle, and a restored Molecule falls back to the global.

One device per process is assumed throughout. `libgpu.init()` returns a handle that spans every GPU visible to the process (`cudaGetDeviceCount`, so `CUDA_VISIBLE_DEVICES` selects a node slice), and each handle allocates its own ERI/JK/LASSCF buffers, so a second handle in the same process means a second full set of allocations.

## Example input file



## Status

The `CUDA`/`cuBLAS`, `SYCL`/`MKL`, and `HIP`/`hipBLAS` backends targeting, respectively, NVIDIA, Intel, and AMD GPUs are all functioning with the same level of capability as tested with several workloads.

Performance of the `SYCL` and `HIP` backends is competitive with `CUDA`. Effort is underway to improve performance of the `SYCL` backend on Intel GPUs.

The `host` backend is fully functional, but only used for development and testing when GPU is not available.

The `OpenMP` backend is stale and should not be used. It remains as a placeholder for possible future testing and development, but will likely be removed in the future.

Any differences observed comparing a CPU-only and GPU-accelerated run should be reported as a bug.

## Deprecated GPU code: the retired `orbital_response` engine

Several `DEPRECATED` comments in `mrh/gpu/src` and in `mrh/my_pyscf` mark blocks of GPU code that have been commented out. They are gathered here so that any one banner can be read on its own without opening the source. The blocks all belong to a single retired subsystem, which is why this is one section rather than nine. 

The retired code is an older GPU "legacy integral engine" built around `Device::orbital_response`, the GPU implementation of the orbital-response step of the LASSCF Hessian/gradient. It was superseded by a numpy implementation and then commented out rather than deleted, so that it can be revived. **This affects only the orbital-response step.** The kernels that dominate LASSCF runtime -- J/K construction, AO2MO, and the density-fitted integral transformations -- are all still GPU-accelerated and unaffected, so ordinary input decks need no changes on account of what follows.

**`src/device/device.cpp`** -- banner above the `#if 0`-wrapped body of `Device::orbital_response()`, the retired implementation itself. The body is preserved verbatim so it can be brought back. It is genuinely gone from the build: `nm` reports zero `orbital_response` symbols in `libgpu.so`.

**`src/device/device.h`** -- banner above the commented-out declaration. The declaration sits directly above the commented-out `fdrv` declaration, so this one comment covers both, and it lists what a revival would require: the `device.cpp` implementation, the `libgpu.h`/`libgpu.cpp` bindings, and the `fdrv` helper.

**`src/libgpu.cpp`** -- banner above the `#if 0`-wrapped `libgpu_orbital_response`, the pybind11 wrapper that used to expose the routine to Python. With it commented out, `libgpu.orbital_response` no longer exists as a module attribute.

**`src/pm/cuda/jk.cpp`, `src/pm/hip/jk.cpp`, `src/pm/sycl/jk.cpp`, `src/pm/host/jk.cpp`, `src/pm/openmp/jk.cpp`** (five identical banners) -- the per-backend definitions of `Device::fdrv`, the scratch helper the legacy engine used to expand tril-packed integrals and contract them against the MO coefficients. Because the legacy engine was its only caller, these are commented out across all five backends together. The `OpenMP` copy additionally has its `get_jk`/`init_get_jk` callers commented out; that backend is stale in any case (see **Status** above).

**`mrh/my_pyscf/mcscf/lasscf_sync_o0.py`** -- the comment in `orbital_response_2cum`, the Python side of the same story. The `use_gpu` branch that called `libgpu.orbital_response` is commented out, so the numpy path beneath it always runs and `self.las.use_gpu` is ignored at that point. **This is a performance change, not a correctness one:** the same mathematics is computed either way, so CPU and GPU runs still agree. The pure-Python `orbital_response` methods in `lasscf_async`, `lasscf_sync_o1` and `lasscf_rdm` were never on this path and are unaffected.

All of these blocks are marked `DEPRECATED` and are reversible, but they must be re-enabled as a set rather than one at a time. Each banner names the specific pieces to restore, and that list is the complete set.

One consequence of the same retirement is worth knowing:

- Calling `libgpu.orbital_response(...)` from Python now raises `AttributeError`. No example under `mrh/examples` does this, but the standalone harness `mrh/gpu/mini-apps/orbital_response/main.py` still does, at the uncommented calls on lines 53 and 81, so that mini-app no longer runs against the current build.

*Last Updated : 10-01-2026*
