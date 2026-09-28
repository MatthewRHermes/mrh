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

Deprecated: the following input-modifications are now deprecated.

- `mol=gto.M(use_gpu=gpu, atom=...` : this is the key usage of the gpu handle by which most of the underlying code and algorithms in PySCF and mrh can access the gpu library.
- `las=LASSCF(mf, list((2,)*nfrags),list((2,)*nfrags), use_gpu=gpu)` : this is currently required, but expected to not be necessary soon...

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

*Last Updated : 9-28-2026*
