# GPU-enabled LASSCF

The following is a short summary documenting how to run LASSCF (and similar) calculations using the `mrh` code accelerated by `gpu4mrh`, which supports multiple backends targeting GPUs from different vendors.

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
from mrh.my_pyscf import libgpu
from gpu4mrh import patch_pyscf
from pyscf import gto, scf

gpu = libgpu.libgpu_init()
libgpu.libgpu_set_verbose_(1)

mol = gto.M(atom'geom.xyz', basis=basis, use_gpu=gpu)
mf = scf.ROHF(mol).density_fit().run()

from mrh.my_pyscf.mcscf.lasscf_async import LASSCF
las = LASSCF(mf, (no1, no2), (ne1, ne2), use_gpu=gpu)
lo = las.set_fragments_((atom_list1, atom_list2), mf.m_coeff)

las.kernel(lo)
libgpu.libgpu_destroy_device(gpu)

```

Key modifications to a "normal" LASSCF input file are as follows.
- `from gpu4mrh import patch_pyscf` : enable monkey patching for a select number of PySCF source files, such as updating the Molecule object to track the new `use_gpu` variable.
- `from mrh.my_pyscf.gpu import libgpu` : enable access to the libgpu interface 
- `gpu = libgpu.libgpu_init()` : initialize the gpu library and return a handle. This gpu handle is to be passed to a small number of functions (and likely smaller in the future).
- `libgpu.libgpu_set_verbose_(1)` : (optional) enables outputting additional information on CPU affinity, devices used, timing summaries, and memory statistics for ERI blocks. This function needs to be called immediately after `libgpu_init()` for timing summaries to be complete. 
- `mol=gto.M(use_gpu=gpu, atom=...` : this is the key usage of the gpu handle by which most of the underlying code and algorithms in PySCF and mrh can access the gpu library.
- `las=LASSCF(mf, list((2,)*nfrags),list((2,)*nfrags), use_gpu=gpu)` : this is currently required, but expected to not be necessary soon...
- `libgpu.libgpu_destroy_device(gpu)` : always good to clean up after ourselves and prevent out-of-memory issues in more complex workflows. Also prints additional information if requested via `libgpu_set_verbose_(1)`.


## Example input file



## Status

The `CUDA`/`cuBLAS`, `SYCL`/`MKL`, and `HIP`/`hipBLAS` backends targeting, respectively, NVIDIA, Intel, and AMD GPUs are all functioning with the same level of capability as tested with several workloads.

Performance of the `SYCL` and `HIP` backends is competitive with `CUDA`. Effort is underway to improve performance of the `SYCL` backend on Intel GPUs.

The `host` and `OpenMP` backends should not be used. They remain for testing and development, but will likely be removed in the future.

Any differences observed comparing a CPU-only and GPU-accelerated run should be reported as a bug.

*Last Updated : 3-07-2025*
