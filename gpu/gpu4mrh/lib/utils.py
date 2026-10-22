# gpu4mrh is a plugin to use NVIDIA/Intel GPUs in PySCF/MRH package
import functools
from mrh.my_pyscf.gpu.context import resolve_device


def resolve_use_gpu(instance):
    '''Resolve the GPU device handle for a patched kernel call.

    A Molecule/LASSCF records its device when it is built, so a later
    ``libgpu.init()`` assigning a different handle to the process-global
    ``lib.param.use_gpu`` cannot retarget an already-built calculation.

    Thin wrapper over :func:`mrh.my_pyscf.gpu.context.resolve_device`, which owns
    the resolution order so plugin and non-plugin callers cannot drift apart.
    '''
    return resolve_device(instance)


def patch_cpu_kernel(cpu_kernel):
    '''Generate a decorator to patch cpu function to gpu function'''
    def patch(gpu_kernel):
        @functools.wraps(cpu_kernel)
        def hybrid_kernel(method, *args, **kwargs):
            if resolve_use_gpu(method) is not None:
                return gpu_kernel(method, *args, **kwargs)
            else:
                return cpu_kernel(method, *args, **kwargs)
        hybrid_kernel.__package__ = 'gpu4mrh'
        return hybrid_kernel
    return patch