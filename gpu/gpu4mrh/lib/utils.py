# gpu4mrh is a plugin to use NVIDIA/Intel GPUs in PySCF/MRH package
import functools
from mrh.my_pyscf.gpu.context import current_device, object_device


def resolve_use_gpu(instance):
    '''Resolve the GPU device handle for a patched kernel call.

    A Molecule/LASSCF records its device when it is built, so a later
    ``libgpu.init()`` assigning a different handle to the process-global
    ``lib.param.use_gpu`` cannot retarget an already-built calculation.

    Resolution order: the instance's own ``use_gpu``, then the ``mol`` it
    carries, then the active context/global device. Only a recorded handle
    counts, so a Molecule built without one falls back like any other caller.
    '''
    device = object_device(instance)
    if device is not None:
        return device
    return current_device()


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