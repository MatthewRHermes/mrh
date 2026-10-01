# gpu4mrh is a plugin to use NVIDIA/Intel GPUs in PySCF/MRH package
import functools
from pyscf import lib


def _own_attr(obj, key):
    '''Read *key* straight out of ``obj.__dict__``.

    Deliberately bypasses ``type(obj).__getattr__``: ``Mole.__getattr__``
    imports ``pyscf.scf``/``pyscf.dft`` and runs method resolution on every
    failed public lookup, so ``getattr`` on a Mole is neither cheap nor safe.
    '''
    if obj is None:
        return None
    try:
        return object.__getattribute__(obj, '__dict__').get(key)
    except AttributeError:
        return None


def resolve_use_gpu(instance):
    '''Resolve the GPU device handle for a patched kernel call.

    A Molecule/LASSCF records its device when it is built, so a later
    ``libgpu.init()`` assigning a different handle to the process-global
    ``lib.param.use_gpu`` cannot retarget an already-built calculation.

    Resolution order: the instance's own ``use_gpu``, then the ``mol`` it
    carries, then the process-global. Only a recorded handle counts, so a
    Molecule built without one falls back to the global like any other caller.
    '''
    for obj in (instance, _own_attr(instance, 'mol')):
        gpu = _own_attr(obj, 'use_gpu')
        if gpu is not None:
            return gpu
    return getattr(lib.param, 'use_gpu', None)


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