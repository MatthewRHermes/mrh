from gpu4mrh.mcscf import df
_gpu_init = df._ERIS.__init__
from pyscf.mcscf.df import _ERIS
_cpu_init = _ERIS.__init__

from gpu4mrh.lib.utils import resolve_use_gpu

print(f'{_gpu_init} inside CASSCF DF ERIS monkey-patched to GPU accelerated version')

def _hybrid_init(self, casscf, *args, **kwargs):
    '''Dispatch on the owning casscf, not on the _ERIS instance.

    patch_cpu_kernel's generic decorator resolves the device on ``method``,
    which here is the brand-new _ERIS: it records neither ``use_gpu`` nor
    ``mol``, so a device recorded only on the Mole would send the call to the
    CPU implementation and the ``resolve_use_gpu(casscf)`` inside
    df._ERIS.__init__ would never run. The casscf is the object that carries
    the recorded device, so both the dispatch and the kernel branch resolve
    the same handle and can never disagree.
    '''
    if resolve_use_gpu(casscf) is not None:
        return _gpu_init(self, casscf, *args, **kwargs)
    return _cpu_init(self, casscf, *args, **kwargs)

_ERIS.__init__ = _hybrid_init
