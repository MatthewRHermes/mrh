#!/usr/bin/env python
import contextlib
import contextvars

from pyscf import lib

_device = contextvars.ContextVar('mrh_gpu_device', default=None)


def current_device():
    '''Resolve the GPU device handle that is active right now.

    A context-scoped device set by :func:`gpu_scope` takes priority; otherwise
    fall back to the process-global ``lib.param.use_gpu``, which is only ever
    written by user scripts. Returns None when no device is active, which every
    caller already treats as "run on the CPU".
    '''
    device = _device.get()
    if device is None:
        device = getattr(lib.param, 'use_gpu', None)
    return device


@contextlib.contextmanager
def gpu_scope(handle):
    '''Pin *handle* as the active device for the duration of the block.

    Needed by call paths that reach a GPU kernel without carrying a Molecule or
    a recorded ``use_gpu``, such as the FCI RDM/TDM routines.

    Passing None is a deliberate no-op rather than an override, so CPU-only code
    paths never have to be special-cased. The previous device is restored on exit,
    including when the block raises, and scopes may be nested.
    '''
    if handle is None:
        yield
        return
    token = _device.set(handle)
    try:
        yield
    finally:
        _device.reset(token)


def _own_attr(obj, key):
    '''Read *key* straight out of ``obj.__dict__``.

    Bypasses ``type(obj).__getattr__`` deliberately: ``Mole.__getattr__`` imports
    ``pyscf.scf``/``pyscf.dft`` and runs method resolution on every failed public
    lookup, so plain ``getattr`` on a Mole is neither cheap nor safe.
    '''
    if obj is None:
        return None
    try:
        return object.__getattribute__(obj, '__dict__').get(key)
    except AttributeError:
        return None


def object_device(instance):
    '''The device recorded on *instance*, or on the Molecule it carries.

    Returns None when nothing was recorded, which callers treat as "fall back to
    the active context/global device".
    '''
    for obj in (instance, _own_attr(instance, 'mol')):
        device = _own_attr(obj, 'use_gpu')
        if device is not None:
            return device
    return None


def resolve_device(instance):
    '''The device for a calculation: recorded on *instance* or its Molecule, else
    the active context/global device.

    This is the single implementation of the resolution order -- explicit instance
    ``use_gpu``, then recorded ``mol.use_gpu``, then context, then the process
    global. It lives here rather than in ``gpu4mrh`` because this module is
    CPU-safe: callers outside the plugin (e.g. lassi ``hsi``) need the same chain
    without importing ``gpu4mrh``, which would drag in the native library.
    '''
    device = object_device(instance)
    if device is not None:
        return device
    return current_device()