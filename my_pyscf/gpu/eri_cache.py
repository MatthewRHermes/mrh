'''Lifetime of libgpu's device cache of DF 3-center integral blocks.

libgpu caches the blocks of ``dfobj.loop()`` keyed on ``id(dfobj)``. An id is only
unique among live objects, so a DF object's blocks are dropped (``evict``) when

  * it discards its integrals (``DF.reset``: geometry optimizers, scans, finite
    differences; patched in ``gpu4mrh.df.patch_pyscf``), and
  * it is garbage collected (``track`` registers a finalizer), otherwise a new DF
    object that reuses the address would be handed the old integrals.

libgpu only caches blocks of DF objects registered by ``track``, so a code path that
forgets it fails loudly (RuntimeError) instead of silently risking stale integrals.

Replacing ``_cderi`` in place is not detected: a DF object that has already run on the
GPU keeps its old blocks. After assigning a new ``_cderi`` to such an object, call
``evict(dfobj)`` or ``dfobj.reset()``. (The LAS impurity rebuild in
``lasscf_async/crunch.py`` refreshes its blocks with ``set_update_dfobj_(gpu, 1)`` instead.)

Kept free of ``gpu4mrh`` imports so CPU-safe callers (e.g. ``las_ao2mo``) can use it;
``libgpu`` is only imported when a block actually has to be dropped.
'''
import weakref

_finalizers = {} # id(dfobj) -> weakref.finalize

def _evict (addr):
    '''Free addr's blocks on all devices and unregister it (track() registers it again).'''
    _finalizers.pop (addr, None)
    from mrh.my_pyscf.gpu import libgpu
    return libgpu.evict_dfobj (addr)

def track (dfobj):
    '''Drop dfobj's cached blocks when it is garbage collected, and register it with
    libgpu. Required before any libgpu call that passes host integrals with id(dfobj)
    (i.e. may cache blocks: compute_get_jk, get_h2eff_df_v*); libgpu raises
    RuntimeError when asked to cache blocks of an unregistered dfobj.'''
    addr = id (dfobj)
    if addr in _finalizers: return
    from mrh.my_pyscf.gpu import libgpu
    libgpu.register_dfobj (addr)
    fin = weakref.finalize (dfobj, _evict, addr)
    fin.atexit = False # devices may be gone at interpreter shutdown; nothing to free then
    _finalizers[addr] = fin

def evict (dfobj):
    '''Drop dfobj's cached blocks on all devices now (it stays tracked if it caches
    again). Returns the number of blocks dropped.'''
    fin = _finalizers.get (id (dfobj))
    if fin is not None:
        return fin () # runs _evict once and detaches the finalizer
    return _evict (id (dfobj))
