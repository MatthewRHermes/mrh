'''Lifetime of libgpu's device cache of DF 3-center integral blocks.

libgpu caches the blocks of ``dfobj.loop()`` keyed on ``id(dfobj)``. An id is only
unique among live objects, so a DF object's blocks are dropped (``evict``) when

  * it discards its integrals (``DF.reset``: geometry optimizers, scans, finite
    differences; patched in ``gpu4mrh.df.patch_pyscf``), and
  * it is garbage collected (``track`` registers a finalizer), otherwise a new DF
    object that reuses the address would be handed the old integrals.

Kept free of ``gpu4mrh`` imports so CPU-safe callers (e.g. ``las_ao2mo``) can use it;
``libgpu`` is only imported when a block actually has to be dropped.
'''
import weakref

_finalizers = {} # id(dfobj) -> weakref.finalize

def _evict (addr):
    _finalizers.pop (addr, None)
    from mrh.my_pyscf.gpu import libgpu
    return libgpu.evict_dfobj (addr)

def track (dfobj):
    '''Drop dfobj's cached blocks when it is garbage collected. Call before handing
    id(dfobj) to libgpu with host data that may be cached.'''
    addr = id (dfobj)
    if addr in _finalizers: return
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
