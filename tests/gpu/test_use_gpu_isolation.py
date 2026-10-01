'''Multi-device handle isolation.

``lib.param.use_gpu`` is a single process-global slot, and only a user script
should write it. A script that runs more than one calculation allocates a
device per calculation, so a second ``libgpu.init()`` overwrites the global. A
Molecule/LASSCF therefore records its device at construction and the patched
kernels read that record, so an already-built calculation cannot be silently
retargeted by unrelated code writing the global later.

The first block deliberately uses only the public surface (``gto.M``,
``patch_cpu_kernel``, real PySCF wrapper objects) so it fails for the right
reason on the old global-only behaviour. The second block covers the resolver
in detail.
'''
import warnings

import numpy as np
import pytest

from gpu4mrh import patch_pyscf
from gpu4mrh.lib.utils import patch_cpu_kernel, resolve_use_gpu
from mrh.my_pyscf.gpu.context import current_device, gpu_scope

from pyscf import df, gto, lib, mcscf, scf

ATOM = 'H 0 0 0; F 0 0 1'
BASIS = 'sto-3g'


class Handle:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name


def build_mol(use_gpu=None):
    if use_gpu is None:
        return gto.M(atom=ATOM, basis=BASIS, verbose=0)
    return gto.M(atom=ATOM, basis=BASIS, verbose=0, use_gpu=use_gpu)


def make_hybrid():
    '''Dispatcher spy reporting which device the GPU branch would use.'''

    def cpu_kernel(method, *args, **kwargs):
        return 'cpu'

    def gpu_kernel(method, *args, **kwargs):
        return 'gpu:' + repr(resolve_use_gpu(method))

    return patch_cpu_kernel(cpu_kernel)(gpu_kernel)


@pytest.fixture
def hybrid():
    return make_hybrid()


def test_gto_m_does_not_modify_the_global(hybrid):
    handle = Handle('a')
    wrapper = df.DF(build_mol(handle))

    assert getattr(lib.param, 'use_gpu', None) is None
    assert hybrid(wrapper) == 'gpu:a'


def test_gto_m_records_the_global_at_construction(hybrid):
    handle = Handle('a')
    lib.param.use_gpu = handle

    wrapper = df.DF(build_mol())

    assert hybrid(wrapper) == 'gpu:a'


def test_object_without_recorded_handle_follows_the_global(hybrid):
    wrapper = df.DF(build_mol())

    lib.param.use_gpu = Handle('late')

    assert hybrid(wrapper) == 'gpu:late'


def test_assigned_object_routes_to_gpu_with_no_global(hybrid):
    wrapper = df.DF(build_mol(Handle('a')))

    lib.param.use_gpu = None

    assert hybrid(wrapper) == 'gpu:a'


def test_two_calculations_keep_separate_handles(hybrid):
    lib.param.use_gpu = Handle('a')
    wrapper_a = df.DF(build_mol())

    lib.param.use_gpu = Handle('b')
    wrapper_b = df.DF(build_mol())

    lib.param.use_gpu = Handle('c')

    assert hybrid(wrapper_a) == 'gpu:a'
    assert hybrid(wrapper_b) == 'gpu:b'


def test_explicit_argument_overrides_the_global(hybrid):
    lib.param.use_gpu = Handle('global')

    wrapper = df.DF(build_mol(Handle('explicit')))

    assert hybrid(wrapper) == 'gpu:explicit'


def test_unassigned_object_routes_to_cpu_with_no_global(hybrid):
    wrapper = df.DF(build_mol())

    lib.param.use_gpu = None

    assert hybrid(wrapper) == 'cpu'


def test_live_get_jk_is_a_gpu4mrh_hybrid_kernel():
    from pyscf.df import df_jk

    assert getattr(df_jk.get_jk, '__package__', None) == 'gpu4mrh'


def test_gto_m_stores_use_gpu():
    handle = object()

    assert vars(build_mol(handle)).get('use_gpu') is handle


def test_gto_m_without_a_device_records_nothing():
    lib.param.use_gpu = None

    assert 'use_gpu' not in vars(build_mol())


def test_resolver_prefers_instance_then_mol_then_global():
    from gpu4mrh.lib.utils import resolve_use_gpu

    lib.param.use_gpu = Handle('global')
    mol = build_mol(Handle('a'))

    assert resolve_use_gpu(mol) is mol.use_gpu
    assert resolve_use_gpu(df.DF(mol)) is mol.use_gpu

    mol_b = build_mol(Handle('b'))
    assert resolve_use_gpu(df.DF(mol_b)) is mol_b.use_gpu

    lib.param.use_gpu = Handle('later')
    assert resolve_use_gpu(mol) is mol.use_gpu


def test_resolver_reaches_mol_through_casscf():
    from gpu4mrh.lib.utils import resolve_use_gpu
    from mrh.my_pyscf.gpu.context import _own_attr

    mol = build_mol(Handle('a'))
    casscf = mcscf.CASSCF(scf.RHF(mol), 2, 2)

    assert _own_attr(casscf, 'mol') is mol
    assert resolve_use_gpu(casscf) is mol.use_gpu


def test_resolver_falls_back_to_global_for_bare_mole():
    from gpu4mrh.lib.utils import resolve_use_gpu
    from pyscf.gto.mole import Mole

    lib.param.use_gpu = Handle('global')
    bare = Mole()
    bare.build(atom=ATOM, basis=BASIS, verbose=0)

    assert resolve_use_gpu(bare) is lib.param.use_gpu


@pytest.mark.parametrize('value', ['FCImake_rdm1a', np.zeros(3), None, 7])
def test_resolver_tolerates_non_objects(value):
    from gpu4mrh.lib.utils import resolve_use_gpu

    lib.param.use_gpu = Handle('global')

    assert resolve_use_gpu(value) is lib.param.use_gpu


LASSCF_FACTORIES = [
    'mrh.my_pyscf.mcscf.lasscf_sync_o0',
    'mrh.my_pyscf.mcscf.lasscf_sync_o1',
    'mrh.my_pyscf.mcscf.lasscf_async',
    'mrh.my_pyscf.mcscf.lasscf_rdm',
]

SMALL_ATOM = 'H 0 0 0; H 0 0 0.74'


def make_las(module_name, mol_gpu=None, las_gpu=None):
    from importlib import import_module

    LASSCF = import_module(module_name).LASSCF
    if mol_gpu is None:
        mol = gto.M(atom=SMALL_ATOM, basis=BASIS, verbose=0)
    else:
        mol = gto.M(atom=SMALL_ATOM, basis=BASIS, verbose=0, use_gpu=mol_gpu)
    if las_gpu is None:
        return LASSCF(mol, [1, 1], [1, 1])
    return LASSCF(mol, [1, 1], [1, 1], use_gpu=las_gpu)


@pytest.mark.parametrize('module_name', LASSCF_FACTORIES)
def test_lasscf_inherits_mol_device(module_name):
    las = make_las(module_name, mol_gpu=Handle('a'))

    assert las.use_gpu.name == 'a'


@pytest.mark.parametrize('module_name', LASSCF_FACTORIES)
def test_lasscf_inherits_global_when_mol_unassigned(module_name):
    lib.param.use_gpu = Handle('global')
    las = make_las(module_name)

    assert las.use_gpu is lib.param.use_gpu


@pytest.mark.parametrize('module_name', LASSCF_FACTORIES)
def test_lasscf_explicit_argument_beats_mol_and_global(module_name):
    lib.param.use_gpu = Handle('global')
    las = make_las(module_name, mol_gpu=Handle('from_mol'), las_gpu=Handle('explicit'))

    assert las.use_gpu.name == 'explicit'


@pytest.mark.parametrize('module_name', LASSCF_FACTORIES)
def test_lasscf_explicit_argument_beats_global(module_name):
    lib.param.use_gpu = Handle('global')
    las = make_las(module_name, las_gpu=Handle('explicit'))

    assert las.use_gpu.name == 'explicit'


@pytest.mark.parametrize('module_name', LASSCF_FACTORIES)
def test_lasscf_records_none_when_nothing_assigned(module_name):
    lib.param.use_gpu = None

    assert make_las(module_name).use_gpu is None


@pytest.mark.parametrize('module_name', LASSCF_FACTORIES)
def test_lasscf_use_gpu_is_no_longer_deprecated(module_name):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        make_las(module_name, las_gpu=Handle('a'))

    assert [w for w in caught if 'use_gpu' in str(w.message)] == []


class TestGpuScope():
    '''gpu_scope lets object-less call paths (FCI RDM/TDM) see a device.'''

    def test_scope_overrides_the_global(self):
        lib.param.use_gpu = Handle('global')

        with gpu_scope(Handle('scoped')):
            assert current_device().name == 'scoped'

        assert current_device().name == 'global'

    def test_scope_with_none_leaves_the_global_alone(self):
        lib.param.use_gpu = Handle('global')

        with gpu_scope(None):
            assert current_device().name == 'global'

    def test_scope_with_none_on_a_cpu_run_reports_no_device(self):
        lib.param.use_gpu = None

        with gpu_scope(None):
            assert current_device() is None

    def test_scopes_nest_and_unwind_in_order(self):
        outer = Handle('outer')

        with gpu_scope(outer):
            with gpu_scope(Handle('inner')):
                assert current_device().name == 'inner'
            assert current_device() is outer

        assert current_device() is None

    def test_scope_is_restored_when_the_block_raises(self):
        lib.param.use_gpu = Handle('global')

        with pytest.raises(RuntimeError):
            with gpu_scope(Handle('scoped')):
                raise RuntimeError('boom')

        assert current_device().name == 'global'

    def test_recorded_handle_beats_the_active_scope(self):
        mol = build_mol(Handle('recorded'))

        with gpu_scope(Handle('scoped')):
            assert resolve_use_gpu(mol) is mol.use_gpu

    def test_object_without_a_recorded_handle_follows_the_scope(self):
        from pyscf.gto.mole import Mole

        bare = Mole()

        with gpu_scope(Handle('scoped')):
            assert resolve_use_gpu(bare).name == 'scoped'

    def test_scoped_device_reaches_a_patched_kernel_with_no_object(self, hybrid):
        # Stands in for _make_rdm1_spin1, which is handed a filename and vectors
        # rather than a Molecule, so the scope is its only channel.
        with gpu_scope(Handle('scoped')):
            assert hybrid('FCItrans_rdm1a') == 'gpu:scoped'

    def test_fci_kernel_falls_back_to_global_outside_any_scope(self, hybrid):
        lib.param.use_gpu = Handle('global')

        assert hybrid('FCItrans_rdm1a') == 'gpu:global'


def test_casscf_grad_scopes_the_device_from_its_mol():
    '''grad_elec must find the device through mc, not the process global.'''
    from mrh.my_pyscf.gpu.context import object_device

    mol = build_mol(Handle('from-mol'))
    casscf = mcscf.CASSCF(scf.RHF(mol), 2, 2)

    lib.param.use_gpu = Handle('global')

    assert object_device(casscf) is mol.use_gpu
