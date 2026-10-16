import pytest
from pyscf import lib


@pytest.fixture(autouse=True)
def _reset_use_gpu():
    yield
    lib.param.use_gpu = None