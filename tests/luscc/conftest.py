import pytest
from pyscf import gto, scf

from mrh.my_pyscf.lassi import lassis
from mrh.my_pyscf.mcscf.lasscf_sync_o0 import LASSCF


@pytest.fixture(scope="module")
def h4_las():
    mol = gto.M(
        atom="H 0 0 0; H 1 0 0; H 3 0 0; H 4 0 0",
        basis="sto-3g",
        symmetry=False,
        verbose=0,
        output="/dev/null",
    )
    mf = scf.RHF(mol).run()
    las = LASSCF(mf, (2, 2), (2, 2), spin_sub=(1, 1))
    mo_coeff = las.localize_init_guess(((0, 1), (2, 3)), mf.mo_coeff)
    las.kernel(mo_coeff)
    yield las
    mol.stdout.close()


@pytest.fixture(scope="module")
def h4_lassis(h4_las):
    return lassis.LASSIS(h4_las).run()
