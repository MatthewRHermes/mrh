import unittest
import numpy as np

from pyscf.pbc import gto as pgto
from pyscf.pbc import scf

from mrh.my_pyscf.pbc import mcscf
from mrh.my_pyscf.pbc.mcscf import avas

def get_xyz(nU=1, dintra= 1.3, dinter=1.4):
    unit = [("H", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, dintra)]
    repeat = 2.0 * dintra + dinter
    translated = [(elem, x, y, z + t * repeat) 
                  for t in range(nU) 
                  for elem, x, y, z in unit]
    return translated

def get_cell(nU=1, dintra=1.3, dinter=1.4, basis='6-31G', pseudo=None, maxMem=4000, verbose=0):
    cell = pgto.Cell(atom = get_xyz(nU, dintra, dinter),
                    a = np.diag([17.5, 17.5, (dinter+dintra)*nU]),
                    basis = basis,
                    pseudo = pseudo,
                    precision = 1e-10,
                    verbose = verbose,
                    max_memory = maxMem,
                    ke_cutoff = 40)
    cell.build()
    return cell

def get_cell2():
    cell = pgto.Cell(
        a=np.diag([3.0, 10.0, 10.0]),
        atom="N 0.0 0.0 0.0; N 0.0 0.0 1.1",
        basis="STO-3G",
        unit="Angstrom",
        ke_cutoff=50,
        precision=1e-12,
        verbose=0,)
    return cell

def get_cell3():
    cell = pgto.Cell(
        a=np.diag([3.0, 10.0, 10.0]),
        atom="Be 0.0 0.0 0.0",
        basis="6-31G",
        unit="Angstrom",
        max_memory=100000,
        ke_cutoff=50,
        precision=1e-12,
        verbose=0,
    )
    return cell

class KnownValues(unittest.TestCase):

    def test_krhf_limit_ncore0(self):
        cell = get_cell(nU=1, dintra=1.3, dinter=1.4)
        kmesh1D = [3, 1, 1]
        kpts = cell.make_kpts(kmesh1D, wrap_around=True)
        kmf = scf.KRHF(cell, kpts=kpts).density_fit(auxbasis='def2-svp-jkfit')
        kmf.max_cycle=100
        kmf.exxdiv = None
        e_scf = kmf.kernel()

        kmc = mcscf.CASSCF(kmf, 1, 2)
        e_cas = kmc.kernel(kmf.mo_coeff)[0]

        msg = "KRHF energy for the limit of 1 unit cell does not match with the RHF energy."
        self.assertAlmostEqual(e_scf, e_cas, places=6, msg=msg)

        del cell, kpts, kmf, kmc

    def test_krhf_limit(self):
        cell = get_cell2()
        cell.build()
        kmesh1D = [3, 1, 1]
        kpts = cell.make_kpts(kmesh1D, wrap_around=True)
        kmf = scf.KRHF(cell, kpts=kpts).density_fit(auxbasis='def2-svp-jkfit')
        kmf.max_cycle=100
        kmf.exxdiv = None
        e_scf = kmf.kernel()

        kmc = mcscf.CASSCF(kmf, 1, 2)
        e_cas = kmc.kernel(kmf.mo_coeff)[0]

        msg = "KRHF energy for the limit of 1 unit cell does not match with the RHF energy."
        self.assertAlmostEqual(e_scf, e_cas, places=6, msg=msg)

        del cell, kpts, kmf, kmc

    def test_kCASSCF_ncore0(self):
        cell = get_cell(nU=1, dintra=1.3, dinter=1.4)
        kmesh1D = [3, 1, 1]
        kpts = cell.make_kpts(kmesh1D, wrap_around=True)
        kmf = scf.KRHF(cell, kpts=kpts).density_fit(auxbasis='def2-svp-jkfit')
        kmf.max_cycle=100
        kmf.exxdiv = None
        e_scf = kmf.kernel()

        # Computed using k-CASSCF code.
        e_ref = -1.80061296232592

        kmc = mcscf.CASSCF(kmf, 2, 2)
        e_cas = kmc.kernel(kmf.mo_coeff)[0]

        msg = "KRHF energy for the limit of 1 unit cell does not match with the RHF energy."
        self.assertAlmostEqual(e_ref, e_cas, places=6, msg=msg)
    
    def test_kCASSCF(self):
        cell = get_cell3()
        cell.build()
        kmesh1D = [2, 1, 1]
        kpts = cell.make_kpts(kmesh1D, wrap_around=True)
        kmf = scf.KRHF(cell, kpts=kpts).density_fit(auxbasis='def2-svp-jkfit')
        kmf.max_cycle=100
        kmf.exxdiv = None
        e_scf = kmf.kernel()

        e_ref = -14.26898884190738
        mo_coeff = avas.kernel(kmf, ['Be 2s', 'Be 2p'], minao=cell.basis)[2]
        mo_coeff = np.array(mo_coeff)

        kmc = mcscf.CASSCF(kmf, 4, (1, 1))
        e_cas = kmc.kernel(mo_coeff)[0]

        msg = "KRHF energy for the limit of 1 unit cell does not match with the RHF energy."
        self.assertAlmostEqual(e_ref, e_cas, places=6, msg=msg)

if __name__ == '__main__':
    unittest.main()