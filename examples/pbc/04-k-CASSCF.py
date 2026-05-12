import numpy as np

from pyscf import lib
from pyscf.pbc import scf
from pyscf.pbc import gto as pgto
from mrh.my_pyscf.pbc.mcscf import avas
from mrh.my_pyscf.pbc import mcscf


intraH = 0.74
interH = 1.5
nx = 1
ny = 1
vac = 17.5

ax = nx * (intraH + interH)
by = ny * (intraH + interH)
cz = vac

cell = pgto.Cell()
cell.a = np.diag([ax, by, cz])
cell.atom = [
    ["H", (0.0, 0.0, vac / 2.0)],
    ["H", (intraH, 0.0, vac / 2.0)],
]
cell.basis = "631G"
cell.unit = "Angstrom"
cell.max_memory = 100000
cell.ke_cutoff = 100
cell.precision = 1e-10
cell.verbose = lib.logger.INFO
cell.build()

kmesh2D = [3, 1, 1] 
kpts = cell.make_kpts(kmesh2D, wrap_around=True)

kmf = scf.KRHF(cell, kpts=kpts).density_fit(auxbasis='def2-svp-jkfit')
kmf.exxdiv = None
kmf.conv_tol = 1e-10
kmf.kernel()

from mrh.my_pyscf.pbc.fci import csf_solver
kmc = mcscf.CASSCF(kmf, 2, 2)
kmc.kpts = kpts  
kmc.kmesh = kmesh2D
kmc.fcisolver = csf_solver(cell, smult=1)
kmc.max_cycle_macro = 50
kmc.kernel(kmf.mo_coeff)

nkpts = np.prod(kmesh2D)
print(f"k-RHF energy: {kmf.e_tot.real:12.8f}")
print(f"k-CASSCF energy: {kmc.e_tot.real:12.8f}")