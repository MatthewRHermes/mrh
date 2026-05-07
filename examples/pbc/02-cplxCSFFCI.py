import numpy as np
from ase import Atoms
from pyscf import lib
from pyscf.pbc import scf
from pyscf.pbc import gto as pgto
from mrh.my_pyscf.pbc import mcscf

def make_h2_1D(intraH=1.0, interH=1.5, nx=1, vacuum=17.5):
    Lx = nx * (intraH + interH)
    cell = np.diag([Lx, vacuum, vacuum])
    y0 = vacuum / 2.0
    z0 = vacuum / 2.0
    positions = []
    symbols = []
    for i in range(nx):
        x0 = i * (intraH + interH)
        positions.append([x0,        y0, z0])   # H1
        positions.append([x0+intraH, y0, z0])   # H2
        symbols += ['H', 'H']
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, )
    atoms.center()
    return atoms


# H2 Molecule in 1D
atoms1D = make_h2_1D(intraH=0.74, interH=1.5, nx = 1, vacuum=17.5)
cell = pgto.Cell()
cell.a = atoms1D.cell.array
pos = atoms1D.get_positions()
sym = atoms1D.get_chemical_symbols()
cell.atom = [(sym[i], tuple(pos[i])) for i in range(len(sym))]
cell.basis = 'STO-6G'
cell.unit = 'Angstrom'
cell.max_memory = 100000
cell.ke_cutoff = 100
cell.precision = 1e-10
cell.verbose = lib.logger.INFO
cell.build()

kmesh1D = [5, 1, 1]

kpts = cell.make_kpts(kmesh1D, wrap_around=True)

kmf = scf.KRHF(cell, kpts=kpts).density_fit(auxbasis='def2-svp-jkfit')
kmf.max_cycle=1000
kmf.exxdiv = None
kmf.conv_tol = 1e-10
kmf.kernel()

# This is equivalent to (10e, 10o) in the supercell
from mrh.my_pyscf.pbc.fci import csf_solver
kmc = mcscf.CASCI(kmf, 2, 2)
kmc.fcisolver = csf_solver(cell, smult=1)
kmc.kernel(kmf.mo_coeff)

