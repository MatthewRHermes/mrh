
import os
import numpy as np
from ase import Atoms
from pyscf import lib
from pyscf.pbc import scf, df
from pyscf.pbc import gto as pgto
from mrh.my_pyscf.pbc.mcscf import avas
from mrh.my_pyscf.pbc import mcscf

def make_h2_2D(intraH=1.0, interH=1.5, nx=1, ny=1, vacuum=17.5):
    ax = nx * (intraH + interH)
    by = ny * (intraH + interH)
    cz = vacuum
    cell = np.diag([ax, by, cz])
    
    positions = []
    symbols = []
    for ix in range(nx):
        for iy in range(ny):
            x0 = ix * (intraH + interH)
            y0 = iy * (intraH + interH)
            positions.append([x0, y0, vacuum / 2.0])
            positions.append([x0 + intraH, y0, vacuum / 2.0])
            symbols += ['H', 'H']

    atoms = Atoms(symbols, positions=positions, cell=cell, pbc=True)
    atoms.center()
    return atoms

# H2 Molecule in 2D 
atoms2D = make_h2_2D(intraH=0.74, interH=1.5, nx = 1, ny = 1, vacuum=17.5)
cell = pgto.Cell()
cell.a = atoms2D.cell.array
pos = atoms2D.get_positions()
sym = atoms2D.get_chemical_symbols()
cell.atom = [(sym[i], tuple(pos[i])) for i in range(len(sym))]
cell.basis = '631G'
cell.unit = 'Angstrom'
cell.max_memory = 100000
cell.ke_cutoff = 100
cell.precision = 1e-10
cell.verbose = lib.logger.INFO
cell.output = 'h2_2D_CASCI.log'
cell.build()

kmesh2D = [2, 2, 1] 
kpts = cell.make_kpts(kmesh2D, wrap_around=True)

kmf = scf.KRHF(cell, kpts=kpts).density_fit(auxbasis='def2-svp-jkfit')
kmf.max_cycle=1000
kmf.exxdiv = None
kmf.conv_tol = 1e-10
kmf.kernel()

# This is equivalent to (8e, 8o) in the supercell
kmc = mcscf.CASSCF(kmf, 2, 2)
kmc.kernel(kmf.mo_coeff)

# Different FCISolvers can be plugged in too.
