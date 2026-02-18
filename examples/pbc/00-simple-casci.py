
import os
import numpy as np
from ase import Atoms
from pyscf import lib
from pyscf.pbc import scf, df
from pyscf.pbc import gto as pgto
from mrh.my_pyscf.pbc.mcscf import avas
from mrh.my_pyscf.pbc import mcscf

# Example to run the CASCI for kmf object (1D, 2D or 3D)

# TODO: Write one general function to make the H2 chain in 1D, 2D and 3D. 
# The current code is a bit redundant.
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

def make_h2_3D(intraH=1.0, interH=1.5, nx=1, ny=1, nz=1, vacuum=17.5):
    pitch = intraH + interH
    ax = nx * pitch
    by = ny * pitch
    cz = nz * pitch
    cell = np.diag([ax, by, cz])

    positions = []
    symbols = []

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                x0 = ix * pitch
                y0 = iy * pitch
                z0 = iz * pitch
                positions.append([x0,          y0, z0])
                positions.append([x0 + intraH, y0, z0])
                symbols += ['H', 'H']

    atoms = Atoms(symbols, positions=positions, cell=cell, pbc=True)
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
cell.output = 'h2_1D_CASCI.log'
cell.build()

kmesh1D = [7, 1, 1]

kpts = cell.make_kpts(kmesh1D, wrap_around=True)

kmf = scf.KRHF(cell, kpts=kpts).density_fit(auxbasis='def2-svp-jkfit')
kmf.max_cycle=1000
kmf.exxdiv = None
kmf.conv_tol = 1e-10
kmf.kernel()

# This is equivalent to (14e, 14o) in the supercell
kmc = mcscf.CASCI(kmf, 2, 2)
kmc.kernel(kmf.mo_coeff)


del cell, kpts, kmf, kmc

# H2 Molecule in 2D 
atoms2D = make_h2_2D(intraH=0.74, interH=1.5, nx = 1, ny = 1, vacuum=17.5)
cell = pgto.Cell()
cell.a = atoms2D.cell.array
pos = atoms2D.get_positions()
sym = atoms2D.get_chemical_symbols()
cell.atom = [(sym[i], tuple(pos[i])) for i in range(len(sym))]
cell.basis = 'STO-6G'
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
kmc = mcscf.CASCI(kmf, 2, 2)
kmc.kernel(kmf.mo_coeff)


# H2 Molecule in 2D 
atoms2D = make_h2_2D(intraH=0.74, interH=1.5, nx = 1, ny = 1, vacuum=17.5)
cell = pgto.Cell()
cell.a = atoms2D.cell.array
pos = atoms2D.get_positions()
sym = atoms2D.get_chemical_symbols()
cell.atom = [(sym[i], tuple(pos[i])) for i in range(len(sym))]
cell.basis = 'STO-6G'
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
kmc = mcscf.CASCI(kmf, 2, 2)
kmc.kernel(kmf.mo_coeff)

del cell, kpts, kmf, kmc

# H2 Molecule in 3D 
atoms3D = make_h2_3D(intraH=0.74, interH=1.5, nx = 1, ny = 1, nz=1, vacuum=17.5)
cell = pgto.Cell()
cell.a = atoms3D.cell.array
pos = atoms3D.get_positions()
sym = atoms3D.get_chemical_symbols()
cell.atom = [(sym[i], tuple(pos[i])) for i in range(len(sym))]
cell.basis = 'STO-6G'
cell.unit = 'Angstrom'
cell.max_memory = 100000
cell.ke_cutoff = 100
cell.precision = 1e-10
cell.verbose = lib.logger.INFO
cell.output = 'h2_3D_CASCI.log'
cell.build()

kmesh3D = [2, 2, 2] 
kpts = cell.make_kpts(kmesh3D, wrap_around=True)

kmf = scf.KRHF(cell, kpts=kpts).density_fit(auxbasis='def2-svp-jkfit')
kmf.max_cycle=1000
kmf.exxdiv = None
kmf.conv_tol = 1e-10
kmf.kernel()

# This is equivalent to (16e, 16o) in the supercell
# Note: this is very slow: c-CASCI beyond (14e,14o) is very expensive.
kmc = mcscf.CASCI(kmf, 2, 2)
kmc.kernel(kmf.mo_coeff)