
import numpy as np
from ase import Atoms

from pyscf import lib
from pyscf.pbc import scf, gto as pgto

from mrh.my_pyscf.pbc import mcscf
from mrh.my_pyscf.pbc.fci import direct_spin1_cplx_opt, direct_spin0_cplx, direct_spin1_cplx

'''
In this file, we are comparing different type of cplx FCI solvers.
1. direct_spin1_cplx.FCISolver: 
2. direct_spin1_cplx_opt.FCISolver: This is the optimized version of the above code. (Low level optimization of the contract_2e function.)
3. direct_spin0_cplx.FCISolver: This is the optimized version of the above code for spin0/singlet symmetry.

I would also added the time taken by the different solvers to compare the performance.
'''


def make_h2_1d_cell(intraH=0.74, interH=1.5, nx=1, vacuum=17.5,
                    basis='6-31G', ke_cutoff=100, precision=1e-10, verbose=4):
    Lx = nx * (intraH + interH)

    atoms = Atoms(
        symbols=['H'] * (2 * nx),
        positions=[
            [i * (intraH + interH) + dx, vacuum / 2, vacuum / 2]
            for i in range(nx)
            for dx in (0.0, intraH)
        ],
        cell=np.diag([Lx, vacuum, vacuum]),
    )
    atoms.center()

    cell = pgto.Cell()
    cell.a = atoms.cell.array
    cell.atom = list(zip(atoms.get_chemical_symbols(), map(tuple, atoms.get_positions())))
    cell.basis = basis
    cell.unit = 'Angstrom'
    cell.max_memory = 100000
    cell.ke_cutoff = ke_cutoff
    cell.precision = precision
    cell.verbose = verbose
    cell.build()

    return cell

cell = make_h2_1d_cell()
cell.build()

kmesh1D = [7, 1, 1]

kpts = cell.make_kpts(kmesh1D, wrap_around=True)

kmf = scf.KRHF(cell, kpts=kpts).density_fit(auxbasis='def2-svp-jkfit')
kmf.max_cycle=1000
kmf.exxdiv = None
kmf.conv_tol = 1e-10
kmf.kernel()

logger = lib.logger
log = lib.logger.Logger(cell.stdout, lib.logger.TIMER_LEVEL)
cput1 = cput0 = (logger.process_clock(), logger.perf_counter())

with lib.temporary_env(kmf):
    kmc = mcscf.CASCI(kmf, 2, 2)
    kmc.fcisolver = direct_spin1_cplx.FCISolver(cell)
    kmc.kernel(kmf.mo_coeff)
    e1 = kmc.e_tot
    print("Direct Spin1 Complex FCI Energy: ", e1)

    del kmc

cput1 = log.timer('direct_spin1_cplx takes: ', *cput1)

with lib.temporary_env(kmf):
    kmc = mcscf.CASCI(kmf, 2, 2)
    kmc.fcisolver = direct_spin1_cplx_opt.FCISolver(cell)
    kmc.kernel(kmf.mo_coeff)
    e2 = kmc.e_tot
    print("Direct Spin1 Complex Optimized FCI Energy: ", e2)
    del kmc

cput1 = log.timer('direct_spin1_cplx_opt takes: ', *cput1)

with lib.temporary_env(kmf):
    kmc = mcscf.CASCI(kmf, 2, 2)
    kmc.fcisolver = direct_spin0_cplx.FCISolver(cell)
    kmc.kernel(kmf.mo_coeff)
    e3 = kmc.e_tot
    print("Direct Spin0 Complex FCI Energy: ", e3)
cput1 = log.timer('direct_spin0_cplx takes: ', *cput1)

# Now compare the energy and time taken by the different solvers.
print("\n------Summary of Energies:---------")
print(f"Direct Spin1 Complex FCI Energy          : {e1.real:.12f}")
print(f"Direct Spin1 Complex Optimized FCI Energy: {e2.real:.12f}")
print(f"Direct Spin0 Complex FCI Energy          : {e3.real:.12f}")

# Currently (commit d8eb4113d4055e5e851a3deaa2221665767a3770) 
# on 36 threads and 100 GB of memory: 
# The noted time (total wall time) is for one CASCI iteration.

# Case-1: for (12e, 12o)
# direct_spin1_cplx takes    :  113.89 seconds (CPU time)
# direct_spin1_cplx_opt takes:  24.33 seconds (CPU time)
# direct_spin0_cplx takes    :  16.33 seconds (CPU time)

# Molecular code for the same size of active space
# direct_spin1 takes    :  2.75 seconds (CPU time)
# direct_spin0 takes    :  3.36 seconds (CPU time)


# Case-2: for (14e, 14o)
# direct_spin1_cplx takes    :  1307.45 seconds (CPU time)
# direct_spin1_cplx_opt takes:  388.26 seconds (CPU time)
# direct_spin0_cplx takes    :  214.01 seconds (CPU time)

# Molecular code for the same size of active space
# direct_spin1 takes    :  194.41 seconds (CPU time)
# direct_spin0 takes    :  143.03 seconds (CPU time)

