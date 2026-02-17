import numpy as np
import os
from pyscf.pbc import scf, df, ao2mo
from pyscf.pbc import gto as pgto
from mrh.my_pyscf.pbc.mcscf import avas

'''
Run the CASCI calculation for the PA chain system.
'''

def get_xyz(nU=1, d= 2.47):
    '''
    Generate atomic coordinates for a system with nU unit cells.
    args:
        nU: Number of unit cells
        d:  lattice vector of UC
    '''
    coords = [
    ("C", -0.5892731038,  0.3262391909,  0.0000000000),
    ("H", -0.5866101958,  1.4126530287,  0.0000000000),
    ("C",  0.5916281105, -0.3261693897,  0.0000000000),
    ("H",  0.5889652025, -1.4125832275,  0.0000000000)]

    translated_coords = []
    for t in range(nU):
        shift = t * d
        translated_coords.extend([(elem, x + shift, y, z)
            for elem, x, y, z in coords])
    return translated_coords

def get_gdf(filename, kpts=None, restart=True):
    """
    Calculate the 2e Integrals using the Gaussian Density Fitting.
    """
    if not os.path.exists(filename) or restart:
        gdf = df.GDF(cell, kpts=kpts)
        gdf._cderi_to_save = filename
        gdf.build()
    return filename

# Build the Cell object
nU = 1
d = 2.47
basis = '6-31G'
pseudo = None
maxMem = 120000

cell = pgto.Cell(atom = get_xyz(nU, d),
                 a = np.diag([d*nU, 17.5, 17.5]),
    basis = basis,
    pseudo = pseudo,
    precision = 1e-10,
    verbose = 3, #lib.logger.INFO,
    max_memory = maxMem,
    ke_cutoff = 40,
)
cell.build()

nk = [3, 1, 1]
kpts = cell.make_kpts(nk)
nC = nU * nk[0]

kmf = scf.KRHF(cell, kpts=kpts).density_fit(auxbasis='def2-svp-jkfit')
kmf.max_cycle=1000
kmf.chkfile = f'PAchain.{nC}.chk'
kmf.with_df._cderi = get_gdf(f'PAchain.{nC}.gdf', kpts, restart=False)
kmf.exxdiv = None
kmf.init_guess = 'chk'
kmf.conv_tol = 1e-10
meanfieldenergy = kmf.kernel()

# Use the right AVAS, for the active space selection.
ncas, nelecas, mo_coeff = avas.kernel(kmf, ['C 2pz'], minao=cell.basis, threshold=0.01, canonicalize=True)[:3]

# Print the molden file for the active space orbitals.
from pyscf.tools import molden
molden.from_mo(kmf.cell, f'PAchain.{nC}.molden', np.hstack([mo_coeff[k][:, 6:8] for k in range(len(kpts))]).real ) 

from mrh.my_pyscf.pbc import mcscf
kmc = mcscf.CASCI(kmf, 2, 2)
kmc.kernel(mo_coeff)
