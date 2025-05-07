from pyscf import lib
from pyscf.pbc import gto, scf, df
from mrh.my_pyscf.pdmet import runpDMET 
import numpy as np
import h5py
np.set_printoptions(precision=4)

# Define the cell
cell = gto.Cell(basis = 'gth-tzvp',pseudo = 'gth-pade', a = np.eye(3) * 12, max_memory = 5000)
cell.atom = '''
N 0 0 0
N 0 0 1.1
Ne 0 0 20
'''
cell.verbose = 4
cell.build()

# Integral generation
gdf = df.GDF(cell)
gdf_file = 'N2.h5'
gdf._cderi_to_save = gdf_file
gdf.build()

mf = scf.RHF(cell).density_fit()
mf.exxdiv = None
mf.with_df._cderi = gdf_file
mf.kernel()

# Now, the aim is to create a new h5 file, where we are changing the eri integrals
# and using that eri integrals we want to get the mf energy: If this is done properly,
# then mf.e_tot = newmf.e_tot 

from pyscf import ao2mo
from functools import reduce

def _get_cderi_transformed(kmf, mo):
    """
    Transforms CDERI integrals from AO to MO basis.
    Lpq---> Lij 
    Args:
        mo: np.array (nao*neo)
    Returns:
        Transformed CDERI integrals (Lij).
    """
    assert mo.ndim == 2, "MO_coeff should be a 2D array"

    nmo = mo.shape[-1]
    
    Lij = np.empty((kmf.with_df.get_naoaux(), nmo * (nmo + 1) // 2), dtype=mo.dtype)
    
    ijmosym, mij_pair, moij, ijslice = ao2mo.incore._conc_mos(mo, mo, compact=True)
    b0 = 0
    for eri1 in kmf.with_df.loop():
        b1 = b0 + eri1.shape[0]
        eri2 = Lij[b0:b1]
        eri2 = ao2mo._ao2mo.nr_e2(eri1, moij, ijslice, aosym='s2', mosym=ijmosym, out=eri2)
        b0 = b1
    return Lij

def _get_eri_transformed(kmf, mo, oldfile):
    Lij = _get_cderi_transformed(kmf, mo)
    old_gdf = h5py.File(oldfile, 'r')
    new_gdf = h5py.File(f'df_{oldfile}', 'w') 
    for key in old_gdf.keys():
        if key == 'j3c':
            new_gdf['j3c/0/0'] = Lij
        else:
            old_gdf.copy(old_gdf[key], new_gdf, key)
    old_gdf.close()
    new_gdf.close()
    return f'df_{oldfile}'

def _get_h1e_transformed(kmf, mo):
    """
    Transforms the one-electron integrals from AO to MO basis.
    Args:
        kmf: pyscf object
        ao2eo: np.array (nao*neo)
            Transformation matrix from AO to EO
    Returns:
        h1e: np.array (nmo*nmo)
            Transformed one-electron integrals
    """
    h1e = kmf.get_hcore()
    h1e = np.einsum('pi,ij,jq->pq', mo.T, h1e, mo)
    return h1e

def _get_dm_mo(kmf, mo):
    dm = kmf.make_rdm1()
    s = kmf.get_ovlp()
    mo = kmf.mo_coeff
    dm_mo = reduce(np.dot, (mo.T, s,dm, s,mo))
    return dm_mo

def _get_vhf(kmf, erifile):
    """
    Get the two-electron integrals in the MO basis.
    Args:
        kmf: pyscf object
        erifile: str
            Filename of the ERI file
    Returns:
        vhf: np.array (nmo*nmo)
            Transformed two-electron integrals
    """
    cell = kmf.cell
    mo = kmf.mo_coeff
    dm_mo =  _get_dm_mo(kmf, mo)

    mftemp = scf.RHF(cell, exxdiv = None).density_fit()
    mftemp.with_df._cderi = erifile
    veff = mftemp.get_veff(dm=dm_mo, hermi=1)
    return veff


dm_mo = _get_dm_mo(mf, mf.mo_coeff) 
h1e = _get_h1e_transformed(mf, mf.mo_coeff)
vhf = _get_vhf(mf, _get_eri_transformed(mf, mf.mo_coeff, gdf_file))
e_tot = scf.hf.energy_elec(mf, dm=dm_mo, h1e=h1e, vhf=vhf)[0]
e_tot += mf.energy_nuc()

assert abs(mf.e_tot - e_tot) < 1e-7, f"Something went wrong. {mf.e_tot} != {e_tot}"

neo = h1e.shape[0]
emb_mf = scf.ROHF(cell).density_fit()
erifile = _get_eri_transformed(mf, mf.mo_coeff, gdf_file)
emb_mf.with_df._cderi = erifile
emb_mf.exxdiv = None
emb_mf.get_hcore = lambda *args: h1e
emb_mf.get_ovlp  = lambda *args: np.eye(neo)
emb_mf.conv_tol = 1e-12
emb_mf.energy_nuc = lambda *args: mf.energy_nuc() # Later, will add through core-energy contribution
emb_mf.kernel(dm_mo)

print(emb_mf.e_tot-mf.e_tot)
