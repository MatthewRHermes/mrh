# Moved to PySCF-Forge in the LASSCF migration (see MIGRATION.md). The mrh classes add gpu4mrh support.
# This module re-exports all of pyscf.mcscf.las.lasscf_rdm (including private names) so that existing
# imports keep working.
import pyscf.mcscf.las.lasscf_rdm as _forge
globals ().update ({k: v for k, v in vars (_forge).items () if not k.startswith ('__')})
from mrh.my_pyscf.mcscf import lasci, lasscf_sync_o0

class LASSCF_HessianOperator (_forge.LASSCF_HessianOperator, lasscf_sync_o0.LASSCF_HessianOperator):
    pass

class LASSCFNoSymm (lasscf_sync_o0._LASSCFMixin, _forge.LASSCFNoSymm):
    _hop = LASSCF_HessianOperator

def LASSCF (mf_or_mol, ncas_sub, nelecas_sub, **kwargs):
    if isinstance(mf_or_mol, gto.Mole):
        mf = scf.RHF(mf_or_mol)
    else:
        mf = mf_or_mol
    if mf.mol.symmetry: 
        raise NotImplementedError ("point-group symmetry")
        #las = LASSCFSymm (mf, ncas_sub, nelecas_sub, **kwargs)
    else:
        las = LASSCFNoSymm (mf, ncas_sub, nelecas_sub, **kwargs)
    if getattr (mf, 'with_df', None):
        las = lasci.density_fit (las, with_df = mf.with_df) 
    return las
