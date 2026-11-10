# Moved to PySCF-Forge in the LASSCF migration (see MIGRATION.md). The mrh-only parts (gpu4mrh, lassi) stay here.
# This module re-exports all of pyscf.mcscf.las.lasci (including private names) so that existing
# imports keep working.
import pyscf.mcscf.las.lasci as _forge
globals ().update ({k: v for k, v in vars (_forge).items () if not k.startswith ('__')})
from pyscf import lib
from mrh.my_pyscf.mcscf import las_ao2mo as _mrh_las_ao2mo

class _LASCIMixin:
    '''mrh-only features of the LASCI classes: gpu4mrh support and the lassi method.
    It goes first in the bases, so its methods wrap the Forge ones through super ().'''

    def __init__(self, mf, *args, **kwargs):
        mol = getattr(mf, 'mol', None)
        default_use_gpu = getattr(mol, '__dict__', {}).get('use_gpu')
        if default_use_gpu is None:
            default_use_gpu = getattr(lib.param, 'use_gpu', None)
        self.use_gpu = kwargs.pop('use_gpu', default_use_gpu)
        super ().__init__(mf, *args, **kwargs)

    def _get_h2eff_df (self, mo_coeff):
        if self.use_gpu: return _mrh_las_ao2mo.get_h2eff_gpu_v2 (self, mo_coeff)#the full version is not working yet.
        return super ()._get_h2eff_df (mo_coeff)

    def lassi(self, mo_coeff=None, ci=None, veff_c=None, h2eff_sub=None, orbsym=None,
              soc=False, break_symmetry=False, opt=1, **kwargs):
        #import warnings
        #lassi_kernel_warn = "Now LASSI have kernel, which takes las instance as input. This [las.lassi()] function " \
        #                    "will be removed soon."
        #warnings.warn(lassi_kernel_warn, stacklevel=3)
        from mrh.my_pyscf.lassi import lassi
        mylassi = lassi.LASSI(self, mo_coeff=mo_coeff, ci=ci, soc=soc, opt=opt,
                              break_symmetry=break_symmetry, **kwargs)
        return mylassi.kernel(mo_coeff=mo_coeff, ci=ci, veff_c=veff_c, h2eff_sub=h2eff_sub,
                              orbsym=orbsym)

class LASCINoSymm (_LASCIMixin, _forge.LASCINoSymm):
    pass

class LASCISymm (LASCINoSymm, _forge.LASCISymm):
    pass

def LASCI (mf_or_mol, ncas_sub, nelecas_sub, **kwargs):
    if isinstance(mf_or_mol, gto.Mole):
        mf = scf.RHF(mf_or_mol)
    else:
        mf = mf_or_mol
    if mf.mol.symmetry: 
        las = LASCISymm (mf, ncas_sub, nelecas_sub, **kwargs)
    else:
        las = LASCINoSymm (mf, ncas_sub, nelecas_sub, **kwargs)
    if getattr (mf, 'with_df', None):
        las = density_fit (las, with_df = mf.with_df) 
    return las
