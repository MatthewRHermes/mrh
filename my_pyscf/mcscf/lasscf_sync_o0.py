# Moved to PySCF-Forge in the LASSCF migration (see MIGRATION.md). The gpu4mrh parts stay here.
# This module re-exports all of pyscf.mcscf.las.lasscf (including private names) so that existing
# imports keep working.
import pyscf.mcscf.las.lasscf as _forge
globals ().update ({k: v for k, v in vars (_forge).items () if not k.startswith ('__')})
from pyscf import lib
from mrh.my_pyscf.mcscf import lasci

class LASSCF_HessianOperator (_forge.LASSCF_HessianOperator):
    def _init_df_(self):
        if getattr (self.las, 'use_gpu', None):
            lib.logger.info (self.las, 'bPpj construction is bypassed in Hessian constructor')
        _init_df_(self)

    def _update_h2eff_sub_gpu(self,gpu,mo1,umat,h2eff_sub):
        # Currently uncalled. Kept for a follow-up PR. This and _update_h2eff_sub_debug both
        # rotate an already-built compact h2eff_sub by umat, and such a rotation does not
        # preserve the 8-fold permutation symmetry that get_h2eff_slice now depends on (it
        # rebuilds the full tensor with ao2mo.restore, which requires that symmetry), so the
        # two are incompatible. update_mo_ci_eri always calls _update_h2eff_sub (a full
        # recompute via las.ao2mo) until the rotation path is fixed.
        from mrh.my_pyscf.gpu import libgpu
        ncore, ncas, nocc, nmo = self.ncore, self.ncas, self.nocc, self.nmo
        #ucas = umat[ncore:nocc, ncore:nocc]
        bmPu = None
        #if hasattr (h2eff_sub, 'bmPu'): bmPu = h2eff_sub.bmPu
        libgpu.update_h2eff_sub(gpu,ncore,ncas,nocc,nmo,umat, h2eff_sub)
        #if bmPu is not None:
        #    bmPu = np.dot (bmPu, ucas)
        #    h2eff_sub = lib.tag_array (h2eff_sub, bmPu = bmPu)
        return h2eff_sub 

    def _update_h2eff_sub_debug(self, mo1, umat, h2eff_sub):
        # Currently uncalled. Kept for a follow-up PR; see the note in
        # _update_h2eff_sub_gpu for why the rotation cannot currently be used.
        # This code is outlining the algorithm taken in the GPU branch.
        ncore, ncas, nocc, nmo = self.ncore, self.ncas, self.nocc, self.nmo
        ucas = umat[ncore:nocc, ncore:nocc]
        h2eff_sub = h2eff_sub.reshape (nmo*ncas, ncas*(ncas+1)//2)
        h2eff_sub = lib.numpy_helper.unpack_tril (h2eff_sub)
        h2eff_sub = h2eff_sub.reshape (nmo, ncas, ncas, ncas)
        h2eff_sub = np.tensordot (h2eff_sub, ucas, axes=((2),(0))) # qbab
        h2eff_sub = np.tensordot (h2eff_sub, ucas, axes=((2),(0))) # qbbb
        h2eff_sub=h2eff_sub.transpose((2,3,1,0))#new  #gpu code does qbab and qbbb lines first, and then does the next four lines because batching is easier.
        h2eff_sub=np.einsum('iI,JKip->JKIp',ucas,h2eff_sub)#,ucas)
        h2eff_sub=np.einsum('JKIp,pP->JKIP',h2eff_sub,umat)#new
        h2eff_sub=h2eff_sub.transpose((3,2,0,1))#new
        ix_i, ix_j = np.tril_indices (ncas)
        h2eff_sub = h2eff_sub.reshape (nmo, ncas, ncas*ncas)
        h2eff_sub = h2eff_sub[:,:,(ix_i*ncas)+ix_j]
        h2eff_sub = h2eff_sub.reshape (nmo, -1)
        return h2eff_sub

def _init_df_(h_op):
    from mrh.my_pyscf.mcscf.lasci import _DFLASCI
    if isinstance (h_op.las, _DFLASCI):
        h_op.with_df = h_op.las.with_df
        if getattr (h_op.las, 'use_gpu', None):
           pass
        elif h_op.bPpj is None: h_op.bPpj = np.ascontiguousarray (
                h_op.las.cderi_ao2mo (h_op.mo_coeff, h_op.mo_coeff[:,:h_op.nocc],
                compact=False))

class _LASSCFMixin (lasci._LASCIMixin):
    '''mrh-only features of the LASSCF classes, in addition to those of LASCI.'''

    _hop = LASSCF_HessianOperator

    def fast_veffa (self, casdm1s_sub, bmPu, mo_coeff=None, ci=None):
        # gpu4mrh: the fast path is not implemented with GPUs, so take the get_veff path, as
        # passing bmPu=None does
        if self.use_gpu: bmPu = None
        return super ().fast_veffa (casdm1s_sub, bmPu, mo_coeff=mo_coeff, ci=ci)

class LASSCFNoSymm (_LASSCFMixin, lasci.LASCINoSymm, _forge.LASSCFNoSymm):
    pass

class LASSCFSymm (LASSCFNoSymm, lasci.LASCISymm, _forge.LASSCFSymm):
    pass

density_fit = lasci.density_fit
def LASSCF (mf_or_mol, ncas_sub, nelecas_sub, **kwargs):
    if isinstance(mf_or_mol, gto.Mole):
        mf = scf.RHF(mf_or_mol)
    else:
        mf = mf_or_mol
    if mf.mol.symmetry: 
        las = LASSCFSymm (mf, ncas_sub, nelecas_sub, **kwargs)
    else:
        las = LASSCFNoSymm (mf, ncas_sub, nelecas_sub, **kwargs)
    if getattr (mf, 'with_df', None):
        las = density_fit (las, with_df = mf.with_df) 
    return las
