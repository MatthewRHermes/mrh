from pyscf.mcscf import newton_casscf
from pyscf.grad import rks as rks_grad
from pyscf.dft import gen_grid
from pyscf.lib import logger, pack_tril, current_memory, tag_array
#from mrh.my_pyscf.grad import sacasscf
from pyscf.grad import sacasscf
from pyscf.mcscf.casci import cas_natorb
from mrh.my_pyscf.mcpdft.otpd import get_ontop_pair_density, _grid_ao2mo
from mrh.my_pyscf.mcpdft.pdft_veff import _contract_vot_rho, _contract_ao_vao
from mrh.util.rdm import get_2CDM_from_2RDM
from functools import reduce
from scipy import linalg
import numpy as np
import time, gc

def kernel (ci_opt):

    mc_1root = mc
    mc_1root = mcscf.CASCI (mc._scf, mc.ncas, mc.nelecas)
    mc_1root.fcisolver = fci.solver (mc._scf.mol, singlet = False, symm = False)
    mc_1root.mo_coeff = mc.mo_coeff
    nao, nmo = mc.mo_coeff.shape
    ncas, ncore = mc.ncas,mc.ncore
    nocc = ncas + ncore
    mo_cas = mc.mo_coeff[:,ncore:nocc]
 

    casdm1 = mc.fcisolver.states_make_rdm1 (mc.ci,mc_1root.ncas,mc_1root.nelecas)
    dm1 = np.dot(casdm1,mo_cas.T)
    dm1 = np.dot(mo_cas,dm1).transpose(1,0,2)
    aeri = ao2mo.restore (1, mc.get_h2eff (mc.mo_coeff), mc.ncas)
    rows,col = np.tril_indices(nroots,k=-1)
    pairs = len(rows)
    ci_array = np.array(mc.ci)
    u = np.identity(nroots)
    t = np.zeros((nroots,nroots))
    t_old = np.zeros((nroots,nroots))

    trans12_tdm1, trans12_tdm2 = mc.fcisolver.states_trans_rdm12(ci_array[col],ci_array[rows],mc_1root.ncas,mc_1root.nelecas)
    trans12_tdm1_array = np.array(trans12_tdm1)
    tdm1 = np.dot(trans12_tdm1_array,mo_cas.T)
    tdm1 = np.dot(mo_cas,tdm1).transpose(1,0,2)

    log = lib.logger.new_logger (mc, mc.verbose)
    log.info ("Entering grad cmspdft.kernel")

    rowscol2ind = np.zeros ((nroots, nroots), dtype=np.integer)
    rowscol2ind[(rows,col)] = list (range (pairs)) # 0,1,2,3,...
    rowscol2ind += rowscol2ind.T # Now it can handle both k>l and l>k
    rowscol2ind[np.diag_indices(nroots)] = -1 # Makes sure it crashes if you loo

    def w_klmn(k,l,m,n,dm,tdm):
          d = dm[k] if k==l else tdm[rowscol2ind[k,l]]
          dm1_g = mc_1root._scf.get_j (dm=d)
          d = dm[m] if m==n else tdm[rowscol2ind[m,n]]
          w = (dm1_g*d).sum ((0,1))
          return w


