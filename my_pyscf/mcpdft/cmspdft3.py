import numpy as np
import time
from itertools import product
from scipy import linalg
from pyscf import gto, dft, ao2mo, fci, mcscf, lib
from pyscf.lib import logger, temporary_env
from pyscf.mcscf import mc_ao2mo
from pyscf.mcscf.addons import StateAverageMCSCFSolver, state_average_mix, state_average_mix_, state_average
from mrh.util.rdm import get_2CDM_from_2RDM, get_2CDMs_from_2RDMs
from mrh.my_pyscf.mcpdft import pdft_veff
from mrh.my_pyscf.mcpdft.otpd import get_ontop_pair_density
from mrh.my_pyscf.mcpdft.otfnal import otfnal, transfnal, ftransfnal
from mrh.my_pyscf import mcpdft
from mrh.my_pyscf.grad.mcpdft import Gradients
# MRH: this ^ import can cause recursive inheritance crashes if it appears
# higher in this list, before other mcpdft modules. I need to fix it but in the
# meantime just make sure it's the last import and it should be OK.

def kernel (mc,nroots=None):
# MRH: made nroots a kwarg so that this function can be called more simply
    if nroots is None: nroots = mc.fcisolver.nroots

#Initialization
        
    nao, nmo = mc.mo_coeff.shape
    ncas, ncore = mc.ncas,mc.ncore
    nocc = ncas + ncore
    mo_cas = mc.mo_coeff[:,ncore:nocc]
    casdm1 = mc.fcisolver.states_make_rdm1 (mc.ci,mc.ncas,mc.nelecas)
    dm1 = np.dot(casdm1,mo_cas.T)
    dm1 = np.dot(mo_cas,dm1).transpose(1,0,2)
    aeri = ao2mo.restore (1, mc.get_h2eff (mc.mo_coeff), mc.ncas)
    row,col = np.tril_indices(nroots,k=-1)
    npairs = len(row)
    ci_array = np.array(mc.ci)
    t = np.zeros((nroots,nroots))

    log = lib.logger.new_logger (mc, mc.verbose)
    log.info ("Entering cmspdft3.kernel")

#Print the First Coulomb Energy
    j = mc._scf.get_j (dm=dm1)
    e_coul = (j*dm1).sum((1,2)) / 2
    log.debug ("Reference state e_coul {}".format (e_coul)) 
    log.info ("Reference state e_coul sum = %f",e_coul.sum())
    rowcol2ind = np.zeros ((nroots, nroots), dtype=np.integer)
    rowcol2ind[(row,col)] = list (range (npairs)) # 0,1,2,3,...
    rowcol2ind += rowcol2ind.T # Now it can handle both k>l and l>k 
    rowcol2ind[np.diag_indices(nroots)] = npairs  # Makes sure it crashes if you look
                                                  # for k==l, since that's the density 
                                                  # matrix and we compute that with a 
                                                  # different function.
    def w_klmn(k,l,m,n,dm1,tdm1):
        d = dm1[k] if k==l else tdm1[rowcol2ind[k,l]]
        dm1_g = mc._scf.get_j (dm=d)
        d = dm1[m] if m==n else tdm1[rowcol2ind[m,n]]
        w = (dm1_g*d).sum ((0,1))
        return w

    def v_klmn(k,l,m,n,dm1,tdm1):
        if l==m:
            v = (w_klmn(k,n,k,k,dm1,tdm1)
                -w_klmn(k,n,l,l,dm1,tdm1)
                +w_klmn(n,k,n,n,dm1,tdm1)
                -w_klmn(k,n,m,m,dm1,tdm1)
              -4*w_klmn(k,l,m,n,dm1,tdm1))
        else:
            v = 0
        return v

#Loop Initializations
    maxiter = 50
    ci_rot = ci_array
    thrs = 1.0e-06
    e_coul_old = e_coul
    conv = False

#################
#Begin Loop 
#################


    for it in range(maxiter):
        log.info ("****iter {} ***********".format (it))
       
#       Form U
        U = linalg.expm(t) 
      
#       Rotate T
        ci_rot = np.tensordot(U,ci_rot,1)

#       Form New DM1s
        casdm1_rot = mc.fcisolver.states_make_rdm1 (ci_rot,mc.ncas,mc.nelecas)
        dm1_cirot = np.dot(casdm1_rot,mo_cas.T)
        dm1_cirot = np.dot(mo_cas,dm1_cirot).transpose(1,0,2)
        dm1_cirot = np.array(dm1_cirot)

#       Form New TDM
        trans12_tdm1_rot, trans12_tdm2 = mc.fcisolver.states_trans_rdm12(ci_rot[col],ci_rot[row],mc.ncas,mc.nelecas)
        trans12_tdm1_array = np.array(trans12_tdm1_rot)
        tdm1 = np.dot(trans12_tdm1_array,mo_cas.T)
        tdm1 = np.dot(mo_cas,tdm1).transpose(1,0,2)

#       Print New E coul and difference
        j = mc._scf.get_j (dm=dm1_cirot)
        e_coul_new = (j*dm1_cirot).sum((1,2)) / 2
        log.info ("Sum e_coul = {} ; difference = {}".format (e_coul_new.sum(), 
            e_coul_new.sum()-e_coul_old.sum()))
 
#       Compute Gradient
        dg = mc._scf.get_j (dm=tdm1)
        grad1 = (dg*dm1_cirot[row]).sum((1,2))
        grad2 = (dg*dm1_cirot[col]).sum((1,2))
        grad = 2*(grad1 - grad2)
        grad_norm = np.linalg.norm(grad)
        log.debug ("grad: {}".format (grad))
        log.info ("grad norm = %e", grad_norm)

#       Hessian
        hess = np.zeros((npairs, npairs))
        for (i, (k,l)), (j, (m,n)) in product (enumerate (zip (row, col)), repeat=2):
            hess[i,j] = (v_klmn(k,l,m,n,dm1_cirot,tdm1)
                        +v_klmn(l,k,n,m,dm1_cirot,tdm1)
                        -v_klmn(k,l,n,m,dm1_cirot,tdm1)
                        -v_klmn(l,k,m,n,dm1_cirot,tdm1))

#       Convergence check
        evals, evecs = linalg.eigh (hess)
        log.info ("Hessian eigenvalues: {}".format (evals))
        if (grad_norm < thrs) and (np.all (evals < 0)):
            conv = True
            break

#       Effective Hessian that moves us away from minima
        evals = -np.abs (evals) 
        hess = np.dot (evecs * evals[None,:], evecs.T)

        t_add = linalg.solve(hess,-grad)
        t[:] = 0
        t[np.tril_indices(t.shape[0], k = -1)] = t_add

        t = t - t.T
        t_old = t.copy()

#       Reset Old Values

        e_coul_old = e_coul_new

#########################
# End Loop

    if conv:
        log.note ("CMS-PDFT intermediate state determination CONVERGED")
    else:
        log.note (("CMS-PDFT intermediate state determination did not converge"
                   " after {} cycles").format (it))

# Intermediate Energies 
# Run MC-PDFT
    E_int = np.asarray ([mcpdft.mcpdft.kernel (mc, ot=mc.otfnal, ci=c)[0] 
        for c in ci_rot])
    log.info ("CMS-PDFT intermediate state energies: {}".format (E_int))


    return conv, E_int, ci_rot


if __name__ == '__main__':
    # This ^ is a convenient way to debug code that you are working on. The
    # code in this block will only execute if you run this python script as the
    # input directly: "python cmspdft3.py".

    from pyscf import scf
    from mrh.my_pyscf.tools import molden # My version is better for MC-SCF
    from mrh.my_pyscf.fci import csf_solver
    xyz = '''O  0.00000000   0.08111156   0.00000000
             H  0.78620605   0.66349738   0.00000000
             H -0.78620605   0.66349738   0.00000000'''
    mol = gto.M (atom=xyz, basis='sto-3g', symmetry=False, output='cmspdft3.log', verbose=lib.logger.DEBUG)
    mf = scf.RHF (mol).run ()
    mc = mcpdft.CASSCF (mf, 'tPBE', 4, 4).set (fcisolver = csf_solver (mol, 1))
    mc = mc.state_average ([1.0/3,]*3).run ()
    molden.from_sa_mcscf (mc, 'h2o_sapdft_sa.molden', cas_natorb=True)
    # ^ molden file with state-averaged NOs
    for i in range (3):
        fname = 'h2o_sapdft_ref{}.molden'.format (i)
        # ^ molden file with ith reference state NOs
        molden.from_sa_mcscf (mc, fname, state=i, cas_natorb=True)

    conv, E_int, ci_int = kernel (mc, 3)
    print ("The iteration did{} converge".format ((' not','')[int (conv)]))
    print ("The intermediate-state energies are",E_int)
    print (("Molden files with intermediate-state NOs are in "
            "h2o_sapdft_int?.molden"))
    with lib.temporary_env (mc, ci=ci_int):
        # ^ See line 258
        for i in range (3):
            fname = 'h2o_sapdft_int{}.molden'.format (i)
            molden.from_sa_mcscf (mc, fname, state=i, cas_natorb=True)
    
    print ("See cmspdft3.log for more information")

