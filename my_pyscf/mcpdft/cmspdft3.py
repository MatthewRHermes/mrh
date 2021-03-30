import numpy as np
import time
from scipy import linalg
from pyscf import gto, dft, ao2mo, fci, mcscf, lib
from pyscf.lib import logger, temporary_env
from pyscf.mcscf import mc_ao2mo
from pyscf.mcscf.addons import StateAverageMCSCFSolver, state_average_mix, state_average_mix_, state_average
from mrh.my_pyscf.grad.mcpdft import Gradients
from mrh.util.rdm import get_2CDM_from_2RDM, get_2CDMs_from_2RDMs
from mrh.my_pyscf.mcpdft import pdft_veff
from mrh.my_pyscf.mcpdft.otpd import get_ontop_pair_density
from mrh.my_pyscf.mcpdft.otfnal import otfnal, transfnal, ftransfnal
from mrh.my_pyscf import mcpdft

def kernel (mc,nroots):

#Initialization
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

#Print the First Coulomb Energy
    j = mc_1root._scf.get_j (dm=dm1)
    e_coul = (j*dm1).sum((1,2)) / 2
    print ("e_coul", e_coul)
    print("e_coul sum",e_coul.sum())
   
#Hessian Functions
    def w_klmn(k,l,m,n,ci):
        casdm1 = mc.fcisolver.states_make_rdm1 (ci,mc_1root.ncas,mc_1root.nelecas)
        trans12_tdm1, trans12_tdm2 = mc.fcisolver.states_trans_rdm12(ci[col],ci[rows],mc_1root.ncas,mc_1root.nelecas)
        if k==l:
            dm1_g = mc_1root._scf.get_j (dm=dm1[k])
        else:
            for i in range (pairs):
                if rows[i]==k:
                    if col[i]==l:
                        ind=i
                if col[i]==k:
                    if rows[i]==l:
                        ind=i

            tdm1_2 = np.dot(trans12_tdm1[ind],mo_cas.T)
            tdm1_2 = np.dot(mo_cas,tdm1_2).transpose(1,0)
            dm1_g = mc_1root._scf.get_j(dm=tdm1_2)
        if m==n:
            w  = (dm1_g*dm1[n]).sum((0,1))
        else:
            for i in range (pairs):
               if rows[i]==m:
                   if col[i]==n:
                       ind2=i
               if col[i]==m:
                   if rows[i]==n:
                       ind2=i
            tdm1_2 = np.dot(trans12_tdm1_array[ind2],mo_cas.T)
            tdm1_2 = np.dot(mo_cas,tdm1_2).transpose(1,0)
            w = (dm1_g*tdm1_2).sum((0,1))
        return w

    def v_klmn(k,l,m,n,ci):
        if l==m:
            v = w_klmn(k,n,k,k,ci)-w_klmn(k,n,l,l,ci)+w_klmn(n,k,n,n,ci)-w_klmn(k,n,m,m,ci)-4*w_klmn(k,l,m,n,ci)
        else:
            v = 0
        return v

#Loop Initializations
    dm1_old = dm1
    maxiter = 5
    ci_old = ci_array
    thrs = 1.0e-06
    e_coul_old = e_coul


#################
#Begin Loop 
#################


    for it in range(maxiter):
        print ("****iter ",it,"***********")
       
#       Form U
        U = linalg.expm(t) 
      
#       Rotate T
        ci_rot = np.tensordot(U,ci_old,1)

#       Form New DM1s
        casdm1_rot = mc.fcisolver.states_make_rdm1 (ci_rot,mc_1root.ncas,mc_1root.nelecas)
        dm1_cirot = np.dot(casdm1_rot,mo_cas.T)
        dm1_cirot = np.dot(mo_cas,dm1_cirot).transpose(1,0,2)
        dm1_cirot = np.array(dm1_cirot)

#       Form New TDM
        trans12_tdm1_rot, trans12_tdm2 = mc.fcisolver.states_trans_rdm12(ci_rot[col],ci_rot[rows],mc_1root.ncas,mc_1root.nelecas)
        trans12_tdm1_array = np.array(trans12_tdm1_rot)
        tdm1 = np.dot(trans12_tdm1_array,mo_cas.T)
        tdm1 = np.dot(mo_cas,tdm1).transpose(1,0,2)

#       Print New E coul and difference
        j = mc_1root._scf.get_j (dm=dm1_cirot)
        e_coul_new = (j*dm1_cirot).sum((1,2)) / 2
        print("Sum e_coul",e_coul_new.sum(), "difference", e_coul_new.sum()-e_coul_old.sum())
 
#       Compute Gradient
        dg = mc_1root._scf.get_j (dm=tdm1)
        grad1 = (dg*dm1_cirot[rows]).sum((1,2))
        grad2 = (dg*dm1_cirot[col]).sum((1,2))
        grad = 4*(grad1 + grad2)
        grad_norm = np.linalg.norm(grad)
        print("grad norm", grad_norm)


        if grad_norm < thrs:
            print("CONVERGED")
            ci_final = ci_rot
            break

#       Hessian
        hess = np.zeros((pairs, pairs))
        for i in range (pairs):
            k=rows[i]
            l=col[i]
            for j in range(pairs):
                m=rows[j]
                n=col[j]
                hess[i,j] = v_klmn(k,l,m,n,ci_rot)+v_klmn(l,k,n,m,ci_rot)-v_klmn(k,l,n,m,ci_rot)-v_klmn(l,k,m,n,ci_rot)

#       Make T

        t_add = linalg.solve(hess,-grad)
        t[:] = 0
        t[np.tril_indices(t.shape[0], k = -1)] = t_add

        t = t - t.T
        t = t + t_old
        t_old = t.copy()

#       Reset Old Values

        ci_old=ci_rot
        e_coul_old = e_coul_new

#########################
# End Loop

# Intermediate Energies 
# Run MC-PDFT
    mc.ci = ci_final
    E_int = np.zeros((nroots)) 
    for i in range(nroots):    
        E_int [i]= mcpdft.mcpdft.kernel(mc,mc.otfnal,i)[0]
    print ("E_int", E_int)


    return


