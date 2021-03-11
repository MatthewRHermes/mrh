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

#Intializations
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

#Print Initial Classical Coulomb Energy
    j = mc_1root._scf.get_j (dm=dm1)
    e_coul = (j*dm1).sum((1,2)) / 2
    print("e_coul_1",e_coul)

#Initialize  U and T
    u = np.identity(nroots)
    print ("U :", u)
    t = np.zeros((nroots,nroots))

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

##########################################

#Begin Loop

##########################################
    maxiter = 5
    ci_old = ci_array
    thrs = 1.0e-06
    for it in range(maxiter):
        print ("iter",it)
        U = linalg.expm(t)
#        U=t
        print ("U :", U)
#Rotate States
        ci_rot = np.einsum('abc, ai->bci', ci_old, U).transpose(2,1,0)    
#        print("diff",ci_rot-ci_old)
#        print("norm",np.linalg.norm(ci_old))
#        print("norm",np.linalg.norm(ci_rot))
#Dm1s
        casdm1_rot = mc.fcisolver.states_make_rdm1 (ci_rot,mc_1root.ncas,mc_1root.nelecas)
        dm1_cirot = np.dot(casdm1_rot,mo_cas.T)
        dm1_cirot = np.dot(mo_cas,dm1_cirot).transpose(1,0,2)

#Compute transition densities
        trans12_tdm1_rot, trans12_tdm2 = mc.fcisolver.states_trans_rdm12(ci_rot[col],ci_rot[rows],mc_1root.ncas,mc_1root.nelecas)
        trans12_tdm1_array = np.array(trans12_tdm1_rot)
        tdm1 = np.dot(trans12_tdm1_array,mo_cas.T)
        tdm1 = np.dot(mo_cas,tdm1).transpose(1,0,2)

        j = mc_1root._scf.get_j (dm=dm1_cirot)
        e_coul2 = (j*dm1_cirot).sum((1,2)) / 2
        print("iter", it, "e coul", e_coul2)

        grad1 = np.zeros((nroots,pairs))
#Gradient
        for m in range(nroots):
            for n in range(pairs):   
                if rows[n]==m or col[n]==m:
                    grad1 [m,n] = w_klmn(m,m,m,n,ci_rot)
        print("grad1", grad1)
        dg = mc_1root._scf.get_j (dm=tdm1)
#        grad1 = 4*(dg*dm1_cirot[rows]).sum((1,2))
        grad2 = 4*(dg*dm1_cirot[col]).sum((1,2))
#        grad = grad1 + grad2
        print("grad",grad2)
        grad_norm = np.linalg.norm(grad1)
        print("gradnorm", grad_norm)
        if grad_norm < thrs:      
            print("yay", dm1_cirot)
            break
#Hessian
        hess = np.zeros((pairs, pairs))
        for i in range (pairs):
            k=rows[i]
            l=col[i]
            for j in range(pairs):
                m=rows[j]
                n=col[j]
                hess[i,j] = v_klmn(k,l,m,n,ci_rot)+v_klmn(l,k,n,m,ci_rot)-v_klmn(k,l,n,m,ci_rot)-v_klmn(l,k,m,n,ci_rot)

        print ("hess", hess)    
        inv_hess = np.linalg.inv(hess)

        t_add = inv_hess.dot(grad1)
        print("tadd", t_add)
        t[np.tril_indices(t.shape[0], k = -1)] = t_add
        t = t + t.T
        print("t",t)      

    return

