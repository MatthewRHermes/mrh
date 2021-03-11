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

    mc_1root = mc
    mc_1root = mcscf.CASCI (mc._scf, mc.ncas, mc.nelecas)
    mc_1root.fcisolver = fci.solver (mc._scf.mol, singlet = False, symm = False)
    mc_1root.mo_coeff = mc.mo_coeff
    nao, nmo = mc.mo_coeff.shape
    ncas, ncore = mc.ncas,mc.ncore
    nocc = ncas + ncore
    mo_cas = mc.mo_coeff[:,ncore:nocc]

#Compute Classical Couloumb Energy

    casdm1 = mc.fcisolver.states_make_rdm1 (mc.ci,mc_1root.ncas,mc_1root.nelecas)
    dm1 = np.dot(casdm1,mo_cas.T)
    dm1 = np.dot(mo_cas,dm1).transpose(1,0,2)
    
    print("dm1",np.shape(dm1))
#    print("dm1",dm1)
    j = mc_1root._scf.get_j (dm=dm1)
#    print("j",j)
    e_coul = (j*dm1).sum((1,2)) / 2
    print("e_coul_1",e_coul)

#Transition Density

    for i in range(0,nroots):
        ci_coeff = mc.ci[i]
        print ("ci",i, ci_coeff)
    print("mc.ci",type(mc.ci))

    rows,col = np.tril_indices(nroots,k=-1)
    pairs = len(rows)
    print("rows", rows,"col", col,"pairs", pairs)
    print("mc.ci shape",np.shape(mc.ci))
    mc.ci_array = np.array(mc.ci)
    row_ci = mc.ci_array[rows]
#    print ("mc.ci[rows]",row_ci)
    col_ci = mc.ci_array[col]
    trans12_tdm1, trans12_tdm2 = mc.fcisolver.states_trans_rdm12(col_ci,row_ci,mc_1root.ncas,mc_1root.nelecas)
    print("trans12_tdm1",np.shape(trans12_tdm1))

#Load in the two-electron integrals
    aeri = ao2mo.restore (1, mc.get_h2eff (mc.mo_coeff), mc.ncas)
#    print("aeri", aeri)
#    print("eri shape", np.shape(aeri))


#Initialize rotation matrix
    u = np.identity(mc.fcisolver.nroots)
    print ("U :", u)
    
    t = np.zeros((nroots,nroots))
    
    u_lt = linalg.expm(t)

    print("u_lt",u_lt)

######################################################
#Begin Loop
######################################################  
        
    

#Gradients
    trans12_tdm1_array = np.array(trans12_tdm1)
    grad = np.zeros(pairs)
    for i in range(pairs):
        ind = rows[i]   
        grad[i] = (casdm1[ind] * trans12_tdm1_array[i]*aeri).sum((0,1,2,3))
        print('ind,i', ind, ',',i)
    grad = 4 * grad
    print('grad',grad)

#    gradnorm = np.linalg.norm(grad)
#    print ("grad norm", gradnorm)
    print ("grad shape", np.shape(grad))
#Try 2

#    j1 = mc_1root._scf.get_jk(mc._scf.mol, dm1, 'ijkl,lk->ij', intor='int2e_ip1_sph', aosym='s2kl', comp=3)
#    print("j1",j1)

#Gradient try 3
       
#    grad3 =  trans12_tdm1_array*casdm1 * aeri
#    print ("grad3",grad3)
#    print("dm1",np.shape(dm1))
    grad3=np.zeros(pairs)
#    for i in range (pairs):

#    dg = mc_1root._scf.get_j (dm=dm1[col])
#    print("dg shape",np.shape(dg))
    tdm1 = np.dot(trans12_tdm1_array,mo_cas.T)
    tdm1 = np.dot(mo_cas,tdm1).transpose(1,0,2)
    dg = mc_1root._scf.get_j (dm=tdm1)
    print ("tdm1_array",trans12_tdm1)
#    tdm1_1 = tdm1[i]
#    grad3 = (dg*tdm1).sum((1,2))
    grad3 = 4*(dg*dm1[rows]).sum((1,2))
    print ("grad 3 shape", np.shape(grad3))
#    grad3=grad3.sum((1,2))
    print("grad3 rows",grad3)
#    print("grad3*4", grad3*4)
    gradnorm3 = np.linalg.norm(grad3)
    print ("grad norm", gradnorm3)

    grad4 = 4*(dg*dm1[col]).sum((1,2))
    print("grad3 col",grad4)

    gradnorm4 = np.linalg.norm(grad4)
    print ("grad norm", gradnorm4)

    print("grad norm 3+4", gradnorm4+gradnorm3)

    gradsum = grad4+grad3
    print("grad sum",gradsum)
    print("grad sum norm", np.linalg.norm(gradsum))

#GRADIENT TRY 5
#    grad5 = np.zeros((nroots,pairs))
#    for i in range(nroots):
#        dg = mc_1root._scf.get_j(dm=dm1[i])
#        for j in range(pairs):
#            if rows[j]==i or col[j]==i:
#                grad5[i,j]=(dg*tdm1[j]).sum((0,1))
#    print("grad5",grad5)


#Hessian
    
    def w_klmn(k,l,m,n):
        if k==l:
            dm1_g = mc_1root._scf.get_j (dm=dm1[k])
        else:
            for i in range (pairs):
                if rows[i]==k:
                    if col[i]==l:
                        ind=i
#                        print("ind",ind)
                if col[i]==k:
                    if rows[i]==l:
                        ind=i    
#                        print("ind",ind)

            tdm1_2 = np.dot(trans12_tdm1[ind],mo_cas.T)
#            print("tdm1",np.shape(tdm1_2))
            tdm1_2 = np.dot(mo_cas,tdm1_2).transpose(1,0)
#            print("tdm1", np.shape(tdm1_2))
            dm1_g = mc_1root._scf.get_j(dm=tdm1_2)
        if m==n:
            w  = (dm1_g*dm1[n]).sum((0,1))
        else:
            for i in range (pairs):
               if rows[i]==m:
                   if col[i]==n:
                       ind2=i
#                       print("ind2",ind2)
               if col[i]==m:
                   if rows[i]==n:
                       ind2=i
#                       print("ind2",ind2)

            tdm1_2 = np.dot(trans12_tdm1_array[ind2],mo_cas.T)
            tdm1_2 = np.dot(mo_cas,tdm1_2).transpose(1,0)
            w = (dm1_g*tdm1_2).sum((0,1))
        return w

#    for k in range (nroots):
#        for l in range(nroots):
#            for m in range(nroots):
#                for n in range nroots:

    def v_klmn(k,l,m,n):
        if l==m:
            v = w_klmn(k,n,k,k)-w_klmn(k,n,l,l)+w_klmn(n,k,n,n)-w_klmn(k,n,m,m)-4*w_klmn(k,l,m,n)
        else:
            v = 0
        return v

    vv=v_klmn(0,0,0,1)
    print("v shape", np.shape(vv))
    print("v",vv)

    hess = np.zeros((pairs, pairs))
    for i in range (pairs):
        k=rows[i]
        l=col[i]
        print("k,l", k,",",l)
        for j in range(pairs):
            m=rows[j]
            n=col[j]
            print("m,n", m,",",n)
            hess[i,j] = v_klmn(k,l,m,n)+v_klmn(l,k,n,m)-v_klmn(k,l,n,m)-v_klmn(l,k,m,n)     

    print ("hess", hess)
     
    inv_hess = np.linalg.inv(hess)
    
    print("inv_hessian", inv_hess)
    
    t_add = inv_hess.dot(gradsum)    

    print("t_add",t_add)

#    t[rows] =  t
    t = np.zeros((nroots,nroots))
    t[np.tril_indices(t.shape[0], k = -1)] = t_add
    t = t + t.T 
    
    print ("t", t)

    u_new = linalg.expm(t)

    print("u_new", u_new)
    
    rot_one =np.identity((nroots))    
    rot_half = np.identity((nroots))*0.5    
    ci_rot= np.einsum( 'abc, ai->bci',mc.ci_array,u_new).transpose(2,1,0)
    casdm1_rot = mc.fcisolver.states_make_rdm1 (ci_rot,mc_1root.ncas,mc_1root.nelecas)
    dm1_cirot = np.dot(casdm1_rot,mo_cas.T)
    dm1_cirot = np.dot(mo_cas,dm1_cirot).transpose(1,0,2)    
    
    dm1_rot = np.einsum('abc, ai->bci', dm1, u_new).transpose(2,1,0) 


    print("no rot", dm1_rot-dm1_cirot)
    
    
    return
