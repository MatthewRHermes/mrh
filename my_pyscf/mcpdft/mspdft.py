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

#   lroots=nroots+1
    for i in range(0,nroots):
        ci_coeff = mc.ci[i]
        print ("ci",i,nroots, ci_coeff)
 
    amo = mc.mo_coeff[:,mc.ncore:mc.ncore+mc.ncas]
    # make_rdm12s returns (a, b), (aa, ab, bb)

    mc_1root = mc
    mc_1root = mcscf.CASCI (mc._scf, mc.ncas, mc.nelecas)
    mc_1root.fcisolver = fci.solver (mc._scf.mol, singlet = False, symm = False)
    mc_1root.mo_coeff = mc.mo_coeff
    nao, nmo = mc.mo_coeff.shape
  
#    dm1s = np.asarray (mc_1root.state_average.states_make_rdm1s ())
    adm1s = np.stack (mc.fcisolver.make_rdm1s (mc.ci, mc_1root.ncas,mc_1root.nelecas), axis=0)
    dm1 = mc.states_make_rdm1()
    dm1_hold = np.ones((nroots,nao,nao))   
 
    print ("dm1_hold",np.shape(dm1_hold))
    print ("dm1",np.shape(dm1))
    print ("adms",np.shape(adm1s))
    print ("dm1",dm1[1])
    print ("dm1",dm1[2])
    print ("dm1",dm1[3])
    print ("dm1",dm1[4])

#    adm1s = np.stack (mc_1root.fcisolver.make_rdm1s (mc.ci, mc.ncas, mc.nelecas), axis=0)
#    adm2 = get_2CDM_from_2RDM (mc_1root.fcisolver.make_rdm12 (mc_1root.ci, mc.ncas, mc.nelecas)[1], adm1s)
#    adm2s = get_2CDMs_from_2RDMs (mc_1root.fcisolver.make_rdm12s (mc_1root.ci, mc.ncas, mc.nelecas)[1], adm1s)

#    adm1 = dm1s[0] + dm1s[1]

    print("dm1s",dm1)

#   lroots=nroots+1
    for i in range(0,nroots):
        ci_coeff = mc.ci[i]
        print ("ci",i,nroots, ci_coeff)

#   print ("casdm1 :", casdm1, np.shape(casdm1))
#    print ("dm1s", dm1s)
#    print ("casdm2 :", casdm2)

#Calculates State Pairs
    Pair = 0
    istate = 1
    jstate = 1
    npairs = int( nroots*(nroots-1)//2)

    statepair = np.zeros((npairs,2))
    print("state pairs", statepair)
    print ("npairs", npairs, " nroots ", nroots)

    ipair = 0
    for i in range(nroots):
        print("i",i)
        for j in range(i):
            print("j", j)
            print("ipair", ipair,i,j)
            statepair[ipair,0]=i
            statepair[ipair,1]=j
            ipair = ipair+1
    print("state pair",statepair)

#    nci_sum = ci[0] 
#    nci_sum2 = ci[1]
#    nci_sum3 = np.append(nci_sum2,ci[0])
#    nci_sum4 = np.append(nci_sum,ci[1])

                


#    print("result",newci)
#    print("shape",np.shape(newci))
#    print ("shape ci", np.shape(ci))
#    print("nci_sum2", nci_sum4)

    trans12_tdm1, trans12_tdm2 = mc.fcisolver.states_trans_rdm12(mc.ci,mc.ci,mc_1root.ncas,mc_1root.nelecas)
    print("trans12 tdm1 :", trans12_tdm1)
#    print("trans12 tdm2 :", trans12_tdm2)

#Load in the two-electron integrals
    aeri = ao2mo.restore (1, mc.get_h2eff (mc.mo_coeff), mc.ncas)
#    print("aeri", aeri)
    print("eri shape", np.shape(aeri))

#Initialize rotation matrix

    u = np.identity(mc.fcisolver.nroots)
    print ("U :", u)

#Rotate the States and Corresponding Density Matrices
    converged = False
    cmsthresh = 1e-06
    cmsiter = 1    
    print("nao",nao)
#    print("dm1s",dm1s[1,1,1])
    adm1s = np.stack (mc.fcisolver.make_rdm1s (mc.ci, mc_1root.ncas,mc_1root.nelecas), axis=0)
    adm2 = get_2CDM_from_2RDM (mc.fcisolver.make_rdm12 (mc.ci,mc_1root.ncas,mc_1root.nelecas)[1], adm1s)
    E_c = np.tensordot (aeri, adm2, axes=4) / 2
    print("e_c",np.shape(e_c))
    print ("e_c",E_c)

#Calculates old VeeSum

#    def calcvee(rmat,ddg):
#        vee=np.zeros(nroots)
#        
#        for istate in range (nroots):
#            for j in range (nroots):
#                for k in range (nroots):
#                    for l in range (nroots):
#                        for m in range (nroots):
#                           print("i,j,k,l,m", i,j,k,l,m)
#                           vee[i]= vee[i]+rmat[istate,j]*rmat[istate,k]*rmat[istate,l]*rmat[istate,m]*ddg[j,k,l,m]
#            vee[i]=vee[i]/2     
   
#    def getddg(eri):
#        ddg=np.zeros(nroots**4).reshape(nroots,nroots,nroots,nroots)
#        adm1s = np.stack (mc.fcisolver.make_rdm1s (mc.ci, mc_1root.ncas,mc_1root.nelecas), axis=0)
#        adm1s = adm1s[0]+adm1s[1]
#        print("adm1s shape", np.shape(adm1s))
#        print("aeri",np.shape(aeri))
#        for i in range (nroots):
#            for j in range (nroots): 
#                if j > i :
#                    jj=i
#                    ii=j1
#                else:
#                    ii=i
#                    jj=j
#                for k in range (nroots):
#                    for l in range (nroots):
#                        if  l > k :
#                            kk=l
#                            ll=k
#                        else:
#                            kk=k
#                            ll=l
#                        for t in range(mc.ncas):
#                            for u in range(mc.ncas):
#                                for v in range(mc.ncas):
#                                    for x in range(mc.ncas):
#                                        ii = ii*(ii-1)//2+jj
#                                        kk = kk*(kk-1)//2+ll 
#                                        ddg[i,j,k,l]=aeri[t,u,v,x] 
#                                        ddg[i,j,k,l]=adm1s[ii,t,u]*adm1s[kk,v,x]*aeri[t,u,v,x]
#                                        print(i,ii,j,jj,k,kk,l,ll,t,u,v,x)
#                                        print("ddg", ddg[i,j,k,l]) 
#                                        print("dm1_hold", dm1_hold[kk,t,u])
 
#    ddg=getddg(aeri)      
         
#    print("ddg", ddg)
#    veesum = calcvee(u,ddg)
#    print("vsum",veesum)
#    dm1s = np.stack (mc.fcisolver.make_rdm1s (mc.ci[1],mc_1root.ncas,mc_1root.nelecas), axis=0)
#    dm1 = dm1s[0] + dm1s[1]
#    j = mc_1root._scf.get_j (dm=dm1)
#    for i in range(nroots):
    j = mc_1root._scf.get_j (dm=dm1)
    e_coul = np.tensordot (j, dm1s, axes=2) / 2
    print("e_coul_1",i,e_coul)
#Gradient
#Example
#np.einsum ('rsij,r->ij', veff_a, las.weights)
  
    w = aeri*adm1s
    print("w",np.shape(w))
#    for i in range(statepairs.len()):
#        trans12_tdm2 = trans12_tdm2 + mc.fcisolver.states_trans_rdm12(mc.ci,mc.ci,mc_1root.ncas,mc_1root.nelecas)
    
    
#    g_noscale = np.tensordot (w,np.appent(trans12_tdm2,axes=4)
#    print("g_noscale",g_noscale)
#    g = 4*g_noscale
#
#    print ("g",g)    


    return 
