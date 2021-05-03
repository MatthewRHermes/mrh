import numpy as np
import time
from itertools import product
from scipy import linalg
from pyscf import gto, dft, ao2mo, fci, mcscf, lib
from pyscf.lib import logger, temporary_env
from pyscf.mcscf import mc_ao2mo
from pyscf.mcscf.addons import StateAverageMCSCFSolver, state_average_mix, state_average_mix_, state_average
from mrh.util.rdm import get_2CDM_from_2RDM, get_2CDMs_from_2RDMs
from pyscf.fci import direct_spin1
from mrh.my_pyscf.grad.mcpdft import Gradients
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
        

def kernel (mc,nroots):

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
    log.info ("Entering cmspdft3.kernel")
# MRH: PySCF convention is never to use the "print" function in method code.
# All I/O in published PySCF code goes through the pyscf.lib.logger module.
# This prints data to an output file specified by the user like
# mol = gto.M (atom = ..., verbose = lib.logger.INFO, output = 'fname.log')
# If no output file is specified, it goes to the STDOUT as if it were a print
# command. The different pyscf.lib.logger functions, "note", "info", "debug",
# and some others, correspond to different levels of verbosity. In the above
# example, log.note and log.info commands would print information to 
# 'fname.log', but "debug" commands, which are a higher level, are skipped.
# I changed all the print commands in this function to log.debug or log.info
# commands.

#Print the First Coulomb Energy
    j = mc_1root._scf.get_j (dm=dm1)
    e_coul = (j*dm1).sum((1,2)) / 2
    log.debug ("Reference state e_coul {}".format (e_coul)) 
    log.info ("Reference state e_coul sum = %f",e_coul.sum())
# MRH: There are a couple of things going on here. First of all, the two
# statements share different levels of detail, so I use different verbosities:
# "debug" (only printed if the user has verbose=DEBUG or higher) for the list
# of all Coulomb energies and "info" (verbose=INFO is the default if an output
# file is specified) for the sum. Secondly, I am showing two separate ways of
# formatting strings in Python. log.xxx () knows how to parse the "old" Python
# string-formatting rules, which is what I've used in the log.info command, but
# it doesn't know how to interpret a list of arguments the way "print" does, so
# you have to pass it only one single string. The other way to do this is with
# "new" python formatting, which I think is simpler:
# print ("{} and {} and {}".format (a, b, c))
# is basically identical to
# print (a, "and", b, "and", c)
# but you can only use the former in log.xxx:
# log.xxx ("{} and {} and {}".format (a, b, c))

#Hessian Functions
# MRH: Here's a way do not do those nested loops and conditionals and therefore
# have less to worry about re indentation. Print out "rowscol2ind" and it should
# be clear how this works.
    rowscol2ind = np.zeros ((nroots, nroots), dtype=np.integer)
    rowscol2ind[(rows,col)] = list (range (pairs)) # 0,1,2,3,...
    rowscol2ind += rowscol2ind.T # Now it can handle both k>l and l>k 
    rowscol2ind[np.diag_indices(nroots)] = -1 # Makes sure it crashes if you look
                                              # for k==l, since that's the density 
                                              # matrix and we compute that with a 
                                              # different function.
    def w_klmn(k,l,m,n,dm,tdm):
#        casdm1 = mc.fcisolver.states_make_rdm1 (ci,mc_1root.ncas,mc_1root.nelecas)
        # MRH: don't you also need to put casdm1 into the AO basis/recompute dm1?
#        dm1 = np.dot(casdm1,mo_cas.T)
#        dm1 = np.dot(mo_cas,dm1).transpose(1,0,2)
        # MRH: IMO it's more elegant to rewrite these functions to take the density
        # matrices dm1_cirot and tdm1 as you compute them for the gradient downstairs,
        # since it's the same density matrices for both derivatives. Then this whole
        # function could be like 5 lines long:
          d = dm[k] if k==l else tdm[rowscol2ind[k,l]]
          dm1_g = mc_1root._scf.get_j (dm=d)
          d = dm[m] if m==n else tdm[rowscol2ind[m,n]]
          w = (dm1_g*d).sum ((0,1))
          return w
        # But I'll leave it like this so you can see what's going on more clearly in
        # the github "files changed" tab.
#        trans12_tdm1, trans12_tdm2 = mc.fcisolver.states_trans_rdm12(ci[col],ci[rows],mc_1root.ncas,mc_1root.nelecas)
#        if k==l:
#            dm1_g = mc_1root._scf.get_j (dm=dm1[k])
#        else:
#            ind = rowscol2ind[k,l]
#            tdm1_2 = np.dot(trans12_tdm1[ind],mo_cas.T)
#            tdm1_2 = np.dot(mo_cas,tdm1_2).transpose(1,0)
#            dm1_g = mc_1root._scf.get_j(dm=tdm1_2)
#        if m==n:
#            w  = (dm1_g*dm1[n]).sum((0,1))
#        else:
#            ind2 = rowscol2ind[m,n]
#            tdm1_2 = np.dot(trans12_tdm1_array[ind2],mo_cas.T)
#            tdm1_2 = np.dot(mo_cas,tdm1_2).transpose(1,0)
#            w = (dm1_g*tdm1_2).sum((0,1))

    def v_klmn(k,l,m,n,dm,tdm):
        if l==m:
            v = w_klmn(k,n,k,k,dm,tdm)-w_klmn(k,n,l,l,dm,tdm)+w_klmn(n,k,n,n,dm,tdm)-w_klmn(k,n,m,m,dm,tdm)-4*w_klmn(k,l,m,n,dm,tdm)

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

#Rotate to XMS States
   
    h1, h0 = mc.get_h1eff ()

#    print("fvecs_nstates",fvecs_nstates)         
     
#    ci_array = np.tensordot(fvecs_nstates, ci_array, 1)  

    casdm1 = mc.fcisolver.states_make_rdm1 (ci_array,mc_1root.ncas,mc_1root.nelecas)
    dm1 = np.dot(casdm1,mo_cas.T)
    dm1 = np.dot(mo_cas,dm1).transpose(1,0,2)

#Loop Initializations
    dm1_old = dm1
    maxiter = 150
    ci_old = ci_array
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
        log.info ("Sum e_coul = {} ; difference = {}".format (e_coul_new.sum(), 
            e_coul_new.sum()-e_coul_old.sum()))
 
#       Compute Gradient
        dg = mc_1root._scf.get_j (dm=tdm1)
        grad1 = (dg*dm1_cirot[rows]).sum((1,2))
        grad2 = (dg*dm1_cirot[col]).sum((1,2))
        grad = 2*(grad1 - grad2)
        grad_norm = np.linalg.norm(grad)
        log.debug ("grad: {}".format (grad))
        log.info ("grad norm = %f", grad_norm)

#        if grad_norm < thrs:
#            conv = True
            # ci_final = ci_rot # best just to use ci_rot
#            break
#        print("hello")
#       Hessian
        hess = np.zeros((pairs, pairs))
#       MRH: you can do this whole nested loop, defining all six indices, in one (1) line:
        for (i, (k,l)), (j, (m,n)) in product (enumerate (zip (rows, col)), repeat=2):
            # To explain:
            # for k,l in zip (rows, col) 
            #  ^ Puts elements of "rows" in "k" and elements of "col" in "l"
            #    Advances through "rows" and "col" simultaneously, so it's always
            #    the nth element of "rows" and the nth element of "col"
            #    So obviously, "rows" and "col" have to be the same size.
            #    You don't need parentheses on the right-hand side at this point
            # for i, (k, l) in enumerate (zip (rows,col)):
            #  ^ Iterates over the zip and puts its elements in (k, l), and also 
            #    counts upwards from zero and puts the count in "i". Here, you do
            #    need parentheses so that the interpreter understands that "k"
            #    and "l" are grouped together, separate from "i".
            # the uncommented line
            #  ^ "Product" is like "zip", except instead of advancing through the
            #    arguments simultaneously, it gives you all combinations of all 
            #    elements. Also, I've only entered one argument and asked it to
            #    repeat it twice. I would have gotten the same result with, i.e.,
            #    product (enumerate, enumerate). Note that I had to import the
            #    function "product" from the built-in Python module "itertools",
            #    which happens on line 3. Again, you need parentheses to show
            #    which indices are grouped together.
            # Using stuff like this really helps keep the indentations under
            # control. Nested loops and conditionals are really slow in Python,
            # so it's a good idea to combine them using tools like this whenever
            # possible.
            hess[i,j] = v_klmn(k,l,m,n,dm1_cirot,tdm1)+v_klmn(l,k,n,m,dm1_cirot,tdm1)-v_klmn(k,l,n,m,dm1_cirot,tdm1)-v_klmn(l,k,m,n,dm1_cirot,tdm1)

#       MRH: print some diagnostic stuff, but only do the potentially-expensive
#       diagonalization if the user-specified verbosity is high enough
#        if log.verbose >= lib.logger.DEBUG:
        evals, evecs = linalg.eigh (hess)
        evecs = np.array(evecs)
        log.info ("Hessian eigenvalues: {}".format (evals))
        hess1=hess        
#        print("hess1",hess)            
#        print("evals",evals)

        for i in range(pairs):
            if 1E-09 > evals[i] and evals[i] > -1E-09:
               log.info ("Hess is singular!")
            if evals[i] > 0 :
                neg = False
                break
            neg = True
        log.info ("Hess diag is neg? {}".format (neg))
#        If the diagonals are positive make them negative:
        if neg == False :
            for i in range(pairs):
                if evals[i] > 0 :
                    evals[i]=-evals[i]
#      Remake the Hessian
        diag = np.identity(pairs)*evals
        hess = np.dot(np.dot(evecs,diag),evecs.T)
        hess2= hess
#        print("hess2", hess)
#        print("difference between hessian", hess2-hess1)         

#       Make T

        t_add = linalg.solve(hess,-grad)
        t[:] = 0
        t[np.tril_indices(t.shape[0], k = -1)] = t_add

        t = t - t.T
        # t = t + t_old
        # MRH: I don't think you add them. They're expressed in different 
        # bases, and anyway ci_rot already has the effect of t_old in it. On
        # iteration zero, say ci_rot is called ci0. Then on iteration 1, it's
        # ci1 = expm (t1) ci0
        # Then on iteration 2, it's
        # ci2 = expm (t2) ci1 = expm (t2) expm (t1) ci0
        # and so forth. So you don't need to keep the running sum. 
        t_old = t.copy()

#       Reset Old Values

        ci_old=ci_rot
        e_coul_old = e_coul_new


        if grad_norm < thrs and neg == True:
                conv = True
                break

#########################
# End Loop

    if conv:
        log.note ("CMS-PDFT intermediate state determination CONVERGED")
    else:
        log.note (("CMS-PDFT intermediate state determination did not converge"
                   " after {} cycles").format (it))

# Intermediate Energies 
# Run MC-PDFT
    #mc.ci = ci_final
    #E_int = np.zeros((nroots)) 
    #with lib.temporary_env (mc, ci=ci_rot):
    #    # This ^ ~temporarily sets mc.ci to ci_rot
    #    # As soon as you leave this indent block, it returns to
    #    # whatever it was before. Convenient! Of course, it would
    #    # be WAY BETTER for me to just implement ci as a kwarg
    #    # in mcpdft.kernel.
    #    for i in range(nroots):    
    #        E_int [i]= mcpdft.mcpdft.kernel(mc,mc.otfnal,root=i)[0]
    E_int = np.asarray ([mcpdft.mcpdft.kernel (mc, ot=mc.otfnal, ci=c)[0]
        for c in ci_rot])
    log.info ("CMS-PDFT intermediate state energies: {}".format (E_int))

   #Compute the full Hamiltonian

    h1, h0 = mc.get_h1eff ()

    h2 = mc.get_h2eff ()

    h2eff = direct_spin1.absorb_h1e (h1, h2, mc.ncas, mc.nelecas, 0.5)

    hc_all = [direct_spin1.contract_2e (h2eff, c, mc.ncas, mc.nelecas) for c in ci_rot]

    Ham = np.tensordot (ci_rot, hc_all, axes=((1,2),(1,2)))

 #   print ("ham", Ham)
   
    for i in range(nroots):
        Ham[i,i]=E_int[i]
#    print("ham",Ham)

    log.info ("Effective hamiltonian: {}".format (Ham))

    e_cms, e_vecs = linalg.eigh(Ham) 
    
#    print("e_cms", e_cms)

    log.info ("CMS-PDFT final state energies: {}".format (e_cms))

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
