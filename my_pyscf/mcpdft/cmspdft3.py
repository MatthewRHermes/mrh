import numpy as np
import time
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
        grad = 4*(grad1 + grad2)
        grad_norm = np.linalg.norm(grad)
        log.debug ("grad: {}".format (grad))
        log.info ("grad norm = %f", grad_norm)


        if grad_norm < thrs:
            conv = True
            # ci_final = ci_rot # best just to use ci_rot
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

#       MRH: print some diagnostic stuff, but only do the potentially-expensive
#       diagonalization if the user-specified verbosity is high enough
        if log.verbose >= lib.logger.DEBUG:
            evals, evecs = linalg.eigh (hess)
            log.debug ("Hessian eigenvalues: {}".format (evals))

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
    E_int = np.zeros((nroots)) 
    with lib.temporary_env (mc, ci=ci_rot):
        # This ^ ~temporarily sets mc.ci to ci_rot
        # As soon as you leave this indent block, it returns to
        # whatever it was before. Convenient! Of course, it would
        # be WAY BETTER for me to just implement ci as a kwarg
        # in mcpdft.kernel.
        for i in range(nroots):    
            E_int [i]= mcpdft.mcpdft.kernel(mc,mc.otfnal,i)[0]
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
        # ^ See line 225 
        for i in range (3):
            fname = 'h2o_sapdft_int{}.molden'.format (i)
            molden.from_sa_mcscf (mc, fname, state=i, cas_natorb=True)
    
    print ("See cmspdft3.log for more information")

