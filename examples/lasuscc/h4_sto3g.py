# Author: Shreya Verma shreyav@uchicago.edu
# This is a sample script to run LAS-USCCSD for the H4 molecule with the polynomial-scaling algorithm to select cluster excitations
# (2e,2o)+(2e,1o)
# This is not a VQE calculation with statevector simulator, rather the classical emulator is used 

import numpy as np
import pyscf
from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.exploratory.unitary_cc import lasuccsd
from mrh.exploratory.unitary_cc.uccsd_sym0 import get_uccsd_op
from mrh.exploratory.citools import grad, lasci_ominus1

# Initializing the molecule with RHF
#===================================
xyz = '''H 0.0 0.0 0.0;
            H 1.0 0.0 0.0;
            H 0.2 1.6 0.1;
            H 1.159166 1.3 -0.1'''
mol = gto.M (atom = xyz, basis = 'sto-3g', output='h4_sto3g.log.py',
    verbose=0)
mf = scf.RHF (mol).run ()
ref = mcscf.CASSCF (mf, 4, 4).run () # = FCI

# Running LASSCF
#===================================
las = LASSCF (mf, (2,1), (2,2), spin_sub=(1,1))
las.verbose = 4
frag_atom_list = ((0,1),(2,3))
mo_loc = las.localize_init_guess (frag_atom_list, mf.mo_coeff)
las.kernel (mo_loc)

# Extracting vpqrs from LAS object
#=================================
nmo = las.mo_coeff.shape[1]
ncas, ncore = las.ncas, las.ncore
nocc = ncore + ncas
h2e = lib.numpy_helper.unpack_tril (las.get_h2eff().reshape (nmo*ncas,ncas*(ncas+1)//2)).reshape (nmo, ncas, ncas, ncas)[ncore:nocc,:,:,:]

# Constructing h: includes h0, h1 from las (las.h1e_for_cas()[0]), h2 from las (h2e)
#===================================================================================
h1las, h0las = las.h1e_for_cas(mo_coeff=las.mo_coeff)
h2las = h2e
hlas = [h0las,h1las,h2las]
#print ("h1e_for_las = ", las.h1e_for_cas(), las.h1e_for_cas(mo_coeff=las.mo_coeff))

#Extracting 1-,2-RDMs from LAS object
#====================================
rdm1s = las.make_casdm1s()
rdm2s = las.make_casdm2s()
rdm3s = las.make_casdm3s()

#Extracting single and double t amplitudes indices
#=================================================
norb = las.ncas                                                        
nlas = las.ncas_sub                                                    
uop = lasuccsd.gen_uccsd_op(norb,nlas)                                 
a_idxs = uop.a_idxs                                                    
i_idxs = uop.i_idxs   

#Getting gradient for all cluster excitations through LAS-UCCSD gradients, may use your desired epsilon
#==========================================================================================
all_g, all_gen_indices = grad.get_grad_exact(a_idxs,i_idxs,hlas, rdm1s, rdm2s, rdm3s, epsilon=0.0)
print ("All_g = ", all_g)

#Selecting cluster excitations through LAS-UCCSD gradients, may use your desired epsilon
#==========================================================================================
excitations = []
(
        g,
        gen_indices,
        a_idxs_new,
        i_idxs_new,
        num_a_idxs,
        num_i_idxs,
    ) = grad.grad_select(all_g, all_gen_indices, a_idxs, i_idxs, epsilon=0.001)

print ("Selected gradients = ", g)

for a, i in zip(a_idxs_new, i_idxs_new):
    excitations.append((tuple(i), tuple(a[::-1])))

print ("Selected excitations = ", excitations)

#Computing energy through the LAS-UCC kernel using selected excitations
#==========================================================================================
epsilon=0.001
mc_uscc = mcscf.CASCI(mf, 3, 4)
mc_uscc.mo_coeff = las.mo_coeff
lasci_ominus1.GLOBAL_MAX_CYCLE = 15000
mc_uscc.fcisolver = lasuccsd.FCISolver2(mol, a_idxs_new, i_idxs_new)
mc_uscc.fcisolver.norb_f = [2,1]
mc_uscc.kernel()
print("Epsilon: {:.9f} | Number of parameters: {:.0f} | LASUSCCSD energy: {:.9f}".format(epsilon, len(a_idxs_new), mc_uscc.e_tot))
