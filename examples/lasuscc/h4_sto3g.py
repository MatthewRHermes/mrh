# This is a sample script to run LAS-USCCSD for the H4 molecule with the polynomial-scaling algorithm
# (2e,2o)+(2e,1o)

import numpy as np
import pyscf
from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.exploratory.unitary_cc import lasuccsd
from mrh.exploratory.unitary_cc.uccsd_sym0 import get_uccsd_op
from mrh.exploratory.citools import grad

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

#Selecting cluster excitations through LAS-UCCSD gradients, may use your desired epsilon
#==========================================================================================
all_g, all_gen_indices = grad.get_grad_exact(a_idxs,i_idxs,hlas, rdm1s, rdm2s, rdm3s, epsilon=0.0)

print ("All_g = ", all_g)

