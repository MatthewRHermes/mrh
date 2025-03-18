import numpy as np
import pyscf
from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.exploratory.unitary_cc import lasuccsd
from mrh.exploratory.unitary_cc.uccsd_sym0 import get_uccsd_op
from mrh.exploratory.citools import fockspace, lasci_ominus1, grad
import unittest


def molecule_initialize(nelesub, norbsub, spinsub):
    """ Define the molecule"""
    xyz = '''H 0.0 0.0 0.0;
            H 1.0 0.0 0.0;
            H 0.2 1.6 0.1;
            H 1.159166 1.3 -0.1'''
    mol = gto.M (atom = xyz, basis = 'sto-3g', output='h4_sto3g.log.py', verbose=0)
    mf = scf.RHF (mol).run ()
    las = LASSCF (mf, norbsub, nelesub, spin_sub=(1,1))
    las.verbose = 4
    frag_atom_list = ((0,1),(2,3))
    mo_loc = las.localize_init_guess (frag_atom_list, mf.mo_coeff)
    las.kernel (mo_loc)
    return mol, mf, las

def construct_hlas_expanded(las):
    """ Construct the expanded hamiltonian"""
    nmo = las.mo_coeff.shape[1]
    ncas, ncore = las.ncas, las.ncore
    nocc = ncore + ncas
    h2e = lib.numpy_helper.unpack_tril (las.get_h2eff().reshape (nmo*ncas,ncas*(ncas+1)//2)).reshape (nmo, ncas, ncas, ncas)[ncore:nocc,:,:,:]
    h1las, h0las = las.h1e_for_cas(mo_coeff=las.mo_coeff)
    h2las = h2e
    hlas = [h0las,h1las,h2las]
    return hlas

def generate_cluster_excitations(las):
    """ Generate the cluster excitations across fragments"""
    norb = las.ncas                                                        
    nlas = las.ncas_sub                                                    
    uop = lasuccsd.gen_uccsd_op(norb,nlas)                                 
    a_idxs = uop.a_idxs                                                    
    i_idxs = uop.i_idxs   
    return a_idxs, i_idxs

def generate_fci_grads(mol, mf, las, norbcas, nelecas, epsilon=0.0):
    """ LAS-UCC gradients wrt cluster excitations with the FCI-based exponential scaling algorithm"""
    lasci_ominus1.GLOBAL_MAX_CYCLE = 0
    mc = mcscf.CASCI (mf, norbcas, nelecas)
    mc.mo_coeff = las.mo_coeff
    mc.fcisolver = lasuccsd.FCISolver (mol)
    mc.fcisolver.norb_f = las.ncas_sub#[2,1]
    mc.verbose=5
    ci0_f = [np.squeeze (fockspace.hilbert2fock (ci[0], no, ne))
            for ci, no, ne in zip (las.ci, las.ncas_sub, las.nelecas_sub)]
    mc.fcisolver.get_init_guess = lambda *args: ci0_f
    mc.kernel ()
    psi = mc.fcisolver.psi
    x = psi.x
    assert(np.all(x==0))

    h1, h0 = mc.get_h1eff ()
    h2 = mc.get_h2eff ()
    hlas = [h0, h1, h2]

    all_g_fci, g, gen_indices, a_idxs_new, i_idxs_new, num_a_idxs, num_i_idxs = psi.get_grad_t1(x, hlas, epsilon=0.0)
    all_g = np.array(all_g_fci)[:, 0]

    return all_g

def generate_poly_grads(a_idxs, i_idxs, hlas, rdm1s, rdm2s, rdm3s, epsilon=0.0):
    """ LAS-UCC gradients wrt cluster excitations with the polynomial scaling algorithm"""
    all_g, all_gen_indices = grad.get_grad_exact(a_idxs, i_idxs, hlas, rdm1s, rdm2s, rdm3s, epsilon=0.0) 
    return all_g

def collect_grads (mol, mf, las, epsilon=0.0):
    """ Call both kinds of gradient functions"""
    rdm1s, rdm2s, rdm3s = las.make_casdm1s(), las.make_casdm2s(), las.make_casdm3s()

    hlas_poly = construct_hlas_expanded(las)
    a_idxs, i_idxs = generate_cluster_excitations(las)

    all_g_poly = generate_poly_grads(a_idxs, i_idxs, hlas_poly, rdm1s, rdm2s, rdm3s, epsilon=0.0)
    all_g_fci = generate_fci_grads(mol, mf, las, las.ncas, las.nelecas, epsilon=0.0)

    return all_g_poly, all_g_fci

    

class KnownValues(unittest.TestCase): 

    def test_fci_poly_las1_spin1 (self):
        """ LAS H4 with (2,2)+(2,1)"""
        nelesub = (2,2)
        norbsub = (2,1)
        spinsub = (1,1)
        mol, mf, las = molecule_initialize(nelesub, norbsub, spinsub)
        all_g_poly, all_g_fci = collect_grads (mol, mf, las)
        self.assertAlmostEqual (lib.fp (all_g_poly), lib.fp(all_g_fci))

    def test_fci_poly_las2 (self):
        """ LAS H4 with (2,2)+(2,2)"""
        nelesub = (2,2)
        norbsub = (2,2)
        spinsub = (1,1)
        mol, mf, las = molecule_initialize(nelesub, norbsub, spinsub)
        all_g_poly, all_g_fci = collect_grads (mol, mf, las)
        self.assertAlmostEqual (lib.fp (all_g_poly), lib.fp(all_g_fci))

    def test_fci_poly_las1_spin2 (self):
        """ LAS H4 with (2,2)+(2,1) triplet"""
        nelesub = (2,2)
        norbsub = (2,1)
        spinsub = (3,1)
        mol, mf, las = molecule_initialize(nelesub, norbsub, spinsub)
        all_g_poly, all_g_fci = collect_grads (mol, mf, las)
        self.assertAlmostEqual (lib.fp (all_g_poly), lib.fp(all_g_fci))

if __name__ == "__main__":
    print("Full Tests for LAS-USCC polynomial scaling algorithm of H2 dimer")
    unittest.main()

