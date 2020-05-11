#from rhf import monkeypatch_setup
#monkeypatch_teardown = monkeypatch_setup ()
import math
import numpy as np
from scipy import linalg
from pyscf import scf, gto, lib, mcscf, df
from mrh.my_pyscf.df.grad import dfsacasscf as casscf_grad
from mrh.my_pyscf.grad import numeric as numeric_grad
from mrh.my_pyscf.fci import csf_solver

def bond_length (carts, i, j):
    return linalg.norm (carts[i] - carts[j])
def bond_angle (carts, i, j, k):
    rij = carts[i] - carts[j]
    rkj = carts[k] - carts[j]
    res = max (min (1.0, np.dot (rij, rkj) / linalg.norm (rij) / linalg.norm (rkj)), -1.0)
    return math.acos (res) * 180 / math.pi
def out_of_plane_angle (carts, i, j, k, l):
    eji = carts[j] - carts[i]
    eki = carts[k] - carts[i]
    eli = carts[l] - carts[i]
    eji /= linalg.norm (eji)
    eki /= linalg.norm (eki)
    eli /= linalg.norm (eli)
    return -math.asin (np.dot (eji, (np.cross (eki, eli) / math.sin (bond_angle (carts, j, i, k) * math.pi / 180)))) * 180 / math.pi
def h2co_geom_analysis (carts):
    print ("rCO = {:.4f} Angstrom".format (bond_length (carts, 1, 0)))
    print ("rCH1 = {:.4f} Angstrom".format (bond_length (carts, 2, 0)))
    print ("rCH2 = {:.4f} Angstrom".format (bond_length (carts, 3, 0)))
    print ("tOCH1 = {:.2f} degrees".format (bond_angle (carts, 1, 0, 2)))
    print ("tOCH2 = {:.2f} degrees".format (bond_angle (carts, 1, 0, 3)))
    print ("tHCH = {:.2f} degrees".format (bond_angle (carts, 3, 0, 2)))
    print ("eta = {:.2f} degrees".format (out_of_plane_angle (carts, 0, 2, 3, 1)))
def my_call (env):
    carts = env['mol'].atom_coords () * lib.param.BOHR
    h2co_geom_analysis (carts)
conv_params = {
    'convergence_energy': 1e-6,  # Eh
    'convergence_grms': 5.0e-5,  # Eh/Bohr
    'convergence_gmax': 7.5e-5,  # Eh/Bohr
    'convergence_drms': 1.0e-4,  # Angstrom
    'convergence_dmax': 1.5e-4,  # Angstrom
}

h2co_casscf66_631g_xyz = '''C  0.534004  0.000000  0.000000
O -0.676110  0.000000  0.000000
H  1.102430  0.000000  0.920125
H  1.102430  0.000000 -0.920125'''
mol = gto.M (atom = h2co_casscf66_631g_xyz, basis = '6-31g', symmetry = False, verbose = lib.logger.INFO, output = 'h2co_sa2_casscf66_631g_grad.log')
mf_conv = scf.RHF (mol).run ()
mc_conv = mcscf.CASSCF (mf_conv, 6, 6)
mc_conv.fcisolver = csf_solver (mol, smult=1)
mc_conv = mc_conv.state_average_([0.5,0.5])
mc_conv.conv_tol = 1e-10
mc_conv.kernel ()

mf = scf.RHF (mol).density_fit (auxbasis = df.aug_etb (mol)).run ()
mc_df = mcscf.CASSCF (mf, 6, 6)
mc_df.fcisolver = csf_solver (mol, smult=1)
mc_df = mc_df.state_average_([0.5,0.5])
mc_df.conv_tol = 1e-10
mc_df.kernel ()

de_conv_0 = mc_conv.nuc_grad_method ().kernel (state = 0)
de_df_0 = casscf_grad.Gradients (mc_df).kernel (state = 0)
de_conv_1 = mc_conv.nuc_grad_method ().kernel (state = 1)
de_df_1 = casscf_grad.Gradients (mc_df).kernel (state = 1)

print ("Gradient of state 0 with conventional ERIs:\n", de_conv_0)
print ("Gradient of state 0 with DF ERIs:\n", de_df_0)
print ("Gradient of state 1 with conventional ERIs:\n", de_conv_1)
print ("Gradient of state 1 with DF ERIs:\n", de_df_1)

