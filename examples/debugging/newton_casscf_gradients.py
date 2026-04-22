import os
import numpy as np
from pyscf import gto, scf, lib, mcscf, ao2mo
from pyscf.mcscf import newton_casscf, mc1step
from pyscf.csf_fci import csf_solver
from scipy import linalg
from scipy.sparse import linalg as sparse_linalg
from mrh.util.debugging.gradients import GradientDebugger
from mrh.util.debugging.hessians import HessianDebugger
from mrh.util.la import vector_error

try:
    folder = 'newton_casscf_gradients'
    os.mkdir (folder)
except OSError as e:
    print (e)

mol = gto.M (atom="""
    O  0.000000  0.000000  0.000000
    H  0.758602  0.000000  0.504284
    H  -0.758602  0.000000  0.504284
""", basis='631g',
output='newton_casscf_gradients.log',
verbose=lib.logger.DEBUG)
mf = scf.RHF (mol).run ()

solver = csf_solver (mol, smult=1)
mc = mcscf.CASSCF (mf, 6, 8).run (fcisolver=solver).newton ()
mc.internal_rotation = True
# ^ This is mandatory if the CI vector isn't optimized
# There is a frame-rotation effect that can't be computed
# if nonzero active-active orbital rotations are discarded
# by the indexing
mc.mo_coeff = mf.mo_coeff

eris = mc.ao2mo ()
g_vec, g_update, h_op, hdiag_vec = newton_casscf.gen_g_hop (
    mc,
    mc.mo_coeff,
    mc.ci,
    eris
    )
nvar_tot = len (g_vec)
nvar_orb = np.count_nonzero (mc.uniq_var_indices (
    mc.mo_coeff.shape[1],
    mc.ncore,
    mc.ncas,
    mc.frozen
    ))

rng = np.random.default_rng ()
x_random = 1 - (2*rng.random ((nvar_tot)))

ci0 = mc.ci.copy ()
gorb0 = mc.unpack_uniq_var (g_vec[:nvar_orb])

# This doesn't really work at a stationary point
print ("Orbital sector gradient norm:", linalg.norm (g_vec[:nvar_orb]))
print ("CI sector gradient norm:", linalg.norm (g_vec[nvar_orb:]))

def energy_elec (mc):
    h1, h0 = mc.get_h1eff ()
    h2 = ao2mo.restore (1, mc.get_h2eff (), mc.ncas)
    dm1, dm2 = mc.fcisolver.make_rdm12 (mc.ci, mc.ncas, mc.nelecas)
    energy_elec = h0
    energy_elec += np.dot (h1.ravel (), dm1.ravel ())
    energy_elec += np.dot (h2.ravel (), dm2.ravel ()) / 2
    return energy_elec

e0 = energy_elec (mc)
x_trial = x_random.copy ()
mc1 = mc.copy ()
def e_op (x):
    u, ci = newton_casscf.extract_rotation (mc, x, 1, ci0)
    mc1.mo_coeff = mc.mo_coeff @ u
    mc1.ci = ci
    return energy_elec (mc1) - e0

dbg = GradientDebugger (e_op, g_vec, x=x_trial).run ()
dbg.name = 'Gradient'
print ('Gradient:\n', dbg.sprintf_results ())
dbg.plot (os.path.join (folder, 'gradient.eps'))

dbgorb, dbgci = dbg.split ([nvar_orb], ('orb','CI'))
dbgorb.run ()
print ('Gorb:\n', dbgorb.sprintf_results ())
dbgorb.plot (os.path.join (folder, 'gorb.eps'))
dbgci.run ()
print ('GCI:\n', dbgci.sprintf_results ())
dbgci.plot (os.path.join (folder, 'gci.eps'))

def rotate_gorb (x, gorb, gci):
    # Express gorb in the original orbital basis, not the updated orbital basis
    # The CI sector doesn't have this problem because determinants don't change
    xorb = mc.unpack_uniq_var (x[:nvar_orb])
    nocc = mc.ncore + mc.ncas
    gorb += (xorb @ gorb0 - gorb0 @ xorb) / 2 
    gc = gci.ravel ().dot (ci0.ravel ())
    gci -= gc * ci0.ravel ()
    g = np.append (mc.pack_uniq_var (gorb), gci.ravel ())
    return g

def g_op (x):
    u, ci = newton_casscf.extract_rotation (mc, x, 1, ci0)
    mc1.mo_coeff = mc.mo_coeff @ u
    mc1.ci = ci
    g1 = newton_casscf.gen_g_hop (
        mc1,
        mc1.mo_coeff,
        mc1.ci,
        mc1.ao2mo ()
    )[0]
    gorb = mc1.unpack_uniq_var (g1[:nvar_orb])
    gci = g1[nvar_orb:]
    g1 = rotate_gorb (x, gorb, gci)
    return g1

dbg = HessianDebugger (g_op, h_op, x=x_trial, dtype=float, shape=(nvar_tot,nvar_tot)).run ()
dbg.name = 'Hessian'
print ('Hessian:\n', dbg.sprintf_results ())
dbg.plot (os.path.join (folder, 'hessian.eps'))
dbgoo, dbgoc, dbgco, dbgcc = dbg.split ([nvar_orb,], ('orb', 'CI'))
dbgoo.run ()
print ('Horb:\n', dbgoo.sprintf_results ())
dbgoo.plot (os.path.join (folder, 'Horb.eps'))
dbgoo.plotall (os.path.join (folder, 'Horb.eps'))
dbgoc.run ()
print ('Horb,CI:\n', dbgoc.sprintf_results ())
dbgoc.plot (os.path.join (folder, 'HorbCI.eps'))
dbgoc.plotall (os.path.join (folder, 'HorbCI.eps'))
dbgco.run ()
print ('HCI,orb:\n', dbgco.sprintf_results ())
dbgco.plot (os.path.join (folder, 'HCIorb.eps'))
dbgco.plotall (os.path.join (folder, 'HCIorb.eps'))
dbgcc.run ()
print ('HCI:\n', dbgcc.sprintf_results ())
dbgcc.plot (os.path.join (folder, 'HCI.eps'))
dbgcc.plotall (os.path.join (folder, 'HCI.eps'))

