from pyscf import gto, scf, mcscf, lib
from pyscf.tools import molden
from mrh.my_pyscf import mcpdft
from mrh.my_pyscf.tools.molden import from_sa_mcscf

mol = gto.M (atom = 'Li 0 0 0\nH 1.5 0 0', basis = 'sto3g',
             output = 'LiH.log', verbose = lib.logger.INFO)
mf = scf.RHF (mol).run ()
mc0 = mcscf.CASSCF (mf, 5, 2).run ()

mc = []
mc.append (mcpdft.CASSCF (mol, 'tPBE', 5, 2).set (mo_coeff=mf.mo_coeff).run ())
mc.append (mcpdft.CASSCF (mf, 'tPBE', 5, 2).run ())
mc.append (mcpdft.CASSCF (mc0, 'tPBE', 5, 2).run ())
mc.append (mcpdft.CASCI (mol, 'tPBE', 5, 2).set (mo_coeff=mc[-1].mo_coeff).run ())
mc.append (mcpdft.CASCI (mf, 'tPBE', 5, 2).set (mo_coeff=mc[-1].mo_coeff).run ())
mc.append (mcpdft.CASCI (mc0, 'tPBE', 5, 2).run ())
for m in mc:
    print ('{:.9f} {}'.format (m.e_tot, m.converged))

mc_grad = []
for i, m in enumerate (mc):
    try:
        m_grad = m.nuc_grad_method ()
        de = m_grad.kernel ()
        mc_grad.append ('{}\n{}'.format (m_grad.converged, de))
    except NotImplementedError as e:
        mc_grad.append (str (e))
for m in mc_grad:
    print (m)



