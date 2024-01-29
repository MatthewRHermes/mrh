import numpy as np
from pyscf import gto, scf
from pyscf.lib import logger
from mrh.my_pyscf import mcpdft

mol = gto.M (atom='Li 0 0 0; H 1.2 0 0', basis='sto-3g', verbose=logger.DEBUG2,
             output='lih_tpbe22_sto3g.log')
mf = scf.RHF (mol).run ()
mc = mcpdft.CASSCF (mf, 'tPBE', 2, 2, grids_level=1).run ()
mc_grad = mc.nuc_grad_method ().run ()



