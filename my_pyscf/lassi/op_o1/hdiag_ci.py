import numpy as np
from pyscf import lib
from pyscf.lib import logger
from mrh.my_pyscf.lassi.op_o1 import frag
from mrh.my_pyscf.lassi.op_o1.hci.schcs import ContractHamCI_SHS

class HdiagCI (ContractHamCI_SHS):

    def _crunch_all_(self):
        for row in self.exc_1d: self._crunch_env_(self._crunch_1d_, *row)
        for row in self.exc_2d: self._crunch_env_(self._crunch_2d_, *row)

    def get_vecs (self):
        t1, w1 = logger.process_clock (), logger.perf_counter ()
        hessd_fr_plc = []
        for inti in self.ints:
            hessd_r_plc = inti._hessdiag ()
            hessd_fr_plc.append ([hessd_r_plab[i] for i in self.mask_bra_space])
        dt, dw = logger.process_clock () - t1, logger.perf_counter () - w1
        self.dt_p, self.dw_p = self.dt_p + dt, self.dw_p + dw
        return hessd_fr_plc

def hessdiag_ci (las, h1, h2, ci_fr, nelec_frs, si_bra, si_ket, h0=0, soc=0):
    log = lib.logger.new_logger (las, las.verbose)
    nlas = las.ncas_sub
    hopping_index, ints, lroots = frag.make_ints (las, ci, nelec_frs, nlas=nlas)
    max_memory = getattr (las, 'max_memory', las.mol.max_memory)
    contracter = HdiagCI (las, ints, nlas, hopping_index, lroots, h0, h1, h2, si_bra=si_bra,
                          si_ket=si_ket, dtype=ci[0][0].dtype, max_memory=max_memory, log=log)
    hessd_fr_plc, t0 = contracter.kernel ()
    return hessd_fr_plc

