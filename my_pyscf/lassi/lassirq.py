import numpy as np
from pyscf.lib import logger
from mrh.my_pyscf.lassi import LASSI
from mrh.my_pyscf.lassi.states import spin_shuffle, spin_shuffle_ci
from mrh.my_pyscf.lassi.states import all_single_excitations, SingleLASRootspace
from mrh.my_pyscf.mcscf.lasci import get_space_info

def prepare_states_spin_shuffle (lsi):
    las_ss = lsi._las.get_single_state_las (state=0)
    if np.all (get_space_info (las_ss)[2] == 1):
        return las_ss
    las_ss = spin_shuffle (las_ss, equal_weights=True)
    las_ss.ci = spin_shuffle_ci (las_ss, las_ss.ci)
    las_ss.converged = lsi._las.converged
    return las_ss

def prepare_states (lsi):
    # 1. Spin shuffle step
    log = logger.new_logger (lsi, lsi.verbose)
    las = prepare_states_spin_shuffle (lsi)
    for ir in range (lsi.r):
        las = all_single_excitations (las)
    las = lsi.filter_spaces (las)
    lroots = lsi.make_lroots (las)
    las.lasci_(lroots=lroots)
    return las.converged, las

def make_lroots (lsi, las, q=None):
    if q is None: q = lsi.q
    ncsf = las.get_ugg ().ncsf_sub
    return np.minimum (ncsf, q)

class LASSIrq (LASSI):
    def __init__(self, las, r=0, q=1, opt=1, **kwargs):
        self.r = r
        self.q = q
        LASSI.__init__(self, las, opt=opt, **kwargs)

    def kernel (self, **kwargs):
        self.converged, las = self.prepare_states ()
        #self.__dict__.update(las.__dict__) # Unsafe
        self.fciboxes = las.fciboxes
        self.ci = las.ci
        self.nroots = las.nroots
        self.weights = las.weights
        self.e_lexc = las.e_lexc
        self.e_states = las.e_states
        return LASSI.kernel (self, **kwargs)

    def filter_spaces (self, las):
        # Hook for child methods
        return las

    make_lroots = make_lroots
    prepare_states = prepare_states

class LASSIrqCT (LASSIrq):
    def make_lroots (self, las, q=None):
        lroots = LASSIrq.make_lroots (self, las, q=q)
        charges_fr = get_space_info (las)[0].T
        lroots[charges_fr==0] = 1
        return lroots


