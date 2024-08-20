import numpy as np
from pyscf import lib,gto
from pyscf.lib import logger
from mrh.my_pyscf.lassi import LASSI
from mrh.my_pyscf.lassi.spaces import spin_shuffle, spin_shuffle_ci
from mrh.my_pyscf.lassi.spaces import all_single_excitations, SingleLASRootspace
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

def as_scanner(lsi):
    '''Generating a scanner for LASSIrq PES.
    
    The returned solver is a function. This function requires one argument
    "mol" as input and returns total LASSIrq energy.

    The solver will automatically use the results of last calculation as the
    initial guess of the new calculation.  All parameters of LASSIrq object
    are automatically applied in the solver.
    
    Note scanner has side effects.  It may change many underlying objects
    (_scf, with_df, with_x2c, ...) during calculation.
    ''' 
    if isinstance(lsi, lib.SinglePointScanner):
        return lsi
        
    logger.info(lsi, 'Create scanner for %s', lsi.__class__)
    name = lsi.__class__.__name__ + LASSIrq_Scanner.__name_mixin__
    return lib.set_class(LASSIrq_Scanner(lsi), (LASSIrq_Scanner, lsi.__class__), name)
        
class LASSIrq_Scanner(lib.SinglePointScanner):
    def __init__(self, lsi, state=0):
        self.__dict__.update(lsi.__dict__)
        self._las = lsi._las.as_scanner()
        self._scan_state = state

    def __call__(self, mol_or_geom, **kwargs):
        if isinstance(mol_or_geom, gto.MoleBase):
            mol = mol_or_geom
        else:
            mol = self.mol.set_geom_(mol_or_geom, inplace=False)
    
        self.reset (mol)
        for key in ('with_df', 'with_x2c', 'with_solvent', 'with_dftd3'):
            sub_mod = getattr(self, key, None)
            if sub_mod:
                sub_mod.reset(mol)

        las_scanner = self._las
        las_scanner(mol)
        self.mol = mol
        self.mo_coeff = las_scanner.mo_coeff
        e_tot = self.kernel()[0][self._scan_state]
        if hasattr (e_tot, '__len__'):
            e_tot = np.average (e_tot)
        return e_tot


class LASSIrq (LASSI):
    def __init__(self, las, r=0, q=1, opt=2, **kwargs):
        self.r = r
        self.q = q
        LASSI.__init__(self, las, opt=opt, **kwargs)

    def prepare_states_(self):
        self.converged, las = self.prepare_states ()
        #self.__dict__.update(las.__dict__) # Unsafe
        self.fciboxes = las.fciboxes
        self.ci = las.ci
        self.nroots = las.nroots
        self.weights = las.weights
        self.e_lexc = las.e_lexc
        self.e_states = las.e_states

    def kernel (self, **kwargs):
        self.prepare_states_()
        return LASSI.kernel (self, **kwargs)

    def filter_spaces (self, las):
        # Hook for child methods
        return las

    make_lroots = make_lroots
    prepare_states = prepare_states
    as_scanner=as_scanner


class LASSIrqCT (LASSIrq):
    def make_lroots (self, las, q=None):
        lroots = LASSIrq.make_lroots (self, las, q=q)
        charges_fr = get_space_info (las)[0].T
        lroots[charges_fr==0] = 1
        return lroots


