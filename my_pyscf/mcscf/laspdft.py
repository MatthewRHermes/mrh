from pyscf import ao2mo, lib, scf, gto
import numpy as np
from scipy import linalg
from copy import deepcopy

class LASPDFT:

    def __init__(self, mc):
        self.mc = mc
        self.otfnal = None
        self.mc.get_h2eff = self.get_h2eff
        self.mc.make_one_casdm1s = self.make_casdm1s
        self.mc.make_one_casdm2 = self.mc.make_casdm2
        self.mc.e_mcscf = mc.e_tot
        
    def get_h2eff(self, mo_coeff=None):
        '''Compute the active space two-particle Hamiltonian.
        '''
        mc = self.mc
        ncore = mc.ncore
        ncas = mc.ncas
        nocc = ncore + ncas

        if mo_coeff is None:
            ncore = mc.ncore
            mo_coeff = mc.mo_coeff[:,ncore:nocc]
        elif mo_coeff.shape[1] != ncas:
            mo_coeff = mo_coeff[:,ncore:nocc]

        if hasattr(mc._scf, '_eri') and mc._scf._eri is not None:
            eri = ao2mo.full(mc._scf._eri, mo_coeff,
                                max_memory=mc.max_memory)
        else:
            eri = ao2mo.full(mc.mol, mo_coeff, verbose=mc.verbose,
                                max_memory=mc.max_memory)
        return eri

    def make_casdm1s(self, ci=None, state=None, **kwargs):
        ''' Make the full-dimensional casdm1s spanning the collective active space '''
        #casdm1frs = self.mc.states_make_casdm1s_sub(ci=self.mc.ci, ncas_sub=self.mc.ncas_sub, nelecas_sub=self.mc.nelecas_sub, **kwargs)
        #w = self.mc.weights
        casdm1s_sub = self.make_casdm1s_sub (**kwargs)
        #casdm1s_sub = [np.einsum ('rspq,r->spq', dm1, w) for dm1 in casdm1frs]
        casdm1a = linalg.block_diag (*[dm[0] for dm in casdm1s_sub])
        casdm1b = linalg.block_diag (*[dm[1] for dm in casdm1s_sub])
        return np.stack ([casdm1a, casdm1b], axis=0)

    def make_casdm1s_sub(self, ci=None, ncas_sub=None, nelecas_sub=None,
            casdm1frs=None, w=None, **kwargs):
        if casdm1frs is None: casdm1frs = self.mc.states_make_casdm1s_sub (ci=ci,
            ncas_sub=ncas_sub, nelecas_sub=nelecas_sub, **kwargs)
        if w is None: w = self.mc.weights
        return [np.einsum ('rspq,r->spq', dm1, w) for dm1 in casdm1frs]
        
    def update_kernel(self, obj):
        obj.kernel = modified_kernel(obj)
        return obj

def modified_kernel(self, mo_coeff=None, ci0=None, otxc=None, grids_attr=None,
        grids_level=None, **kwargs ):
    #self.optimize_mcscf_(mo_coeff=mo_coeff, ci0=ci0, **kwargs)
    self.compute_pdft_energy_(otxc=otxc, grids_attr=grids_attr,
                                grids_level=grids_level, **kwargs)
    return (self.e_tot, self.e_ot, self.e_mcscf, self.e_cas, self.ci,
    self.mo_coeff, self.mo_energy)
    
class _lasscf_env (object):
    '''Prevent MC-SCF step of MC-PDFT from overwriting redefined
    quantities e_states and e_tot '''
    def __init__(self, mc):
        self.mc = mc
        self.e_tot = deepcopy(self.mc.e_tot)
        self.e_states = deepcopy(getattr (self.mc, 'e_states', None))
    
    def __enter__(self):
        self.mc._in_mcscf_env = True

    def __exit__(self, type, value, traceback):
        self.mc.e_tot = self.e_tot
        if getattr (self.mc, 'e_states', None) is not None:
            self.mc.e_mcscf = np.array (self.mc.e_states)
        if self.e_states is not None:
            try:
                self.mc.e_states = self.e_states
            except AttributeError as e:
                self.mc.fcisolver.e_states = self.e_states
                assert (self.mc.e_states is self.e_states), str (e)
            # TODO: redesign this. MC-SCF e_states is stapled to
            # fcisolver.e_states, but I don't want MS-PDFT to be 
            # because that makes no sense
        self.mc._in_mcscf_env = False