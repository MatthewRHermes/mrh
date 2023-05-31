import inspect
from pyscf import ao2mo, lib
import numpy as np
import copy
from scipy import linalg
from types import MethodType
from copy import deepcopy
from mrh.my_pyscf.df.sparse_df import sparsedf_array

try:
    from pyscf.mcpdft.mcpdft import _PDFT, _mcscf_env
except ImportError:
        msg = "For performing LASPDFT, you will require pyscf-forge.\n" +\
        "pyscf-forge can be found at : https://github.com/pyscf/pyscf-forge"
        raise ImportError(msg)

class _LASPDFT(_PDFT):
    'MC-PDFT energy for a LASSCF wavefunction'
    _mc_class = _PDFT._mc_class
    def __init__(self, scf, ncas, nelecas, my_ot=None, grids_level=None,
            grids_attr=None, **kwargs):

        del _PDFT.optimize_mcscf_, _PDFT.kernel
        _PDFT.__init__(self, scf, ncas, nelecas, my_ot)
        self.e_mcscf = copy.deepcopy(self.e_tot)
        self.e_states = copy.deepcopy(self.e_states)
        
    def get_h2eff(self, mo_coeff=None):
        'Compute the active space two-particle Hamiltonian.'
        ncore = self.ncore
        ncas = self.ncas
        nocc = ncore + ncas
        if mo_coeff is None: mo_coeff = self.mo_coeff[:,ncore:nocc]
        elif mo_coeff.shape[1] != ncas: mo_coeff = mo_coeff[:,ncore:nocc]

        if getattr (self._scf, '_eri', None) is not None:
            eri = ao2mo.full(self._scf._eri, mo_coeff,
                                max_memory=self.max_memory)
        else:
            eri = ao2mo.full(self.mol, mo_coeff, verbose=self.verbose,
                                max_memory=self.max_memory)
        return eri

    def make_casdm1s_wrapper(self, ci=None, state=None, **kwargs):
       ''' Make the full-dimensional casdm1s spanning the collective active space '''
       if ci is None: ci = self.ci
       if state is None: state = self.state 
       casdm1s_sub = self.make_casdm1s_sub (**kwargs)
       casdm1a = linalg.block_diag (*[dm[0] for dm in casdm1s_sub])
       casdm1b = linalg.block_diag (*[dm[1] for dm in casdm1s_sub])
       return np.stack ([casdm1a, casdm1b], axis=0)
    

def get_mcpdft_child_class(mc, ot, **kwargs):
    mc_doc = (mc.__class__.__doc__ or 'No docstring for MC-SCF parent method')
   
    class PDFT(_LASPDFT, mc.__class__):
        __doc__= mc_doc + '\n\n' + _LASPDFT.__doc__
        _mc_class = mc.__class__

        def get_h2eff(self, mo_coeff=None):
            if self._in_mcscf_env: return mc.__class__.get_h2eff(self, mo_coeff=mo_coeff)
            else: return _LASPDFT.get_h2eff(self, mo_coeff=mo_coeff)
        
        def make_one_casdm1s(self, ci=None, state=None, **kwargs):
             if not self._in_mcscf_env:
                 return _LASPDFT.make_casdm1s_wrapper(self, ci=ci, state=state, **kwargs)

        def make_one_casdm2(self, ci=None, state=None, **kwargs):
            if not self._in_mcscf_env:
                return self.make_casdm2(ci=ci, state=state, **kwargs)
        
        def optimize_mcscf_(self, mo_coeff=None, ci0=None, **kwargs):
            '''Optimize the MC-SCF wave function underlying an MC-PDFT calculation.
            Has the same calling signature as the parent kernel method. '''
            with _mcscf_env(self):
                self.e_mcscf, self.e_cas, self.ci, self.mo_coeff, self.mo_energy = \
                    self._mc_class.kernel(self, mo_coeff, ci0=ci0, **kwargs)[:-2]
            return self.e_mcscf, self.e_cas, self.ci, self.mo_coeff, self.mo_energy
    
        def kernel(self, mo_coeff=None, ci0=None, otxc=None, grids_attr=None,
                    grids_level=None, **kwargs):
            self.optimize_mcscf_(mo_coeff=mo_coeff, ci0=ci0, **kwargs)
            self.compute_pdft_energy_(otxc=otxc, grids_attr=grids_attr,
                                      grids_level=grids_level, **kwargs)
            return (self.e_tot, self.e_ot, self.e_mcscf, self.e_cas, self.ci,
                self.mo_coeff, self.mo_energy)

    pdft = PDFT(mc._scf, mc.ncas_sub, mc.nelecas_sub, my_ot=ot, **kwargs)

    _keys = pdft._keys.copy()
    pdft.__dict__.update (mc.__dict__)
    pdft._keys = pdft._keys.union(_keys)
    return pdft

