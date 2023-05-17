from pyscf import ao2mo
import numpy as np
import copy
from scipy import linalg
from types import MethodType
from copy import deepcopy

try:
    from pyscf.mcpdft.mcpdft import _PDFT, _mcscf_env
except ImportError:
        msg = "For performing LASPDFT, you will require pyscf-forge.\n" +\
        "pyscf-forge can be found at : https://github.com/pyscf/pyscf-forge"
        raise ImportError(msg)

class _LASPDFT(_PDFT):
    'MC-PDFT energy for a LASSCF wavefunction'

    def __init__(self, scf, ncas, nelecas, my_ot=None, grids_level=None,
            grids_attr=None, **kwargs):
        _PDFT.__init__(self, scf, ncas, nelecas, my_ot)
        self.e_mcscf = copy.deepcopy(self.e_tot) # MRH commentary: I think self.e_states goes here
        # MRH design commentary:
        # The line below, and the redefinition of optimize_mcscf_ and kernel further down, are a
        # consequence of my use of "super()" in the MC-PDFT metaclass template, which is something
        # I absolutely should not have done. I will try to improve upon this so that this
        # awkwardness isn't necessary in the future.
        del _PDFT.optimize_mcscf_, _PDFT.kernel
        
    def get_h2eff(self, mo_coeff=None):
        'Compute the active space two-particle Hamiltonian.'
        ncore = self.ncore
        ncas = self.ncas
        nocc = ncore + ncas

        if mo_coeff is None:
            ncore = self.ncore # MRH commentary: This line is redundant (look 5 lines above!)
            mo_coeff = self.mo_coeff[:,ncore:nocc]
        elif mo_coeff.shape[1] != ncas:
            mo_coeff = mo_coeff[:,ncore:nocc]

        # MRH commentary: you can make the line below shorter:
        #     if getattr (self._scf, '_eri', None) is not None: 
        # does exactly the same thing.
        if hasattr(self._scf, '_eri') and self._scf._eri is not None:
            eri = ao2mo.full(self._scf._eri, mo_coeff,
                                max_memory=self.max_memory)
        else:
            eri = ao2mo.full(self.mol, mo_coeff, verbose=self.verbose,
                                max_memory=self.max_memory)
        return eri

    def make_casdm1s(self, ci=None, state=None, **kwargs):
       ''' Make the full-dimensional casdm1s spanning the collective active space '''
       # MRH commentary: Note again that the kwarg "state" does nothing at all.
       # I don't understand what this redefinition is for. Why do you need this?
       # It breaks multi-state functionality.
       casdm1s_sub = self.make_casdm1s_sub (**kwargs)
       casdm1a = linalg.block_diag (*[dm[0] for dm in casdm1s_sub])
       casdm1b = linalg.block_diag (*[dm[1] for dm in casdm1s_sub])
       return np.stack ([casdm1a, casdm1b], axis=0)
    
    def optimize_mcscf_(self, mo_coeff=None, ci0=None, **kwargs):
        '''Optimize the MC-SCF wave function underlying an MC-PDFT calculation.
        Has the same calling signature as the parent kernel method. '''
        with _mcscf_env(self):
            self.e_mcscf, self.e_cas, self.ci, self.mo_coeff, self.mo_energy = \
                super().kernel(mo_coeff, ci0=ci0, **kwargs)[:-2]
            return self.e_mcscf, self.e_cas, self.ci, self.mo_coeff, self.mo_energy
    
    def kernel(self, mo_coeff=None, ci0=None, otxc=None, grids_attr=None,
                grids_level=None, **kwargs):
        self.optimize_mcscf_(mo_coeff=mo_coeff, ci0=ci0, **kwargs)
        self.compute_pdft_energy_(otxc=otxc, grids_attr=grids_attr,
                                  grids_level=grids_level, **kwargs)
        return (self.e_tot, self.e_ot, self.e_mcscf, self.e_cas, self.ci,
            self.mo_coeff, self.mo_energy)


def get_mcpdft_child_class(mc, ot, **kwargs):
    mc_doc = (mc.__class__.__doc__ or 'No docstring for MC-SCF parent method')

    class PDFT(_LASPDFT, mc.__class__):
        __doc__= mc_doc + '\n\n' + _LASPDFT.__doc__

        def get_h2eff(self, mo_coeff=None):
            if self._in_mcscf_env: return mc.__class__.get_h2eff(self, mo_coeff=mo_coeff)
            else: return _LASPDFT.get_h2eff(self, mo_coeff=mo_coeff)
        
        def make_one_casdm1s(self, ci=None, state=None, **kwargs):
            # MRH commentary: the arguments "ci" and "state" passed by caller
            # disappear because you don't pass them on to the function below.
            # You should probably have "ci=ci, state=state" in the line below.
            # Design commentary: the right way to do this is to make this function
            # access the "state"th return values of mc.__class__.states_make_casdm1s
            return _LASPDFT.make_casdm1s(self, ci=None, state=None, **kwargs)

        def make_one_casdm2(self, ci=None, state=None, **kwargs):
            # MRH commentary: the arguments "ci" and "state" passed by caller
            # disappear because you don't pass them on to the function below.
            # You should probably have "ci=ci, state=state" in the line below.
            # Design commentary: the right way to do this is to make this function
            # access the "state"th return values of mc.__class__.states_make_casdm2
            return self.make_casdm2(ci=None, state=None, **kwargs)
        
        def optimize_mcscf_(self, mo_coeff=None, ci0=None, **kwargs):
            # MRH commentary: this does nothing, because optimize_mcscf_ is inherited
            # from the parent _LASPDFT class already
            return _LASPDFT.optimize_mcscf_(self, mo_coeff=None, ci0=None, **kwargs)
        
        def kernel(self, mo_coeff=None, ci0=None, otxc=None, grids_attr=None,
                grids_level=None, **kwargs):
            # MRH commentary: this does nothing, because kernel is inherited
            # from the parent _LASPDFT class already
            return _LASPDFT.kernel(self, mo_coeff=None, ci0=None, otxc=None, grids_attr=None,
                grids_level=None, **kwargs)
        
    pdft = PDFT(mc._scf, mc.ncas_sub, mc.nelecas_sub, my_ot=ot, **kwargs)
    _keys = pdft._keys.copy()
    pdft.__dict__.update (mc.__dict__)
    pdft._keys = pdft._keys.union(_keys)
    return pdft

