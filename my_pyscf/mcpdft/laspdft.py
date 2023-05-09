from pyscf import ao2mo
import numpy as np
from scipy import linalg

class laspdfthelper:

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
        casdm1s_sub = self.make_casdm1s_sub (**kwargs)
        casdm1a = linalg.block_diag (*[dm[0] for dm in casdm1s_sub])
        casdm1b = linalg.block_diag (*[dm[1] for dm in casdm1s_sub])
        return np.stack ([casdm1a, casdm1b], axis=0)

    def make_casdm1s_sub(self, ci=None, ncas_sub=None, nelecas_sub=None,
            casdm1frs=None, w=None, **kwargs):
        if casdm1frs is None: casdm1frs = self.mc.states_make_casdm1s_sub (ci=ci,
            ncas_sub=ncas_sub, nelecas_sub=nelecas_sub, **kwargs)
        if w is None: w = self.mc.weights
        return [np.einsum ('rspq,r->spq', dm1, w) for dm1 in casdm1frs]
        