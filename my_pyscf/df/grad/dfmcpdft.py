from mrh.my_pyscf.grad import mcpdft as mcpdft_grad
from mrh.my_pyscf.df.grad import dfsacasscf as dfsacasscf_grad

# I need to resolve the __init__ and get_ham_response members. Otherwise everything should be fine! 
class Gradients (mcpdft_grad.Gradients, dfsacasscf_grad.Gradients):
    
    def __init__(self, pdft):
        self.auxbasis_response = True
        mcpdft_grad.Gradients.__init__(self, pdft)

    def get_ham_response (self, state=None, atmlst=None, verbose=None, mo=None, ci=None, eris=None, mf_grad=None, veff1=None, veff2=None, **kwargs):
        if state is None: state = self.state
        if atmlst is None: atmlst = self.atmlst
        if verbose is None: verbose = self.verbose
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        if (veff1 is None) or (veff2 is None):
            assert (False), kwargs
            veff1, veff2 = self.base.get_pdft_veff (mo, ci[state], incl_coul=True, paaa_only=True)
        fcasscf = self.make_fcasscf ()
        fcasscf.mo_coeff = mo
        fcasscf.ci = ci[state]
        return mcpdft_HellmanFeynman_grad (fcasscf, self.base.otfnal, veff1, veff2, mo_coeff=mo, ci=ci[state], atmlst=atmlst, mf_grad=mf_grad, verbose=verbose, auxbasis_response=self.auxbasis_response)


