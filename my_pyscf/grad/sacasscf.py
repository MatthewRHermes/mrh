from mrh.my_pyscf.grad import lagrange
from mrh.my_pyscf.mcscf import newton_sacasscf
from pyscf.mcscf import mc1step, newton_casscf
from pyscf.grad import casscf as casscf_grad
import numpy as np
import copy

class Gradients (lagrange.Gradients):

    def __init__(self, mc):
        self.ngorb = np.count_nonzero (mc.uniq_var_indices)
        self.nci = mc.fcisolver.nroots * mc.ci[0].size
        self.nroots = mc.fcisolver.nroots
        self.iroot = mc.grad_nuc_iroot
        self.eris = None
        self.weights = np.array ([1])
        if hasattr (mc, 'weights'):
            self.weights = np.asarray (mc.weights)
        assert (len (self.weights) == self.nroots)
        lagrange.Gradients.__init__(self, mc.mol, self.ngorb+self.nci, mc)

    def kernel (self, iroot=None, atmlst=None, verbose=None, mo=None, ci=None, eris=None, mf_grad=None, **kwargs):
        if iroot is None: iroot = self.iroot
        if atmlst is None: atmlst = self.atmlst
        if verbose is None: atmlst = self.verbose
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        if eris is None and self.eris is None:
            eris = self.eris = self.base.ao2mo (mo)
        elif eris is None:
            eris = self.eris
        if mf_grad is None: mf_grad = self.base._scf.nuc_grad_method ()
        return super().kernel (iroot=iroot, atmlst=atmlst, verbose=verbose, mo=mo, ci=ci, eris=eris, mf_grad=mf_grad)

    def get_wfn_response (self, atmlst=None, iroot=None, verbose=None, mo=None, ci=None, eris=None, **kwargs):
        if iroot is None: iroot = self.iroot
        if atmlst is None: atmlst = self.atmlst
        if verbose is None: atmlst = self.verbose
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci[iroot]
        if eris is None and self.eris is None:
            eris = self.eris = self.base.ao2mo (mo)
        elif eris is None:
            eris = self.eris
        fcasscf = copy.copy (self.base)
        if hasattr (self.base, 'ss_fcisolver'):
            fcasscf.fcisolver = self.base.ss_fcisolver
        g_all = newton_casscf.gen_g_hop (fcasscf, mo, ci, eris, verbose)[0]
        return g_all

    def get_Lop_Ldiag (self, atmlst=None, verbose=None, mo=None, ci=None, eris=None, **kwargs):
        if atmlst is None: atmlst = self.atmlst
        if verbose is None: atmlst = self.verbose
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        if eris is None and self.eris is None:
            eris = self.eris = self.base.ao2mo (mo)
        elif eris is None:
            eris = self.eris
        Lop, Ldiag = newton_sacasscf.gen_g_hop (self.base, mo, ci, eris, verbose)[2:]
        return Lop, Ldiag

    def get_nuc_response (self, iroot=None, atmlst=None, verbose=None, mo=None, ci=None, eris=None, mf_grad=None, **kwargs):
        if iroot is None: iroot = self.iroot
        if atmlst is None: atmlst = self.atmlst
        if verbose is None: atmlst = self.verbose
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci[iroot]
        if eris is None and self.eris is None:
            eris = self.eris = self.base.ao2mo (mo)
        elif eris is None:
            eris = self.eris
        return casscf_grad.kernel (self.base, mo_coeff=mo, ci=ci, atmlst=atmlst, mf_grad=mf_grad, verbose=verbose)

    def get_LdotJnuc (self, Lvec, atmlst=None, verbose=None, mo=None, ci=None, eris=None, mf_grad=None, **kwargs):
        if iroot is None: iroot = self.iroot
        if atmlst is None: atmlst = self.atmlst
        if verbose is None: atmlst = self.verbose
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci[iroot]
        if eris is None and self.eris is None:
            eris = self.eris = self.base.ao2mo (mo)
        elif eris is None:
            eris = self.eris
        ncas = self.base.ncas
        nelecas = self.base.nelecas
        if getattr(self.base.fcisolver, 'gen_linkstr', None):
            linkstrl = casscf.fcisolver.gen_linkstr(ncas, nelecas, True)
            linkstr  = casscf.fcisolver.gen_linkstr(ncas, nelecas, False)
        else:
            linkstrl = linkstr  = None

        # Just sum the weights now... Lorb can be implicitly summed
        # Lci may be in the csf basis
        Lorb = self.mc.unpack_uniq_var (Lvec[:self.ngorb])
        Lci = Lvec[self.ngorb:].reshape (self.nroots, -1)
        ci = np.ravel (ci).reshape (self.nroots, -1)

        # CI part
        Lcasdm1 = np.zeros ((self.nroots, ncas, ncas))
        Lcasdm2 = np.zeros ((self.nroots, ncas, ncas, ncas, ncas))
        for ir in range (self.nroots):
            Lcasdm1[ir], Lcasdm2[ir] = self.base.fcisolver.trans_rdm12 (Lci[ir], ci[ir], ncas, nelecas, link_index=linkstr)
        Lcasdm1 = (Lcasdm1 * self.weights[:,None,None]).sum (0)
        Lcasdm2 = (Lcasdm2 * self.weights[:,None,None,None,None]).sum (0)

        # Orb part
        casdm1, casdm2 = self.base.fcisolver.make_rdm12 (ci, ncas, nelecas, link_index=linkstr)
        Lcasdm1 += Lorb @ casdm1 - casdm1 @ Lorb
        Lcasdm2 += np.tensordot (Lorb, casdm2, axes=1) - np.tensordot (casdm2, Lorb, axes=1)
        #Lcasdm2 += np.einsum ('kp,ijpl->ijkl', Lorb, casdm2)
        Lcasdm2 += np.tensordot (Lorb, casdm2, axes=(1,2)).transpose (1,2,0,3)
        #Lcasdm2 -= np.einsum ('ipkl,pj->ijkl', casdm2, Lorb)
        Lcasdm2 -= np.tensordot (casdm2, Lorb, axes=(1,0)).transpose (0,3,1,2)

        fcasscf = copy.copy (self.base)
        fcasscf.fcisolver = fci.solver (self.mol)
        fcasscf.make_rdm12 = lambda *args, **kwargs: Lcasdm1, Lcasdm2

        return casscf_grad.kernel (fcasscf, mo_coeff=mo, ci=ci, atmlst=atmlst, mf_grad=mf_grad, verbose=verbose)
        
    


