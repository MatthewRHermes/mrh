from pyscf.mcscf import newton_casscf
from mrh.my_pyscf.grad import sacasscf
from functools import reduce
from scipy import linalg
import numpy as np

def mcpdft_HellmanFeynman_grad (mc, veff1, veff2, mo_coeff=None, ci=None, atmlst=None, mf_grad=None, verbose=None):
    ''' Modification of pyscf.grad.casscf.kernel to compute instead the Hellman-Feynman gradient
        terms of MC-PDFT. From the differentiated Hamiltonian matrix elements, only the core and
        Coulomb energy parts remain. For the renormalization terms, the effective Fock matrix is as in
        CASSCF, but with the same Hamiltonian substutition that is used for the energy response terms. '''
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci is None: ci = mc.ci
    if mf_grad is None: mf_grad = mc._scf.nuc_grad_method()
    if mc.frozen is not None:
        raise NotImplementedError

    mol = mc.mol
    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas
    nelecas = mc.nelecas
    nao, nmo = mo_coeff.shape
    nao_pair = nao * (nao+1) // 2

    mo_occ = mo_coeff[:,:nocc]
    mo_core = mo_coeff[:,:ncore]
    mo_cas = mo_coeff[:,ncore:nocc]

    casdm1, casdm2 = mc.fcisolver.make_rdm12(ci, ncas, nelecas)

# gfock = Generalized Fock, Adv. Chem. Phys., 69, 63
    dm_core = np.dot(mo_core, mo_core.T) * 2
    dm_cas = reduce(np.dot, (mo_cas, casdm1, mo_cas.T))
    # MRH: I need to replace aapa with the equivalent array from veff2
    # I'm not sure how the outcore file-paging system works, but hopefully I can do this
    # I also need to generate vhf_c and vhf_a from veff2 rather than the molecule's actual integrals
    # The true Coulomb repulsion should already be in veff1, but I need to generate the "fake"
    # vj - vk/2 from veff2
    h1e_mo = mo_coeff.T @ (mc.get_hcore() + veff1) @ mo_coeff + veff2.vhf_c
    aapa = np.zeros ((ncas,ncas,nmo,ncas), dtype=h1e_mo.dtype)
    vhf_a = np.zeros ((nmo,nmo), dtype=h1e_mo.dtype)
    for i in range (nmo):
        jbuf = veff2.ppaa[i]
        kbuf = veff2.papa[i]
        aapa[:,:,i,:] = jbuf[ncore:nocc,:,:]
        vhf_a[i] = np.tensordot (jbuf, casdm1, axes=2)
    vhf_a *= 0.5
    # for this potential, vj = vk: vj - vk/2 = vj - vj/2 = vj/2
    gfock = np.zeros ((nmo, nmo))
    gfock[:,:ncore] = (h1e_mo[:,:ncore] + vhf_a[:,:ncore]) * 2
    gfock[:,ncore:nocc] = h1e_mo[:,ncore:nocc] @ casdm1
    gfock[:,ncore:nocc] += np.einsum('uviw,vuwt->it', aapa, casdm2)
    dme0 = reduce(np.dot, (mo_coeff, (gfock+gfock.T)*.5, mo_coeff.T))
    aapa = vhf_a = h1e_mo = gfock = None

    dm1 = dm_core + dm_cas
    # MRH: vhf1c and vhf1a should be the TRUE vj_c and vj_a (no vk!)
    vj = mf_grad.get_jk (dm=dm1)[0]
    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)

    # MRH: the whole 2RDM part doesn't matter!

    if atmlst is None:
        atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()
    de_hcore = np.zeros((len(atmlst),3))
    de_renorm = np.zeros((len(atmlst),3))
    de_eri = np.zeros((len(atmlst),3))
    de = np.zeros((len(atmlst),3))

    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        h1ao = hcore_deriv(ia) # MRH: this should be the TRUE hcore
        de_hcore[k] += np.einsum('xij,ij->x', h1ao, dm1)
        de_renorm[k] -= np.einsum('xij,ij->x', s1[:,p0:p1], dme0[p0:p1]) * 2

        # MRH: the whole eri part is just the Coulomb energy!
        de_eri[k] += np.einsum('xij,ij->x', vj[:,p0:p1], dm1[p0:p1]) * 2

    de_nuc = mf_grad.grad_nuc(mol, atmlst)
    print ("MC-PDFT Hellmann-Feynman nuclear :\n{}".format (de_nuc))
    print ("MC-PDFT Hellmann-Feynman hcore component:\n{}".format (de_hcore))
    print ("MC-PDFT Hellmann-Feynman renorm component:\n{}".format (de_renorm))
    print ("MC-PDFT Hellmann-Feynman eri component:\n{}".format (de_eri))

    de = de_nuc + de_hcore + de_eri + de_renorm
    return de


class Gradients (sacasscf.Gradients):

    def __init__(self, pdft):
        super().__init__(pdft)

    def get_wfn_response (self, atmlst=None, iroot=None, verbose=None, mo=None, ci=None, veff1=None, veff2=None, **kwargs):
        if iroot is None: iroot = self.iroot
        if atmlst is None: atmlst = self.atmlst
        if verbose is None: verbose = self.verbose
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        if (veff1 is None) or (veff2 is None):
            assert (False), kwargs
            veff1, veff2 = self.base.get_pdft_veff (mo, ci[iroot], incl_coul=True)
        ndet = ci[iroot].size
        fcasscf = self.make_fcasscf ()
        fcasscf.mo_coeff = mo
        fcasscf.ci = ci[iroot]
        def my_hcore ():
            return self.base.get_hcore () + veff1
        fcasscf.get_hcore = my_hcore

        g_all_iroot = newton_casscf.gen_g_hop (fcasscf, mo, ci[iroot], veff2, verbose)[0]

        g_all = np.zeros (self.nlag)
        g_all[:self.ngorb] = g_all_iroot[:self.ngorb]
        # No need to reshape or anything, just use the magic of repeated slicing
        g_all[self.ngorb:][ndet*iroot:][:ndet] = g_all_iroot[self.ngorb:]

        # Do I need to project away a component of the gradient here? Very much maybe.
        return g_all

    def get_ham_response (self, iroot=None, atmlst=None, verbose=None, mo=None, ci=None, eris=None, mf_grad=None, veff1=None, veff2=None, **kwargs):
        if iroot is None: iroot = self.iroot
        if atmlst is None: atmlst = self.atmlst
        if verbose is None: verbose = self.verbose
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        if (veff1 is None) or (veff2 is None):
            assert (False), kwargs
            veff1, veff2 = self.base.get_pdft_veff (mo, ci[iroot], incl_coul=True)
        fcasscf = self.make_fcasscf ()
        fcasscf.mo_coeff = mo
        fcasscf.ci = ci[iroot]
        return mcpdft_HellmanFeynman_grad (fcasscf, veff1, veff2, mo_coeff=mo, ci=ci[iroot], atmlst=atmlst, mf_grad=mf_grad, verbose=verbose)

    def kernel (self, **kwargs):
        ''' Cache the effective Hamiltonian terms so you don't have to calculate them twice '''
        iroot = kwargs['iroot'] if 'iroot' in kwargs else self.iroot
        mo = kwargs['mo'] if 'mo' in kwargs else self.base.mo_coeff
        ci = kwargs['ci'] if 'ci' in kwargs else self.base.ci
        if isinstance (ci, np.ndarray): ci = [ci] # hack hack hack...
        kwargs['ci'] = ci
        kwargs['veff1'], kwargs['veff2'] = self.base.get_pdft_veff (mo, ci[iroot], incl_coul=True)
        return super().kernel (**kwargs)



