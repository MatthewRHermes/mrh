from mrh.my_pyscf.grad import lagrange
from mrh.my_pyscf.fci.csfstring import CSFTransformer
from pyscf.grad.mp2 import _shell_prange
from pyscf.mcscf import mc1step, mc1step_symm, newton_casscf
from pyscf.grad import casscf as casscf_grad
from pyscf.grad import rhf as rhf_grad
from pyscf.fci.direct_spin1 import _unpack_nelec
from pyscf.fci.spin_op import spin_square0
from pyscf import lib, ao2mo
import numpy as np
import copy
from functools import reduce
from scipy import linalg

def Lorb_dot_dgorb_dx (Lorb, mc, mo_coeff=None, ci=None, atmlst=None, mf_grad=None, verbose=None):
    ''' Modification of pyscf.grad.casscf.kernel to compute instead the orbital
    Lagrange term nuclear gradient (sum_pq Lorb_pq d2_Ecas/d_lambda d_kpq)
    This involves making the substitution
    (D_[p]q + D_p[q]) -> D_pq
    (d_[p]qrs + d_pq[r]s + d_p[q]rs + d_pqr[s]) -> d_pqrs
    Where [] around an index implies contraction with Lorb from the left, so that the external index
    (regardless of whether the index on the rdm is bra or ket) is always the bra of Lorb. '''

    # dmo = smoT.dao.smo
    # dao = mo.dmo.moT

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

    # MRH: new 'effective' MO coefficients including contraction from the Lagrange multipliers
    s0 = mc._scf.get_ovlp ()
    moL_coeff = mo_coeff @ Lorb
    smo_coeff = mc._scf.get_ovlp () @ mo_coeff
    smoL_coeff = smo_coeff @ Lorb
    moL_occ = moL_coeff[:,:nocc]
    moL_core = moL_coeff[:,:ncore]
    moL_cas = moL_coeff[:,ncore:nocc]
    smo_occ = smo_coeff[:,:nocc]
    smo_core = smo_coeff[:,:ncore]
    smo_cas = smo_coeff[:,ncore:nocc]
    smoL_occ = smoL_coeff[:,:nocc]
    smoL_core = smoL_coeff[:,:ncore]
    smoL_cas = smoL_coeff[:,ncore:nocc]

    # MRH: these SHOULD be state-averaged! Use the actual sacasscf object!
    casdm1, casdm2 = mc.fcisolver.make_rdm12(ci, ncas, nelecas)

    # gfock = Generalized Fock, Adv. Chem. Phys., 69, 63
    # MRH: each index exactly once!
    dm_core = np.dot(mo_core, mo_core.T) * 2
    dm_cas = reduce(np.dot, (mo_cas, casdm1, mo_cas.T))
    # MRH: new density matrix terms
    dmL_core = np.dot(moL_core, mo_core.T) * 2
    dmL_cas = reduce(np.dot, (moL_cas, casdm1, mo_cas.T))
    dmL_core += dmL_core.T
    dmL_cas += dmL_cas.T
    dm1 = dm_core + dm_cas
    dm1L = dmL_core + dmL_cas
    # MRH: end new density matrix terms
    # MRH: wrap the integral instead of the density matrix. I THINK the sign is the same!
    # mo sets 0 and 2 should be transposed, 1 and 3 should be not transposed; this will lead to correct sign
    # Except I can't do this for the external index, because the external index is contracted to ovlp matrix,
    # not the 2RDM
    aapaL  = ao2mo.kernel(mol, (moL_cas, mo_cas, mo_coeff, mo_cas), compact=False)
    aapaL += ao2mo.kernel(mol, (mo_cas, moL_cas, mo_coeff, mo_cas), compact=False) 
    aapaL += ao2mo.kernel(mol, (mo_cas, mo_cas, mo_coeff, moL_cas), compact=False) 
    aapaL  = aapaL.reshape(ncas,ncas,nmo,ncas) 
    aapa = ao2mo.kernel(mol, (mo_cas, mo_cas, mo_coeff, mo_cas), compact=False) 
    aapa = aapa.reshape(ncas,ncas,nmo,ncas) 
    # MRH: new vhf terms
    vj, vk   = mc._scf.get_jk(mol, (dm_core,  dm_cas))
    vjL, vkL = mc._scf.get_jk(mol, (dmL_core, dmL_cas))
    h1 = mc.get_hcore()
    vhf_c = vj[0] - vk[0] * .5
    vhf_a = vj[1] - vk[1] * .5
    vhfL_c = vjL[0] - vkL[0] * .5
    vhfL_a = vjL[1] - vkL[1] * .5
    # MRH: I rewrote this Feff calculation completely, double-check it
    gfock  = h1 @ dm1L # h1e 
    gfock += (vhf_c + vhf_a) @ dmL_core # core-core and active-core, 2nd 1RDM linked
    gfock += (vhfL_c + vhfL_a) @ dm_core # core-core and active-core, 1st 1RDM linked
    gfock += vhfL_c @ dm_cas # core-active, 1st 1RDM linked
    gfock += vhf_c @ dmL_cas # core-active, 2nd 1RDM linked
    gfock[:] = 0
    gfock += mo_coeff @ np.einsum('uviw,uvtw->it', aapaL, casdm2) @ mo_cas.T # active-active
    # MRH: I have to contract this external 2RDM index explicitly on the 2RDM but fortunately I can do so here
    gfock += mo_coeff @ np.einsum('uviw,vuwt->it', aapa, casdm2) @ moL_cas.T 
    # MRH: As of 04/18/2019, the two-body part of this is including aapaL is definitely, unambiguously correct
    dme0 = (gfock+gfock.T)*.5 # This transpose is for the overlap matrix later on
    aapa = vj = vk = vhf_c = vhf_a = None

    vhf1c, vhf1a, vhf1cL, vhf1aL = mf_grad.get_veff(mol, (dm_core, dm_cas, dmL_core, dmL_cas))
    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)

    diag_idx = np.arange(nao)
    diag_idx = diag_idx * (diag_idx+1) // 2 + diag_idx
    casdm2_cc = casdm2 + casdm2.transpose(0,1,3,2)
    dm2buf = ao2mo._ao2mo.nr_e2(casdm2_cc.reshape(ncas**2,ncas**2), mo_cas.T,
                                (0, nao, 0, nao)).reshape(ncas**2,nao,nao)
    # MRH: contract the final two indices of the active-active 2RDM with L as you change to AOs
    # note tensordot always puts indices in the order of the arguments.
    dm2Lbuf = np.zeros ((ncas**2,nmo,nmo))
    dm2Lbuf[:,:,ncore:nocc]  = np.tensordot (Lorb[:,ncore:nocc], casdm2, axes=(1,2)).transpose (1,2,0,3).reshape (ncas**2,nmo,ncas)
    dm2Lbuf[:,ncore:nocc,:] += np.tensordot (Lorb[:,ncore:nocc], casdm2, axes=(1,3)).transpose (1,2,3,0).reshape (ncas**2,ncas,nmo) # This term transposes the L
    dm2Lbuf += dm2Lbuf.transpose (0,2,1) # This term transposes the derivative later on
    dm2Lbuf = np.ascontiguousarray (dm2Lbuf)
    dm2Lbuf = ao2mo._ao2mo.nr_e2(dm2Lbuf.reshape (ncas**2,nmo**2), mo_coeff.T,
                                (0, nao, 0, nao)).reshape(ncas**2,nao,nao)
    dm2buf = lib.pack_tril(dm2buf)
    dm2buf[:,diag_idx] *= .5
    dm2buf = dm2buf.reshape(ncas,ncas,nao_pair)
    dm2Lbuf = lib.pack_tril(dm2Lbuf)
    dm2Lbuf[:,diag_idx] *= .5
    dm2Lbuf = dm2Lbuf.reshape(ncas,ncas,nao_pair)

    if atmlst is None:
        atmlst = list (range(mol.natm))
    aoslices = mol.aoslice_by_atom()
    de_hcore = np.zeros((len(atmlst),3))
    de_renorm = np.zeros((len(atmlst),3))
    de_eri = np.zeros((len(atmlst),3))
    de = np.zeros((len(atmlst),3))

    max_memory = mc.max_memory - lib.current_memory()[0]
    blksize = int(max_memory*.9e6/8 / ((aoslices[:,3]-aoslices[:,2]).max()*nao_pair))
    blksize = min(nao, max(2, blksize))

    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        h1ao = hcore_deriv(ia)
        # MRH: h1e and Feff terms
        de_hcore[k] += np.einsum('xij,ij->x', h1ao, dm1L)
        de_renorm[k] -= np.einsum('xij,ij->x', s1[:,p0:p1], dme0[p0:p1]) * 2

        q1 = 0
        for b0, b1, nf in _shell_prange(mol, 0, mol.nbas, blksize):
            q0, q1 = q1, q1 + nf
            dm2_ao  = lib.einsum('ijw,pi,qj->pqw', dm2Lbuf, mo_cas[p0:p1], mo_cas[q0:q1])
            # MRH: now contract the first two indices of the active-active 2RDM with L as you go from MOs to AOs
            dm2_ao += lib.einsum('ijw,pi,qj->pqw', dm2buf, moL_cas[p0:p1], mo_cas[q0:q1])
            dm2_ao += lib.einsum('ijw,pi,qj->pqw', dm2buf, mo_cas[p0:p1], moL_cas[q0:q1])
            shls_slice = (shl0,shl1,b0,b1,0,mol.nbas,0,mol.nbas)
            eri1 = mol.intor('int2e_ip1', comp=3, aosym='s2kl',
                             shls_slice=shls_slice).reshape(3,p1-p0,nf,nao_pair)
            # MRH: I still don't understand why there is a minus here!
            de_eri[k] -= np.einsum('xijw,ijw->x', eri1, dm2_ao) * 2
            eri1 = None
        # MRH: core-core and core-active 2RDM terms
        de_eri[k] += np.einsum('xij,ij->x', vhf1c[:,p0:p1], dm1L[p0:p1]) * 2
        de_eri[k] += np.einsum('xij,ij->x', vhf1cL[:,p0:p1], dm1[p0:p1]) * 2
        # MRH: active-core 2RDM terms
        de_eri[k] += np.einsum('xij,ij->x', vhf1a[:,p0:p1], dmL_core[p0:p1]) * 2
        de_eri[k] += np.einsum('xij,ij->x', vhf1aL[:,p0:p1], dm_core[p0:p1]) * 2

    # MRH: deleted the nuclear-nuclear part to avoid double-counting
    # lesson learned from debugging - mol.intor computes -1 * the derivative and only
    # for one index
    # on the other hand, mf_grad.hcore_generator computes the actual derivative of
    # h1 for both indices and with the correct sign

    print ("Orb lagrange hcore component:\n{}".format (de_hcore))
    print ("Orb lagrange renorm component:\n{}".format (de_renorm))
    print ("Orb lagrange eri component:\n{}".format (de_eri))
    de = de_hcore + de_renorm + de_eri

    s1a = np.zeros ((len (atmlst), 3, s1.shape[1], s1.shape[2]))
    all_h1eX = np.zeros_like (s1a)
    all_eriX = np.zeros ((len (atmlst), 3, nmo, nmo, nmo, nmo))
    for k, ia in enumerate (atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        s1a[k,:,p0:p1,:] -= s1[:,p0:p1,:] 
        s1a[k,:,:,p0:p1] -= s1[:,p0:p1,:].transpose (0,2,1) 
        all_h1eX[k] = hcore_deriv(ia)
        shls_slice = (shl0,shl1,0,mol.nbas,0,mol.nbas,0,mol.nbas)
        eri1 = mol.intor ('int2e_ip1', comp=3, aosym='s1', shls_slice=shls_slice).reshape (3,p1-p0,nao,nao,nao)
        all_eriX[k,:,p0:p1,:,:,:] -= eri1
        all_eriX[k,:,:,p0:p1,:,:] -= eri1.transpose (0,2,1,4,3)
        all_eriX[k,:,:,:,p0:p1,:] -= eri1.transpose (0,3,4,1,2)
        all_eriX[k,:,:,:,:,p0:p1] -= eri1.transpose (0,4,3,2,1)
        for alp in range (all_eriX.shape[1]):
            all_h1eX[k,alp] = mo_coeff.T @ all_h1eX[k,alp] @ mo_coeff
            all_eriX[k,alp] = ao2mo.incore.full (all_eriX[k,alp], mo_coeff, compact=False).reshape (nmo,nmo,nmo,nmo)

    s1a /= 2
    test_gfock = np.einsum ('abpq,pq->ab',s1a,gfock+gfock.T)
    print ("Orb lagrange renorm test gfock:\n{}".format (test_gfock))

    s1a = mo_coeff.T @ s1a @ mo_coeff
    h1e_mo = mo_coeff.T @ h1 @ mo_coeff
    all_h1eS  = np.einsum ('abpx,xq->abpq', s1a, h1e_mo)
    all_h1eS += np.einsum ('abqx,px->abpq', s1a, h1e_mo)
    all_dm1 = np.zeros_like (dm1L)
    all_dm1[:,ncore:nocc] += np.einsum ('px,xq->pq',Lorb[:,ncore:nocc],casdm1)
    all_dm1[ncore:nocc,:] += np.einsum ('qx,px->pq',Lorb[:,ncore:nocc],casdm1)
    all_eri = ao2mo.kernel (mol, (mo_coeff, mo_coeff, mo_coeff, mo_coeff), compact=False).reshape (nmo,nmo,nmo,nmo)
    all_eriS  = np.einsum ('abpx,xqrs->abpqrs', s1a, all_eri)
    all_eriS += np.einsum ('abqx,pxrs->abpqrs', s1a, all_eri)
    all_eriS += np.einsum ('abrx,pqxs->abpqrs', s1a, all_eri)
    all_eriS += np.einsum ('absx,pqrx->abpqrs', s1a, all_eri)
    all_dm2 = np.zeros_like (all_eri)
    all_dm2[:,ncore:nocc,ncore:nocc,ncore:nocc] += np.einsum ('px,xqrs->pqrs',Lorb[:,ncore:nocc],casdm2) 
    all_dm2[ncore:nocc,:,ncore:nocc,ncore:nocc] += np.einsum ('qx,pxrs->pqrs',Lorb[:,ncore:nocc],casdm2)
    all_dm2[ncore:nocc,ncore:nocc,:,ncore:nocc] += np.einsum ('rx,pqxs->pqrs',Lorb[:,ncore:nocc],casdm2)
    all_dm2[ncore:nocc,ncore:nocc,ncore:nocc,:] += np.einsum ('sx,pqrx->pqrs',Lorb[:,ncore:nocc],casdm2)
    de_renorm1_debug = np.einsum ('abpq,pq->ab', all_h1eS, all_dm1) 

    print ("Orb lagrange renorm component 1 DEBUG:\n{}".format (de_renorm1_debug))
    de_renorm2_debug = np.einsum ('abpqrs,pqrs->ab',all_eriS,all_dm2) / 2
    print ("Orb lagrange renorm component 2 DEBUG:\n{}".format (de_renorm2_debug))
    print ("Orb lagrange renorm component 1+2 DEBUG:\n{}".format (de_renorm1_debug+de_renorm2_debug))

    de_hcore_debug = np.einsum ('abpq,pq->ab', all_h1eX, all_dm1) 
    print ("Orb lagrange hcore component DEBUG:\n{}".format (de_hcore_debug))

    de_eri_debug = np.einsum ('abpqrs,pqrs->ab',all_eriX,all_dm2) / 2
    print ("Orb lagrange eri component DEBUG:\n{}".format (de_eri_debug))

    return de

def Lci_dot_dgci_dx (Lci, weights, mc, mo_coeff=None, ci=None, atmlst=None, mf_grad=None, verbose=None):
    ''' Modification of pyscf.grad.casscf.kernel to compute instead the CI
    Lagrange term nuclear gradient (sum_IJ Lci_IJ d2_Ecas/d_lambda d_PIJ)
    This involves removing all core-core and nuclear-nuclear terms and making the substitution
    sum_I w_I<L_I|p'q|I> + c.c. -> <0|p'q|0>
    sum_I w_I<L_I|p'r'sq|I> + c.c. -> <0|p'r'sq|0>
    The active-core terms (sum_I w_I<L_I|x'iyi|I>, sum_I w_I <L_I|x'iiy|I>, c.c.) must be retained.'''
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
    nroots = ci.shape[0]

    mo_occ = mo_coeff[:,:nocc]
    mo_core = mo_coeff[:,:ncore]
    mo_cas = mo_coeff[:,ncore:nocc]

    # MRH: TDMs + c.c. instead of RDMs
    casdm1 = np.zeros ((nroots, ncas, ncas))
    casdm2 = np.zeros ((nroots, ncas, ncas, ncas, ncas))
    for iroot in range (nroots):
        #print ("norm of Lci, ci for root {}: {} {}".format (iroot, linalg.norm (Lci[iroot]), linalg.norm (ci[iroot])))
        casdm1[iroot], casdm2[iroot] = mc.fcisolver.trans_rdm12 (Lci[iroot], ci[iroot], ncas, nelecas)
    casdm1 = (casdm1 * weights[:,None,None]).sum (0)
    casdm2 = (casdm2 * weights[:,None,None,None,None]).sum (0)
    casdm1 += casdm1.transpose (1,0)
    casdm2 += casdm2.transpose (1,0,3,2)

# gfock = Generalized Fock, Adv. Chem. Phys., 69, 63
    dm_core = np.dot(mo_core, mo_core.T) * 2
    dm_cas = reduce(np.dot, (mo_cas, casdm1, mo_cas.T))
    aapa = ao2mo.kernel(mol, (mo_cas, mo_cas, mo_coeff, mo_cas), compact=False)
    aapa = aapa.reshape(ncas,ncas,nmo,ncas)
    vj, vk = mc._scf.get_jk(mol, (dm_core, dm_cas))
    h1 = mc.get_hcore()
    vhf_c = vj[0] - vk[0] * .5
    vhf_a = vj[1] - vk[1] * .5
    # MRH: delete h1 + vhf_c from the first line below (core and core-core stuff)
    # Also extend gfock to span the whole space
    gfock = np.zeros_like (dm_cas)
    gfock[:,:nocc]   = reduce(np.dot, (mo_coeff.T, vhf_a, mo_occ)) * 2
    gfock[:,ncore:nocc]  = reduce(np.dot, (mo_coeff.T, h1 + vhf_c, mo_cas, casdm1))
    gfock[:,ncore:nocc] += np.einsum('uvpw,vuwt->pt', aapa, casdm2)
    dme0 = reduce(np.dot, (mo_coeff, (gfock+gfock.T)*.5, mo_coeff.T))
    aapa = vj = vk = vhf_c = vhf_a = h1 = gfock = None

    vhf1c, vhf1a = mf_grad.get_veff(mol, (dm_core, dm_cas))
    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)

    diag_idx = np.arange(nao)
    diag_idx = diag_idx * (diag_idx+1) // 2 + diag_idx
    casdm2_cc = casdm2 + casdm2.transpose(0,1,3,2)
    dm2buf = ao2mo._ao2mo.nr_e2(casdm2_cc.reshape(ncas**2,ncas**2), mo_cas.T,
                                (0, nao, 0, nao)).reshape(ncas**2,nao,nao)
    dm2buf = lib.pack_tril(dm2buf)
    dm2buf[:,diag_idx] *= .5
    dm2buf = dm2buf.reshape(ncas,ncas,nao_pair)
    casdm2 = casdm2_cc = None

    if atmlst is None:
        atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()
    de_hcore = np.zeros((len(atmlst),3))
    de_renorm = np.zeros((len(atmlst),3))
    de_eri = np.zeros((len(atmlst),3))
    de = np.zeros((len(atmlst),3))

    max_memory = mc.max_memory - lib.current_memory()[0]
    blksize = int(max_memory*.9e6/8 / ((aoslices[:,3]-aoslices[:,2]).max()*nao_pair))
    blksize = min(nao, max(2, blksize))

    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        h1ao = hcore_deriv(ia)
        # MRH: dm1 -> dm_cas in the line below
        de_hcore[k] += np.einsum('xij,ij->x', h1ao, dm_cas)
        de_renorm[k] -= np.einsum('xij,ij->x', s1[:,p0:p1], dme0[p0:p1]) * 2

        q1 = 0
        for b0, b1, nf in _shell_prange(mol, 0, mol.nbas, blksize):
            q0, q1 = q1, q1 + nf
            dm2_ao = lib.einsum('ijw,pi,qj->pqw', dm2buf, mo_cas[p0:p1], mo_cas[q0:q1])
            shls_slice = (shl0,shl1,b0,b1,0,mol.nbas,0,mol.nbas)
            eri1 = mol.intor('int2e_ip1', comp=3, aosym='s2kl',
                             shls_slice=shls_slice).reshape(3,p1-p0,nf,nao_pair)
            de_eri[k] -= np.einsum('xijw,ijw->x', eri1, dm2_ao) * 2
            eri1 = None
        # MRH: dm1 -> dm_cas in the line below. Also eliminate core-core terms
        de_eri[k] += np.einsum('xij,ij->x', vhf1c[:,p0:p1], dm_cas[p0:p1]) * 2
        de_eri[k] += np.einsum('xij,ij->x', vhf1a[:,p0:p1], dm_core[p0:p1]) * 2

    print ("CI lagrange hcore component:\n{}".format (de_hcore))
    print ("CI lagrange renorm component:\n{}".format (de_renorm))
    print ("CI lagrange eri component:\n{}".format (de_eri))
    de = de_hcore + de_renorm + de_eri
    return de


class Gradients (lagrange.Gradients):

    def __init__(self, mc):
        self.__dict__.update (mc.__dict__)
        nmo = mc.mo_coeff.shape[-1]
        self.ngorb = np.count_nonzero (mc.uniq_var_indices (nmo, mc.ncore, mc.ncas, mc.frozen))
        self.nci = mc.fcisolver.nroots * mc.ci[0].size
        self.nroots = mc.fcisolver.nroots
        self.iroot = mc.nuc_grad_iroot
        self.eris = None
        self.weights = np.array ([1])
        self.e_avg = mc.e_tot
        self.e_states = np.asarray (mc.e_tot)
        if hasattr (mc, 'weights'):
            self.weights = np.asarray (mc.weights)
            self.e_avg = (self.weights * self.e_states).sum ()
        assert (len (self.weights) == self.nroots), '{} {}'.format (self.weights, self.nroots)
        lagrange.Gradients.__init__(self, mc.mol, self.ngorb+self.nci, mc)

    def make_fcasscf (self, casscf_attr={}, fcisolver_attr={}):
        ''' Make a fake CASSCF object for ostensible single-state calculations '''
        if isinstance (self.base, mc1step_symm.CASSCF):
            fcasscf = mc1step_symm.CASSCF (self.base._scf, self.base.ncas, self.base.nelecas)
        else:
            fcasscf = mc1step.CASSCF (self.base._scf, self.base.ncas, self.base.nelecas)
        fcasscf.__dict__.update (self.base.__dict__)
        if hasattr (self.base, 'weights'):
            fcasscf.fcisolver = self.base.fcisolver._base_class (self.base.mol)
            fcasscf.nroots = 1
        fcasscf.__dict__.update (casscf_attr)
        fcasscf.fcisolver.__dict__.update (fcisolver_attr)
        return fcasscf

    def kernel (self, iroot=None, atmlst=None, verbose=None, mo=None, ci=None, eris=None, mf_grad=None, e_states=None, e_avg=None, level_shift=None, **kwargs):
        if iroot is None: iroot = self.iroot
        if atmlst is None: atmlst = self.atmlst
        if verbose is None: verbose = self.verbose
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        if eris is None and self.eris is None:
            eris = self.eris = self.base.ao2mo (mo)
        elif eris is None:
            eris = self.eris
        if mf_grad is None: mf_grad = self.base._scf.nuc_grad_method ()
        if e_states is None: e_states = self.e_states
        if e_avg is None: e_avg = self.e_avg
        if level_shift is None: level_shift=self.level_shift
        return super().kernel (iroot=iroot, atmlst=atmlst, verbose=verbose, mo=mo, ci=ci, eris=eris, mf_grad=mf_grad, e_states=e_states, e_avg=e_avg, level_shift=level_shift)

    def get_wfn_response (self, atmlst=None, iroot=None, verbose=None, mo=None, ci=None, **kwargs):
        if iroot is None: iroot = self.iroot
        if atmlst is None: atmlst = self.atmlst
        if verbose is None: verbose = self.verbose
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        ndet = ci[iroot].size
        fcasscf = self.make_fcasscf ()
        fcasscf.mo_coeff = mo
        fcasscf.ci = ci[iroot]
        eris = fcasscf.ao2mo (mo)
        g_all_iroot = newton_casscf.gen_g_hop (fcasscf, mo, ci[iroot], eris, verbose)[0]
        g_all = np.zeros (self.nlag)
        g_all[:self.ngorb] = g_all_iroot[:self.ngorb]
        # No need to reshape or anything, just use the magic of repeated slicing
        g_all[self.ngorb:][ndet*iroot:][:ndet] = g_all_iroot[self.ngorb:]
        return g_all

    def get_Aop_Adiag (self, atmlst=None, iroot=None, verbose=None, mo=None, ci=None, eris=None, level_shift=None, **kwargs):
        if atmlst is None: atmlst = self.atmlst
        if verbose is None: verbose = self.verbose
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        if eris is None and self.eris is None:
            eris = self.eris = self.base.ao2mo (mo)
        elif eris is None:
            eris = self.eris
        Aop, Adiag = newton_casscf.gen_g_hop (self.base, mo, ci, eris, verbose)[2:]
        # Eliminate the component of Aop (x) which is parallel to the state-average space
        # The Lagrange multiplier equations are not defined there
        def my_Aop (x):
            Ax = Aop (x)
            Ax_ci = Ax[self.ngorb:].reshape (self.nroots, -1)
            ci_arr = np.asarray (ci).reshape (self.nroots, -1)
            ovlp = ci_arr.conjugate () @ Ax_ci.T
            Ax_ci -= ovlp.T @ ci_arr
            Ax[self.ngorb:] = Ax_ci.ravel ()
            return Ax
        return my_Aop, Adiag

    def get_ham_response (self, iroot=None, atmlst=None, verbose=None, mo=None, ci=None, eris=None, mf_grad=None, **kwargs):
        if iroot is None: iroot = self.iroot
        if atmlst is None: atmlst = self.atmlst
        if verbose is None: verbose = self.verbose
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        if eris is None and self.eris is None:
            eris = self.eris = self.base.ao2mo (mo)
        elif eris is None:
            eris = self.eris
        fcasscf = self.make_fcasscf ()
        fcasscf.mo_coeff = mo
        fcasscf.ci = ci[iroot]
        return casscf_grad.kernel (fcasscf, mo_coeff=mo, ci=ci[iroot], atmlst=atmlst, mf_grad=mf_grad, verbose=verbose)

    def get_LdotJnuc (self, Lvec, iroot=None, atmlst=None, verbose=None, mo=None, ci=None, eris=None, mf_grad=None, **kwargs):
        if iroot is None: iroot = self.iroot
        if atmlst is None: atmlst = self.atmlst
        if verbose is None: verbose = self.verbose
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci[iroot]
        if eris is None and self.eris is None:
            eris = self.eris = self.base.ao2mo (mo)
        elif eris is None:
            eris = self.eris
        ncas = self.base.ncas
        nelecas = self.base.nelecas
        if getattr(self.base.fcisolver, 'gen_linkstr', None):
            linkstr  = self.base.fcisolver.gen_linkstr(ncas, nelecas, False)
        else:
            linkstr  = None

        # Just sum the weights now... Lorb can be implicitly summed
        # Lci may be in the csf basis
        Lorb = self.base.unpack_uniq_var (Lvec[:self.ngorb])
        Lci = Lvec[self.ngorb:].reshape (self.nroots, -1)
        ci = np.ravel (ci).reshape (self.nroots, -1)

        # CI part
        de_Lci = Lci_dot_dgci_dx (Lci, self.weights, self.base, mo_coeff=mo, ci=ci, atmlst=atmlst, mf_grad=mf_grad, verbose=verbose)
        lib.logger.info(self, '--------------- %s gradient Lagrange CI response ---------------',
                    self.base.__class__.__name__)
        if verbose >= lib.logger.INFO: rhf_grad._write(self, self.mol, de_Lci, atmlst)
        lib.logger.info(self, '----------------------------------------------------------------')

        # Orb part
        de_Lorb = Lorb_dot_dgorb_dx (Lorb, self.base, mo_coeff=mo, ci=ci, atmlst=atmlst, mf_grad=mf_grad, verbose=verbose)
        lib.logger.info(self, '--------------- %s gradient Lagrange orbital response ---------------',
                    self.base.__class__.__name__)
        if verbose >= lib.logger.INFO: rhf_grad._write(self, self.mol, de_Lorb, atmlst)
        lib.logger.info(self, '----------------------------------------------------------------------')

        return de_Lci + de_Lorb
    
    def debug_lagrange (self, Lvec, bvec, Aop, Adiag, iroot=None, mo=None, ci=None, **kwargs):
        if iroot is None: iroot = self.iroot
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        lib.logger.info (self, '{} gradient: iroot = {}'.format (self.base.__class__.__name__, iroot))
        ngorb = self.ngorb
        nci = self.nci
        nroots = self.nroots
        ndet = nci // nroots
        ncore = self.base.ncore
        ncas = self.base.ncas
        nelecas = self.base.nelecas
        nocc = ncore + ncas
        nlag = self.nlag
        ci = np.asarray (self.base.ci).reshape (nroots, -1)
        err = Aop (Lvec) + bvec
        eorb = self.base.unpack_uniq_var (err[:ngorb])
        eci = err[ngorb:].reshape (nroots, -1)
        borb = self.base.unpack_uniq_var (bvec[:ngorb])
        bci = bvec[ngorb:].reshape (nroots, -1)
        Lorb = self.base.unpack_uniq_var (Lvec[:ngorb])
        Lci = Lvec[ngorb:].reshape (nroots, ndet)
        Lci_ci_ovlp = np.asarray (ci).reshape (nroots,-1).conjugate () @ Lci.T
        Lci_Lci_ovlp = Lci.conjugate () @ Lci.T
        ci_ci_ovlp = ci.conjugate () @ ci.T
        lib.logger.debug (self, "{} gradient RHS, inactive-active orbital rotations:\n{}".format (
            self.base.__class__.__name__, borb[:ncore,ncore:nocc]))
        lib.logger.debug (self, "{} gradient RHS, inactive-external orbital rotations:\n{}".format (
            self.base.__class__.__name__, borb[:ncore,nocc:]))
        lib.logger.debug (self, "{} gradient RHS, active-external orbital rotations:\n{}".format (
            self.base.__class__.__name__, borb[ncore:nocc,nocc:]))
        lib.logger.debug (self, "{} gradient residual, inactive-active orbital rotations:\n{}".format (
            self.base.__class__.__name__, eorb[:ncore,ncore:nocc]))
        lib.logger.debug (self, "{} gradient residual, inactive-external orbital rotations:\n{}".format (
            self.base.__class__.__name__, eorb[:ncore,nocc:]))
        lib.logger.debug (self, "{} gradient residual, active-external orbital rotations:\n{}".format (
            self.base.__class__.__name__, eorb[ncore:nocc,nocc:]))
        lib.logger.debug (self, "{} gradient Lagrange factor, inactive-active orbital rotations:\n{}".format (
            self.base.__class__.__name__, Lorb[:ncore,ncore:nocc]))
        lib.logger.debug (self, "{} gradient Lagrange factor, inactive-external orbital rotations:\n{}".format (
            self.base.__class__.__name__, Lorb[:ncore,nocc:]))
        lib.logger.debug (self, "{} gradient Lagrange factor, active-external orbital rotations:\n{}".format (
            self.base.__class__.__name__, Lorb[ncore:nocc,nocc:]))
        '''
        lib.logger.debug (self, "{} gradient RHS, inactive-inactive orbital rotations (redundant!):\n{}".format (
            self.base.__class__.__name__, borb[:ncore,:ncore]))
        lib.logger.debug (self, "{} gradient RHS, active-active orbital rotations (redundant!):\n{}".format (
            self.base.__class__.__name__, borb[ncore:nocc,ncore:nocc]))
        lib.logger.debug (self, "{} gradient RHS, external-external orbital rotations (redundant!):\n{}".format (
            self.base.__class__.__name__, borb[nocc:,nocc:]))
        lib.logger.debug (self, "{} gradient Lagrange factor, inactive-inactive orbital rotations (redundant!):\n{}".format (
            self.base.__class__.__name__, Lorb[:ncore,:ncore]))
        lib.logger.debug (self, "{} gradient Lagrange factor, active-active orbital rotations (redundant!):\n{}".format (
            self.base.__class__.__name__, Lorb[ncore:nocc,ncore:nocc]))
        lib.logger.debug (self, "{} gradient Lagrange factor, external-external orbital rotations (redundant!):\n{}".format (
            self.base.__class__.__name__, Lorb[nocc:,nocc:]))
        '''
        lib.logger.debug (self, "{} gradient Lagrange factor, CI part overlap with true CI SA space:\n{}".format ( 
            self.base.__class__.__name__, Lci_ci_ovlp))
        lib.logger.debug (self, "{} gradient Lagrange factor, CI part self overlap matrix:\n{}".format ( 
            self.base.__class__.__name__, Lci_Lci_ovlp))
        lib.logger.debug (self, "{} gradient Lagrange factor, CI vector self overlap matrix:\n{}".format ( 
            self.base.__class__.__name__, ci_ci_ovlp))
        neleca, nelecb = _unpack_nelec (nelecas)
        spin = neleca - nelecb + 1
        csf = CSFTransformer (ncas, neleca, nelecb, spin)
        ecsf = csf.vec_det2csf (eci, normalize=False, order='C')
        err_norm_det = linalg.norm (err)
        err_norm_csf = linalg.norm (np.append (eorb, ecsf.ravel ()))
        lib.logger.debug (self, "{} gradient: determinant residual = {}, CSF residual = {}".format (
            self.base.__class__.__name__, err_norm_det, err_norm_csf))
        ci_lbls, ci_csf   = csf.printable_largest_csf (ci,  10, isdet=True, normalize=True,  order='C')
        bci_lbls, bci_csf = csf.printable_largest_csf (bci, 10, isdet=True, normalize=False, order='C')
        eci_lbls, eci_csf = csf.printable_largest_csf (eci, 10, isdet=True, normalize=False, order='C')
        Lci_lbls, Lci_csf = csf.printable_largest_csf (Lci, 10, isdet=True, normalize=False, order='C')
        ncsf = bci_csf.shape[1]
        for iroot in range (self.nroots):
            lib.logger.debug (self, "{} gradient Lagrange factor, CI part root {} spin square: {}".format (
                self.base.__class__.__name__, iroot, spin_square0 (Lci[iroot], ncas, nelecas)))
            lib.logger.debug (self, "Base CI vector")
            for icsf in range (ncsf):
                lib.logger.debug (self, '{} {}'.format (ci_lbls[iroot,icsf], ci_csf[iroot,icsf]))
            lib.logger.debug (self, "CI gradient:")
            for icsf in range (ncsf):
                lib.logger.debug (self, '{} {}'.format (bci_lbls[iroot,icsf], bci_csf[iroot,icsf]))
            lib.logger.debug (self, "CI residual:")
            for icsf in range (ncsf):
                lib.logger.debug (self, '{} {}'.format (eci_lbls[iroot,icsf], eci_csf[iroot,icsf]))
            lib.logger.debug (self, "CI Lagrange vector:")
            for icsf in range (ncsf):
                lib.logger.debug (self, '{} {}'.format (Lci_lbls[iroot,icsf], Lci_csf[iroot,icsf]))
        Afull = np.zeros ((nlag, nlag))
        dum = np.zeros ((nlag))
        for ix in range (nlag):
            dum[ix] = 1
            Afull[ix,:] = Aop (dum)
            dum[ix] = 0
        Afull_orborb = Afull[:ngorb,:ngorb]
        Afull_orbci = Afull[:ngorb,ngorb:].reshape (ngorb, nroots, ndet)
        Afull_ciorb = Afull[ngorb:,:ngorb].reshape (nroots, ndet, ngorb)
        Afull_cici = Afull[ngorb:,ngorb:].reshape (nroots, ndet, nroots, ndet).transpose (0, 2, 1, 3)
        print ("Orb-orb Hessian:\n{}".format (Afull_orborb))
        for iroot in range (nroots):
            print ("Orb-ci Hessian root {}:\n{}".format (iroot, Afull_orbci[:,iroot,:]))
            print ("Ci-orb Hessian root {}:\n{}".format (iroot, Afull_ciorb[iroot,:,:]))
            for jroot in range (nroots):
                print ("Ci-ci Hessian roots {},{}:\n{}".format (iroot, jroot, Afull_cici[iroot,jroot,:,:]))


    def get_lagrange_precond (self, Adiag, level_shift=None, ci=None, **kwargs):
        ''' The preconditioner needs to keep the various CI roots orthogonal to the whole SA space.
        If I understand correctly, for root I of the CI problem the preconditioner should be
        R_I * (1 - |J> S(I)_JK^-1 <K|R_I)
        where R_I = (Hdiag-E)^-1
        I,J, and K are true CI vectors (not the x Lagrange cofactor guesses)
        and S(I)_JK = <J|R_I|K> which must be inverted. The x that gets passed is the 
        change to the vector; I need to add the last-iteration guess in order to keep the
        CI guesses orthogonal'''
        if level_shift is None: level_shift = self.level_shift
        if ci is None: ci = self.base.ci
        ci = np.asarray (ci).reshape (self.nroots, -1)
        AorbD = Adiag[:self.ngorb]
        AciD = Adiag[self.ngorb:].reshape (self.nroots, -1)
        eorb = self.e_avg
        eci = self.e_states 
        '''
        Aorb = np.zeros ((self.ngorb, self.ngorb))
        fdum = np.zeros ((self.nlag))
        for idum in range (self.ngorb):
            fdum[idum] = 1
            Aorb[idum,:] = Aop (fdum)[:self.ngorb]
            fdum[idum] = 0
        Aorb_inv = linalg.inv (Aorb)
        '''
        Rorb = AorbD.copy () 
        Rorb[abs(Rorb)<1e-8] = 1e-8
        Rorb = 1./Rorb
        Rci = AciD - ((eci - level_shift) * self.weights)[:,None]
        Rci[abs(Rci)<1e-8] = 1e-8
        Rci = 1./Rci
        # R_I|J> 
        # Indices: I, det, J
        Rci_cross = Rci[:,:,None] * ci.T[None,:,:]
        # S(I)_JK = <J|R_I|K> (first index of CI contract with middle index of R_I|J> and reshape to put I first)
        Sci = np.tensordot (ci.conjugate (), Rci_cross, axes=(1,1)).transpose (1,0,2)
        # R_I|J> S(I)_JK^-1 (can only loop explicitly because of necessary call to linalg.inv)
        # Indices: I, det, K
        Rci_fix = np.zeros_like (Rci_cross)
        for iroot in range (self.nroots):
            Rci_fix[iroot] = Rci_cross[iroot] @ linalg.inv (Sci[iroot]) 

        def my_precond (x):
            # Orb part
            xorb = x[:self.ngorb]
            xorb = Rorb * xorb

            # CI part
            xci = x[self.ngorb:].reshape (self.nroots, -1)
            # R_I|H I> (indices: I, det)
            Rx = Rci * xci
            # <J|R_I|H I> (indices: J, I)
            from_right = ci.conjugate () @ Rx.T 
            # R_I|J> S(I)_JK^-1 <K|R_I|H I> (indices: I, det)
            Rx_sub = np.zeros_like (Rx)
            for iroot in range (self.nroots): 
                Rx_sub[iroot] = np.dot (Rci_fix[iroot], from_right[:,iroot])
            xci = Rx - Rx_sub

            # Make CI vectors orthogonal.  Need to refer to Lvec_last b/c x is just the change in Lvec
            '''
            xci_tot = xci + Lvec_op ()[self.ngorb:].reshape (self.nroots, -1)
            ovlp = xci_tot.conjugate () @ xci_tot.T
            norms = np.diag (ovlp)
            for iroot in range (1, self.nroots):
                ov = ovlp[:iroot,iroot] / norms[:iroot]
                xci_tot[iroot,:] -= (ov[:,None] * xci_tot[:iroot,:]).sum (0)
                ovlp = xci_tot.conjugate () @ xci_tot.T
                norms = np.diag (ovlp)
            xci = xci_tot - Lvec_op ()[self.ngorb:].reshape (self.nroots, -1)
            '''

            return np.append (xorb, xci.ravel ())
        return my_precond

    def get_lagrange_callback (self, Lvec_last, itvec, geff_op):
        def my_call (x):
            itvec[0] += 1
            geff = geff_op (x)
            deltax = x - Lvec_last
            gorb = geff[:self.ngorb]
            gci = geff[self.ngorb:]
            deltaorb = deltax[:self.ngorb]
            deltaci = deltax[self.ngorb:]
            lib.logger.debug (self, ('Lagrange optimization iteration {}, |gorb| = {}, |gci| = {}, '
                '|dLorb| = {}, |dLci| = {}').format (itvec[0], linalg.norm (gorb), linalg.norm (gci),
                linalg.norm (deltaorb), linalg.norm (deltaci))) 
            Lvec_last[:] = x[:]
        return my_call


