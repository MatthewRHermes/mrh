from pyscf.mcscf import newton_casscf
from pyscf.grad import rks as rks_grad
from pyscf.dft import gen_grid
from pyscf.lib import logger
from mrh.my_pyscf.grad import sacasscf
from mrh.my_pyscf.mcpdft.otpd import get_ontop_pair_density
from mrh.util.rdm import get_2CDM_from_2RDM
from functools import reduce
from scipy import linalg
import numpy as np
import time

def mcpdft_HellmanFeynman_grad (mc, ot, veff1, veff2, mo_coeff=None, ci=None, atmlst=None, mf_grad=None, verbose=None):
    ''' Modification of pyscf.grad.casscf.kernel to compute instead the Hellman-Feynman gradient
        terms of MC-PDFT. From the differentiated Hamiltonian matrix elements, only the core and
        Coulomb energy parts remain. For the renormalization terms, the effective Fock matrix is as in
        CASSCF, but with the same Hamiltonian substutition that is used for the energy response terms. '''
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci is None: ci = mc.ci
    if mf_grad is None: mf_grad = mc._scf.nuc_grad_method()
    if mc.frozen is not None:
        raise NotImplementedError
    t0 = (time.clock (), time.time ())

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

    t0 = logger.timer (mc, 'PDFT HlFn gfock', *t0)
    dm1 = dm_core + dm_cas
    # MRH: vhf1c and vhf1a should be the TRUE vj_c and vj_a (no vk!)
    vj = mf_grad.get_jk (dm=dm1)[0]
    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)

    if atmlst is None:
        atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()
    de_hcore = np.zeros ((len(atmlst),3))
    de_renorm = np.zeros ((len(atmlst),3))
    de_coul = np.zeros ((len(atmlst),3))
    de_xc = np.zeros ((len(atmlst),3))
    de_grid = np.zeros ((len(atmlst),3))
    de_wgt = np.zeros ((len(atmlst),3))
    de = np.zeros ((len(atmlst),3))

    # MRH: Now I have to compute the gradient of the exchange-correlation energy
    # This involves derivatives of the orbitals that construct rho and Pi and therefore another
    # set of potentials. It also involves the derivatives of quadrature grid points which
    # propagate through the densities and therefore yet another set of potentials.
    # The orbital-derivative part includes all the grid points and some of the orbitals (- sign);
    # the grid-derivative part includes all of the orbitals and some of the grid points (+ sign).
    # I'll do a loop over grid sections and make arrays of type (3,nao, nao) and (3,nao, ncas, ncas, ncas).
    # I'll contract them within the grid loop for the grid derivatives and in the following
    # orbital loop for the xc derivatives
    dm1s = mc.make_rdm1s ()
    casdm1s = np.stack (mc.fcisolver.make_rdm1s (ci, ncas, nelecas), axis=0)
    twoCDM = get_2CDM_from_2RDM (casdm2, casdm1s)
    casdm1s = None
    make_rho = tuple (ot._numint._gen_rho_evaluator (mol, dm1s[i], 1) for i in range(2))
    dv1 = np.zeros ((3,nao,nao)) # Term which should be contracted with the whole density matrix
    dv1_a = np.zeros ((3,nao,nao)) # Term which should only be contracted with the core density matrix
    dv2 = np.zeros ((3,nao,ncas,ncas,ncas))
    idx = np.array ([[1,4,5,6],[2,5,7,8],[3,6,8,9]], dtype=np.int_) # For addressing particular ao derivatives
    if ot.xctype == 'LDA': idx = idx[:,0] # For LDAs no second derivatives
    diag_idx = np.arange(ncore, dtype=np.int_) * (ncore + 1) # for pqii
    casdm2_puvx = np.tensordot (mo_cas, casdm2, axes=1)
    full_atmlst = -np.ones (mol.natm, dtype=np.int_)
    t0 = logger.timer (mc, 'PDFT HlFn quadrature setup', *t0)
    for k, ia in enumerate (atmlst):
        full_atmlst[ia] = k
    for ia, (coords, w0, w1) in enumerate (rks_grad.grids_response_cc (ot.grids)):
        # For the xc potential derivative, I need every grid point in the entire molecule regardless of atmlist. (Because that's about orbitals.)
        # For the grid and weight derivatives, I only need the gridpoints that are in atmlst
        mask = gen_grid.make_mask (mol, coords)
        ao = ot._numint.eval_ao (mol, coords, deriv=ot.dens_deriv+1, non0tab=mask) # Need 1st derivs for LDA, 2nd for GGA, etc.
        if ot.xctype == 'LDA': # Might confuse the rho and Pi generators if I don't slice this down
            aoval = ao[:1]
        elif ot.xctype == 'GGA':
            aoval = ao[:4]
        rho = np.asarray ([m[0] (0, aoval, mask, ot.xctype) for m in make_rho])
        Pi = get_ontop_pair_density (ot, rho, aoval, dm1s, twoCDM, mo_cas, ot.dens_deriv)
        # Make sure that w1 only spans the atoms of interest

        t0 = logger.timer (mc, 'PDFT HlFn quadrature atom {} rho/Pi calc'.format (ia), *t0)
        for comp in range (3):
            # Weight response
            for k, ja in enumerate (atmlst):
                de_wgt[k,comp] += ot.get_E_ot (rho, Pi, w1[ja,comp])
            dao = ao[idx[comp]]

            # Vpq + Vpqii
            k = full_atmlst[ia]
            moval = np.tensordot (aoval, mo_core, axes=1)
            tmp_dv = ot.get_veff_1body (rho, Pi, [dao, aoval], w0)
            tmp_dv += ot.get_veff_2body (rho, Pi, [dao, aoval, moval, moval],
                w0).reshape (nao,nao,ncore*ncore)[:,:,diag_idx].sum(2) # Note that this is implicitly dm_core / 2 
            if k >= 0: de_grid[k,comp] += 2 * (tmp_dv * dm1).sum () # All orbitals, only some grid points
            dv1[comp] -= tmp_dv # d/dr = -d/dR

            # Vpquv * Duv (contract with core dm only)
            moval = np.tensordot (aoval, mo_cas, axes=1)
            tmp_dv = ot.get_veff_2body (rho, Pi, [dao, aoval, moval, moval], w0)
            tmp_dv = np.tensordot (tmp_dv, casdm1, axes=2) # Since this is explicitly casdm1, I now have a factor of 2 that needs to cancel.
            if k >= 0: de_grid[k,comp] += (tmp_dv * dm_core).sum () # All orbitals, only some grid points
            dv1_a[comp] -= tmp_dv # d/dr = -d/dR

            # Vpuvx
            tmp_dv = ot.get_veff_2body (rho, Pi, [dao, moval, moval, moval], w0)
            if k >= 0: de_grid[k,comp] += 2 * (tmp_dv * casdm2_puvx).sum () # All orbitals, only some grid points
            dv2[comp] -= tmp_dv # d/dr = -d/dR
            t0 = logger.timer (mc, 'PDFT HlFn quadrature atom {} component {}'.format (ia, comp), *t0)

    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        h1ao = hcore_deriv(ia) # MRH: this should be the TRUE hcore
        de_hcore[k] += np.einsum('xij,ij->x', h1ao, dm1)
        de_renorm[k] -= np.einsum('xij,ij->x', s1[:,p0:p1], dme0[p0:p1]) * 2
        de_coul[k] += np.einsum('xij,ij->x', vj[:,p0:p1], dm1[p0:p1]) * 2
        de_xc[k] += np.einsum ('xij,ij->x', dv1[:,p0:p1], dm1[p0:p1]) * 2 # Full quadrature, only some orbitals
        de_xc[k] += np.einsum ('xij,ij->x', dv1_a[:,p0:p1], dm_core[p0:p1]) # Ditto
        de_xc[k] += np.einsum ('xijkl,ijkl->x', dv2[:,p0:p1], casdm2_puvx[p0:p1]) * 2 # Ditto

    de_nuc = mf_grad.grad_nuc(mol, atmlst)

    logger.debug (mc, "MC-PDFT Hellmann-Feynman nuclear :\n{}".format (de_nuc))
    logger.debug (mc, "MC-PDFT Hellmann-Feynman hcore component:\n{}".format (de_hcore))
    logger.debug (mc, "MC-PDFT Hellmann-Feynman coulomb component:\n{}".format (de_coul))
    logger.debug (mc, "MC-PDFT Hellmann-Feynman xc component:\n{}".format (de_xc))
    logger.debug (mc, "MC-PDFT Hellmann-Feynman quadrature point component:\n{}".format (de_grid))
    logger.debug (mc, "MC-PDFT Hellmann-Feynman quadrature weight component:\n{}".format (de_wgt))
    logger.debug (mc, "MC-PDFT Hellmann-Feynman renorm component:\n{}".format (de_renorm))

    de = de_nuc + de_hcore + de_coul + de_renorm + de_xc + de_grid + de_wgt
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
        return mcpdft_HellmanFeynman_grad (fcasscf, self.base.otfnal, veff1, veff2, mo_coeff=mo, ci=ci[iroot], atmlst=atmlst, mf_grad=mf_grad, verbose=verbose)

    def kernel (self, **kwargs):
        ''' Cache the effective Hamiltonian terms so you don't have to calculate them twice '''
        iroot = kwargs['iroot'] if 'iroot' in kwargs else self.iroot
        mo = kwargs['mo'] if 'mo' in kwargs else self.base.mo_coeff
        ci = kwargs['ci'] if 'ci' in kwargs else self.base.ci
        if isinstance (ci, np.ndarray): ci = [ci] # hack hack hack...
        kwargs['ci'] = ci
        kwargs['veff1'], kwargs['veff2'] = self.base.get_pdft_veff (mo, ci[iroot], incl_coul=True)
        return super().kernel (**kwargs)



