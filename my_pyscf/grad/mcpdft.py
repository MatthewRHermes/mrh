from pyscf.mcscf import newton_casscf
from pyscf.grad import rks as rks_grad
from pyscf.dft import gen_grid
from pyscf.lib import logger, pack_tril, current_memory
from mrh.my_pyscf.grad import sacasscf
from mrh.my_pyscf.mcpdft.otpd import get_ontop_pair_density
from mrh.my_pyscf.mcpdft.pdft_veff import _contract_vot_rho, _contract_ao_vao
from mrh.util.rdm import get_2CDM_from_2RDM
from functools import reduce
from scipy import linalg
import numpy as np
import time, gc

BLKSIZE = gen_grid.BLKSIZE

def mcpdft_HellmanFeynman_grad (mc, ot, veff1, veff2, mo_coeff=None, ci=None, atmlst=None, mf_grad=None, verbose=None, max_memory=None):
    ''' Modification of pyscf.grad.casscf.kernel to compute instead the Hellman-Feynman gradient
        terms of MC-PDFT. From the differentiated Hamiltonian matrix elements, only the core and
        Coulomb energy parts remain. For the renormalization terms, the effective Fock matrix is as in
        CASSCF, but with the same Hamiltonian substutition that is used for the energy response terms. '''
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci is None: ci = mc.ci
    if mf_grad is None: mf_grad = mc._scf.nuc_grad_method()
    if mc.frozen is not None:
        raise NotImplementedError
    if max_memory is None: max_memory = mc.max_memory
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
    make_rho_c = ot._numint._gen_rho_evaluator (mol, dm_core, 1) 
    make_rho_a = ot._numint._gen_rho_evaluator (mol, dm_cas, 1) 
    dv1 = np.zeros ((3,nao,nao)) # Term which should be contracted with the whole density matrix
    dv1_a = np.zeros ((3,nao,nao)) # Term which should only be contracted with the core density matrix
    dv2 = np.zeros ((3,nao))
    idx = np.array ([[1,4,5,6],[2,5,7,8],[3,6,8,9]], dtype=np.int_) # For addressing particular ao derivatives
    if ot.xctype == 'LDA': idx = idx[:,0] # For LDAs no second derivatives
    diag_idx = np.arange(ncas) # for puvx
    diag_idx = diag_idx * (diag_idx+1) // 2 + diag_idx
    casdm2_pack = (casdm2 + casdm2.transpose (0,1,3,2)).reshape (ncas**2, ncas, ncas)
    casdm2_pack = pack_tril (casdm2_pack).reshape (ncas, ncas, -1)
    casdm2_pack[:,:,diag_idx] *= 0.5
    diag_idx = np.arange(ncore, dtype=np.int_) * (ncore + 1) # for pqii
    full_atmlst = -np.ones (mol.natm, dtype=np.int_)
    t1 = logger.timer (mc, 'PDFT HlFn quadrature setup', *t0)
    for k, ia in enumerate (atmlst):
        full_atmlst[ia] = k
    for ia, (coords, w0, w1) in enumerate (rks_grad.grids_response_cc (ot.grids)):
        mask = gen_grid.make_mask (mol, coords)
        # For the xc potential derivative, I need every grid point in the entire molecule regardless of atmlist. (Because that's about orbitals.)
        # For the grid and weight derivatives, I only need the gridpoints that are in atmlst
        # Estimated memory footprint: [2*ndao*(nao+nocc) + 3*ndpi*ncas^2 + O(ncas^0,nao^0,nocc^0)]*ngrids
        gc.collect ()
        ngrids = coords.shape[0]
        ndao = (1,4,10,19)[ot.dens_deriv+1]
        ndrho = (1,4,10,19)[ot.dens_deriv]
        ndpi = (1,4)[ot.Pi_deriv]
        ncols = 1.05 * (ndao*(nao+nocc) + max(ndao*nao,3*ndpi*ncas*ncas))
        remaining_floats = (max_memory - current_memory ()[0]) * 1e6 / 8
        blksize = int (remaining_floats / (ncols*BLKSIZE)) * BLKSIZE
        blksize = max (BLKSIZE, min (blksize, ngrids, BLKSIZE*1200))
        for ip0 in range (0, ngrids, blksize):
            ip1 = min (ngrids, ip0+blksize)
            logger.info (mc, 'PDFT gradient atom {} slice {}-{} of {} total'.format (ia, ip0, ip1, ngrids))
            ao = ot._numint.eval_ao (mol, coords[ip0:ip1], deriv=ot.dens_deriv+1, non0tab=mask) # Need 1st derivs for LDA, 2nd for GGA, etc.
            if ot.xctype == 'LDA': # Might confuse the rho and Pi generators if I don't slice this down
                aoval = ao[:1]
            elif ot.xctype == 'GGA':
                aoval = ao[:4]
            rho = np.asarray ([m[0] (0, aoval, mask, ot.xctype) for m in make_rho])
            Pi = get_ontop_pair_density (ot, rho, aoval, dm1s, twoCDM, mo_cas, ot.dens_deriv)

            t1 = logger.timer (mc, 'PDFT HlFn quadrature atom {} rho/Pi calc'.format (ia), *t1)
            moval_occ = np.tensordot (aoval, mo_occ, axes=1)
            moval_core = moval_occ[...,:ncore]
            moval_cas = moval_occ[...,ncore:]
            t1 = logger.timer (mc, 'PDFT HlFn quadrature atom {} ao2mo grid'.format (ia), *t1)
            eot, vrho, vot = ot.eval_ot (rho, Pi, weights=w0[ip0:ip1])
            puvx_mem = 2 * ndpi * (ip1-ip0) * ncas * ncas * 8 / 1e6
            remaining_mem = max_memory - current_memory ()[0]
            logger.info (mc, 'PDFT gradient memory note: working on {} grid points; estimated puvx usage = {:.1f} of {:.1f} remaining MB'.format ((ip1-ip0), puvx_mem, remaining_mem))

            # Weight response
            de_wgt += np.tensordot (eot, w1[atmlst,...,ip0:ip1], axes=(0,2))
            t1 = logger.timer (mc, 'PDFT HlFn quadrature atom {} weight response'.format (ia), *t1)

            # Find the atoms that are a part of the atomlist - grid correction shouldn't be added if they aren't there
            # The last stuff to vectorize is in get_veff_2body!
            k = full_atmlst[ia]

            # Vpq + Vpqii
            vrho = _contract_vot_rho (vot, make_rho_c [0] (0, aoval, mask, ot.xctype), add_vrho=vrho)
            tmp_dv = np.stack ([ot.get_veff_1body (rho, Pi, [ao[ix], aoval], w0[ip0:ip1], kern=vrho) for ix in idx], axis=0)
            if k >= 0: de_grid[k] += 2 * np.tensordot (tmp_dv, dm1.T, axes=2) # Grid response
            dv1 -= tmp_dv # XC response
            vrho = tmp_dv = None
            t1 = logger.timer (mc, 'PDFT HlFn quadrature atom {} Vpq + Vpqii'.format (ia), *t1)

            # Viiuv * Duv
            vrho_a = _contract_vot_rho (vot, make_rho_a [0] (0, aoval, mask, ot.xctype))
            tmp_dv = np.stack ([ot.get_veff_1body (rho, Pi, [ao[ix], aoval], w0[ip0:ip1], kern=vrho_a) for ix in idx], axis=0)
            if k >= 0: de_grid[k] += 2 * np.tensordot (tmp_dv, dm_core.T, axes=2) # Grid response
            dv1_a -= tmp_dv # XC response
            vrho_a = tmp_dv = None
            t1 = logger.timer (mc, 'PDFT HlFn quadrature atom {} Viiuv'.format (ia), *t1)

            # Vpuvx
            tmp_dv = ot.get_veff_2body_kl (rho, Pi, moval_cas, moval_cas, w0[ip0:ip1], symm=True, kern=vot) # ndpi,ngrids,ncas*(ncas+1)//2
            tmp_dv = np.tensordot (tmp_dv, casdm2_pack, axes=(-1,-1)) # ndpi, ngrids, ncas, ncas
            tmp_dv[0] = (tmp_dv[:ndpi] * moval_cas[:ndpi,:,None,:]).sum (0) # Chain and product rule
            tmp_dv[1:ndpi] *= moval_cas[0,:,None,:] # Chain and product rule
            tmp_dv = tmp_dv.sum (-1) # ndpi, ngrids, ncas
            tmp_dv = np.tensordot (ao[idx[:,:ndpi]], tmp_dv, axes=((1,2),(0,1))) # comp, nao (orb), ncas (dm2)
            tmp_dv = np.einsum ('cpu,pu->cp', tmp_dv, mo_cas) # comp, ncas (it's ok to not vectorize this b/c the quadrature grid is gone)
            if k >= 0: de_grid[k] += 2 * tmp_dv.sum (1)
            dv2 -= tmp_dv # XC response
            tmp_dv = None
            t1 = logger.timer (mc, 'PDFT HlFn quadrature atom {} Vpuvx'.format (ia), *t1)

            rho = Pi = eot = vot = ao = aoval = moval_occ = moval_core = moval_cas = None
            gc.collect ()

    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        h1ao = hcore_deriv(ia) # MRH: this should be the TRUE hcore
        de_hcore[k] += np.einsum('xij,ij->x', h1ao, dm1)
        de_renorm[k] -= np.einsum('xij,ij->x', s1[:,p0:p1], dme0[p0:p1]) * 2
        de_coul[k] += np.einsum('xij,ij->x', vj[:,p0:p1], dm1[p0:p1]) * 2
        de_xc[k] += np.einsum ('xij,ij->x', dv1[:,p0:p1], dm1[p0:p1]) * 2 # Full quadrature, only some orbitals
        de_xc[k] += np.einsum ('xij,ij->x', dv1_a[:,p0:p1], dm_core[p0:p1]) * 2 # Ditto
        de_xc[k] += dv2[:,p0:p1].sum (1) * 2 # Ditto

    de_nuc = mf_grad.grad_nuc(mol, atmlst)

    logger.debug (mc, "MC-PDFT Hellmann-Feynman nuclear :\n{}".format (de_nuc))
    logger.debug (mc, "MC-PDFT Hellmann-Feynman hcore component:\n{}".format (de_hcore))
    logger.debug (mc, "MC-PDFT Hellmann-Feynman coulomb component:\n{}".format (de_coul))
    logger.debug (mc, "MC-PDFT Hellmann-Feynman xc component:\n{}".format (de_xc))
    logger.debug (mc, "MC-PDFT Hellmann-Feynman quadrature point component:\n{}".format (de_grid))
    logger.debug (mc, "MC-PDFT Hellmann-Feynman quadrature weight component:\n{}".format (de_wgt))
    logger.debug (mc, "MC-PDFT Hellmann-Feynman renorm component:\n{}".format (de_renorm))

    de = de_nuc + de_hcore + de_coul + de_renorm + de_xc + de_grid + de_wgt

    t1 = logger.timer (mc, 'PDFT HlFn total', *t0)

    return de

class Gradients (sacasscf.Gradients):

    def __init__(self, pdft):
        super().__init__(pdft)
        self.e_mcscf = self.base.e_mcscf

    def get_wfn_response (self, atmlst=None, iroot=None, verbose=None, mo=None, ci=None, veff1=None, veff2=None, **kwargs):
        if iroot is None: iroot = self.iroot
        if atmlst is None: atmlst = self.atmlst
        if verbose is None: verbose = self.verbose
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        if (veff1 is None) or (veff2 is None):
            assert (False), kwargs
            veff1, veff2 = self.base.get_pdft_veff (mo, ci[iroot], incl_coul=True, paaa_only=True)
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
        # Eliminate gradient of self-rotation
        gci_iroot = g_all_iroot[self.ngorb:]
        ci_arr = np.asarray (ci).reshape (self.nroots, -1)
        gci_sa = np.dot (ci_arr[iroot], gci_iroot)
        gci_iroot -= gci_sa * gci_iroot
        gci = g_all[self.ngorb:].reshape (self.nroots, -1)
        gci[iroot] += gci_iroot 

        return g_all

    def get_ham_response (self, iroot=None, atmlst=None, verbose=None, mo=None, ci=None, eris=None, mf_grad=None, veff1=None, veff2=None, **kwargs):
        if iroot is None: iroot = self.iroot
        if atmlst is None: atmlst = self.atmlst
        if verbose is None: verbose = self.verbose
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        if (veff1 is None) or (veff2 is None):
            assert (False), kwargs
            veff1, veff2 = self.base.get_pdft_veff (mo, ci[iroot], incl_coul=True, paaa_only=True)
        fcasscf = self.make_fcasscf ()
        fcasscf.mo_coeff = mo
        fcasscf.ci = ci[iroot]
        return mcpdft_HellmanFeynman_grad (fcasscf, self.base.otfnal, veff1, veff2, mo_coeff=mo, ci=ci[iroot], atmlst=atmlst, mf_grad=mf_grad, verbose=verbose)

    def get_init_guess (self, bvec, Adiag, Aop, precond):
        ''' Initial guess should solve the problem for SA-SA rotations '''
        ci_arr = np.asarray (self.base.ci).reshape (self.nroots, -1)
        ndet = ci_arr.shape[-1]
        b_ci = bvec[self.ngorb:].reshape (self.nroots, ndet)
        x0 = np.zeros_like (bvec)
        if self.nroots > 1:
            b_sa = np.dot (ci_arr.conjugate (), b_ci[self.iroot])
            A_sa = 2 * self.weights[self.iroot] * (self.e_mcscf - self.e_mcscf[self.iroot])
            A_sa[self.iroot] = 1
            b_sa[self.iroot] = 0
            x0_sa = -b_sa / A_sa # Hessian is diagonal so: easy
            ovlp = ci_arr.conjugate () @ b_ci.T
            logger.debug (self, 'Linear response SA-SA part:\n{}'.format (ovlp))
            logger.debug (self, 'Linear response SA-CI norms:\n{}'.format (linalg.norm (
                b_ci.T - ci_arr.T @ ovlp, axis=1)))
            logger.debug (self, 'Linear response orbital norms:\n{}'.format (linalg.norm (bvec[:self.ngorb])))
            logger.debug (self, 'SA-SA Lagrange multiplier for root {}:\n{}'.format (self.iroot, x0_sa))
            x0[self.ngorb:][ndet*self.iroot:][:ndet] = np.dot (x0_sa, ci_arr)
        r0 = bvec + Aop (x0)
        r0_ci = r0[self.ngorb:].reshape (self.nroots, ndet)
        ovlp = ci_arr.conjugate () @ r0_ci.T
        logger.debug (self, 'Lagrange residual SA-SA part after solving SA-SA part:\n{}'.format (ovlp))
        logger.debug (self, 'Lagrange residual SA-CI norms after solving SA-SA part:\n{}'.format (linalg.norm (
            r0_ci.T - ci_arr.T @ ovlp, axis=1)))
        logger.debug (self, 'Lagrange residual orbital norms after solving SA-SA part:\n{}'.format (linalg.norm (r0[:self.ngorb])))
        x0 += precond (-r0)
        r1 = bvec + Aop (x0)
        r1_ci = r1[self.ngorb:].reshape (self.nroots, ndet)
        ovlp = ci_arr.conjugate () @ r1_ci.T
        logger.debug (self, 'Lagrange residual SA-SA part after first precondition:\n{}'.format (ovlp))
        logger.debug (self, 'Lagrange residual SA-CI norms after first precondition:\n{}'.format (linalg.norm (
            r1_ci.T - ci_arr.T @ ovlp, axis=1)))
        logger.debug (self, 'Lagrange residual orbital norms after first precondition:\n{}'.format (linalg.norm (r1[:self.ngorb])))
        return x0

    def kernel (self, **kwargs):
        ''' Cache the effective Hamiltonian terms so you don't have to calculate them twice '''
        iroot = kwargs['iroot'] if 'iroot' in kwargs else self.iroot
        mo = kwargs['mo'] if 'mo' in kwargs else self.base.mo_coeff
        ci = kwargs['ci'] if 'ci' in kwargs else self.base.ci
        if isinstance (ci, np.ndarray): ci = [ci] # hack hack hack...
        kwargs['ci'] = ci
        kwargs['veff1'], kwargs['veff2'] = self.base.get_pdft_veff (mo, ci[iroot], incl_coul=True, paaa_only=True)
        return super().kernel (**kwargs)

    def project_Aop (self, Aop, ci, iroot):
        ''' Wrap the Aop function to project out redundant degrees of freedom for the CI part.  What's redundant
            changes between SA-CASSCF and MC-PDFT so modify this part in child classes. '''
        try:
            A_sa = 2 * self.weights[iroot] * (self.e_mcscf - self.e_mcscf[iroot])
        except IndexError as e:
            assert (self.nroots == 1), e
            A_sa = 0
        ci_arr = np.asarray (ci).reshape (self.nroots, -1)
        def my_Aop (x):
            Ax = Aop (x)
            x_ci = x[self.ngorb:].reshape (self.nroots, -1)
            Ax_ci = Ax[self.ngorb:].reshape (self.nroots, -1)
            ovlp = ci_arr.conjugate () @ Ax_ci.T
            Ax_ci -= np.dot (ovlp.T, ci_arr)
            # Add back in the SA rotation part but from the true energy conditions
            x_sa = np.dot (ci_arr.conjugate (), x_ci[iroot])
            Ax_ci[iroot] += np.dot (x_sa * A_sa, ci_arr)
            Ax[self.ngorb:] = Ax_ci.ravel ()
            return Ax
        return my_Aop

'''
    def get_lagrange_precond (self, Adiag, level_shift=None, ci=None, **kwargs):
        if level_shift is None: level_shift = self.level_shift
        if ci is None: ci = self.base.ci
        return PDFTLagPrec (nroots=self.nroots, nlag=self.nlag, ngorb=self.ngorb, Adiag=Adiag, 
            level_shift=level_shift, ci=ci, **kwargs)

class PDFTLagPrec (sacasscf.SACASLagPrec):
     MC-PDFT Lagrange gradient preconditioner. Nearly the same as the SACAS preconditioner except that SA-SA rotations are allowed
    but must be antisymmetric (z_JI = -z_IJ).  Therefore the CI part is slightly different. 

    def __init__(self, nroots=None, nlag=None, ngorb=None, Adiag=None, ci=None, level_shift=None, **kwargs):
        super().__init__(nroots=nroots,nlag=nlag,ngorb=ngorb,Adiag=Adiag,ci=ci,level_shift=level_shift,**kwargs)

    def ci_prec (self, x):
        xci = x[self.ngorb:].reshape (self.nroots, -1)
        # R_I|H I> (indices: I, det)
        Rx = self.Rci * xci
        # <J|R_I|H I> (indices: J, I)
        sa_ovlp = self.ci.conjugate () @ Rx.T
        # R_I|J> S(I)_JK^-1 <K|R_I|H I> (indices: I, det)
        Rx_sub = np.zeros_like (Rx)
        for iroot in range (self.nroots):
            Rx_sub[iroot] = self.Rci_sa[iroot,:,iroot] * sa_ovlp[iroot,iroot]
        return Rx - Rx_sub
'''




