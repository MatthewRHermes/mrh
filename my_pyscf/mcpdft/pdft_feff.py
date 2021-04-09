import time
import numpy as np
from scipy import linalg
from pyscf import lib
from pyscf.dft.gen_grid import BLKSIZE
from pyscf.dft.numint import _contract_rho
from pyscf.mcscf import mc1step
from pyscf.scf import hf
from mrh.my_pyscf.mcpdft.otpd import *
from mrh.my_pyscf.mcpdft.otpd import _grid_ao2mo
from mrh.my_pyscf.mcpdft.tfnal_derivs import contract_fot
from mrh.my_pyscf.mcpdft.pdft_veff import _contract_vot_ao, _contract_vot_rho

def _contract_rho_all (bra, ket):
    if bra.ndim == 2: bra = bra[None,:,:]
    if ket.ndim == 2: ket = ket[None,:,:]
    nderiv, ngrids, norb = bra.shape
    rho = np.empty ((nderiv, ngrids), dtype=bra.dtype)
    if norb==0:
        rho[:] = 0.0
        return rho
    rho[0] = _contract_rho (bra[0], ket[0])
    for ideriv in range (1,min(nderiv,4)):
        rho[ideriv]  = _contract_rho (bra[ideriv], ket[0])
        rho[ideriv] += _contract_rho (bra[0], ket[ideriv])
    if nderiv > 4: raise NotImplementedError (('ftGGA or Colle-Salvetti type '
        'functionals'))
    return rho

# PySCF's overall sign convention is
#   de = h.D - D.h
#   dD = x.D - D.x
# which is consistent with
#   |trial> = exp(k_pq E_pq)|0>
# where k_pq is an antihermitian matrix
# However, in various individual terms the two conventions in 
# h_op are simultaneously reversed:
#   de = D.h - h.D
#   dD = D.x - x.D
# Specifically, update_jk_in_ah has this return signature. So I have
# to return -hx.T specifically.
# I'm also really confused by factors of 2. It looks like PySCF does
# the full commutator
#   dD = x.D - D.x
# in update_jk_in_ah, but only does 
#   dD = x.D
# everywhere else? Why would the update_jk_in_ah terms be
# multiplied by 2?

class EotOrbitalHessianOperator (object):
    ''' Callable object for computing
        (f.x)_pq = (int fot * drho/dk_pq drho/dk_rs x_rs dr)
        where fot is the second functional derivative of an on-top energy
        functional, and "rho" is any of the density, the on-top pair density,
        or their derivatives, in the context of an MC-PDFT calculation using
        the ncore, ncas, mo_coeff.

        Does not compute any contribution of the type
        (int vot * d^2 rho / dk_pq dk_rs x_rs dr)
        except optionally for that related to the Pi = (1/4) rho^2 cumulant
        approximation. All other vot terms are more efficiently computed using
        cached effective Hamiltonian tensors that map straightforwardly to
        those involved in CASSCF orbital optimization.
    '''
    
    def __init__(self, mc, ot=None, mo_coeff=None, ncore=None, ncas=None,
            casdm1=None, casdm2=None, max_memory=None, do_cumulant=True,
            incl_d2rho=False):
        if ot is None: ot = mc.otfnal
        if mo_coeff is None: mo_coeff = mc.mo_coeff
        if ncore is None: ncore = mc.ncore
        if ncas is None: ncas = mc.ncas
        if max_memory is None: max_memory = mc.max_memory
        if (casdm1 is None) or (casdm2 is None):
            dm1, dm2 = mc.fcisolver.make_rdm12 (mc.ci, ncas, mc.nelecas)
            if casdm1 is None: casdm1 = dm1
            if casdm2 is None: casdm2 = dm2

        self.ot = ot
        self.verbose, self.stdout = ot.verbose, ot.stdout
        self.log = lib.logger.new_logger (self, self.verbose)
        self.ni, self.xctype = ni, xctype = ot._numint, ot.xctype
        self.rho_deriv, self.Pi_deriv = ot.dens_deriv, ot.Pi_deriv
        deriv = ot.dens_deriv
        self.nderiv_rho = (1,4,10)[int (ot.dens_deriv)]
        self.nderiv_Pi = (1,4)[int (ot.Pi_deriv)]
        self.nderiv_ao = (deriv+1)*(deriv+2)*(deriv+3)//6
        self.mo_coeff = mo_coeff
        self.nao, self.nmo = nao, nmo = mo_coeff.shape
        self.ncore = ncore
        self.ncas = ncas
        self.nocc = nocc = ncore + ncas
        self.casdm2 = casdm2
        self.casdm1s = casdm1s = np.stack ([casdm1, casdm1], axis=0)/2
        self.cascm2 = cascm2 = get_2CDM_from_2RDM (casdm2, casdm1)
        self.max_memory = max_memory        
        self.do_cumulant = do_cumulant
        self.incl_d2rho = incl_d2rho

        dm1 = 2 * np.eye (nocc, dtype=casdm1.dtype)
        dm1[ncore:,ncore:] = casdm1
        occ_coeff = mo_coeff[:,:nocc]
        no_occ, uno = linalg.eigh (dm1)
        no_coeff = occ_coeff @ uno
        dm1 = occ_coeff @ dm1 @ occ_coeff.conj ().T
        self.dm1 = dm1 = lib.tag_array (dm1, mo_coeff=no_coeff, mo_occ=no_occ)

        self.make_rho = ni._gen_rho_evaluator (ot.mol, dm1, 1)[0]
        self.pack_uniq_var = mc.pack_uniq_var
        self.unpack_uniq_var = mc.unpack_uniq_var        

        if incl_d2rho: # Include d^2rho/dk^2 type derivatives
            # Also include a full E_OT value and gradient recalculator
            # for debugging purposes
            self.veff1, self.veff2 = mc.get_pdft_veff (mo=mo_coeff,
                casdm1s=casdm1s, casdm2=casdm2, incl_coul=False)
            get_hcore = lambda * args: self.veff1
            with lib.temporary_env (mc, get_hcore=get_hcore):
                g_orb, _, h_op, h_diag = mc1step.gen_g_hop (mc, mo_coeff,
                    np.eye (nmo), casdm1, casdm2, self.veff2)
            # dressed gen_g_hop objects
            from mrh.my_pyscf.mcpdft.orb_scf import get_gorb_update
            gorb_update_u = get_gorb_update (mc, mo_coeff, ncore=ncore,
                ncas=ncas, eot_only=True)
            def delta_gorb (x):
                u = mc.update_rotate_matrix (x)
                g1 = gorb_update_u (u, mc.ci)
                return g1 - g_orb
            jk_null = np.zeros ((ncas,nmo)), np.zeros ((ncore,nmo-ncore))
            update_jk = lambda * args: jk_null
            def d2rho_h_op (x):
                with lib.temporary_env (mc, update_jk_in_ah=update_jk):
                    return h_op (x)
            self.g_orb = g_orb
            self.delta_gorb = delta_gorb
            self.d2rho_h_op = d2rho_h_op
            self.h_diag = h_diag
            # on-top energy and calculator
            mo_cas = mo_coeff[:,ncore:nocc]
            dm1s = np.dot (mo_cas, casdm1s).transpose (1,0,2)
            dm1s = np.dot (dm1s, mo_cas.conj ().T)
            mo_core = mo_coeff[:,:ncore]
            dm1s += (mo_core @ mo_core.conj ().T)[None,:,:]
            dm_list = (dm1s, (casdm1s, (cascm2, None, None)))
            e_ot = mc.energy_dft (ot=ot, mo_coeff=mo_coeff, dm_list=dm_list)
            def delta_eot (x):
                u = mc.update_rotate_matrix (x)
                mo1 = mo_coeff @ u
                mo_cas = mo1[:,ncore:nocc]
                dm1s = np.dot (mo_cas, casdm1s).transpose (1,0,2)
                dm1s = np.dot (dm1s, mo_cas.conj ().T)
                mo_core = mo1[:,:ncore]
                dm1s += (mo_core @ mo_core.conj ().T)[None,:,:]
                dm_list = (dm1s, (casdm1s, (cascm2, None, None)))
                e1 = mc.energy_dft (ot=ot, mo_coeff=mo1, dm_list=dm_list)
                return e1 - e_ot
            self.e_ot = e_ot
            self.delta_eot = delta_eot


    def get_blocksize (self):
        nderiv_ao, nao = self.nderiv_ao, self.nao
        nderiv_rho, nderiv_Pi = self.nderiv_rho, self.nderiv_Pi
        ncas, nocc = self.ncas, self.nocc
        # Ignore everything that doesn't scale with the size of the molecule
        # or the active space
        nvar = 2 + int (self.rho_deriv) + 2*int (self.Pi_deriv)
        ncol = (nderiv_ao*(2*nao+ncas)        # ao + copy + mo_cas
             + (2+nderiv_Pi)*(ncas**2)        # tensor-product intermediate
             + nocc*(2*nderiv_rho+nderiv_Pi)) # drho_a, drho_b, dPi
        ncol *= 1.1 # fudge factor
        ngrids = self.ot.grids.coords.shape[0]
        remaining_floats = (self.max_memory-lib.current_memory()[0]) * 1e6 / 8
        blksize = int (remaining_floats/(ncol*BLKSIZE))*BLKSIZE
        return max(BLKSIZE,min(blksize,ngrids,BLKSIZE*1200))

    def make_dens0 (self, ao, mask, make_rho=None, casdm1s=None, cascm2=None,
            mo_cas=None):
        if make_rho is None: make_rho = self.make_rho
        if casdm1s is None: casdm1s = self.casdm1s
        if cascm2 is None: cascm2 = self.cascm2
        if mo_cas is None: mo_cas = self.mo_coeff[:,self.ncore:self.nocc]
        if ao.shape[0] == 1 and ao.ndim == 3: ao = ao[0]
        rho = make_rho (0, ao, mask, self.xctype)
        if ao.ndim == 2: ao = ao[None,:,:]
        rhos = np.stack ([rho, rho], axis=0)/2
        Pi = get_ontop_pair_density (self.ot, rhos, ao, casdm1s, cascm2,
            mo_cas, deriv=self.Pi_deriv, non0tab=mask)
        return rho, Pi
        # volatile memory footprint:
        #   nderiv_ao * nao * ngrids            (copying ao in make_rho)
        # + nderiv_ao * ncas * ngrids           (mo_cas on a grid)
        # + (2+nderiv_Pi) * ncas**2 * ngrids    (tensor-product intermediates)

    def make_ddens (self, ao, rho, mask):
        occ_coeff = self.mo_coeff[:,:self.nocc]
        rhos = np.stack ([rho, rho], axis=0)/2
        mo = _grid_ao2mo (self.ot.mol, ao[:self.nderiv_rho], occ_coeff,
            non0tab=mask)
        drhos, dPi = density_orbital_derivative (self.ot, self.ncore, self.ncas,
            self.casdm1s, self.cascm2, rhos, mo, non0tab=mask)
        return drhos.sum (0), dPi
        # persistent memory footprint:
        #   nderiv_rho * nocc * ngrids          (drho)
        # + nderiv_Pi * nocc * ngrids           (dPi)
        # volatile memory footprint:
        #   nderiv_rho * 2 * ngrids             (copying rho)
        # + 2 * nderiv_rho * nocc * ngrids      (mo & copied drho)
        # + (2+nderiv_Pi) * ncas**2 * ngrids    (tensor-product intermediates)

    def make_dens1 (self, ao, drho, dPi, mask, x):
        # In mc1step.update_jk_in_ah:
        #   hx = ddm1 @ h - h @ ddm1
        #   ddm1 = dm1 @ x - x @ dm1
        # the PROPER convention for consistent sign is
        #   hx = h @ ddm1 - ddm1 @ h
        #   ddm1 = x @ dm1 - dm1 @ x
        # The dm1 index is hidden in drho and dPi
        # Therefore,
        # 1) the SECOND index of x contracts with drho
        # 2) we MULTIPLY BY TWO to to account for + transpose
        
        ngrids = drho.shape[-1]
        ncore, nocc = self.ncore, self.nocc
        occ_coeff_1 = self.mo_coeff @ x[:,:self.nocc] * 2
        mo1 = _grid_ao2mo (self.ot.mol, ao, occ_coeff_1, non0tab=mask)
        Pi1 = _contract_rho_all (mo1[:self.nderiv_Pi], dPi)
        mo1 = mo1[:self.nderiv_rho]
        drho_c, mo1_c = drho[:,:,:ncore],     mo1[:,:,:ncore]
        drho_a, mo1_a = drho[:,:,ncore:nocc], mo1[:,:,ncore:nocc]
        rho1_c = _contract_rho_all (mo1_c, drho_c)
        rho1_a = _contract_rho_all (mo1_a, drho_a)
        return rho1_c, rho1_a, Pi1

    def debug_dens1 (self, ao, mask, x, weights, rho0, Pi0, rho1_test, Pi1_test):
        # This requires the full-space 2RDM
        ncore, nocc, nmo = self.ncore, self.nocc, self.nmo
        casdm1 = self.casdm1s.sum (0)
        dm1 = 2 * np.eye (nmo, dtype=self.dm1.dtype)
        dm1[ncore:nocc,ncore:nocc] = casdm1
        dm1[nocc:,nocc:] = 0
        dm2 = np.multiply.outer (dm1, dm1)
        dm2 -= dm2.transpose (0,3,2,1)/2
        dm2[ncore:nocc,ncore:nocc,ncore:nocc,ncore:nocc] += self.cascm2
        dm1 = x @ dm1 - dm1 @ x
        dm2 = np.dot (dm2, x.T)
        dm2 += dm2.transpose (1,0,3,2)
        dm2 += dm2.transpose (2,3,0,1)
        cm2 = get_2CDM_from_2RDM (dm2, dm1)
        dm1s = dm1/2
        dm1s = np.stack ([dm1s, dm1s], axis=0)
        dm1_ao = self.mo_coeff @ dm1 @ self.mo_coeff.conj ().T
        make_rho = self.ni._gen_rho_evaluator (self.ot.mol, dm1_ao, 1)[0]
        rho1_ref, Pi1_ref = self.make_dens0 (ao, mask, make_rho=make_rho, casdm1s=dm1s,
            cascm2=cm2, mo_cas=self.mo_coeff)
        if rho0.ndim == 1: rho0 = rho0[None,:]
        if Pi0.ndim == 1: Pi0 = Pi0[None,:]
        if rho1_ref.ndim == 1: rho1_ref = rho1_ref[None,:]
        if Pi1_ref.ndim == 1: Pi1_ref = Pi1_ref[None,:]
        nderiv_Pi = self.nderiv_Pi
        Pi0 = Pi0[:nderiv_Pi]
        Pi1_test = Pi1_test[:nderiv_Pi]
        Pi1_ref = Pi1_ref[:nderiv_Pi]
        rho1_err = linalg.norm (rho1_test - rho1_ref)
        Pi1_err = linalg.norm (Pi1_test - Pi1_ref)
        x_norm = linalg.norm (x)
        self.log.debug ("shifted dens: |x|, |rho1_err|, |Pi1_err| = %e, %e, %e",
            x_norm, rho1_err, Pi1_err)
        return rho1_err, Pi1_err

    def get_fot (self, rho, Pi, weights):
        rho = np.stack ([rho,rho], axis=0)/2
        eot, vot, fot = self.ot.eval_ot (rho, Pi, dderiv=2, weights=weights)
        return vot, fot

    def get_vot (self, rho, Pi, weights):
        rho = np.stack ([rho,rho], axis=0)/2
        eot, vot, _ = self.ot.eval_ot (rho, Pi, dderiv=1, weights=weights)
        vrho, vPi = vot
        return vrho, vPi

    def debug_fot (self, x, vot, fot, rho0, Pi0, rho1, Pi1, weights, mask):
        log = self.log
        x_norm = linalg.norm (x)
        if rho0.ndim == 1: rho0 = rho0[None,:]
        if Pi0.ndim == 1: Pi0 = Pi0[None,:]
        Pi0 = Pi0[:self.nderiv_Pi,:]
        def gen_cols ():
            rho1_col = np.zeros_like (rho1)
            Pi1_col = np.zeros_like (Pi1)
            rho1_col[0,:] = rho1[0].copy ()
            yield rho1_col, Pi1_col
            rho1_col[:] = 0.0
            Pi1_col[0,:] = Pi1[0,:].copy ()
            yield rho1_col, Pi1_col
            if not self.rho_deriv: return
            Pi1_col[0,:] = 0.0
            rho1_col[1:4,:] = rho1[1:4].copy ()
            yield rho1_col, Pi1_col
            if not self.Pi_deriv: return
            rho1_col[1:4,:] = 0.0
            Pi1_col[1:4,:] = Pi1[1:4,:].copy ()
        def gen_rows (fxrho, fxPi, dvrho, dvPi):
            yield fxrho[0,:], dvrho[0,:]
            yield fxPi[0,:], dvPi[0,:]
            if not self.rho_deriv: return
            yield fxrho[1:4,:], dvrho[1:4,:]
            if not self.Pi_deriv: return
            yield fxPi[1:4,:], dvPi[1:4,:]

        for icol, (rho1_col, Pi1_col) in enumerate (gen_cols ()):
            xlbl = 'x_' + ('rho','Pi','srr')[icol]
            fxrho, fxPi = contract_fot (self.ot, fot, rho0, Pi0,
                rho1_col, Pi1_col)
            dvrho, dvPi = self.get_vot (rho0+rho1_col, Pi0+Pi1_col, weights)
            dvrho -= vot[0]
            dvPi -= vot[1]
            idx = rho0[0] < 2e-10 # TODO: understand why I have to do this
            fxrho[:,idx] = 0.0    # Some ftLDA (f.x)_Pi elements specifically
            fxPi[:,idx] = 0.0     # are very large while the corresponding 
            dvrho[:,idx] = 0.0    # v1-v0 elements are identically zero, unless
            dvPi[:,idx] = 0.0     # I do this. This is despite the fact that
            # the orbital Hessian-vector product of the ftLDA on-top energy
            # appears fine even if I don't do this. I checked and it has
            # NOTHING to do with the mask index.
            fxrho *= weights[None,:]
            fxPi *= weights[None,:]
            dvrho *= weights[None,:]
            dvPi *= weights[None,:]

            for irow, (fx, dv) in enumerate (gen_rows (fxrho, fxPi, dvrho,
                    dvPi)):
                lbl = 'f(' + xlbl + ')_' + ('rho', 'Pi', 'rhop')[irow]
                log.debug (('test '+lbl+': |x| = %e, |fx-dv|/|dv| = %e'),
                    x_norm, linalg.norm (fx-dv)/linalg.norm(dv))

    def get_fxot (self, ao, rho0, Pi0, drho, dPi, x, weights, mask):
        vot, fot = self.get_fot (rho0, Pi0, weights)
        rho1_c, rho1_a, Pi1 = self.make_dens1 (ao, drho, dPi, mask, x)
        rho1 = rho1_c + rho1_a
        if self.verbose >= lib.logger.DEBUG: # TODO: require higher verbosity
            self.debug_dens1 (ao, mask, x, weights, rho0, Pi0, rho1, Pi1)
            self.debug_fot (x, vot, fot, rho0, Pi0, rho1, Pi1, weights, mask)
        fxrho, fxPi = contract_fot (self.ot, fot, rho0, Pi0, rho1, Pi1)
        vrho, vPi = vot
        de = (np.dot (rho1.ravel (), (vrho * weights[None,:]).ravel ())
            + np.dot (Pi1.ravel (), (vPi * weights[None,:]).ravel ()))
        Pi1 = fxrho_a = rho1 = vrho = None
        if self.do_cumulant and self.ncore: # fxrho gets the D_all D_c part
                                            # D_c D_a part has to be separate
            if vPi.ndim == 1: vPi = vPi[None,:]
            fxrho =   _contract_vot_rho (vPi, rho1_c, add_vrho=fxrho)
            fxrho_a = _contract_vot_rho (vPi, rho1_a)
        return de, fxrho, fxPi, fxrho_a

    def contract_v_ddens (self, v, ddens, ao, weights):
        vw = v * weights[None,:]
        vao = _contract_vot_ao (vw, ao)
        return np.tensordot (vao, ddens, axes=((0,1),(0,1)))

    def __call__ (self, x, packed=False):
        if self.incl_d2rho:
            packed = True
            dg_d2rho = self.d2rho_h_op (x)
        if packed: 
            x_packed = x.copy ()
            x = self.unpack_uniq_var (x)
        else:
            x_packed = self.pack_uniq_var (x)
        dg = np.zeros ((self.nocc, self.nao), dtype=x.dtype)
        de = 0
        for ao, mask, weights, coords in self.ni.block_loop (self.ot.mol,
                self.ot.grids, self.nao, self.rho_deriv, self.max_memory,
                blksize=self.get_blocksize ()):
            rho0, Pi0 = self.make_dens0 (ao, mask)
            if ao.ndim == 2: ao = ao[None,:,:]
            drho, dPi = self.make_ddens (ao, rho0, mask)
            dde, fxrho, fxPi, fxrho_a = self.get_fxot (ao, rho0, Pi0, drho, dPi,
                x, weights, mask)
            de += dde
            dg -= self.contract_v_ddens (fxrho, drho, ao, weights).T
            dg -= self.contract_v_ddens (fxPi, dPi, ao, weights).T
            # Transpose because update_jk_in_ah requires this shape
            # Minus because I want to use 1 consistent sign rule here
            if self.do_cumulant and self.ncore: # The D_c D_a part
                drho_c = drho[:,:,:self.ncore]
                dg[:self.ncore] -= self.contract_v_ddens (fxrho_a, drho_c,
                    ao, weights).T
        if self.incl_d2rho:
            de_test = 2 * np.dot (x_packed, self.g_orb)
            # The factor of 2 is because g_orb is evaluated in terms of square
            # antihermitian arrays, but only the lower-triangular parts are
            # stored in x and g_orb.
            self.log.debug (('E from integration: %e; from stored grad: %e; '
                'diff: %e'), de, de_test, de-de_test)
        dg = np.dot (dg, self.mo_coeff) 
        if packed:
            dg_full = np.zeros ((self.nmo, self.nmo), dtype=dg.dtype)
            dg_full[:self.nocc,:] = dg[:,:]
            dg_full -= dg_full.T
            dg = self.pack_uniq_var (dg_full)
            if self.incl_d2rho: dg += dg_d2rho 
        return dg, de

    def e_de_full (self, x):
        # Compare me to __call__ for debugging purposes
        assert (self.incl_d2rho)
        return self.delta_gorb (x), self.delta_eot (x)

class ExcOrbitalHessianOperator (object):
    ''' for comparison '''

    def __init__(self, ks, mo_coeff=None, mo_occ=None):
        if mo_coeff is None: mo_coeff = ks.mo_coeff
        if mo_occ is None: mo_occ = ks.mo_occ
        self.nao, self.nmo = nao, nmo = mo_coeff.shape[-2:]
        self.ks = ks = ks.newton ()
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ

        dm = ks.make_rdm1 (mo_coeff=mo_coeff, mo_occ=mo_occ)
        vxc = ks.get_veff (dm=dm)
        self.exc = exc = vxc.exc
        vxc -= vxc.vj
        self.vxc = vxc
        def no_j (*args, **kwargs): return 0
        def no_jk (*args, **kwargs): return 0, 0
        with lib.temporary_env (ks, get_j=no_j, get_jk=no_jk):
            g_orb, h_op, h_diag = ks.gen_g_hop (mo_coeff, mo_occ, fock_ao=vxc)
        self.g_orb = g_orb
        self.h_op = h_op
        self.h_diag = h_diag

    def __call__(self, x):
        ''' return dg, de; always packed '''
        def no_j (*args, **kwargs): return 0
        def no_jk (*args, **kwargs): return 0, 0
        with lib.temporary_env (self.ks, get_j=no_j, get_jk=no_jk):
            dg = self.h_op (x)
        de = 2*np.dot (self.g_orb.ravel (), x.ravel ())
        return dg, de

    def e_de_full (self, x):
        ks = self.ks
        mo_occ = self.mo_occ
        u = self.ks.update_rotate_matrix (x, mo_occ)
        mo1 = ks.rotate_mo (self.mo_coeff, u)
        dm1 = self.ks.make_rdm1 (mo_coeff=mo1, mo_occ=mo_occ)
        vxc1 = self.ks.get_veff (dm=dm1)
        exc1 = vxc1.exc
        vxc1 -= vxc1.vj
        de = exc1 - self.exc
        g1 = ks.gen_g_hop (mo1, mo_occ, fock_ao=vxc1)[0]
        dg = g1 - self.g_orb 
        return dg, de

    def pack_uniq_var (self, x):
        return hf.pack_uniq_var (x, self.mo_occ)

    def unpack_uniq_var (self, x):
        return hf.unpack_uniq_var (x, self.mo_occ)

if __name__ == '__main__':
    from pyscf import gto, scf, dft
    from mrh.my_pyscf import mcpdft
    from functools import partial
    mol = gto.M (atom = 'H 0 0 0; H 1.2 0 0', basis = '6-31g',
        verbose=lib.logger.DEBUG, output='pdft_feff.log')
    mf = scf.RHF (mol).run ()
    def debug_hess (hop):
        print ("g_orb:", linalg.norm (hop.g_orb))
        print ("h_diag:", linalg.norm (hop.h_diag))
        x0 = -hop.g_orb / hop.h_diag
        conv_tab = np.zeros ((8,3), dtype=x0.dtype)
        print ("x0 = g_orb/h_diag:", linalg.norm (x0))
        print (" n " + ' '.join (['{:>10s}',]*6).format ('de_test','de_ref',
            'de_relerr','dg_test','dg_ref','dg_relerr'))
        for p in range (10):
            fac = 1/(2**p)
            x1 = x0 * fac
            if p==10: x1[:] = 0
            dg_test, de_test = hop (x1)
            dg_ref,  de_ref  = hop.e_de_full (x1)
            e_err = (de_test-de_ref)/de_ref
            idx = np.argmax (np.abs (dg_test-dg_ref))
            dg_test_max = dg_test[idx]
            dg_ref_max = dg_ref[idx]
            g_err = (dg_test_max-dg_ref_max)/dg_ref_max
            row = [p, de_test, de_ref, e_err, dg_test_max, dg_ref_max, g_err]
            print (("{:2d} " + ' '.join (['{:10.3e}',]*6)).format (*row))
        dg_err = dg_test - dg_ref
        denom = dg_ref.copy ()
        denom[np.abs(dg_ref)<1e-8] = 1.0
        dg_err /= denom
        fmt_str = ' '.join (['{:10.3e}',]*hop.nmo)
        print ("dg_test:")
        for row in hop.unpack_uniq_var (dg_test): print (fmt_str.format (*row))
        print ("dg_ref:")
        for row in hop.unpack_uniq_var (dg_ref): print (fmt_str.format (*row))
        fmt_str = ' '.join (['{:6.2f}',]*hop.nmo)
        print ("dg_err (relative):")
        for row in hop.unpack_uniq_var (dg_err): print (fmt_str.format (*row))
        print ("")
    for nelecas, lbl in zip ((2, (2,0)), ('Singlet','Triplet')):
        print (lbl,'case\n')
        #for fnal in 'LDA,VWN3', 'PBE':
        #    ks = dft.RKS (mol).set (xc=fnal).run ()
        #    print ("H2 {} energy:".format (fnal),ks.e_tot)
        #    exc_hop = ExcOrbitalHessianOperator (ks)
        #    debug_hess (exc_hop)
        #for fnal in 'tLDA,VWN3', 'ftLDA,VWN3', 'tPBE':
        for fnal in 'ftLDA,VWN3', 'tPBE':
            mc = mcpdft.CASSCF (mf, fnal, 2, nelecas, grids_level=1).run ()
            print ("H2 {} energy:".format (fnal),mc.e_tot)
            eot_hop = EotOrbitalHessianOperator (mc, incl_d2rho=True)
            debug_hess (eot_hop)
        print ("")
        assert (False)
