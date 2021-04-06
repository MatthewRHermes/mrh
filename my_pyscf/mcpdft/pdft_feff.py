import numpy as np
from scipy import linalg
from pyscf import lib
from pyscf.dft.gen_grid import BLKSIZE
from mrh.my_pyscf.mcpdft.otpd import *
from mrh.my_pyscf.mcpdft.tfnal_derivs import contract_fot
from mrh.my_pyscf.mcpdft.pdft_veff import _contract_vot_ao

def EotOrbitalHessianOperator (object):
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
    
    def __init__(ot, mo_coeff, ncore, ncas, casdm1, casdm2, max_memory,
            do_cumulant=True):
        self.ot = ot
        self.verbose, self.stdout = ot.verbose, ot.stdout
        self.ni, self.xctype = ni, xctype = ot.ni, ot.xctype
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
        self.cascm2 = get_2CDM_from_2RDM (casdm2, casdm1)
        self.max_memory = max_memory        
        self.do_cumulant = do_cumulant

        dm1 = 2 * np.eye (nocc, dtype=casdm1.dtype)
        dm1[ncore:,ncore:] = casdm1
        occ_coeff = mo_coeff[:,:nocc]
        no_occ, uno = linalg.eigh (dm1)
        no_coeff = occ_coeff @ uno
        dm1 = occ_coeff @ dm1 @ occ_coeff.conj ().T
        self.dm1 = dm1 = lib.tag_array (dm1, mo_coeff=no_coeff, mo_occ=no_occ)

        self.make_rho = ni._gen_rho_evaluator (ot.mol, dm1, 1)
        
    def get_blocksize (self):
        nderiv_ao, nao = self.nderiv_ao, self.nao
        nderiv_rho, nderiv_Pi = self.nderiv_rho, self.nderiv_Pi
        ncas, nocc = self.ncas, self.nocc
        nvar = 2 + int (self.rho_deriv) + 2*int (self.Pi_deriv)
        ncol_perm = (4 + nderiv_ao*nao                    # ao, weights, coords
            + (nocc+2) * (nderiv_rho+nderiv_Pi))          # rho + fx + drho
        ncol_vol = (nderiv_ao*(nao+ncas) + 2*nderiv_rho   # rho volatile
            + (1+nderiv_Pi)*(ncas**2))
        ncol_vol = max (ncol_vol,                         # drho volatile
            (2*nderiv_rho*(nocc+1) 
            + (1+nderiv_Pi)*(ncas**2)))
        ncol_vol = max (ncol_vol,                         # fx volatile
            (nvar*(nvar+1)//2 + nderiv_rho + nderiv_Pi
            + 2*nderiv_ao*nocc))
        ncol = ncol_perm + ncol_vol
        ngrids = self.ot.grids.coords.shape[0]
        remaining_floats = (self.max_memory-lib.current_memory())[0] * 1e6 / 8
        blksize = int (remaining_floats/(ncol*BLKSIZE))*BLKSIZE
        return max(BLKSIZE,min(blksize,ngrids,BLKSIZE*1200))

    def make_dens0 (self, ao, mask):
        rho = self.make_rho (0, ao, mask, self.xctype)
        if ao.ndim == 2: ao = ao[None,:,:]
        rhos = np.stack ([rho, rho], axis=0)/2
        Pi = get_ontop_pair_density (self.ot, rhos, ao, self.casdm1s,
            self.cascm2, self.mo_coeff[:,self.ncore:self.nocc],
            deriv=self.Pi_deriv, non0tab=mask)
        return rho, Pi
        # persistent memory footprint: (nderiv_rho + nderiv_Pi) * ngrids
        # volatile memory footprint:
        #   nderiv_ao * nao * ngrids            (copying ao)
        # + 2 * nderiv_rho * ngrids             (copying rho)
        # + nderiv_ao * ncas * ngrids           (mo_cas on a grid)
        # + (1+nderiv_Pi) * ncas**2 * ngrids    (tensor-product intermediates)

    def make_ddens (self, ao, rho, mask):
        occ_coeff = self.mo_coeff[:,:self.nocc]
        rhos = np.stack ([rho, rho], axis=0)/2
        mo = _grid_ao2mo (self.ot.mol, ao[:self.rho_deriv], occ_coeff,
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
        # + (1+nderiv_Pi) * ncas**2 * ngrids    (tensor-product intermediates)

    def make_dens1 (self, ao, drho, dPi, mask, x):
        # the SECOND index of x touches the DENSITY MATRIX
        ngrids = drho.shape[-1]
        occ_coeff_1 = self.mo_coeff @ (x-x.T)[:,:self.nocc]
        mo1 = _grid_ao2mo (self.ot.mol, ao, occ_coeff_1, non0tab=mask)
        rho1 = (mo1[:self.nderiv_rho] * drho[0:1,:,:]).sum (1)
        Pi1 = (mo1[:self.nderiv_Pi] * dPi[0:1,:,:]).sum (1)
        if self.rho_deriv:
            rho1[1:4] += (mo1[0:1,:,:] * drho[1:4,:,:]).sum (1)
        if self.Pi_deriv:
            Pi1[1:4] += (mo1[0:1,:,:] * dPi[1:4,:,:]).sum (1)
        return rho1, Pi1
        # persistent memory footprint: (nderiv_rho + nderiv_Pi) * ngrids
        # volatile memory footprint: 2 * nderiv_ao * nocc * ngrids

    def get_fot (self, rho, Pi, weights):
        eot, vot, fot = self.ot.eval_ot (rho, Pi, dderiv=2, weights=weights, 
            _unpack_vot=False)
        if self.do_cumulant:
            fot[0] += 0.5*vot[2]
            # TODO: density-gradient terms in ftGGA
        return fot
        # persistent memory footprint: nvar * (nvar+1) * ngrids // 2,
        #   where nvar = 2 + rho_deriv + 2*Pi_deriv
        # volatile memory footprint: trivial

    def get_fxot (self, ao, rho0, Pi0, drho, dPi, x, weights, mask):
        fot = self.get_fot (rho0, Pi0, weights)
        rho1, Pi1 = self.make_dens1 (ao, drho, dPi, mask, x)
        return contract_fot (fot, rho0, Pi0, rho1, Pi1)
        # persistent memory footprint: (nderiv_rho + nderiv_Pi) * ngrids
        # volatile memory footprint:
        #   get_fot persistent
        # + make_dens1 persistent
        # + make_dens1 volatile

    def __call__ (self, x):
        dg = np.zeros ((self.nocc, self.nao), dtype=x.dtype)
        for ao, mask, weights, coords in self.ni.block_loop (self.ot.mol,
                self.ot.grids, self.nao, self.rho_deriv, self.max_memory,
                blksize=self.get_blocksize ()):
            rho0, Pi0 = self.make_dens0 (ao, mask)
            if ao.ndim == 2: ao = ao[None,:,:]
            drho, dPi = self.make_ddens (ao, rho0, mask)
            fxrho, fxPi = self.get_fxot (ao, rho0, Pi0, drho, dPi,
                x, weights, mask)
            drho = _contract_vot_ao (fxrho, drho)
            fxrho = None
            dPi = _contract_vot_ao (fxPi, dPi)
            fxPi = None
            dg += np.tensordot (drho, ao[:self.nderiv_rho], 
                axes=((0,1),(0,1)))
            drho = None
            dg += np.tensordot (dPi, ao[:self.nderiv_Pi],
                axes=((0,1),(0,1)))
            dPi = None
        dg = np.dot (dg, self.mo_coeff)
        return dg

