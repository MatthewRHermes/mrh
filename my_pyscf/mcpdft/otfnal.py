import numpy as np
import copy
from pyscf.lib import logger
from pyscf.dft.numint import _NumInt
from mrh.util import params

class otfnal:

    def __init__ (self, mol, **kwargs):
        self.mol = mol

    def get_E_ot (self, rho, Pi, weight):
        r''' get the on-top energy

        Args:
            rho : ndarray of shape (2,*,ngrids)
                containing spin-density [and derivatives]
            Pi : ndarray with shape (*,ngrids)
                containing on-top pair density [and derivatives]
            weight : ndarray of shape (ngrids)
                containing numerical integration weights

        Returns : float
            The on-top exchange-correlation energy for the given on-top xc functional
        '''

        raise RuntimeError("on-top xc functional not defined")
        return 0


class transfnal (otfnal):
    r''' "translated functional" of Li Manni et al., JCTC 10, 3669 (2014).
    '''

    def __init__ (self, ks, **kwargs):
        super().__init__(ks.mol, **kwargs)
        self.otxc = 't' + ks.xc
        self._numint = copy.copy (ks._numint)
        self.grids = copy.copy (ks.grids)
        self.verbose = ks.verbose
        self.stdout = ks.stdout
        self._numint.hybrid_coeff = t_hybrid_coeff.__get__(self._numint)
        self._numint.nlc_coeff = t_nlc_coeff.__get__(self._numint)
        self._numint.rsh_coeff = t_rsh_coeff.__get__(self._numint)
        self._numint.eval_xc = t_eval_xc.__get__(self._numint)
        self._numint._xc_type = t_xc_type.__get__(self._numint)
        #self._numint.rsh_and_hybrid_coeff = t_rsh_and_hybrid_coeff.__get__(self._numint)
        #self.xctype = self._numint._xc_type (self.otxc)
        #self.xc_deriv = ['LDA', 'GGA', 'MGGA'].index (self.xctype)
        logger.info (self, 'Building %s functional', self.otxc)
        omega, alpha, hyb = self._numint.rsh_and_hybrid_coeff(self.otxc, spin=self.mol.spin)
        if hyb > 0:
            logger.info (self, 'Hybrid functional with %s CASSCF exchange', hyb)

    def get_E_ot (self, rho, Pi, weight):
        r''' E_ot[rho, Pi] = V_xc[rho_translated] 
    
            Args:
                rho : ndarray of shape (2,*,ngrids)
                    containing spin-density [and derivatives]
                Pi : ndarray with shape (*,ngrids)
                    containing on-top pair density [and derivatives]
                weight : ndarray of shape (ngrids)
                    containing numerical integration weights
    
            Returns : float
                The on-top exchange-correlation energy, for an on-top xc functional
                which uses a translated density with an otherwise standard xc functional
        '''
        assert (rho.shape[1:] == Pi.shape[:]), "rho.shape={0}, Pi.shape={1}".format (rho.shape, Pi.shape)
        if rho.ndim == 2:
            rho = np.expand_dims (rho, 1)
            Pi = np.expand_dims (Pi, 0)
            
        rho_t = self.get_rho_translated (Pi, rho)
        dexc_ddens = self._numint.eval_xc (self.otxc, (rho_t[0,:,:], rho_t[1,:,:]), spin=1, relativity=0, deriv=0, verbose=self.verbose)[0]
        dens = rho_t[0,0,:] + rho_t[1,0,:]
 
        rho = np.squeeze (rho)
        Pi = np.squeeze (Pi)

        # E_ot[rho,Pi] = \int {dE_ot/ddens}(r) * dens(r) dr
        #              = \sum_i {dE_ot/ddens}_i * dens_i * weight_i
        dens *= weight
        E_ot = np.sum (dexc_ddens * dens)
        logger.debug (self, 'Total number of electrons in (this chunk of) the translated density = %s', np.sum (dens))

        return E_ot

    def get_ratio (self, Pi, rho_avg):
        r''' R = Pi / [rho/2]^2 = Pi / rho_avg^2
            An intermediate quantity when computing the translated spin densities

            Note this function returns 1 for values and 0 for derivatives for every point where the charge density is close to zero (i.e., convention: 0/0 = 1)
        '''
        assert (Pi.shape == rho_avg.shape)
        nderiv = Pi.shape[0]
        if nderiv > 4:
            raise NotImplementedError("derivatives above order 1")

        R = np.zeros_like (Pi)  
        R[0,:] = 1
        idx = rho_avg[0] >= (1e-15 / 2)
        # Chain rule!
        for ideriv in range (nderiv):
            R[ideriv,idx] = Pi[ideriv,idx] / rho_avg[0,idx] / rho_avg[0,idx]
        # Product rule!
        for ideriv in range (1,nderiv):
            R[ideriv,idx] -= 2 * rho_avg[ideriv,idx] * R[0,idx] / rho_avg[0,idx]
        return R

    def get_rho_translated (self, Pi, rho, Rmax=1, zeta_deriv=False):
        r''' original translation, Li Manni et al., JCTC 10, 3669 (2014).
        rho_t[0] = {(rho[0] + rho[1]) / 2} * (1 + zeta)
        rho_t[1] = {(rho[0] + rho[1]) / 2} * (1 - zeta) 
    
        where
    
        zeta = (1-ratio)^(1/2) ; ratio < 1
             = 0               ; otherwise
        with
        ratio = Pi / [{(rho[0] + rho[1]) / 2}^2]
    
            Args:
                Pi : ndarray of shape (*, ngrids)
                    containing on-top pair density [and derivatives]
                rho : ndarray of shape (2, *, ngrids)
                    containing spin density [and derivatives]
    
            Kwargs:
                Rmax : float
                    cutoff for value of ratio in computing zeta; not inclusive
                zeta_deriv : logical
                    whether to include the derivative of zeta in the gradient of rho_t
    
            Returns: ndarray of shape (2,*,ngrids)
                containing translated spin density (and derivatives)
        '''
        assert (Rmax <= 1), "Don't set Rmax above 1.0!"
        nderiv = rho.shape[1]
        nderiv_zeta = nderiv if zeta_deriv else 1
    
        rho_avg = (rho[0,:,:] + rho[1,:,:]) / 2
        rho_t = rho.copy ()

        R = self.get_ratio (Pi[0:nderiv_zeta,:], rho_avg[0:nderiv_zeta,:])

        # For nonzero charge & pair density, set alpha dens = beta dens = 1/2 charge dens
        idx = (rho_avg[0] >= (1e-15 / 2)) & (Pi[0] >= 1e-15) 
        rho_t[0][:,idx] = rho_t[1][:,idx] = rho_avg[:,idx]

        # For 0 <= ratio < 1 and 0 <= rho, correct spin density using on-top density
        idx &= (Rmax > R[0])
        assert (np.all (R[0,idx] >= 0)), np.amin (R[0,idx])
        assert (np.all (R[0,idx] <= Rmax)), np.amax (R[0,idx])
        zeta = np.empty_like (R[:,idx])
        zeta[0] = np.sqrt (1.0 - R[0,idx])

        # Chain rule!
        for ideriv in range (1, nderiv_zeta):
            zeta[ideriv] = -R[ideriv,idx] / zeta[0] / 2
    
        # Chain rule!
        for ideriv in range (nderiv):
            w = rho_avg[ideriv,idx] * zeta[0]
            rho_t[0,ideriv,idx] += w
            rho_t[1,ideriv,idx] -= w
        # Product rule!
        for ideriv in range (1,nderiv_zeta):
            w = rho_avg[0,idx] * zeta[ideriv]
            rho_t[0,ideriv,idx] += w
            rho_t[1,ideriv,idx] -= w

        if self.verbose > logger.DEBUG:
            logger.debug (self, "R < R0")
            ft_continuity_debug (self, R[0,idx], rho_t[:,0,idx], zeta, 0.9, 1.15)
    
        return rho_t



class ftransfnal (transfnal):
    r''' "fully translated functional" of Carlson et al., JCTC 11, 4077 (2015)
    '''

    def __init__ (self, ks, **kwargs):
        super().__init__(ks, **kwargs)
        self.R0=0.9
        self.R1=1.15
        self.A=-475.60656009
        self.B=-379.47331922 
        self.C=-85.38149682

    def get_rho_translated (self, Pi, rho, Rmax=None, zeta_deriv=True):
        r''' "full" translation, Carlson et al., JCTC 11, 4077 (2015)
        rho_t[0] = {(rho[0] + rho[1]) / 2} * (1 + zeta)
        rho_t[1] = {(rho[0] + rho[1]) / 2} * (1 - zeta)
    
        where
        zeta = (1-ratio)^(1/2)                                  ; ratio < R0
           = A*(ratio-R1)^5 + B*(ratio-R1)^4 + C*(ratio-R1)^3 ; R0 <= ratio <= R1
           = 0                                                ; otherwise
    
        Propagate derivatives thru zeta
    
            Args:
                Pi : ndarray of shape (*, ngrids)
                    containing on-top pair density [and derivatives]
                rho : ndarray of shape (2, *, ngrids)
                    containing spin density [and derivatives]
    
            Returns: ndarray of shape (2,*,ngrids)
                containing fully-translated spin density (and derivatives)
    
        '''
        Rmax = Rmax or self.R1
        nderiv = rho.shape[1]
        if nderiv > 4:
            raise NotImplementedError("derivatives above order 1")
        R0, R1, A, B, C = self.R0, self.R1, self.A, self.B, self.C
    
        rho_ft = super().get_rho_translated (Pi, rho, Rmax=R0, zeta_deriv=True)
    
        rho_avg = (rho[0] + rho[1]) / 2
        R = self.get_ratio (Pi, rho_avg)
    
        idx = np.where (np.logical_and (R[0] >= self.R0, R[0] <= self.R1))[0]
        R_m_R1 = np.stack ([np.power (R[0,idx] - R1, n) for n in range (2,6)], axis=0)
        zeta = np.empty_like (R[:,idx])
        zeta[0] = (A*R_m_R1[5-2] + B*R_m_R1[4-2] + C*R_m_R1[3-2])
        # Chain rule!
        for ideriv in range (1, nderiv):
            zeta[ideriv] = R[ideriv,idx] * (5*A*R_m_R1[4-2] + 4*B*R_m_R1[3-2] + 3*C*R_m_R1[2-2])
    

        # Chain rule!
        for ideriv in range (nderiv):
            rho_ft[0,ideriv,idx] *= (1 + zeta[0])
            rho_ft[1,ideriv,idx] *= (1 - zeta[0])
        # Product rule!
        for ideriv in range (1,nderiv):
            rho_ft[0,ideriv,idx] += rho_avg[0,idx] * zeta[ideriv]
            rho_ft[1,ideriv,idx] -= rho_avg[0,idx] * zeta[ideriv]
    
        if self.verbose > logger.DEBUG:
            logger.debug (self, "R0 <= R < R1")
            ft_continuity_debug (self, R[0,idx], rho_ft[:,0,idx], zeta, R0, R1)

        return np.squeeze (rho_ft)

def ft_continuity_debug (ot, R, rho, zeta, R0, R1, nrows=50):
    idx = np.argsort (np.abs (R - R0))
    logger.debug (ot, "Close to R0 (%s)", R0)
    logger.debug (ot, "{:19s} {:19s} {:19s} {:19s} {:19s} {:19s} {:19s}".format ("R", "rho_a", "rho_b", "zeta", "zeta_x", "zeta_y", "zeta_z"))
    for irow in idx[:nrows]:
        debugstr = "{:19.12e} {:19.12e} {:19.12e} {:19.12e} {:19.12e} {:19.12e} {:19.12e}".format (R[irow], *rho[:,irow], *zeta[:,irow]) 
        logger.debug (ot, debugstr)
    idx = np.argsort (np.abs (R - R1))
    logger.debug (ot, "Close to R1 (%s)", R1)
    logger.debug (ot, "{:19s} {:19s} {:19s} {:19s} {:19s} {:19s} {:19s}".format ("R", "rho_a", "rho_b", "zeta", "zeta_x", "zeta_y", "zeta_z"))
    for irow in idx[:nrows]:
        debugstr = "{:19.12e} {:19.12e} {:19.12e} {:19.12e} {:19.12e} {:19.12e} {:19.12e}".format (R[irow], *rho[:,irow], *zeta[:,irow]) 
        logger.debug (ot, debugstr)
    


__t_doc__ = "For 'translated' functionals, otxc string = 't' + xc string\n"
__ft_doc__ = "For 'fully translated' functionals, otxc string = 'ft' + xc string\n"

def t_hybrid_coeff(ni, xc_code, spin=0):
    return _NumInt.hybrid_coeff(ni, xc_code[1:], spin=0)
t_hybrid_coeff.__doc__ = __t_doc__ + str(_NumInt.hybrid_coeff.__doc__)

def t_nlc_coeff(ni, xc_code):
    return _NumInt.nlc_coeff(ni, xc_code[1:])
t_nlc_coeff.__doc__ = __t_doc__ + str(_NumInt.nlc_coeff.__doc__)

def t_rsh_coeff(ni, xc_code):
    return _NumInt.rsh_coeff(ni, xc_code[1:])
t_rsh_coeff.__doc__ = __t_doc__ + str(_NumInt.rsh_coeff.__doc__)

def t_eval_xc(ni, xc_code, rho, spin=0, relativity=0, deriv=1, verbose=None):
    return _NumInt.eval_xc(ni, xc_code[1:], rho, spin=spin, relativity=relativity, deriv=deriv, verbose=verbose)
t_eval_xc.__doc__ = __t_doc__ + str(_NumInt.eval_xc.__doc__)

def t_xc_type(ni, xc_code):
    return _NumInt._xc_type(ni, xc_code[1:])
t_xc_type.__doc__ = __t_doc__ + str(_NumInt._xc_type.__doc__)

def t_rsh_and_hybrid_coeff(ni, xc_code, spin=0):
    return _NumInt.rsh_and_hybrid_coeff (ni, xc_code[1:], spin=spin)
t_rsh_and_hybrid_coeff.__doc__ = __t_doc__ + str(_NumInt.rsh_and_hybrid_coeff.__doc__)

def ft_hybrid_coeff(ni, xc_code, spin=0):
    return _NumInt.hybrid_coeff(ni, xc_code[2:], spin=0)
ft_hybrid_coeff.__doc__ = __ft_doc__ + str(_NumInt.hybrid_coeff.__doc__)

def ft_nlc_coeff(ni, xc_code):
    return _NumInt.nlc_coeff(ni, xc_code[2:])
ft_nlc_coeff.__doc__ = __ft_doc__ + str(_NumInt.nlc_coeff.__doc__)

def ft_rsh_coeff(ni, xc_code):
    return _NumInt.rsh_coeff(ni, xc_code[2:])
ft_rsh_coeff.__doc__ = __ft_doc__ + str(_NumInt.rsh_coeff.__doc__)

def ft_eval_xc(ni, xc_code, rho, spin=0, relativity=0, deriv=1, verbose=None):
    return _NumInt.eval_xc(ni, xc_code[2:], rho, spin=spin, relativity=relativity, deriv=deriv, verbose=verbose)
ft_eval_xc.__doc__ = __ft_doc__ + str(_NumInt.eval_xc.__doc__)

def ft_xc_type(ni, xc_code):
    return _NumInt._xc_type(ni, xc_code[2:])
ft_xc_type.__doc__ = __ft_doc__ + str(_NumInt._xc_type.__doc__)

def ft_rsh_and_hybrid_coeff(ni, xc_code, spin=0):
    return _NumInt.rsh_and_hybrid_coeff (ni, xc_code[2:], spin=spin)
ft_rsh_and_hybrid_coeff.__doc__ = __ft_doc__ + str(_NumInt.rsh_and_hybrid_coeff.__doc__)

