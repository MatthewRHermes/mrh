import numpy as np
from pyscf.lib import logger
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
        self.ks = ks
        self.xc_deriv = ['LDA', 'GGA', 'MGGA'].index (self.ks._numint._xc_type (self.ks.xc))
        self.verbose = ks.verbose
        self.stdout = ks.stdout
        self.otxc = 't' + self.ks.xc
        logger.info (self, 'Building %s functional', self.otxc)

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
        dexc_ddens = self.ks._numint.eval_xc (self.ks.xc, (rho_t[0,:,:], rho_t[1,:,:]), spin=1, relativity=0, deriv=0, verbose=self.ks.verbose)[0]
        dens = rho_t[0,0,:] + rho_t[1,0,:]
 
        rho = np.squeeze (rho)
        Pi = np.squeeze (Pi)

        # E_ot[rho,Pi] = \int {dE_ot/ddens}(r) * dens(r) dr
        #              = \sum_i {dE_ot/ddens}_i * dens_i * weight_i
        dexc_ddens *= dens
        dexc_ddens *= weight
        E_ot = np.sum (dexc_ddens)

        return E_ot

    def get_ratio (self, Pi, rho_avg):
        r''' R = Pi / [rho/2]^2 = Pi / rho_avg^2
            An intermediate quantity when computing the translated spin densities

            Note this function returns 1.0 for every point where the charge density is close to zero (i.e., convention: 0/0 = 1)
        '''
        assert (Pi.shape == rho_avg.shape)
        nderiv = Pi.shape[0]
        if nderiv > 4:
            raise NotImplementedError("derivatives above order 1")

        R = np.ones_like (Pi)
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

        R = self.get_ratio (Pi[0:1,:], rho_avg[0:1,:])

        # For nonzero charge & pair density, set alpha dens = beta dens = 1/2 charge dens
        idx = (Pi[0] >= 1e-15) & (rho_avg[0] >= (1e-15 / 2))
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
            w = rho_t[0,idx] * zeta[ideriv]
            rho_t[0,ideriv,idx] += w
            rho_t[1,ideriv,idx] -= w
    
        return rho_t



class ftransfnal (transfnal):
    r''' "fully translated functional" of Carlson et al., JCTC 11, 4077 (2015)
    '''

    def __init__ (self, ks, **kwargs):
        super().__init__(ks, kwargs)
        self.R0=0.9
        self.R1=1.15
        self.A=-475.60656009
        self.B=-379.47331922 
        self.C=-85.38149682

    def get_rho_translated (self, Pi, rho, Rmax=None, xi_deriv=True):
        r''' "full" translation, Carlson et al., JCTC 11, 4077 (2015)
        rho_t[0] = {(rho[0] + rho[1]) / 2} * (1 + xi)
        rho_t[1] = {(rho[0] + rho[1]) / 2} * (1 - xi)
    
        where
        xi = (1-ratio)^(1/2)                                  ; ratio < R0
           = A*(ratio-R1)^5 + B*(ratio-R1)^4 + C*(ratio-R1)^3 ; R0 <= ratio <= R1
           = 0                                                ; otherwise
    
        Propagate derivatives thru xi
    
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
    
        rho_ft = super().get_rho_translated (Pi, rho, Rmax=R0, xi_deriv=True)
    
        rho_avg = (rho[0] + rho[1]) / 2
        R = self.get_ratio (Pi, rho_avg)
    
        idx = np.where (np.logical_and (R[0] >= self.R0, R[0] <= self.R1))[0]
        R_m_R0 = np.stack ([np.power (R[0,idx] - R0, n) for n in range (2,6)], axis=0)
        xi = np.empty_like (R[:,idx])
        xi[0] = (self.A*R_m_R0[5-2] 
               + self.B*R_m_R0[4-2] 
               + self.C*R_m_R0[3-2])
        # Chain rule!
        for ideriv in range (1, nderiv):
            xi[ideriv] = R[ideriv,idx] * (5*self.A*R_m_R0[4-2] 
                                        + 4*self.B*R_m_R0[3-2] 
                                        + 3*self.C*R_m_R0[2-2])
    
        # Chain rule!
        for ideriv in range (nderiv):
            rho_ft[0,ideriv,idx] *= (1 + xi[0])
            rho_ft[1,ideriv,idx] *= (1 - xi[0])
        # Product rule!
        for ideriv in range (1,nderiv):
            rho_ft[0,ideriv,idx] += rho_ft[0,0,idx] * xi[ideriv]
            rho_ft[1,ideriv,idx] -= rho_ft[1,0,idx] * xi[ideriv]
    
        return np.squeeze (rho_ft)




