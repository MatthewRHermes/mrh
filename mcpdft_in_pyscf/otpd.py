import numpy as np
from mrh.util.rdm import get_2CDM_from_2RDM
from pyscf.dft.numint import eval_rho, eval_ao 
from itertools import product

deriv_dict = {'LDA': 0,
              'GGA': 1,
              'MGGA': 2}

def get_ontop_pair_density (mc, ks, rho=None, ao=None):
    r''' Pi(r) = i(r)*j(r)*k(r)*l(r)*g^ik_jl 
               = rho_up(r)*rho_down(r) + i(r)*j(r)*k(r)*l(r)*l^ik_jl

        Args:
            mc : an instance of a pyscf MC-SCF class
            ks : an instance of a pyscf UKS class

        Kwargs:
            rho : ndarray containing density and derivatives
                see the documentation of pyscf.dft.numint.eval_rho and
                pyscf.dft.libxc.eval_xc for its shape 
            ao : ndarray containing values and derivatives of atomic orbitals at grid points
                see the documentation of pyscf.dft.numint.eval_ao for its shape

        Returns : ndarray of the same shape as rho
            The on-top pair density and its derivatives
    '''
    xctype = ks._numint._xc_type
    deriv = deriv_dict[xctype]
    if deriv > 1:
        raise NotImplementedError("meta-GGA translated functionals")

    if ao is None:
        # This is probably too big an object for most molecules
        ao = eval_ao (ks.mol, ks.grids.coords, deriv=deriv)
    if rho is None:
        rho = np.asarray ([eval_rho (ks.mol, ao, oneCDM, xctype=xctype) for oneCDM in mc.make_rdm1s ()])
    ngrids = rho.shape[-1]
    norbs_ao = ao.shape[-1]

    # First cumulant
    Pi = np.multiply (rho[0], rho[1])

    # Second cumulant
    ao2amo = mc.mo_coeff[:,mc.ncore:mc.ncore+mc.ncas]
    grid2ao = ao.reshape (product (ao.shape[:-1]), ao.shape[-1])
    grid2amo = np.dot (grid2ao, ao2amo)
    oneCDM_amo, twoRDM_amo = mc.fcisolver.make_rdm12 ()
    twoCDM_amo = get_2CDM_from_2RDM (twoRDM_amo, oneCDM_amo)
    oldshape = grid2amo.shape
    newshape = (product(oldshape[:-1]), oldshape[-1])
    Pi_finalshape = oldshape[:-1]
    Pi += np.einsum ('ijkl,ai,aj,ak,al->a', twoCDM_amo, grid2amo, grid2amo, grid2amo, grid2amo).reshape (rho.shape)

    return Pi


