import numpy as np
from mrh.util.rdm import get_2RDMR_from_2RDM
from pyscf.dft.numint import eval_rho, eval_ao 

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

        Returns : 1d ndarray
            The on-top pair density
    '''
    xctype = ks._numint._xc_type

    if ao is None:
        ao = eval_ao (ks.mol, ks.grids.coords, deriv=deriv_dict[xctype])
    if rho is None:
        rho = np.asarray ([eval_rho (ks.mol, ao, oneCDM, xctype=ks._numint._xc_type) for oneCDM in mc.make_rdm1s ()])
    ngrids = rho.shape[-1]
    norbs_ao = ao.shape[-1]

    # First cumulant
    Pi = np.multiply (rho[0].flat[:ngrids], rho[1].flat[:ngrids])

    # Second cumulant
    grid2ao = ao.flat[:ngrids*norbs_ao].reshape (ngrids, norbs_ao)
`   ao2amo = mc.mo_coeff[:,mc.ncore:mc.ncore+mc.ncas]
    grid2amo = np.dot (grid2ao, ao2amo)
    oneCDM_amo, twoRDM_amo = mc.fcisolver.make_rdm12 ()
    twoCDM_amo = get_2RDMR_from_2RDM (twoRDM_amo, oneCDM_amo)
    Pi += np.einsum ('ijkl,ai,aj,ak,al->a', twoCDM_amo, grid2ao, grid2ao, grid2ao, grid2ao)

    return Pi


