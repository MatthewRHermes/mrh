from pyscf import ao2mo
from pyscf.lib import logger, pack_tril, unpack_tril, current_memory
from pyscf.lib import einsum as einsum_threads
from pyscf.dft.gen_grid import BLKSIZE
from mrh.my_pyscf.mcpdft.otpd import get_ontop_pair_density
from scipy import linalg
from os import path
import numpy as np
import time, gc

class _ERIS(object):
    def __init__(self, mo_coeff, ncore, ncas, method='incore', paaa_only=False):
        self.nao, self.nmo = mo_coeff.shape
        self.ncore = ncore
        self.ncas = ncas
        self.vhf_c = np.zeros ((self.nmo, self.nmo), dtype=mo_coeff.dtype)
        self.method = method
        self.paaa_only = paaa_only
        if method == 'incore':
            #npair = self.nmo * (self.nmo+1) // 2
            #self._eri = np.zeros ((npair, npair))
            #npair_cas = ncas * (ncas+1) // 2
            #self.ppaa = np.zeros ((npair, npair_cas), dtype=mo_coeff.dtype)
            #self.ppaa = np.zeros ((self.nmo, self.nmo, ncas, ncas), dtype=mo_coeff.dtype)
            self.papa = np.zeros ((self.nmo, ncas, self.nmo, ncas), dtype=mo_coeff.dtype)
            self.j_pc = np.zeros ((self.nmo, ncore), dtype=mo_coeff.dtype)
        else:
            raise NotImplementedError ("method={} for veff2".format (self.method))

    def _accumulate (self, ot, rho, Pi, mo, weight, rho_c, rho_a, vPi):
        if self.method == 'incore':
            self._accumulate_incore (ot, rho, Pi, mo, weight, rho_c, rho_a, vPi) 
        else:
            raise NotImplementedError ("method={} for veff2".format (self.method))

    def _accumulate_incore (self, ot, rho, Pi, mo, weight, rho_c, rho_a, vPi):
        #self._eri += ot.get_veff_2body (rho, Pi, mo, weight, aosym='s4', kern=vPi)
        ncore, ncas = self.ncore, self.ncas
        nocc = ncore + ncas
        mo_cas = mo[:,:,ncore:nocc]
        # vhf_c
        vrho_c = _contract_vot_rho (vPi, rho_c)
        self.vhf_c += ot.get_veff_1body (rho, Pi, mo, weight, kern=vrho_c)
        if self.paaa_only:
            # 1/2 v_aiuv D_ii D_uv = v^ai_uv D_uv -> F_ai, F_ia needs to be in here since it would otherwise be calculated using ppaa and papa
            vrho_a = _contract_vot_rho (vPi, rho_a.sum (0))
            vhf_a = ot.get_veff_1body (rho, Pi, mo, weight, kern=vrho_a) 
            vhf_a[ncore:nocc,:] = vhf_a[:,ncore:nocc] = 0.0
            self.vhf_c += vhf_a
        # ppaa
        if self.paaa_only:
            paaa = ot.get_veff_2body (rho, Pi, [mo, mo_cas, mo_cas, mo_cas], weight, aosym='s1', kern=vPi)
            #paaa = unpack_tril (paaa.reshape (-1, ncas * (ncas + 1) // 2)).reshape (-1, ncas, ncas, ncas)
            self.papa[:,:,ncore:nocc,:] += paaa
            self.papa[ncore:nocc,:,:,:] += paaa.transpose (2,3,0,1)
            self.papa[ncore:nocc,:,ncore:nocc,:] -= paaa[ncore:nocc,:,:,:]
        else:
            self.papa += ot.get_veff_2body (rho, Pi, [mo, mo_cas, mo, mo_cas], weight, aosym='s1', kern=vPi)
        # j_pc
        mo = _square_ao (mo)
        mo_core = mo[:,:,:ncore]
        self.j_pc += ot.get_veff_1body (rho, Pi, [mo, mo_core], weight, kern=vPi)

    def _finalize (self):
        if self.method == 'incore':
            nmo, ncore, ncas = self.nmo, self.ncore, self.ncas
            nocc = ncore + ncas
            '''
            diag_idx = np.arange (nmo)
            diag_idx = diag_idx * (diag_idx+1) // 2 + diag_idx
            self.j_pc = np.ascontiguousarray (self._eri[np.ix_(diag_idx,diag_idx[:ncore])])
            self.vhf_c = unpack_tril (self._eri[:,diag_idx[:ncore]].sum (-1))
            self._eri = ao2mo.restore (1, self._eri, nmo)
            self.ppaa = np.ascontiguousarray (self._eri[:,:,ncore:nocc,ncore:nocc])
            self._eri = None
            '''
            '''
            self.ppaa = unpack_tril (self.ppaa, axis=0)
            self.ppaa = unpack_tril (self.ppaa, axis=-1).reshape (nmo, nmo, ncas, ncas)
            self.papa = np.ascontiguousarray (self.ppaa.transpose (0,2,1,3))
            '''
            self.ppaa = np.ascontiguousarray (self.papa.transpose (0,2,1,3))
            self.k_pc = self.j_pc.copy ()
        else:
            raise NotImplementedError ("method={} for veff2".format (self.method))
        self.k_pc = self.j_pc.copy ()

def kernel (ot, oneCDMs, twoCDM_amo, mo_coeff, ncore, ncas, max_memory=20000, hermi=1, veff2_mo=None, paaa_only=False):
    ''' Get the 1- and 2-body effective potential from MC-PDFT. Eventually I'll be able to specify
        mo slices for the 2-body part

        Args:
            ot : an instance of otfnal class
            oneCDMs : ndarray of shape (2, nao, nao)
                containing spin-separated one-body density matrices
            twoCDM_amo : ndarray of shape (ncas, ncas, ncas, ncas)
                containing spin-summed two-body cumulant density matrix in an active space
            ao2amo : ndarray of shape (nao, ncas)
                containing molecular orbital coefficients for active-space orbitals

        Kwargs:
            max_memory : int or float
                maximum cache size in MB
                default is 20000
            hermi : int
                1 if 1CDMs are assumed hermitian, 0 otherwise

        Returns : float
            The MC-PDFT on-top exchange-correlation energy

    '''
    if veff2_mo is not None:
        raise NotImplementedError ('Molecular orbital slices for the two-body part')
    ni, xctype, dens_deriv = ot._numint, ot.xctype, ot.dens_deriv
    norbs_ao = mo_coeff.shape[0]
    mo_core = mo_coeff[:,:ncore]
    ao2amo = mo_coeff[:,ncore:][:,:ncas]
    npair = norbs_ao * (norbs_ao + 1) // 2

    veff1 = np.zeros_like (oneCDMs[0])
    veff2 = _ERIS (mo_coeff, ncore, ncas, paaa_only=paaa_only)

    t0 = (time.clock (), time.time ())
    dm_core = mo_core @ mo_core.T * 2
    casdm1a = oneCDMs[0] - dm_core/2
    casdm1b = oneCDMs[1] - dm_core/2
    make_rho_c = ni._gen_rho_evaluator (ot.mol, dm_core, hermi)
    make_rho_a = tuple (ni._gen_rho_evaluator (ot.mol, dm, hermi) for dm in (casdm1a, casdm1b))
    make_rho = tuple (ni._gen_rho_evaluator (ot.mol, oneCDMs[i,:,:], hermi) for i in range(2))
    gc.collect ()
    remaining_floats = (max_memory - current_memory ()[0]) * 1e6 / 8
    nderiv_rho = (1,4,10)[dens_deriv] # ?? for meta-GGA
    nderiv_Pi = (1,4)[ot.Pi_deriv]
    ncols_v2 = norbs_ao*ncas + ncas**2 if paaa_only else 2*norbs_ao*ncas
    ncols = 1 + nderiv_rho * (5 + norbs_ao*2) + nderiv_Pi * (1 + ncols_v2) 
    pdft_blksize = int (remaining_floats / (ncols * BLKSIZE)) * BLKSIZE # something something indexing
    if ot.grids.coords is None:
        ot.grids.build(with_non0tab=True)
    ngrids = ot.grids.coords.shape[0]
    pdft_blksize = max(BLKSIZE, min(pdft_blksize, ngrids, BLKSIZE*1200))
    logger.debug (ot, '{} MB used of {} available; block size of {} chosen for grid with {} points'.format (
        current_memory ()[0], max_memory, pdft_blksize, ngrids))
    for ao, mask, weight, coords in ni.block_loop (ot.mol, ot.grids, norbs_ao, dens_deriv, max_memory, blksize=pdft_blksize):
        rho = np.asarray ([m[0] (0, ao, mask, xctype) for m in make_rho])
        rho_a = np.asarray ([m[0] (0, ao, mask, xctype) for m in make_rho_a])
        rho_c = make_rho_c [0] (0, ao, mask, xctype)
        t0 = logger.timer (ot, 'untransformed densities (core and total)', *t0)
        Pi = get_ontop_pair_density (ot, rho, ao, oneCDMs, twoCDM_amo, ao2amo, dens_deriv)
        t0 = logger.timer (ot, 'on-top pair density calculation', *t0)
        eot, vrho, vPi = ot.eval_ot (rho, Pi, weights=weight)
        t0 = logger.timer (ot, 'effective potential kernel calculation', *t0)
        veff1 += ot.get_veff_1body (rho, Pi, ao, weight, kern=vrho)
        t0 = logger.timer (ot, '1-body effective potential calculation', *t0)
        ao[:,:,:] = np.tensordot (ao, mo_coeff, axes=1)
        t0 = logger.timer (ot, 'ao2mo grid points', *t0)
        veff2._accumulate (ot, rho, Pi, ao, weight, rho_c, rho_a, vPi)
        t0 = logger.timer (ot, '2-body effective potential calculation', *t0)
    veff2._finalize ()
    t0 = logger.timer (ot, 'Finalizing 2-body effective potential calculation', *t0)
    return veff1, veff2

def lazy_kernel (ot, oneCDMs, twoCDM_amo, ao2amo, max_memory=20000, hermi=1, veff2_mo=None):
    ''' Get the 1- and 2-body effective potential from MC-PDFT. Eventually I'll be able to specify
        mo slices for the 2-body part

        Args:
            ot : an instance of otfnal class
            oneCDMs : ndarray of shape (2, nao, nao)
                containing spin-separated one-body density matrices
            twoCDM_amo : ndarray of shape (ncas, ncas, ncas, ncas)
                containing spin-summed two-body cumulant density matrix in an active space
            ao2amo : ndarray of shape (nao, ncas)
                containing molecular orbital coefficients for active-space orbitals

        Kwargs:
            max_memory : int or float
                maximum cache size in MB
                default is 20000
            hermi : int
                1 if 1CDMs are assumed hermitian, 0 otherwise

        Returns : float
            The MC-PDFT on-top exchange-correlation energy

    '''
    if veff2_mo is not None:
        raise NotImplementedError ('Molecular orbital slices for the two-body part')
    ni, xctype, dens_deriv = ot._numint, ot.xctype, ot.dens_deriv
    norbs_ao = ao2amo.shape[0]
    npair = norbs_ao * (norbs_ao + 1) // 2

    veff1 = np.zeros_like (oneCDMs[0])
    veff2 = np.zeros ((npair, npair), dtype=veff1.dtype)

    t0 = (time.clock (), time.time ())
    make_rho = tuple (ni._gen_rho_evaluator (ot.mol, oneCDMs[i,:,:], hermi) for i in range(2))
    for ao, mask, weight, coords in ni.block_loop (ot.mol, ot.grids, norbs_ao, dens_deriv, max_memory):
        rho = np.asarray ([m[0] (0, ao, mask, xctype) for m in make_rho])
        t0 = logger.timer (ot, 'untransformed density', *t0)
        Pi = get_ontop_pair_density (ot, rho, ao, oneCDMs, twoCDM_amo, ao2amo, dens_deriv)
        t0 = logger.timer (ot, 'on-top pair density calculation', *t0)
        eot, vrho, vPi = ot.eval_ot (rho, Pi, weights=weight)
        t0 = logger.timer (ot, 'effective potential kernel calculation', *t0)
        veff1 += ot.get_veff_1body (rho, Pi, ao, weight, kern=vrho)
        t0 = logger.timer (ot, '1-body effective potential calculation', *t0)
        veff2 += ot.get_veff_2body (rho, Pi, ao, weight, aosym='s4', kern=vPi)
        t0 = logger.timer (ot, '2-body effective potential calculation', *t0)
    return veff1, veff2

def get_veff_1body (otfnal, rho, Pi, ao, weight, kern=None, **kwargs):
    r''' get the derivatives dEot / dDpq
    Can also be abused to get semidiagonal dEot / dPppqq if you pass the right kern and
    squared aos/mos

    Args:
        rho : ndarray of shape (2,*,ngrids)
            containing spin-density [and derivatives]
        Pi : ndarray with shape (*,ngrids)
            containing on-top pair density [and derivatives]
        ao : ndarray of shape (*,ngrids,nao)
            contains values and derivatives of nao
        weight : ndarray of shape (ngrids)
            containing numerical integration weights

    Kwargs:
        kern : ndarray of shape (*,ngrids)
            the derivative of the on-top potential with respect to density (vrho)
            If not provided, it is calculated.

    Returns : ndarray of shape (nao,nao)
        The 1-body effective potential corresponding to this on-top pair density
        exchange-correlation functional, in the atomic-orbital basis.
        In PDFT this functional is always spin-symmetric
    '''
    if rho.ndim == 2:
        rho = np.expand_dims (rho, 1)
        Pi = np.expand_dims (Pi, 0)

    w = weight[None,:]
    if isinstance (ao, np.ndarray) and ao.ndim == 3:
        ao = [ao, ao]
    elif len (ao) != 2:
        raise NotImplementedError ("uninterpretable aos!")
    if kern is None: 
        kern = otfnal.get_dEot_drho (rho, Pi, **kwargs) 
    else:
        kern = kern.copy () 
    nderiv = kern.shape[0]
    kern *= weight[None,:]

    # Zeroth and first derivatives
    veff = np.tensordot (kern[0,:,None] * ao[0][0], ao[1][0], axes=(0,0))
    if nderiv > 1:
        veff += np.tensordot ((kern[1:4,:,None] * ao[0][1:4]).sum (0), ao[1][0], axes=(0,0))
        veff += np.tensordot (ao[0][0], (kern[1:4,:,None] * ao[1][1:4]).sum (0), axes=(0,0))

    rho = np.squeeze (rho)
    Pi = np.squeeze (Pi)

    return veff

def get_veff_2body (otfnal, rho, Pi, ao, weight, aosym='s4', kern=None, vao=None, **kwargs):
    r''' get the derivatives dEot / dPijkl

    Args:
        rho : ndarray of shape (2,*,ngrids)
            containing spin-density [and derivatives]
        Pi : ndarray with shape (*,ngrids)
            containing on-top pair density [and derivatives]
        ao : ndarray of shape (*,ngrids,nao) OR list of ndarrays of shape (*,ngrids,*)
            values and derivatives of atomic or molecular orbitals in which
            space to calculate the 2-body veff
            If a list of length 4, the corresponding set of eri-like elements are returned
        weight : ndarray of shape (ngrids)
            containing numerical integration weights

    Kwargs:
        aosym : int or str
            Index permutation symmetry of the desired integrals. Valid options are 
            1 (or '1' or 's1'), 4 (or '4' or 's4'), '2ij' (or 's2ij'), and '2kl' (or 's2kl').
            These have the same meaning as in PySCF's ao2mo module.
            Currently all symmetry exploitation is extremely slow and unparallelizable for some reason
            so trying to use this is not recommended until I come up with a C routine
        kern : ndarray of shape (*,ngrids)
            the derivative of the on-top potential with respect to pair density (vot)
            If not provided, it is calculated.
        vao : ndarray of shape (*,ngrids,nao,nao) or (*,ngrids,nao*(nao+1)//2)
            An intermediate in which the kernel and the k,l orbital indices have been contracted
            Overrides kl_symm if provided.

    Returns : eri-like ndarray
        The two-body effective potential corresponding to this on-top pair density
        exchange-correlation functional or elements thereof, in the provided basis.
    '''

    if isinstance (ao, np.ndarray) and ao.ndim == 3:
        ao = [ao,ao,ao,ao]
    elif len (ao) != 4:
        raise NotImplementedError ('fancy orbital subsets and fast evaluation in get_veff_2body')
    #elif isinstance (ao, np.ndarray) and ao.ndim == 4:
    #    ao = [a for a in ao]

    if isinstance (aosym, int): aosym = str (aosym)
    ij_symm = '4' in aosym or '2ij' in aosym
    kl_symm = '4' in aosym or '2kl' in aosym

    if vao is None: vao = get_veff_2body_kl (otfnal, rho, Pi, ao[2], ao[3], weight, symm=kl_symm, kern=kern)
    nderiv = vao.shape[0]
    ao2 = _contract_ao1_ao2 (ao[0], ao[1], nderiv, symm=ij_symm)
    veff = np.tensordot (ao2, vao, axes=((0,1),(0,1)))
    #veff = einsum_threads ('dgij,dgkl->ijkl', ao2, vao)
    # ^ When I save these arrays to disk and load them, np.tensordot appears to multi-thread successfully
    # However, it appears not to multithread here, in this context
    # numpy_helper.einsum is definitely always multithreaded but has a serial overhead
    # Even when I was doing papa it was reporting 15 seconds clock and 15 seconds wall with np.tensordot
    # So it's hard for me to believe that this is a clock bug

    return veff 

def get_veff_2body_kl (otfnal, rho, Pi, ao_k, ao_l, weight, symm=False, kern=None, **kwargs):
    r''' get the two-index intermediate Mkl of dEot/dPijkl

    Args:
        rho : ndarray of shape (2,*,ngrids)
            containing spin-density [and derivatives]
        Pi : ndarray with shape (*,ngrids)
            containing on-top pair density [and derivatives]
        ao_k : ndarray of shape (*,ngrids,nao) OR list of ndarrays of shape (*,ngrids,*)
            values and derivatives of atomic or molecular orbitals corresponding to index k
        ao_l : ndarray of shape (*,ngrids,nao) OR list of ndarrays of shape (*,ngrids,*)
            values and derivatives of atomic or molecular orbitals corresponding to index l
        weight : ndarray of shape (ngrids)
            containing numerical integration weights

    Kwargs:
        symm : logical
            Index permutation symmetry of the desired integral wrt k,l
        kern : ndarray of shape (*,ngrids)
            the derivative of the on-top potential with respect to pair density (vot)
            If not provided, it is calculated.

    Returns : ndarray of shape (*,ngrids,nao,nao) or (*,ngrids,nao*(nao+1)//2)
        An intermediate for calculating the two-body effective potential corresponding
        to this on-top pair density exchange-correlation functional in the provided basis
    '''
    if rho.ndim == 2:
        rho = np.expand_dims (rho, 1)
        Pi = np.expand_dims (Pi, 0)

    if kern is None:
        kern = otfnal.get_dEot_dPi (rho, Pi, **kwargs) 
    else:
        kern = kern.copy ()
    kern *= weight[None,:]
    nderiv, ngrid = kern.shape

    # Flatten deriv and grid so I can tensordot it all at once
    # Index symmetry can be built into _contract_ao1_ao2
    vao = _contract_vot_ao (kern, ao_l)
    vao = _contract_ao_vao (ao_k, vao, symm=symm)
    return vao

def _square_ao (ao):
    nderiv = ao.shape[0]
    ao_sq = ao * ao[0]
    if nderiv > 1:
        ao_sq[1:4] *= 2
    if nderiv > 4:
        ao_sq[4:10] += ao[1:4]**2
        ao_sq[4:10] *= 2
    return ao_sq

def _contract_ao1_ao2 (ao1, ao2, nderiv, symm=False):
    if symm:
        ix_p, ix_q = np.tril_indices (ao1.shape[-1])
        ao1 = ao1[:nderiv,:,ix_p]
        ao2 = ao2[:nderiv,:,ix_q]
    else:
        ao1 = np.expand_dims (ao1, -1)[:nderiv]
        ao2 = np.expand_dims (ao2, -2)[:nderiv]
    prod = ao1[:nderiv] * ao2[0]
    if nderiv > 1:
        prod[1:4] += ao1[0] * ao2[1:4] # Product rule
    ao2 = None
    return prod 

def _contract_vot_ao (vot, ao):
    nderiv = vot.shape[0]
    vao = ao[0] * vot[:,:,None]
    if nderiv > 1:
        vao[0] += (ao[1:4] * vot[1:4,:,None]).sum (0)
    return vao

def _contract_ao_vao (ao, vao, symm=False):
    r''' Outer-product of ao grid and vot * ao grid
    Can be used with two-orb-dimensional vao if the last two dimensions are flattened into "nao"

    Args:
        ao : ndarray of shape (*,ngrids,nao1)
        vao : ndarray of shape (nderiv,ngrids,nao2)

    Kwargs:
        symm : logical
            If true, nao1 == nao2 must be true

    Returns: ndarray of shape (nderiv,ngrids,nao1,nao2) or (nderiv,ngrids,nao1*(nao1+1)//2)
    '''

    nderiv = vao.shape[0]
    if symm:
        ix_p, ix_q = np.tril_indices (ao.shape[-1])
        ao = ao[:,:,ix_p]
        vao = vao[:,:,ix_q]
    else:
        ao = np.expand_dims (ao, -1)
        vao = np.expand_dims (vao, -2)
    prod = ao[0] * vao
    if nderiv > 1:
        prod[0] += (ao[1:4] * vao[1:4]).sum (0)
    return prod

def _contract_vot_rho (vot, rho, add_vrho=None):
    ''' Make a jk-like vrho from vot and a density. k = j so it's just vot * vrho / 2,
        but the product rule needs to be followed '''
    nderiv = vot.shape[0]
    vrho = vot * rho[0]
    if nderiv > 1:
        vrho[0] += (vot[1:4] * rho[1:4]).sum (0)
    vrho /= 2
    # vot involves lower derivatives than vhro in original translation
    # make sure vot * rho gets added to only the proper component(s)
    if add_vrho is not None:
        add_vrho[:nderiv] += vrho
        vrho = add_vrho
    return vrho



