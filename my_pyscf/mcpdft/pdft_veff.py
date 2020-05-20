from pyscf import ao2mo, __config__, lib
from pyscf.lib import logger, pack_tril, unpack_tril, current_memory, tag_array
from pyscf.lib import einsum as einsum_threads
from pyscf.dft import numint
from pyscf.dft.gen_grid import BLKSIZE
from mrh.my_pyscf.mcpdft.otpd import get_ontop_pair_density, _grid_ao2mo
from mrh.lib.helper import load_library
from scipy import linalg
from os import path
import numpy as np
import time, gc, ctypes

# MRH 05/18/2020: An annoying convention in pyscf.dft.numint that I have to comply with is that the
# AO grid-value arrays and all their derivatives are in NEITHER column-major NOR row-major order;
# they all have ndim = 3 and strides of (8*ngrids*nao, 8, 8*ngrids). Here, for my ndim > 3 objects,
# I choose to generalize this so that a cyclic transpose to the left, ao.transpose (1,2,3,...,0),
# places the array in column-major (FORTRAN-style) order. I have to comply with this awkward convention
# in order to take full advantage of the code in pyscf.dft.numint and libdft.so. The ngrids dimension,
# which is by far the largest, almost always benefits from having the smallest stride.

# MRH 05/19/2020: Actually, I really should just turn the deriv component part of all of these arrays into
# lists and keep everything in col-major order otherwise, because ndarrays have to have regular strides,
# but lists can just be references and less copying is involved. The copying is more expensive and less
# transparently parallel-scalable than the actual math! (The parallel-scaling part of that is stupid but it
# is what it is.

SWITCH_SIZE = getattr(__config__, 'dft_numint_SWITCH_SIZE', 800)
libpdft = load_library('libpdft')

class _ERIS(object):
    def __init__(self, mol, mo_coeff, ncore, ncas, method='incore', paaa_only=False, verbose=0, stdout=None):
        self.mol = mol
        self.mo_coeff = mo_coeff
        self.nao, self.nmo = mo_coeff.shape
        self.ncore = ncore
        self.ncas = ncas
        self.vhf_c = np.zeros ((self.nmo, self.nmo), dtype=mo_coeff.dtype)
        self.method = method
        self.paaa_only = paaa_only
        self.verbose = verbose
        self.stdout = stdout
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

    def _accumulate (self, ot, rho, Pi, mo, weight, rho_c, rho_a, vPi, non0tab=None, shls_slice=None, ao_loc=None):
        if self.method == 'incore':
            self._accumulate_incore (ot, rho, Pi, mo, weight, rho_c, rho_a, vPi, non0tab, shls_slice, ao_loc) 
        else:
            raise NotImplementedError ("method={} for veff2".format (self.method))

    def _accumulate_incore (self, ot, rho, Pi, ao, weight, rho_c, rho_a, vPi, non0tab, shls_slice, ao_loc):
        #self._eri += ot.get_veff_2body (rho, Pi, mo, weight, aosym='s4', kern=vPi)
        # ao is here stored in row-major order = deriv,AOs,grids regardless of what
        # the ndarray object thinks
        mo_coeff = self.mo_coeff
        ncore, ncas = self.ncore, self.ncas
        nocc = ncore + ncas
        mo_cas = _grid_ao2mo (self.mol, ao, mo_coeff[:,ncore:nocc], non0tab)
        # vhf_c
        vrho_c = _contract_vot_rho (vPi, rho_c)
        self.vhf_c += mo_coeff.conjugate ().T @ ot.get_veff_1body (rho, Pi, ao, weight, non0tab=non0tab, shls_slice=shls_slice, ao_loc=ao_loc, hermi=1, kern=vrho_c) @ mo_coeff
        if self.paaa_only:
            # 1/2 v_aiuv D_ii D_uv = v^ai_uv D_uv -> F_ai, F_ia needs to be in here since it would otherwise be calculated using ppaa and papa
            vrho_a = _contract_vot_rho (vPi, rho_a.sum (0))
            vhf_a = ot.get_veff_1body (rho, Pi, ao, weight, non0tab=non0tab, shls_slice=shls_slice, ao_loc=ao_loc, hermi=1, kern=vrho_a) 
            vhf_a = mo_coeff.conjugate ().T @ vhf_a @ mo_coeff
            vhf_a[ncore:nocc,:] = vhf_a[:,ncore:nocc] = 0.0
            self.vhf_c += vhf_a
        # ppaa
        if self.paaa_only:
            paaa = ot.get_veff_2body (rho, Pi, [ao, mo_cas, mo_cas, mo_cas], weight, aosym='s1', kern=vPi)
            paaa = np.tensordot (mo_coeff.T, paaa, axes=1)
            #paaa = unpack_tril (paaa.reshape (-1, ncas * (ncas + 1) // 2)).reshape (-1, ncas, ncas, ncas)
            self.papa[:,:,ncore:nocc,:] += paaa
            self.papa[ncore:nocc,:,:,:] += paaa.transpose (2,3,0,1)
            self.papa[ncore:nocc,:,ncore:nocc,:] -= paaa[ncore:nocc,:,:,:]
        else:
            #self.papa += ot.get_veff_2body (rho, Pi, [mo, mo_cas, mo, mo_cas], weight, aosym='s1', kern=vPi)
            papa = ot.get_veff_2body (rho, Pi, [ao, mo_cas, ao, mo_cas], weight, aosym='s1', kern=vPi)
            papa = np.tensordot (mo_coeff.T, papa, axes=1)
            self.papa += np.tensordot (mo_coeff.T, papa, axes=((1),(2))).transpose (1,2,0,3)
        # j_pc
        if self.verbose > logger.DEBUG:
            raise NotImplementedError ('TODO: fix the three lines below for the new signature of _accumulate_incore')
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

def kernel (ot, oneCDMs_amo, twoCDM_amo, mo_coeff, ncore, ncas, max_memory=20000, hermi=1, veff2_mo=None, paaa_only=False):
    ''' Get the 1- and 2-body effective potential from MC-PDFT. Eventually I'll be able to specify
        mo slices for the 2-body part

        Args:
            ot : an instance of otfnal class
            oneCDMs_amo : ndarray of shape (2, ncas, ncas)
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
    nocc = ncore + ncas
    ni, xctype, dens_deriv = ot._numint, ot.xctype, ot.dens_deriv
    norbs_ao = mo_coeff.shape[0]
    mo_core = mo_coeff[:,:ncore]
    ao2amo = mo_coeff[:,ncore:nocc]
    npair = norbs_ao * (norbs_ao + 1) // 2

    veff1 = np.zeros ((norbs_ao, norbs_ao), dtype=oneCDMs_amo.dtype)
    veff2 = _ERIS (ot.mol, mo_coeff, ncore, ncas, paaa_only=paaa_only, verbose=ot.verbose, stdout=ot.stdout)

    t0 = (time.clock (), time.time ())
    dm_core = mo_core @ mo_core.T 
    dm_cas = np.dot (ao2amo, np.dot (oneCDMs_amo, ao2amo.T)).transpose (1,0,2)
    dm1s = dm_cas + dm_core[None,:,:] 
    dm_core *= 2
    # Can't trust that NOs are the same for alpha and beta. Have to do this explicitly here
    # Begin tag block: dm_core
    imo_occ = np.ones (ncore, dtype=dm_core.dtype) * 2.0
    dm_core = tag_array (dm_core, mo_coeff=mo_core, mo_occ=imo_occ)
    # Begin tag block: dm_cas
    amo_occ = np.zeros ((2,ncas), dtype=dm_cas.dtype)
    amo_coeff = np.stack ([ao2amo.copy (), ao2amo.copy ()], axis=0)
    for i in range (2):
        amo_occ[i], ua = linalg.eigh (oneCDMs_amo[i])
        amo_coeff[i] = amo_coeff[i] @ ua
    dm_cas = tag_array (dm_cas, mo_coeff=amo_coeff, mo_occ=amo_occ)
    # Begin tag block: dm1s
    mo_occ = np.zeros ((2, nocc), dtype=dm1s.dtype)
    mo_occ[:,:ncore] = 1.0
    mo_occ[:,ncore:nocc] = amo_occ
    tag_coeff = np.stack ((mo_coeff[:,:nocc].copy (), mo_coeff[:,:nocc].copy ()), axis=0)
    tag_coeff[:,:,ncore:nocc] = amo_coeff 
    dm1s = tag_array (dm1s, mo_coeff=tag_coeff, mo_occ=mo_occ)
    # End tag block
    make_rho_c, nset_c, nao_c = ni._gen_rho_evaluator (ot.mol, dm_core, hermi)
    make_rho_a, nset_a, nao_a = ni._gen_rho_evaluator (ot.mol, dm_cas, hermi)
    make_rho, nset, nao = ni._gen_rho_evaluator (ot.mol, dm1s, hermi)
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
    shls_slice = (0, ot.mol.nbas)
    ao_loc = ot.mol.ao_loc_nr()
    for ao, mask, weight, coords in ni.block_loop (ot.mol, ot.grids, norbs_ao, dens_deriv, max_memory, blksize=pdft_blksize):
        rho = np.asarray ([make_rho (i, ao, mask, xctype) for i in range(2)])
        rho_a = np.asarray ([make_rho_a (i, ao, mask, xctype) for i in range(2)])
        rho_c = make_rho_c (0, ao, mask, xctype)
        t0 = logger.timer (ot, 'untransformed densities (core and total)', *t0)
        Pi = get_ontop_pair_density (ot, rho, ao, dm1s, twoCDM_amo, ao2amo, dens_deriv, mask)
        t0 = logger.timer (ot, 'on-top pair density calculation', *t0)
        eot, vrho, vPi = ot.eval_ot (rho, Pi, weights=weight)
        t0 = logger.timer (ot, 'effective potential kernel calculation', *t0)
        veff1 += ot.get_veff_1body (rho, Pi, ao, weight, non0tab=mask, shls_slice=shls_slice, ao_loc=ao_loc, hermi=1, kern=vrho)
        t0 = logger.timer (ot, '1-body effective potential calculation', *t0)
        #ao[:,:,:] = np.tensordot (ao, mo_coeff, axes=1)
        #t0 = logger.timer (ot, 'ao2mo grid points', *t0)
        veff2._accumulate (ot, rho, Pi, ao, weight, rho_c, rho_a, vPi, mask, shls_slice, ao_loc)
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
        Pi = get_ontop_pair_density (ot, rho, ao, oneCDMs, twoCDM_amo, ao2amo, dens_deriv, mask)
        t0 = logger.timer (ot, 'on-top pair density calculation', *t0)
        eot, vrho, vPi = ot.eval_ot (rho, Pi, weights=weight)
        t0 = logger.timer (ot, 'effective potential kernel calculation', *t0)
        veff1 += ot.get_veff_1body (rho, Pi, ao, weight, kern=vrho)
        t0 = logger.timer (ot, '1-body effective potential calculation', *t0)
        veff2 += ot.get_veff_2body (rho, Pi, ao, weight, aosym='s4', kern=vPi)
        t0 = logger.timer (ot, '2-body effective potential calculation', *t0)
    return veff1, veff2

def get_veff_1body (otfnal, rho, Pi, ao, weight, kern=None, non0tab=None, shls_slice=None, ao_loc=None, hermi=0, **kwargs):
    r''' get the derivatives dEot / dDpq
    Can also be abused to get semidiagonal dEot / dPppqq if you pass the right kern and
    squared aos/mos

    Args:
        rho : ndarray of shape (2,*,ngrids)
            containing spin-density [and derivatives]
        Pi : ndarray with shape (*,ngrids)
            containing on-top pair density [and derivatives]
        ao : ndarray or 2 ndarrays of shape (*,ngrids,nao)
            contains values and derivatives of nao.
            2 different ndarrays can have different nao but not different ngrids
        weight : ndarray of shape (ngrids)
            containing numerical integration weights

    Kwargs:
        kern : ndarray of shape (*,ngrids)
            the derivative of the on-top potential with respect to density (vrho)
            If not provided, it is calculated.
        non0tab : ndarray of shape (nblk, nbas)
            Identifies blocks of grid points which are nonzero on
            each AO shell so as to exploit sparsity.
            If you want the "ao" array to be in the MO basis, just
            leave this as None. If hermi == 0, it only applies
            to the bra index ao array, even if the ket index ao
            array is the same (so probably always pass hermi = 1
            in that case)
        shls_slice : sequence of integers of len 2
            Identifies starting and stopping indices of AO shells
        ao_loc : ndarray of length nbas
            Offset to first AO of each shell
        hermi : integer or logical
            Toggle whether veff is supposed to be a Hermitian matrix
            You can still pass two different ao arrays for the bra and the 
            ket indices, for instance if one of them is supposed to be
            a higher derivative. They just have to have the same nao
            in that case.

    Returns : ndarray of shape (nao[0],nao[1])
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
    elif ao[0].size < ao[1].size:
        # Life pro-tip: do more operations with smaller arrays and fewer operations with bigger arrays
        ao = [ao[1], ao[0]]
    if kern is None: 
        kern = otfnal.get_dEot_drho (rho, Pi, **kwargs) 
    else:
        kern = kern.copy () 
    nderiv = kern.shape[0]
    kern *= weight[None,:]

    # Zeroth and first derivatives
    vao = _contract_vot_ao (kern, ao[1])
    nterm = vao.shape[0]
    veff = sum ([_dot_ao_mo (otfnal.mol, a, v, non0tab=non0tab, shls_slice=shls_slice, ao_loc=ao_loc, hermi=hermi)
        for a, v in zip (ao[0][0:nterm], vao)])
    # veff = sum ([np.dot (a.T, v) for a, v in zip (ao[0][0:nterm], vao)])
    # ^ Crazy as it sounds, this sum over list comprehension is by far the fastest
    # implementation of this step that I've managed to pull off so far!
    # It's probably because the operation in the list is so well-vectorized.

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
    # Put in column-major order so reshape doesn't make a copy and screw everything up
    ao2 = ao2.transpose (0,3,2,1)
    vao = vao.transpose (0,3,2,1)
    ijkl_shape = list (ao2.shape[1:-1]) + list (vao.shape[1:-1])
    ao2 = ao2.reshape (ao2.shape[0], -1, ao2.shape[-1]).transpose (0,2,1)
    vao = vao.reshape (vao.shape[0], -1, vao.shape[-1]).transpose (0,2,1)
    veff = sum ([_dot_ao_mo (otfnal.mol, a, v) for a, v in zip (ao2, vao)])
    veff = veff.reshape (*ijkl_shape) 

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
    ao1 = ao1.transpose (0,2,1)
    ao2 = ao2.transpose (0,2,1)
    assert (ao1.flags.c_contiguous), 'shape = {} ; strides = {}'.format (ao1.shape, ao1.strides)
    assert (ao2.flags.c_contiguous), 'shape = {} ; strides = {}'.format (ao2.shape, ao2.strides)
    if symm:
        ix_p, ix_q = np.tril_indices (ao1.shape[1])
        ao1 = ao1[:nderiv,ix_p]
        ao2 = ao2[:nderiv,ix_q]
    else:
        ao1 = np.expand_dims (ao1, -2)[:nderiv]
        ao2 = np.expand_dims (ao2, -3)[:nderiv]
    prod = ao1[:nderiv] * ao2[0]
    if nderiv > 1:
        prod[1:4] += ao1[0] * ao2[1:4] # Product rule
    ao2 = None
    if symm:
        prod = prod.transpose (0,2,1)
    else:
        prod = prod.transpose (0,3,2,1)
    return prod 

def _contract_vot_ao (vot, ao, out=None):
    ''' REQUIRES array in shape = (nderiv,nao,ngrids) and data layout = (nderiv,ngrids,nao)/row-major '''
    nderiv = vot.shape[0]
    ao = np.ascontiguousarray (ao.transpose (0,2,1))
    #vao = (ao[0][None,:,:] * vot[:,None,:]).transpose (0,2,1)
    #ao = ao.transpose (0,2,1)
    #if nderiv > 1:
        #vao[0] += (ao[1:4] * vot[1:4,:,None]).sum (0)
        #vao[0] += numint._scale_ao (ao[1:4], vot[1:4])
    nao, ngrids = ao.shape[1:]
    vao = np.ndarray ((nderiv, nao, ngrids), dtype=ao.dtype, buffer=out).transpose (0,2,1)
    ao = ao.transpose (0,2,1)
    vao[0] = numint._scale_ao (ao[:nderiv], vot, out=vao[0])
    if nderiv > 1: 
        for i in range (1,4):
            vao[i] = numint._scale_ao (ao[0:1,:,:], vot[i:i+1,:], out=vao[i])
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
    ao = ao.transpose (0,2,1)
    vao = vao.transpose (0,2,1)
    assert (ao.flags.c_contiguous), 'shape = {} ; strides = {}'.format (ao.shape, ao.strides)
    assert (vao.flags.c_contiguous), 'shape = {} ; strides = {}'.format (vao.shape, vao.strides)

    nderiv = vao.shape[0]
    if symm:
        ix_p, ix_q = np.tril_indices (ao.shape[1])
        ao = ao[:,ix_p]
        vao = vao[:,ix_q]
    else:
        ao = np.expand_dims (ao, -2)
        vao = np.expand_dims (vao, -3)
    prod = ao[0] * vao
    if nderiv > 1:
        prod[0] += (ao[1:4] * vao[1:4]).sum (0)
    if symm:
        prod = prod.transpose (0,2,1)
    else:
        prod = prod.transpose (0,3,2,1)
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

def _dot_ao_mo (mol, ao, mo, non0tab=None, shls_slice=None, ao_loc=None, hermi=0):
    if hermi:
        #print ("_dot_ao_mo punting to numint._dot_ao_ao")
        return numint._dot_ao_ao (mol, ao, mo, non0tab=non0tab, shls_slice=shls_slice, ao_loc=ao_loc, hermi=hermi)
    ngrids, nao = ao.shape
    nmo = mo.shape[-1]

    if nao < SWITCH_SIZE:
        #print ("_dot_ao_mo punting to lib.dot")
        return lib.dot(ao.T.conj(), mo)

    #print ("_dot_ao_mo doing my own thing")

    if not ao.flags.f_contiguous:
        ao = lib.transpose(ao)
    if not mo.flags.f_contiguous:
        mo = lib.transpose(mo)
    if ao.dtype == mo.dtype == np.double:
        fn = libpdft.VOTdot_ao_mo
    else:
        raise NotImplementedError ("Complex-orbital PDFT")

    if non0tab is None or shls_slice is None or ao_loc is None:
        pnon0tab = pshls_slice = pao_loc = lib.c_null_ptr()
    else:
        pnon0tab    = non0tab.ctypes.data_as(ctypes.c_void_p)
        pshls_slice = (ctypes.c_int*2)(*shls_slice)
        pao_loc     = ao_loc.ctypes.data_as(ctypes.c_void_p)

    vv = np.empty((nao,nmo), dtype=ao.dtype)
    fn(vv.ctypes.data_as(ctypes.c_void_p),
       ao.ctypes.data_as(ctypes.c_void_p),
       mo.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int(nao), ctypes.c_int (nmo),
       ctypes.c_int(ngrids), ctypes.c_int(mol.nbas),
       pnon0tab, pshls_slice, pao_loc)
    return vv

    
