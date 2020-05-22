#!/usr/bin/env python
#
# This code was copied from the data generation program of Tencent Alchemy
# project (https://github.com/tencent-alchemy).
#
# 
# #
# # Copyright 2019 Tencent America LLC. All Rights Reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# #
# # Author: Qiming Sun <osirpt.sun@gmail.com>
# #

import time
import numpy
import ctypes
import scipy.linalg
from pyscf import gto
from pyscf import lib
from pyscf import scf
from pyscf import df
from pyscf.ao2mo.outcore import balance_partition
from pyscf.gto.moleintor import getints, make_cintopt
from pyscf.lib import logger
from pyscf.grad import rhf as rhf_grad
from functools import reduce
from itertools import product
from pyscf.ao2mo import _ao2mo

# MRH 05/03/2020 
# There are three problems with get_jk from the perspective of generalizing
# for usage in DF-CASSCF and its children's analytical gradients:
#   1. orbo = c.n^1/2 doesn't work with non-positive-semidefinite density
#      matrices, which appear in DF-SA-CASSCF and DF-PDFT analytical gradients.
#      The solution to this is simple, though: split it into orbor = c.n^1 and
#      orbol = c[:,|n|>0]. The extra memory cost is trivial compared to the
#      size of the ERIs. This does require care around the rhok_oo lines, but
#      it should work so long as I remember to transpose where necessary. 
#   2. For the auxbasis response, the actual contribution to the gradient
#      is returned instead of vj' and vk'. This is understandable, since
#      the auxbasis response parts have size = 3*natm*nao^2, a factor of natm
#      larger than the size of the AO basis response parts. However, CASSCF
#      and its children have asymmetric Coulomb and exchange contributions to
#      the energy; i.e., E += vk2.D1 = g.(D1xD2)^T with D1 =/= D2. Even the cumulant
#      decomposition doesn't save you from this because once you get to SA-CASSCF,
#      you have the orbrot lagrange multipliers which contract to exactly one 
#      orbital index on the 2RDM. The solution to this is to change the logic.
#      The new kwarg ishf defaults to the original behavior, but if
#      switched to False, it causes vj.aux and vk.aux to return as arrays of shape
#      (nset,nset,3,nao,nao) containing terms involving all pairs of dms.
#      The caller can figure things out from there.
#   3. It just assumes that mf_grad.base.mo_coeff and mf_grad.base.mo_occ are
#      the eigendecomposition! I have to make it actually do the eigendecomposition
#      for me!

def get_jk(mf_grad, mol=None, dm=None, hermi=0, with_j=True, with_k=True, ishf=True):
    t0 = (time.clock (), time.time ())
    if mol is None: mol = mf_grad.mol
    if dm is None: dm = mf_grad.base.make_rdm1()
    #TODO: dm has to be the SCF density matrix in this version.  dm should be
    # extended to any 1-particle density matrix
    #dm = mf_grad.base.make_rdm1()

    with_df = mf_grad.base.with_df
    auxmol = with_df.auxmol
    if auxmol is None:
        auxmol = df.addons.make_auxmol(with_df.mol, with_df.auxbasis)
    pmol = mol + auxmol
    ao_loc = mol.ao_loc
    nbas = mol.nbas
    nauxbas = auxmol.nbas

    get_int3c_s1 = _int3c_wrapper(mol, auxmol, 'int3c2e', 's1')
    get_int3c_s2 = _int3c_wrapper(mol, auxmol, 'int3c2e', 's2ij')
    get_int3c_ip1 = _int3c_wrapper(mol, auxmol, 'int3c2e_ip1', 's1')
    get_int3c_ip2 = _int3c_wrapper(mol, auxmol, 'int3c2e_ip2', 's2ij')

    nao = mol.nao
    naux = auxmol.nao
    dms = numpy.asarray(dm)
    out_shape = dms.shape[:-2] + (3,) + dms.shape[-2:]
    dms = dms.reshape(-1,nao,nao)
    nset = dms.shape[0]

    idx = numpy.arange(nao)
    idx = idx * (idx+1) // 2 + idx
    dm_tril = dms + dms.transpose(0,2,1)
    dm_tril = lib.pack_tril(dm_tril)
    dm_tril[:,idx] *= .5

    auxslices = auxmol.aoslice_by_atom()
    aux_loc = auxmol.ao_loc
    max_memory = mf_grad.max_memory - lib.current_memory()[0]
    blksize = int(min(max(max_memory * .5e6/8 / (nao**2*3), 20), naux, 240))
    ao_ranges = balance_partition(aux_loc, blksize)

    if not with_k:

        # (i,j|P)
        rhoj = numpy.empty((nset,naux))
        for shl0, shl1, nL in ao_ranges:
            int3c = get_int3c_s2((0, nbas, 0, nbas, shl0, shl1))  # (i,j|P)
            p0, p1 = aux_loc[shl0], aux_loc[shl1]
            rhoj[:,p0:p1] = lib.einsum('wp,nw->np', int3c, dm_tril)
            int3c = None

        # (P|Q)
        int2c = auxmol.intor('int2c2e', aosym='s1')
        rhoj = scipy.linalg.solve(int2c, rhoj.T, sym_pos=True).T
        int2c = None

        # (d/dX i,j|P)
        vj = numpy.zeros((nset,3,nao,nao))
        for shl0, shl1, nL in ao_ranges:
            int3c = get_int3c_ip1((0, nbas, 0, nbas, shl0, shl1))  # (i,j|P)
            p0, p1 = aux_loc[shl0], aux_loc[shl1]
            vj += lib.einsum('xijp,np->nxij', int3c, rhoj[:,p0:p1])
            int3c = None

        if mf_grad.auxbasis_response:
            # (i,j|d/dX P)
            vjaux = numpy.empty((nset,nset,3,naux))
            for shl0, shl1, nL in ao_ranges:
                int3c = get_int3c_ip2((0, nbas, 0, nbas, shl0, shl1))  # (i,j|P)
                p0, p1 = aux_loc[shl0], aux_loc[shl1]
                vjaux[:,:,:,p0:p1] = lib.einsum('xwp,mw,np->mnxp',
                                              int3c, dm_tril, rhoj[:,p0:p1])
                int3c = None

            # (d/dX P|Q)
            int2c_e1 = auxmol.intor('int2c2e_ip1', aosym='s1')
            vjaux -= lib.einsum('xpq,mp,nq->mnxp', int2c_e1, rhoj, rhoj)

            vjaux = numpy.array ([-vjaux[:,:,:,p0:p1].sum(axis=3) for p0, p1 in auxslices[:,2:]])
            if ishf:
                vjaux = vjaux.sum ((1,2))
            else:
                vjaux = numpy.ascontiguousarray (vjaux.transpose (1,2,0,3))
            vj = lib.tag_array(-vj.reshape(out_shape), aux=numpy.array(vjaux))
        else:
            vj = -vj.reshape(out_shape)
        logger.timer(mf_grad, 'df vj', *t0)
        return vj, None

    # MRH 05/03/2020: uh-oh, we can't have this! I guess I have to use an ishf
    # kwarg here too!
    if ishf:
        mo_coeff = mf_grad.base.mo_coeff
        mo_occ = mf_grad.base.mo_occ
    else:
        s0 = mol.intor ('int1e_ovlp')
        mo_occ = []
        mo_coeff = []
        for dm in dms:
            sdms = reduce (lib.dot, (s0, dm, s0))
            n, c = scipy.linalg.eigh (sdms, b=s0)
            mo_occ.append (n)
            mo_coeff.append (c)
        mo_occ = numpy.stack (mo_occ, axis=0)
    nmo = mo_occ.shape[-1]
    if isinstance(mf_grad.base, scf.rohf.ROHF) and ishf:
        mo_coeff = numpy.vstack((mo_coeff,mo_coeff))
        mo_occa = numpy.array(mo_occ> 0, dtype=numpy.double)
        mo_occb = numpy.array(mo_occ==2, dtype=numpy.double)
        assert(mo_occa.sum() + mo_occb.sum() == mo_occ.sum())
        mo_occ = numpy.vstack((mo_occa, mo_occb))

    mo_coeff = numpy.asarray(mo_coeff).reshape(-1,nao,nmo)
    mo_occ   = numpy.asarray(mo_occ).reshape(-1,nmo)
    rhoj = numpy.zeros((nset,naux))
    f_rhok = lib.H5TmpFile()
    orbor = []
    orbol = []
    nocc = []
    orbor_stack = numpy.zeros ((nao,0), dtype=mo_coeff.dtype, order='F')
    orbol_stack = numpy.zeros ((nao,0), dtype=mo_coeff.dtype, order='F')
    # MRH 05/03/2020: how do I deal with nocc? It looks like the only time
    # this is referenced is when computing blksize below. I think I can
    # replace it with the maximum nocc for all nset dms.
    offs = 0
    for i in range(nset):
        idx = numpy.abs (mo_occ[i])>1e-8
        nocc.append (numpy.count_nonzero (idx))
        c = mo_coeff[i][:,idx]
        orbol_stack = numpy.append (orbol_stack, c, axis=1)
        orbol.append (orbol_stack[:,offs:offs+nocc[-1]])
        cn = lib.einsum('pi,i->pi', c, mo_occ[i][idx])
        orbor_stack = numpy.append (orbor_stack, cn, axis=1)
        orbor.append (orbor_stack[:,offs:offs+nocc[-1]])
        offs += nocc[-1]

    # (P|Q)
    int2c = scipy.linalg.cho_factor(auxmol.intor('int2c2e', aosym='s1'))

    t1 = (time.clock (), time.time ())
    max_memory = mf_grad.max_memory - lib.current_memory()[0]
    blksize = max_memory * .5e6/8 / (naux*nao)
    mol_ao_ranges = balance_partition(ao_loc, blksize)
    nsteps = len(mol_ao_ranges)
    t2 = t1
    for istep, (shl0, shl1, nd) in enumerate(mol_ao_ranges):
        int3c = get_int3c_s1((0, nbas, shl0, shl1, 0, nauxbas))
        t2 = logger.timer (mf_grad, 'df grad intor (P|ij)', *t2)
        p0, p1 = ao_loc[shl0], ao_loc[shl1]
        for i in range(nset):
            # MRH 05/21/2020: De-vectorize this because array contiguity -> parallel scaling
            v = lib.dot(int3c.reshape (nao, -1, order='F').T, orbor[i]).reshape (naux, -1)
            t2 = logger.timer (mf_grad, 'df grad einsum (P|ik) D_jk', *t2)
            rhoj[i] += numpy.dot (v, orbol[i][p0:p1].ravel ())
            t2 = logger.timer (mf_grad, 'df grad einsum (P|ij) D_ij = rho_P', *t2)
            v = scipy.linalg.cho_solve(int2c, v)
            t2 = logger.timer (mf_grad, 'df grad cho_solve (P|Q) D_Qij = (P|ik) D_jk', *t2)
            f_rhok['%s/%s'%(i,istep)] = v.reshape(naux,p1-p0,-1)
            t2 = logger.timer (mf_grad, 'df grad cache D_Qij', *t2)
        int3c = v = None

    rhoj = scipy.linalg.cho_solve(int2c, rhoj.T).T
    int2c = None
    t1 = logger.timer (mf_grad, 'df grad vj and vk AO (P|Q) D_Q = (P|ij) D_ij solve', *t1)
    # MRH 05/20/2020: Per tests, this block has the worst parallel scaling in the function
    # and accounts for the largest portion of its wall time

    def load(set_id, p0, p1):
        buf = numpy.empty((p1-p0,nocc[set_id],nao))
        col1 = 0
        for istep in range(nsteps):
            dat = f_rhok['%s/%s'%(set_id,istep)][p0:p1]
            col0, col1 = col1, col1 + dat.shape[1]
            buf[:p1-p0,:,col0:col1] = dat.transpose(0,2,1)
        return buf

    vj = numpy.zeros((nset,3,nao,nao))
    vk = numpy.zeros((nset,3,nao,nao))
    # (d/dX i,j|P)
    fmmm = _ao2mo.libao2mo.AO2MOmmm_bra_nr_s1 # MO output index slower than AO output index; input AOs are asymmetric
    fdrv = _ao2mo.libao2mo.AO2MOnr_e2_drv # comp and aux indices are slower
    ftrans = _ao2mo.libao2mo.AO2MOtranse2_nr_s1 # input is not tril_packed
    null = lib.c_null_ptr() 
    t2 = t1
    for shl0, shl1, nL in ao_ranges:
        int3c = get_int3c_ip1((0, nbas, 0, nbas, shl0, shl1)).transpose (0,3,2,1)  # (P|ij'), row-major order
        t2 = logger.timer (mf_grad, "df grad intor (P|i'j)", *t2)
        p0, p1 = aux_loc[shl0], aux_loc[shl1]
        #vj += lib.einsum('xpji,np->nxij', int3c, rhoj[:,p0:p1])
        for i in range(nset):
            # MRH 05/21/2020: De-vectorize this because array contiguity -> parallel scaling
            vj[i,0] += numpy.dot (rhoj[i,p0:p1], int3c[0].reshape (p1-p0, -1)).reshape (nao, nao).T
            vj[i,1] += numpy.dot (rhoj[i,p0:p1], int3c[1].reshape (p1-p0, -1)).reshape (nao, nao).T
            vj[i,2] += numpy.dot (rhoj[i,p0:p1], int3c[2].reshape (p1-p0, -1)).reshape (nao, nao).T
            t2 = logger.timer (mf_grad, "df grad einsum (P|i'j) rho_P", *t2)
            tmp = numpy.empty ((3,p1-p0,nocc[i],nao), dtype=orbol_stack.dtype) 
            fdrv(ftrans, fmmm, # xPmn u_mi -> xPin
                 tmp.ctypes.data_as(ctypes.c_void_p),
                 int3c.ctypes.data_as(ctypes.c_void_p),
                 orbol[i].ctypes.data_as(ctypes.c_void_p),
                 ctypes.c_int (3*(p1-p0)), ctypes.c_int (nao),
                 (ctypes.c_int*4)(0, nocc[i], 0, nao),
                 null, ctypes.c_int(0))
            t2 = logger.timer (mf_grad, "df grad einsum (P|i'k) D_jk = v_Pij", *t2)
            rhok = load(i, p0, p1)
            vk[i] += lib.einsum('xpoi,pok->xik', tmp, rhok)
            # MRH 05/21/2020: TODO: double-check the parallel scalability of this einsum.
            # It was somehow pretty good when the einsum string read 'xipo,pok->xik', even
            # though that's not remotely what the data layout was supposed to be.
            t2 = logger.timer (mf_grad, "df grad einsum D_Pik v_Pjk = v_ij", *t2)
            rhok = tmp = None
        int3c = None
    t1 = logger.timer (mf_grad, 'df grad vj and vk AO (P|ij) D_P eval', *t1)
    # MRH 05/20/2020: Per tests, this block has bad parallel scaling and accounts
    # for the third-largest portion of its wall time

    # MRH 05/03/2020: moved the rhok_oo stuff inside of the mf_grad.auxbasis_response 
    # conditional because it's never used outside of that
    if mf_grad.auxbasis_response:
        # Cache (P|uv) D_ui c_vj. Must be include both upper and lower triangles
        # over nset.
        max_memory = mf_grad.max_memory - lib.current_memory()[0]
        blksize = int(min(max(max_memory * .5e6/8 / (nao*max (nocc)), 20), naux))
        rhok_oo = []
        for i, j in product (range (nset), repeat=2):
            tmp = numpy.empty ((naux,nocc[i],nocc[j]))
            for p0, p1 in lib.prange(0, naux, blksize):
                rhok = load(i, p0, p1).reshape ((p1-p0)*nocc[i], nao)
                tmp[p0:p1] = lib.dot (rhok, orbol[j]).reshape (p1-p0, nocc[i], nocc[j])
            rhok_oo.append(tmp)
            rhok = tmp = None
        t1 = logger.timer (mf_grad, 'df grad vj and vk aux (P|Q) d_Qij = (P|kl) D_ki D_lj solve', *t1)
        # MRH 05/20/2020: Per tests, this block has very good parallel scaling

        vjaux = numpy.zeros((nset,nset,3,naux))
        vkaux = numpy.zeros((nset,nset,3,naux))
        # (i,j|d/dX P)
        t2 = t1
        fmmm = _ao2mo.libao2mo.AO2MOmmm_bra_nr_s2 # MO output index slower than AO output index; input AOs are symmetric
        fdrv = _ao2mo.libao2mo.AO2MOnr_e2_drv # comp and aux indices are slower
        ftrans = _ao2mo.libao2mo.AO2MOtranse2_nr_s2 # input is tril_packed
        null = lib.c_null_ptr() 
        for shl0, shl1, nL in ao_ranges:
            int3c = get_int3c_ip2((0, nbas, 0, nbas, shl0, shl1))  # (i,j|P)
            t2 = logger.timer (mf_grad, "df grad intor (P'|ij)", *t2)
            p0, p1 = aux_loc[shl0], aux_loc[shl1]
            drhoj = lib.dot (int3c.transpose (0,2,1).reshape (3*(p1-p0), -1),
                dm_tril.T).reshape (3, p1-p0, -1) # xpij,mij->xpm
            vjaux[:,:,:,p0:p1] = lib.einsum ('xpm,np->mnxp', drhoj, rhoj[:,p0:p1])
            t2 = logger.timer (mf_grad, "df grad einsum (P'|ij) D_ij rho_P", *t2)
            tmp = [numpy.empty ((3, p1-p0, nocc_i, nao), dtype=orbor_stack.dtype) for nocc_i in nocc]
            assert (orbor_stack.flags.f_contiguous), '{} {}'.format (orbor_stack.shape, orbor_stack.strides)
            for orb, buf, nocc_i in zip (orbol, tmp, nocc):
                fdrv(ftrans, fmmm, # gPmn u_ni -> gPim
                     buf.ctypes.data_as(ctypes.c_void_p),
                     int3c.ctypes.data_as(ctypes.c_void_p),
                     orb.ctypes.data_as(ctypes.c_void_p),
                     ctypes.c_int (3*(p1-p0)), ctypes.c_int (nao),
                     (ctypes.c_int*4)(0, nocc_i, 0, nao),
                     null, ctypes.c_int(0))
            int3c = [[lib.dot (buf.reshape (-1, nao), orb).reshape (3, p1-p0, -1, norb)
                for orb, norb in zip (orbor, nocc)] for buf in tmp] # pim,mj,j -> pij
            t2 = logger.timer (mf_grad, "df grad einsum D_Pij u_im u_in = D_Pmn", *t2)
            for i, j in product (range (nset), repeat=2):
                k = (i*nset) + j
                tmp = rhok_oo[k][p0:p1]
                vkaux[i,j,:,p0:p1] += lib.einsum('xpij,pij->xp', int3c[i][j], tmp)
                t2 = logger.timer (mf_grad, "df grad einsum (P'|mn) D_Pmn = v", *t2)
        int3c = tmp = None
        t1 = logger.timer (mf_grad, "df grad vj and vk aux (P'|ij) eval", *t1)

        # (d/dX P|Q)
        int2c_e1 = auxmol.intor('int2c2e_ip1')
        vjaux -= lib.einsum('xpq,mp,nq->mnxp', int2c_e1, rhoj, rhoj)
        for i, j in product (range (nset), repeat=2):
            # MRH 05/03/2020: Transpose so as to leave only one factor
            # of a density matrix on both indices!
            k = (i*nset) + j
            l = (j*nset) + i
            # MRH 05/19/2020: rhok_oo is C-contiguous but you have to
            # transpose its two fastest indices on one factor. Probably
            # nothing can be done about it.
            tmp = lib.einsum('pij,qji->pq', rhok_oo[k], rhok_oo[l])
            vkaux[i,j] -= lib.einsum('xpq,pq->xp', int2c_e1, tmp)
        t1 = logger.timer (mf_grad, "df grad vj and vk aux (P'|Q) eval", *t1)
        # MRH 05/20/2020: Per tests, this block has very good parallel scaling

        vjaux = numpy.array ([-vjaux[:,:,:,p0:p1].sum(axis=3) for p0, p1 in auxslices[:,2:]])
        vkaux = numpy.array ([-vkaux[:,:,:,p0:p1].sum(axis=3) for p0, p1 in auxslices[:,2:]])
        if ishf:
            vjaux = vjaux.sum ((1,2))
            idx = numpy.array (list (range (nset))) * (nset + 1)
            vkaux = vkaux.reshape ((nset**2,3,mol.natm))[idx,:,:].sum (0)
        else:
            vjaux = numpy.ascontiguousarray (vjaux.transpose (1,2,0,3))
            vkaux = numpy.ascontiguousarray (vkaux.transpose (1,2,0,3))
        vj = lib.tag_array(-vj.reshape(out_shape), aux=numpy.array(vjaux))
        vk = lib.tag_array(-vk.reshape(out_shape), aux=numpy.array(vkaux))
    else:
        vj = -vj.reshape(out_shape)
        vk = -vk.reshape(out_shape)
    logger.timer(mf_grad, 'df grad vj and vk', *t0)
    return vj, vk

def _int3c_wrapper(mol, auxmol, intor, aosym):
    nbas = mol.nbas
    pmol = mol + auxmol
    intor = mol._add_suffix(intor)
    opt = make_cintopt(mol._atm, mol._bas, mol._env, intor)
    def get_int3c(shls_slice=None):
        if shls_slice is None:
            shls_slice = (0, nbas, 0, nbas, nbas, pmol.nbas)
        else:
            shls_slice = shls_slice[:4] + (nbas+shls_slice[4], nbas+shls_slice[5])
        return getints(intor, pmol._atm, pmol._bas, pmol._env, shls_slice,
                       aosym=aosym, cintopt=opt)
    return get_int3c


class Gradients(rhf_grad.Gradients):
    '''Restricted density-fitting Hartree-Fock gradients'''
    def __init__(self, mf):
        # Whether to include the response of DF auxiliary basis when computing
        # nuclear gradients of J/K matrices
        self.auxbasis_response = True
        rhf_grad.Gradients.__init__(self, mf)

    get_jk = get_jk

    def get_j(self, mol=None, dm=None, hermi=0):
        return self.get_jk(mol, dm, with_k=False)[0]

    def get_k(self, mol=None, dm=None, hermi=0):
        return self.get_jk(mol, dm, with_j=False)[1]

    def get_veff(self, mol=None, dm=None):
        vj, vk = self.get_jk(mol, dm)
        vhf = vj - vk*.5
        if self.auxbasis_response:
            e1_aux = vj.aux - vk.aux*.5
            logger.debug1(self, 'sum(auxbasis response) %s', e1_aux.sum(axis=0))
            vhf = lib.tag_array(vhf, aux=e1_aux)
        return vhf

    def extra_force(self, atom_id, envs):
        if self.auxbasis_response:
            return envs['vhf'].aux[atom_id]
        else:
            return 0

Grad = Gradients

def monkeypatch_setup ():
    from pyscf.df.grad import rhf
    true_Gradients = rhf.Gradients
    rhf.Gradients = Gradients
    def monkeypatch_teardown ():
        rhf.Gradients = true_Gradients
    return monkeypatch_teardown

if __name__ == '__main__':
    from pyscf import scf
    mol = gto.Mole()
    mol.atom = [
        ['O' , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ]
    mol.basis = '631g'
    mol.build()
    mf = scf.RHF(mol).density_fit(auxbasis='ccpvdz-jkfit').run()
    g = Gradients(mf).set(auxbasis_response=not False).kernel()
    print(lib.finger(g) - 0.0055166381900824879)
    g = Gradients(mf).kernel()
    print(lib.finger(g) - 0.005516638190173352)
    print(abs(g-scf.RHF(mol).run().Gradients().kernel()).max())
# -0.0000000000    -0.0000000000    -0.0241140368
#  0.0000000000     0.0043935801     0.0120570184
#  0.0000000000    -0.0043935801     0.0120570184

    mfs = mf.as_scanner()
    e1 = mfs([['O' , (0. , 0.     , 0.001)],
              [1   , (0. , -0.757 , 0.587)],
              [1   , (0. , 0.757  , 0.587)] ])
    e2 = mfs([['O' , (0. , 0.     ,-0.001)],
              [1   , (0. , -0.757 , 0.587)],
              [1   , (0. , 0.757  , 0.587)] ])
    print((e1-e2)/0.002*lib.param.BOHR)
