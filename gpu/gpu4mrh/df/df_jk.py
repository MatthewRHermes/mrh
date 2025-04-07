#!/usr/bin/env python

import copy

import ctypes
import numpy
import scipy.linalg
from pyscf import lib
from pyscf import scf
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from gpu4mrh.lib.utils import patch_cpu_kernel

from mrh.my_pyscf.gpu import libgpu

# Setting DEBUG = True will execute both CPU (original) and GPU (new) paths checking for consistency 
DEBUG = False

if DEBUG:
    import math
    import traceback
    import sys

def get_jk(dfobj, dm, hermi=1, with_j=True, with_k=True, direct_scf_tol=1e-13):
    gpu = dfobj.mol.use_gpu
    
    assert (with_j or with_k)
    if (not with_k and not dfobj.mol.incore_anyway and
        # 3-center integral tensor is not initialized
        dfobj._cderi is None):
        return get_j(dfobj, dm, hermi, direct_scf_tol), None

    t0 = t1 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(dfobj.stdout, dfobj.verbose)
    fmmm = _ao2mo.libao2mo.AO2MOmmm_bra_nr_s2
    fdrv = _ao2mo.libao2mo.AO2MOnr_e2_drv
    ftrans = _ao2mo.libao2mo.AO2MOtranse2_nr_s2
    null = lib.c_null_ptr()

    dms = numpy.asarray(dm)
    dm_shape = dms.shape
    nao = dm_shape[-1]
    dms = dms.reshape(-1,nao,nao)
    nset = dms.shape[0]
#    vj = 0
    vk = numpy.zeros(dms.shape)

    #t2 = lib.logger.timer(dfobj, 'get_jk setup', *t0)
    blksize = dfobj.blockdim # hard-coded to constant size for now
    
    if with_j:
        #t2 = (logger.process_clock(), logger.perf_counter())
        idx = numpy.arange(nao)
        dmtril = lib.pack_tril(dms + dms.conj().transpose(0,2,1))
        dmtril[:,idx*(idx+1)//2+idx] *= .5
        #t3 = lib.logger.timer(dfobj, 'get_jk with_j',*t2)

        vj = numpy.zeros_like(dmtril)
    else:
        dmtril = numpy.zeros((nset, int(nao*(nao+1)/2))) # shouldn't have to do this; bug in libgpu side
        vj = numpy.zeros((nset, int(nao*(nao+1)/2))) # shouldn't have to do this; bug in libgpu side
        
    dms = [numpy.asarray(x, order='F') for x in dms]
    
    if not with_k:
        #t2 = (logger.process_clock(), logger.perf_counter())

#        dms = [numpy.asarray(x, order='F') for x in dms]
        count = 0
        for eri1 in dfobj.loop():
            #t6 = (logger.process_clock(), logger.perf_counter())
            naux, nao_pair = eri1.shape
            if gpu:
                #if count == 0:
                libgpu.libgpu_init_get_jk(gpu, eri1, dmtril, blksize, nset, nao, 0, count)
                libgpu.libgpu_compute_get_jk(gpu, naux, nao, nset, eri1, dmtril, dms, vj, vk, 0, count, id(dfobj))
            else:
                rho = numpy.einsum('ix,px->ip', dmtril, eri1)
                vj += numpy.einsum('ip,px->ix', rho, eri1)

            #lib.logger.timer(dfobj, 'get_jk not with_k loop iteration',*t6)

            count += 1

        if gpu:
            libgpu.libgpu_pull_get_jk(gpu, vj, vk, nao, nset, 0)
        #t3 = lib.logger.timer(dfobj, 'get_jk not with_k loop full',*t2)

# Commented 2-19-2024 in favor of accelerated implementation below
# Can offload this if need arises.
#    elif getattr(dm, 'mo_coeff', None) is not None:
#        #TODO: test whether dm.mo_coeff matching dm
#        mo_coeff = numpy.asarray(dm.mo_coeff, order='F')
#        mo_occ   = numpy.asarray(dm.mo_occ)
#        nmo = mo_occ.shape[-1]
#        mo_coeff = mo_coeff.reshape(-1,nao,nmo)
#        mo_occ   = mo_occ.reshape(-1,nmo)
#        if mo_occ.shape[0] * 2 == nset: # handle ROHF DM
#            mo_coeff = numpy.vstack((mo_coeff, mo_coeff))
#            mo_occa = numpy.array(mo_occ> 0, dtype=numpy.double)
#            mo_occb = numpy.array(mo_occ==2, dtype=numpy.double)
#            assert (mo_occa.sum() + mo_occb.sum() == mo_occ.sum())
#            mo_occ = numpy.vstack((mo_occa, mo_occb))
#
#        orbo = []
#        for k in range(nset):
#            c = numpy.einsum('pi,i->pi', mo_coeff[k][:,mo_occ[k]>0],
#                             numpy.sqrt(mo_occ[k][mo_occ[k]>0]))
#            orbo.append(numpy.asarray(c, order='F'))
#
#        max_memory = dfobj.max_memory - lib.current_memory()[0]
#        blksize = max(4, int(min(dfobj.blockdim, max_memory*.3e6/8/nao**2)))
#        buf = numpy.empty((blksize*nao,nao))
#        for eri1 in dfobj.loop(blksize):
#            naux, nao_pair = eri1.shape
#            assert (nao_pair == nao*(nao+1)//2)
#            if with_j:
#                rho = numpy.einsum('ix,px->ip', dmtril, eri1)
#                vj += numpy.einsum('ip,px->ix', rho, eri1)
#
#            for k in range(nset):
#                nocc = orbo[k].shape[1]
#                if nocc > 0:
#                    buf1 = buf[:naux*nocc]
#                    fdrv(ftrans, fmmm,
#                         buf1.ctypes.data_as(ctypes.c_void_p),
#                         eri1.ctypes.data_as(ctypes.c_void_p),
#                         orbo[k].ctypes.data_as(ctypes.c_void_p),
#                         ctypes.c_int(naux), ctypes.c_int(nao),
#                         (ctypes.c_int*4)(0, nocc, 0, nao),
#                         null, ctypes.c_int(0))
#                    vk[k] += lib.dot(buf1.T, buf1)
#            t1 = log.timer_debug1('jk', *t1)
    else:
        #:vk = numpy.einsum('pij,jk->pki', cderi, dm)
        #:vk = numpy.einsum('pki,pkj->ij', cderi, vk)
        
        #t2 = (logger.process_clock(), logger.perf_counter())
        rargs = (ctypes.c_int(nao), (ctypes.c_int*4)(0, nao, 0, nao),
                 null, ctypes.c_int(0))
#        dms = [numpy.asarray(x, order='F') for x in dms]
        max_memory = dfobj.max_memory - lib.current_memory()[0]
        #blksize = max(4, int(min(dfobj.blockdim, max_memory*.22e6/8/nao**2)))
#        blksize = dfobj.blockdim # hard-coded to constant size for now
        buf = numpy.empty((2,blksize,nao,nao))
        
        count = 0
        vj = numpy.zeros_like(dmtril) # redundant? 

        #t3 = lib.logger.timer(dfobj, 'get_jk with_k setup',*t2)

        load_eri = True

        # load_eri = False doesn't offer benefit unless deep-copies and so on. disable for now
#        if gpu:
#            arg = numpy.array([-1, -1, -1, -1], dtype=numpy.int32)
#            libgpu.libgpu_get_dfobj_status(gpu, id(dfobj), arg)
#            if arg[2] > -1: load_eri = False

        if load_eri:
        
            for eri1 in dfobj.loop(blksize): # how much time spent unnecessarily copying eri1 data?
                #t6 = (logger.process_clock(), logger.perf_counter())
                naux, nao_pair = eri1.shape

                if gpu:
                    #if count == 0:
                    libgpu.libgpu_init_get_jk(gpu, eri1, dmtril, blksize, nset, nao, naux, count)
                    libgpu.libgpu_compute_get_jk(gpu, naux, nao, nset, eri1, dmtril, dms, vj, vk, 1, count, id(dfobj))
                
                else:
                
                    if with_j:
                        rho = numpy.einsum('ix,px->ip', dmtril, eri1)
                        vj += numpy.einsum('ip,px->ix', rho, eri1)
                    
                    for k in range(nset):
                        buf1 = buf[0,:naux]
                        fdrv(ftrans, fmmm,
                             buf1.ctypes.data_as(ctypes.c_void_p),
                             eri1.ctypes.data_as(ctypes.c_void_p),
                             dms[k].ctypes.data_as(ctypes.c_void_p),
                             ctypes.c_int(naux), *rargs)
                        
                        buf2 = lib.unpack_tril(eri1, out=buf[1])
                        vk[k] += lib.dot(buf1.reshape(-1,nao).T, buf2.reshape(-1,nao))

                count+=1
                #lib.logger.timer(dfobj, 'get_jk with_k loop iteration',*t6)

        else:
            
            nblk = arg[2]
            for count in range( arg[2] ):
                #t6 = (logger.process_clock(), logger.perf_counter())
                arg = numpy.array([-1, -1, count, -1], dtype=numpy.int32)
                libgpu.libgpu_get_dfobj_status(gpu, id(dfobj), arg)
                naux = arg[0]
                nao_pair = arg[1]

                eri1 = numpy.zeros(1)
                if count == 0: libgpu.libgpu_init_get_jk(gpu, eri1, dmtril, blksize, nset, nao, naux, count)
                libgpu.libgpu_compute_get_jk(gpu, naux, nao, nset, eri1, dmtril, dms, vj, vk, 1, count, id(dfobj))
                
                #lib.logger.timer(dfobj, 'get_jk with_k loop iteration',*t6)
                
        #t4 = lib.logger.timer(dfobj, 'get_jk with_k loop full',*t3)
        t1 = log.timer_debug1('jk', *t1)

        if gpu:
            libgpu.libgpu_pull_get_jk(gpu, vj, vk, nao, nset, 1)
        #t5 = lib.logger.timer(dfobj, 'get_jk with_k pull',*t4)
        
    #t2 = (logger.process_clock(), logger.perf_counter())
    if with_j: vj = lib.unpack_tril(vj, 1).reshape(dm_shape)
    if with_k: vk = vk.reshape(dm_shape)
    #lib.logger.timer(dfobj, 'get_jk finalize',*t2)
    logger.timer(dfobj, 'df vj and vk', *t0)
    return vj, vk

def get_jk_debug(dfobj, dm, hermi=1, with_j=True, with_k=True, direct_scf_tol=1e-13):
    ''' Function that runs get_jk with both cpu and gpu. Checks if sum of square of difference of all elements is below threshold [(vj_cpu-vj_gpu)*(vj_cpu-vj_gpu)]  '''
    #traceback.print_stack(file=sys.stdout)
    gpu = dfobj.mol.use_gpu
    
    assert (with_j or with_k)
    if (not with_k and not dfobj.mol.incore_anyway and
        # 3-center integral tensor is not initialized
        dfobj._cderi is None):
        return get_j(dfobj, dm, hermi, direct_scf_tol), None

    t0 = t1 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(dfobj.stdout, dfobj.verbose)
    fmmm = _ao2mo.libao2mo.AO2MOmmm_bra_nr_s2
    fdrv = _ao2mo.libao2mo.AO2MOnr_e2_drv
    ftrans = _ao2mo.libao2mo.AO2MOtranse2_nr_s2
    null = lib.c_null_ptr()

    dms = numpy.asarray(dm)
    dm_shape = dms.shape
    nao = dm_shape[-1]
    dms = dms.reshape(-1,nao,nao)
    nset = dms.shape[0]
    vk = numpy.zeros(dms.shape)

    blksize = dfobj.blockdim # hard-coded to constant size for now
            
    if with_j:
        idx = numpy.arange(nao)
        dmtril = lib.pack_tril(dms + dms.conj().transpose(0,2,1))
        dmtril[:,idx*(idx+1)//2+idx] *= .5
    
        vj = numpy.zeros_like(dmtril)
    else:
        dmtril = numpy.zeros((nset, int(nao*(nao+1)/2))) # shouldn't have to do this; bug in libgpu side
        vj = numpy.zeros((nset, int(nao*(nao+1)/2))) # shouldn't have to do this; bug in libgpu side
        
    dms = [numpy.asarray(x, order='F') for x in dms]
    
    if not with_k:

        vj_tmp = numpy.zeros_like(vj)
        vk_tmp = numpy.zeros_like(vk)
        
#        dms = [numpy.asarray(x, order='F') for x in dms]
        
        count = 0
        for eri1 in dfobj.loop():
            naux, nao_pair = eri1.shape
            
            print("count= ", count, "nao= ", nao, " naux= ", naux, "  nao_pair= ", nao_pair, " blksize= ", 0, " nset= ", nset, " eri1= ", eri1.shape, " dmtril= ", dmtril.shape, " dms= ", numpy.shape(dms))
            print("vj= ", vj_tmp.shape)
            print("addr of dfobj= ", hex(id(dfobj)), "  addr of eri1= ", hex(id(eri1)), " count= ", count)
            
            #if count == 0:
            libgpu.libgpu_init_get_jk(gpu, eri1, dmtril, blksize, nset, nao, 0, count)
            libgpu.libgpu_compute_get_jk(gpu, naux, nao, nset, eri1, dmtril, dms, vj_tmp, vk_tmp, 0, count, id(dfobj))
            
            rho = numpy.einsum('ix,px->ip', dmtril, eri1)
            vj += numpy.einsum('ip,px->ix', rho, eri1)

            count += 1

            
        libgpu.libgpu_pull_get_jk(gpu, vj_tmp, vk_tmp, nao, nset, 0)
        
        print("vj= ", vj.shape)
        vj_err = 0.0
        for i in range(nset):
            for j in range(vj.shape[1]):
                vj_err += (vj[i,j] - vj_tmp[i,j]) * (vj[i,j] - vj_tmp[i,j])
                #print("ij= ", i, j, "  vj= ", vj[i,j], "  vj_tmp= ", vj_tmp[i,j], "  vj_err= ", vj_err)

        stop = False
        if(vj_err > 1e-8): stop = True
        
        vj_err = "{:e}".format( math.sqrt(vj_err) )
        print("count= ", count, "  vj_err= ", vj_err)

        if stop:
            print("ERROR:: Results don't agree!!")
            quit()

# Commented 2-19-2024 in favor of accelerated implementation below
# Can offload this if need arises.
#    elif getattr(dm, 'mo_coeff', None) is not None:
#        #TODO: test whether dm.mo_coeff matching dm
#        mo_coeff = numpy.asarray(dm.mo_coeff, order='F')
#        mo_occ   = numpy.asarray(dm.mo_occ)
#        nmo = mo_occ.shape[-1]
#        mo_coeff = mo_coeff.reshape(-1,nao,nmo)
#        mo_occ   = mo_occ.reshape(-1,nmo)
#        if mo_occ.shape[0] * 2 == nset: # handle ROHF DM
#            mo_coeff = numpy.vstack((mo_coeff, mo_coeff))
#            mo_occa = numpy.array(mo_occ> 0, dtype=numpy.double)
#            mo_occb = numpy.array(mo_occ==2, dtype=numpy.double)
#            assert (mo_occa.sum() + mo_occb.sum() == mo_occ.sum())
#            mo_occ = numpy.vstack((mo_occa, mo_occb))
#
#        orbo = []
#        for k in range(nset):
#            c = numpy.einsum('pi,i->pi', mo_coeff[k][:,mo_occ[k]>0],
#                             numpy.sqrt(mo_occ[k][mo_occ[k]>0]))
#            orbo.append(numpy.asarray(c, order='F'))
#
#        max_memory = dfobj.max_memory - lib.current_memory()[0]
#        blksize = max(4, int(min(dfobj.blockdim, max_memory*.3e6/8/nao**2)))
#        buf = numpy.empty((blksize*nao,nao))
#        for eri1 in dfobj.loop(blksize):
#            naux, nao_pair = eri1.shape
#            assert (nao_pair == nao*(nao+1)//2)
#            if with_j:
#                rho = numpy.einsum('ix,px->ip', dmtril, eri1)
#                vj += numpy.einsum('ip,px->ix', rho, eri1)
#
#            for k in range(nset):
#                nocc = orbo[k].shape[1]
#                if nocc > 0:
#                    buf1 = buf[:naux*nocc]
#                    fdrv(ftrans, fmmm,
#                         buf1.ctypes.data_as(ctypes.c_void_p),
#                         eri1.ctypes.data_as(ctypes.c_void_p),
#                         orbo[k].ctypes.data_as(ctypes.c_void_p),
#                         ctypes.c_int(naux), ctypes.c_int(nao),
#                         (ctypes.c_int*4)(0, nocc, 0, nao),
#                         null, ctypes.c_int(0))
#                    vk[k] += lib.dot(buf1.T, buf1)
#            t1 = log.timer_debug1('jk', *t1)
    else:
#        print(" -- -- Inside else branch inside mrh/gpu/gpu4mrh/df/df_jk.py::get_jk()")
        #:vk = numpy.einsum('pij,jk->pki', cderi, dm)
        #:vk = numpy.einsum('pki,pkj->ij', cderi, vk)
        rargs = (ctypes.c_int(nao), (ctypes.c_int*4)(0, nao, 0, nao),
                 null, ctypes.c_int(0))
#        dms = [numpy.asarray(x, order='F') for x in dms]
        max_memory = dfobj.max_memory - lib.current_memory()[0]
#        blksize = max(4, int(min(dfobj.blockdim, max_memory*.22e6/8/nao**2)))
#        blksize = dfobj.blockdim # hard-coded to constant size for now
#        print(" dfobj.blockdim= ", dfobj.blockdim, "  max_memory*.22e6/8/nao**2= ", max_memory*.22e6/8/nao**2, " blksize= ", blksize)
        buf = numpy.empty((2,blksize,nao,nao))
        
        print(" -- -- -- blksize= ", blksize, " blockdim= ", dfobj.blockdim, "  nao= ", nao)
        
        count = 0

        vj_tmp = numpy.zeros_like(vj)
        vk_tmp = numpy.zeros_like(vk)

        for eri1 in dfobj.loop(blksize):
            naux, nao_pair = eri1.shape
            print("count= ", count, "nao= ", nao, " naux= ", naux, "  nao_pair= ", nao_pair, " blksize= ", blksize, " nset= ", nset, " eri1= ", eri1.shape, " dmtril= ", dmtril.shape, " dms= ", numpy.shape(dms))
            print("vj= ", vj_tmp.shape, " vk= ", vk_tmp.shape)
            print("addr of dfobj= ", hex(id(dfobj)), "  addr of eri1= ", hex(id(eri1)), " count= ", count)
            #if gpu:
            #if count == 0:
            libgpu.libgpu_init_get_jk(gpu, eri1, dmtril, blksize, nset, nao, naux, count)
            libgpu.libgpu_compute_get_jk(gpu, naux, nao, nset, eri1, dmtril, dms, vj_tmp, vk_tmp, 1, count, id(dfobj))
            if count == -1: quit()

            #else:
                
            if with_j:
                rho = numpy.einsum('ix,px->ip', dmtril, eri1)
                vj += numpy.einsum('ip,px->ix', rho, eri1)
                
            for k in range(nset):
                buf1 = buf[0,:naux]
                fdrv(ftrans, fmmm,
                     buf1.ctypes.data_as(ctypes.c_void_p),
                     eri1.ctypes.data_as(ctypes.c_void_p),
                     dms[k].ctypes.data_as(ctypes.c_void_p),
                     ctypes.c_int(naux), *rargs)
                
                buf2 = lib.unpack_tril(eri1, out=buf[1])
                                
                vk[k] += lib.dot(buf1.reshape(-1,nao).T, buf2.reshape(-1,nao))

            count+=1

        #if gpu:
        libgpu.libgpu_pull_get_jk(gpu, vj_tmp, vk_tmp, nao, nset, 1)
            
        print("vj= ", vj.shape, " vk= ", vk.shape)
        vj_err = 0.0
        for i in range(nset):
            for j in range(vj.shape[1]):
                vj_err += (vj[i,j] - vj_tmp[i,j]) * (vj[i,j] - vj_tmp[i,j])
                #print("ij= ", i, j, "  vj= ", vj[i,j], "  vj_tmp= ", vj_tmp[i,j], "  vj_err= ", vj_err)

        vk_err = 0.0
        for i in range(nset):
            for j in range(vk.shape[1]):
                for k in range(vk.shape[2]):
                    vk_err += (vk[i,j,k] - vk_tmp[i,j,k]) * (vk[i,j,k] - vk_tmp[i,j,k])                    
                    #print("ijk= ", i, j, k, "  vk= ", vk[i,j,k], "  vk_tmp= ", vk_tmp[i,j,k], "  vk_err= ", vk_err)

        stop = False
        if(vj_err > 1e-8): stop = True
        if(vk_err > 1e-8): stop = True
        
        vj_err = "{:e}".format( math.sqrt(vj_err) )
        vk_err = "{:e}".format( math.sqrt(vk_err) )
        print("count= ", count, "  vj_err= ", vj_err,"  vk_err= ", vk_err)

        if stop:
            print("ERROR:: Results don't agree!!")
            quit()
        
        t1 = log.timer_debug1('jk', *t1)
        
    if with_j: vj = lib.unpack_tril(vj, 1).reshape(dm_shape)
    if with_k: vk = vk.reshape(dm_shape)
    logger.timer(dfobj, 'df vj and vk', *t0)
    return vj, vk

def get_j(dfobj, dm, hermi=1, direct_scf_tol=1e-13):
    from pyscf.scf import _vhf
    from pyscf.scf import jk
    from pyscf.df import addons
    t0 = t1 = (logger.process_clock(), logger.perf_counter())

    mol = dfobj.mol
    if dfobj._vjopt is None:
        dfobj.auxmol = auxmol = addons.make_auxmol(mol, dfobj.auxbasis)
        opt = _vhf.VHFOpt(mol, 'int3c2e', 'CVHFnr3c2e_schwarz_cond')
        opt.direct_scf_tol = direct_scf_tol

        # q_cond part 1: the regular int2e (ij|ij) for mol's basis
        opt.init_cvhf_direct(mol, 'int2e', 'CVHFsetnr_direct_scf')
        mol_q_cond = lib.frompointer(opt._this.contents.q_cond, mol.nbas**2)

        # Update q_cond to include the 2e-integrals (auxmol|auxmol)
        j2c = auxmol.intor('int2c2e', hermi=1)
        j2c_diag = numpy.sqrt(abs(j2c.diagonal()))
        aux_loc = auxmol.ao_loc
        aux_q_cond = [j2c_diag[i0:i1].max()
                      for i0, i1 in zip(aux_loc[:-1], aux_loc[1:])]
        q_cond = numpy.hstack((mol_q_cond, aux_q_cond))
        fsetqcond = _vhf.libcvhf.CVHFset_q_cond
        fsetqcond(opt._this, q_cond.ctypes.data_as(ctypes.c_void_p),
                  ctypes.c_int(q_cond.size))

        try:
            opt.j2c = j2c = scipy.linalg.cho_factor(j2c, lower=True)
            opt.j2c_type = 'cd'
        except scipy.linalg.LinAlgError:
            opt.j2c = j2c
            opt.j2c_type = 'regular'

        # jk.get_jk function supports 4-index integrals. Use bas_placeholder
        # (l=0, nctr=1, 1 function) to hold the last index.
        bas_placeholder = numpy.array([0, 0, 1, 1, 0, 0, 0, 0],
                                      dtype=numpy.int32)
        fakemol = mol + auxmol
        fakemol._bas = numpy.vstack((fakemol._bas, bas_placeholder))
        opt.fakemol = fakemol
        dfobj._vjopt = opt
        t1 = logger.timer_debug1(dfobj, 'df-vj init_direct_scf', *t1)

    opt = dfobj._vjopt
    fakemol = opt.fakemol
    dm = numpy.asarray(dm, order='C')
    dm_shape = dm.shape
    nao = dm_shape[-1]
    dm = dm.reshape(-1,nao,nao)
    n_dm = dm.shape[0]

    # First compute the density in auxiliary basis
    # j3c = fauxe2(mol, auxmol)
    # jaux = numpy.einsum('ijk,ji->k', j3c, dm)
    # rho = numpy.linalg.solve(auxmol.intor('int2c2e'), jaux)
    nbas = mol.nbas
    nbas1 = mol.nbas + dfobj.auxmol.nbas
    shls_slice = (0, nbas, 0, nbas, nbas, nbas1, nbas1, nbas1+1)
    with lib.temporary_env(opt, prescreen='CVHFnr3c2e_vj_pass1_prescreen',
                           _dmcondname='CVHFsetnr_direct_scf_dm'):
        jaux = jk.get_jk(fakemol, dm, ['ijkl,ji->kl']*n_dm, 'int3c2e',
                         aosym='s2ij', hermi=0, shls_slice=shls_slice,
                         vhfopt=opt)
    # remove the index corresponding to bas_placeholder
    jaux = numpy.array(jaux)[:,:,0]
    t1 = logger.timer_debug1(dfobj, 'df-vj pass 1', *t1)

    if opt.j2c_type == 'cd':
        rho = scipy.linalg.cho_solve(opt.j2c, jaux.T)
    else:
        rho = scipy.linalg.solve(opt.j2c, jaux.T)
    # transform rho to shape (:,1,naux), to adapt to 3c2e integrals (ij|k)
    rho = rho.T[:,numpy.newaxis,:]
    t1 = logger.timer_debug1(dfobj, 'df-vj solve ', *t1)

    # Next compute the Coulomb matrix
    # j3c = fauxe2(mol, auxmol)
    # vj = numpy.einsum('ijk,k->ij', j3c, rho)
    with lib.temporary_env(opt, prescreen='CVHFnr3c2e_vj_pass2_prescreen',
                           _dmcondname=None):
        # CVHFnr3c2e_vj_pass2_prescreen requires custom dm_cond
        aux_loc = dfobj.auxmol.ao_loc
        dm_cond = [abs(rho[:,:,i0:i1]).max()
                   for i0, i1 in zip(aux_loc[:-1], aux_loc[1:])]
        dm_cond = numpy.array(dm_cond)
        fsetcond = _vhf.libcvhf.CVHFset_dm_cond
        fsetcond(opt._this, dm_cond.ctypes.data_as(ctypes.c_void_p),
                  ctypes.c_int(dm_cond.size))

        vj = jk.get_jk(fakemol, rho, ['ijkl,lk->ij']*n_dm, 'int3c2e',
                       aosym='s2ij', hermi=1, shls_slice=shls_slice,
                       vhfopt=opt)

    t1 = logger.timer_debug1(dfobj, 'df-vj pass 2', *t1)
    logger.timer(dfobj, 'df-vj', *t0)
    return numpy.asarray(vj).reshape(dm_shape)

def _get_jk(dfobj, dm, hermi=1, with_j=True, with_k=True, direct_scf_tol=1e-13):

    if DEBUG:
        vj, vk = get_jk_debug(dfobj, dm, hermi, with_j, with_k, direct_scf_tol)
    else: 
        vj, vk = get_jk(dfobj, dm, hermi, with_j, with_k, direct_scf_tol)

    return vj, vk
