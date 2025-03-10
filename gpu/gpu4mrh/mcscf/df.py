import ctypes
from functools import reduce
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from pyscf.mcscf.casci import CASCI
from pyscf import df
from gpu4mrh.lib.utils import patch_cpu_kernel
from mrh.my_pyscf.gpu import libgpu

class _ERIS:
    def __init__(self, casscf, mo, with_df):
        log = logger.Logger(casscf.stdout, casscf.verbose)
        mol = casscf.mol
        nao, nmo = mo.shape
        ncore = casscf.ncore
        ncas = casscf.ncas
        nocc = ncore + ncas
        naoaux = with_df.get_naoaux()
        mem_incore, mem_outcore, mem_basic = _mem_usage(ncore, ncas, nmo)
        mem_now = lib.current_memory()[0]
        max_memory = max(3000, casscf.max_memory*.9-mem_now)
        if max_memory < mem_basic:
            log.warn('Calculation needs %d MB memory, over CASSCF.max_memory (%d MB) limit',
                     (mem_basic+mem_now)/.9, casscf.max_memory)

        self.feri = lib.H5TmpFile()
        self.ppaa = self.feri.create_dataset('ppaa', (nmo,nmo,ncas,ncas), 'f8')
        self.papa = self.feri.create_dataset('papa', (nmo,ncas,nmo,ncas), 'f8')
        self.j_pc = numpy.zeros((nmo,ncore))
        self.k_pc = numpy.zeros((nmo,ncore))
        k_cp = numpy.zeros((ncore,nmo))
        gpu=casscf.mol.use_gpu
        
        mo = numpy.asarray(mo, order='F')
        #fxpp = lib.H5TmpFile()
        fxpp = numpy.empty ((nmo, nmo, naoaux))

#        blksize = max(4, int(min(with_df.blockdim, (max_memory*.95e6/8-naoaux*nmo*ncas)/3/nmo**2)))
        blksize = with_df.blockdim

        bufpa = numpy.empty((naoaux,nmo,ncas))
        t1 = t0 = (logger.process_clock(), logger.perf_counter())
        if gpu: 
            ppaa = numpy.empty((nmo, nmo, ncas, ncas))
            libgpu.libgpu_push_mo_coeff(gpu, mo, nao*nmo)
            libgpu.libgpu_init_jk_ao2mo(gpu, ncore, nmo) 
            libgpu.libgpu_init_ints_ao2mo_v3(gpu, naoaux, nmo, ncas) #initializes bufpa on pinned memory
            libgpu.libgpu_init_ppaa_ao2mo(gpu, nmo, ncas) #initializes ppaa on pinned memory
            t1 = log.timer('init_ao2mo', *t1)
            count = 0
            for k, eri1 in enumerate(with_df.loop(blksize)):
                pass;
            for count in range(k+1):
                arg = numpy.array([-1, -1, count, -1], dtype = numpy.int32)
                libgpu.libgpu_get_dfobj_status(gpu, id(with_df),arg)
                naux = arg[0]
                libgpu.libgpu_df_ao2mo_v3(gpu,blksize,nmo,nao,ncore,ncas,naux,eri1,count,id(with_df)) # includes pulling bufpa, ppaa to pinned memory. eri1 no longer used, can be removed in a later version. 
            t1 = log.timer('compute_ao2mo', *t1)
            libgpu.libgpu_pull_jk_ao2mo (gpu, self.j_pc, k_cp, nmo, ncore)
            libgpu.libgpu_pull_ints_ao2mo_v3(gpu, bufpa, blksize, naoaux, nmo, ncas)
            libgpu.libgpu_pull_ppaa_ao2mo(gpu, ppaa, nmo, ncas) #pull ppaa
            self.ppaa = ppaa
            #libgpu.libgpu_pull_ints_ao2mo(gpu, fxpp, bufpa, blksize, naoaux, nmo, ncas)
            t1 = log.timer('pull_ao2mo', *t1)
        else:
            fmmm = _ao2mo.libao2mo.AO2MOmmm_nr_s2_iltj
            fdrv = _ao2mo.libao2mo.AO2MOnr_e2_drv
            ftrans = _ao2mo.libao2mo.AO2MOtranse2_nr_s2
            fxpp_keys = []
            b0 = 0
            bufs1 = numpy.empty((blksize,nmo,nmo))
            for k, eri1 in enumerate(with_df.loop(blksize)):
                naux = eri1.shape[0]
                bufpp = bufs1[:naux]
                fdrv(ftrans, fmmm,
                 bufpp.ctypes.data_as(ctypes.c_void_p),
                 eri1.ctypes.data_as(ctypes.c_void_p),
                 mo.ctypes.data_as(ctypes.c_void_p),
                 ctypes.c_int(naux), ctypes.c_int(nao),
                 (ctypes.c_int*4)(0, nmo, 0, nmo),
                 ctypes.c_void_p(0), ctypes.c_int(0))
                fxpp_keys.append([k, b0, b0+naux])
                fxpp[:,:,b0:b0+naux] = bufpp.transpose(1,2,0)  
                bufpa[b0:b0+naux] = bufpp[:,:,ncore:nocc]
                bufd = numpy.einsum('kii->ki', bufpp)
                self.j_pc += numpy.einsum('ki,kj->ij', bufd, bufd[:,:ncore])
                k_cp += numpy.einsum('kij,kij->ij', bufpp[:,:ncore], bufpp[:,:ncore])
                b0 += naux
                t0 = log.timer('rest of the calculation', *t0)
            bufs1 = bufpp = None
            bufaa = bufpa[:,ncore:nocc,:].copy().reshape(-1,ncas**2)
            t1 = log.timer('density fitting ao2mo pass1', *t1)
            mem_now = lib.current_memory()[0]
            nblk = int(max(8, min(nmo, (max_memory-mem_now)*1e6/8/(nmo*naoaux+ncas**2*nmo))))
            bufs1 = numpy.empty((nblk,nmo,naoaux))
            bufs2 = numpy.empty((nblk,nmo,ncas,ncas))
            for p0, p1 in prange(0, nmo, nblk):
                nrow = p1 - p0
                buf = bufs1[:nrow]
                tmp = bufs2[:nrow].reshape(-1,ncas**2)
                for key, col0, col1 in fxpp_keys:
                    #buf[:nrow,:,col0:col1] = fxpp[key][p0:p1]
                    buf[:nrow,:,col0:col1] = fxpp[:,:,col0:col1][p0:p1]
                lib.dot(buf.reshape(-1,naoaux), bufaa, 1, tmp)
                self.ppaa[p0:p1] = tmp.reshape(p1-p0,nmo,ncas,ncas)
            bufs1 = bufs2 = buf = None
            t1 = log.timer('density fitting ppaa pass2', *t1)
            #fxpp.close()
            fxpp=None



                
        self.k_pc = k_cp.T.copy()
           
        mem_now = lib.current_memory()[0]
        nblk = int(max(8, min(nmo, ((max_memory-mem_now)*1e6/8-bufpa.size)/(ncas**2*nmo))))
        bufs1 = numpy.empty((nblk,ncas,nmo,ncas))
        dgemm = lib.numpy_helper._dgemm
        for p0, p1 in prange(0, nmo, nblk):
            tmp = numpy.dot(bufpa[:,p0:p1].reshape(naoaux,-1).T,
                            bufpa.reshape(naoaux,-1))
            #tmp = bufs1[:p1-p0]
            #dgemm('T', 'N', (p1-p0)*ncas, nmo*ncas, naoaux,
            #      bufpa.reshape(naoaux,-1), bufpa.reshape(naoaux,-1),
            #      tmp.reshape(-1,nmo*ncas), 1, 0, p0*ncas, 0, 0)
            self.papa[p0:p1] = tmp.reshape(p1-p0,ncas,nmo,ncas)
        t1 = log.timer('density fitting papa pass2', *t1)
        
        #bufaa = bufpa[:,ncore:nocc,:].copy().reshape(-1,ncas**2)
        bufs1 = bufpa = None
        ####   ####    Removing the ppaa calculation for cpu and moved it up to the branching####    ####
        #mem_now = lib.current_memory()[0]
        #nblk = int(max(8, min(nmo, (max_memory-mem_now)*1e6/8/(nmo*naoaux+ncas**2*nmo))))
        #bufs1 = numpy.empty((nblk,nmo,naoaux))
        #bufs2 = numpy.empty((nblk,nmo,ncas,ncas))
        #for p0, p1 in prange(0, nmo, nblk):
        #    nrow = p1 - p0
        #    buf = bufs1[:nrow]
        #    tmp = bufs2[:nrow].reshape(-1,ncas**2)
        #    for key, col0, col1 in fxpp_keys:
        #        #buf[:nrow,:,col0:col1] = fxpp[key][p0:p1]
        #        buf[:nrow,:,col0:col1] = fxpp[:,:,col0:col1][p0:p1]
        #    lib.dot(buf.reshape(-1,naoaux), bufaa, 1, tmp)
        #    self.ppaa[p0:p1] = tmp.reshape(p1-p0,nmo,ncas,ncas)
        #bufs1 = bufs2 = buf = None

        self.feri.flush()

        dm_core = numpy.dot(mo[:,:ncore], mo[:,:ncore].T)
        vj, vk = casscf.get_jk(mol, dm_core)
        self.vhf_c = reduce(numpy.dot, (mo.T, vj*2-vk, mo))
        t0 = log.timer('density fitting ao2mo', *t0)

    def __del__(self):
        self.feri.close()

def _mem_usage(ncore, ncas, nmo):
    outcore = basic = ncas**2*nmo**2*2 * 8/1e6
    incore = outcore + (ncore+ncas)*nmo**3*4/1e6
    return incore, outcore, basic

def prange(start, end, step):
    for i in range(start, end, step):
        yield i, min(i+step, end)

