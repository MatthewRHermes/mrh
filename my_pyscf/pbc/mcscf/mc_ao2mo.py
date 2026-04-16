import numpy as np
from functools import reduce
import itertools
from pyscf import lib
from pyscf.ao2mo import _ao2mo
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.df import df_ao2mo
from pyscf.pbc.df.df import _load3c

_mo_as_complex = df_ao2mo._mo_as_complex
_conc_mos = df_ao2mo._conc_mos
logger = lib.logger

# The 2e integrals transformation to MO basis for the orbital optimization.mk

def get_nauxlist(mydf, kpts, nkpts):
    nauxlist = {}
    for k1 in range(nkpts):
        for k2 in range(nkpts):
            kpti_kptj = np.vstack((kpts[k1], kpts[k2]))
            assert kpti_kptj.shape == (2, 3)
            with _load3c(mydf._cderi, mydf._dataname, kpti_kptj) as j3c:
                nauxlist[(k1,k2)] = j3c.shape[0]
    return nauxlist

def _do_ao2mo_direct(kcasscf, mo_kpts, nkpts, ncore, ncas, nmo, level=1):
    cell = kcasscf._scf.cell
    kpts = kcasscf._scf.kpts
    mydf = kcasscf._scf.with_df
    nocc = ncore + ncas
    dtype = mo_kpts[0].dtype
    assert len(mo_kpts) == nkpts
    log = lib.logger.Logger(kcasscf.stdout, kcasscf.verbose)
    t1 = t0 = (logger.process_clock(), logger.perf_counter())

    ppaa = np.empty((nkpts, nkpts, nkpts, nmo, nmo, ncas, ncas), dtype=dtype)
    papa = np.empty((nkpts, nkpts, nkpts, nmo, ncas, nmo, ncas), dtype=dtype)
    paap = np.empty((nkpts, nkpts, nkpts, nmo, ncas, ncas, nmo), dtype=dtype)
    paap_ppmm = np.empty((nkpts, nkpts, nkpts, nmo, ncas, ncas, nmo), dtype=dtype)

    kconserv = kpts_helper.get_kconserv(cell, kpts)
    for k1, k2, k3 in kpts_helper.loop_kkk(nkpts):
        k4 = kconserv[k1, k2, k3]
        ppaa[k1, k2, k3] = mydf.ao2mo([mo_kpts[k1], mo_kpts[k2], mo_kpts[k3][:, ncore:nocc], mo_kpts[k4][:, ncore:nocc]],
                          [kpts[i] for i in (k1, k2, k3, k4)], 
                          compact=False).reshape(nmo, nmo, ncas, ncas)
    t1 = log.timer('density fitting ao2mo ppaa', *t0)

    for k1, k2, k3 in kpts_helper.loop_kkk(nkpts):
        k4 = kconserv[k1, k2, k3]
        papa[k1, k2, k3] = mydf.ao2mo([mo_kpts[k1], mo_kpts[k2][:, ncore:nocc], mo_kpts[k3], mo_kpts[k4][:, ncore:nocc]],
                        [kpts[i] for i in (k1, k2, k3, k4)],
                        compact=False).reshape(nmo, ncas, nmo, ncas)
    t2 = log.timer('density fitting ao2mo papa', *t1)

    for k1, k2, k3 in kpts_helper.loop_kkk(nkpts):
        k4 = kconserv[k1, k2, k3]
        paap[k1, k2, k3] = mydf.ao2mo([mo_kpts[k1], mo_kpts[k2][:, ncore:nocc], mo_kpts[k3][:, ncore:nocc], mo_kpts[k4]],
                        [kpts[i] for i in (k1, k2, k3, k4)],
                        compact=False).reshape(nmo, ncas, ncas, nmo)
    t3 = log.timer('density fitting ao2mo paap', *t2)

    for kp, kw, kx in kpts_helper.loop_kkk(nkpts):
        kq = kconserv[kp, kx, kw]
        buf = mydf.ao2mo([mo_kpts[kp], mo_kpts[kx][:, ncore:nocc], 
                                     mo_kpts[kw][:, ncore:nocc], mo_kpts[kq]],
                                     [kpts[i] for i in (kp, kx, kw, kq)], compact=False).reshape(nmo, ncas, ncas, nmo)
        paap_ppmm[kp, kw, kx] = buf.transpose(0, 2, 1, 3)
    
    buf = None

    t4 = log.timer('density fitting ao2mo paap_ppmm', *t3)

    # This is very naive implementation, would require a lot of optimization.
    if level == 1:
        if ncore == 0:
            j_pc = np.zeros((nkpts, nmo, ncore), dtype=dtype)
            k_pc = np.zeros((nkpts, nmo, ncore), dtype=dtype)
        else:
            j_pc = np.empty((nkpts, nmo, ncore), dtype=dtype)
            k_pc = np.empty((nkpts, nmo, ncore), dtype=dtype)
            for k in range(nkpts):
                mo_ppaa = [mo_kpts[k], mo_kpts[k], mo_kpts[k][:, :ncore], mo_kpts[k][:, :ncore]]
                temp = mydf.ao2mo(mo_ppaa, [kpts[k]]*4, compact=False).reshape(nmo, nmo, ncore, ncore)
                j_pc[k] = np.einsum('ppjj->pj', temp)
                mo_papa = [mo_kpts[k], mo_kpts[k][:, :ncore], mo_kpts[k], mo_kpts[k][:, :ncore]]
                temp = mydf.ao2mo(mo_papa, [kpts[k]]*4, compact=False).reshape(nmo, ncore, nmo, ncore)
                k_pc[k] = np.einsum('pjpj->pj', temp)
    else:
        j_pc = k_pc = None
    log.timer('density fitting ao2mo j_pc, k_pc', *t2)
    return ppaa, papa, paap, paap_ppmm, j_pc, k_pc

def _do_ao2mo_disk(kcasscf, mo_kpts, nkpts, ncore, ncas, nmo, level=1):
    cell = kcasscf._scf.cell
    kpts = kcasscf._scf.kpts
    nkpts = kcasscf.nkpts
    mydf = kcasscf._scf.with_df
    nocc = ncore + ncas
    dtype = mo_kpts[0].dtype

    assert len(mo_kpts) == nkpts

    # Note: There are two h5py files created, one temp file which hold intermediates for the ppaa and papa integrals, 
    # and another which will store those integrals. The first file will be deleted after these integrals
    # are constructed, and only the second file will be used for accessing the integrals.
    erifile = lib.H5TmpFile()
    erifile.require_group("ppaa")
    erifile.require_group("papa")
    erifile.require_group("paap")
    erifile.require_group("paap_ppmm")

    t1 = t0 = (logger.process_clock(), logger.perf_counter())
    log = lib.logger.Logger(kcasscf.stdout, kcasscf.verbose)
    
    mem_now = lib.current_memory()[0]
    
    # I am not sure wheather the naoaux will be same for all k1, k2 pairs. So I am taking the maximum naoaux among all pairs.
    # naoaux = mydf.get_naoaux()
    nauxlist = get_nauxlist(mydf, kpts, nkpts)
    naoaux = max(nauxlist.values())

    # Note the datatype is complex double, so 16 bytes.
    mem_required = naoaux * nmo * nmo * 16 / 1e6
    if mem_now < 2.0 * mem_required:
        raise MemoryError(f"Not enough memory for intermediate arrays for ao2mo transformation. \
                          Required: {mem_required} MB, Current: {mem_now} MB.")
    
    # For complex integrals: I am using compact=False as of now to avoid any conj/sign issues.
    # However, I haven't encountered the code in PySCF which uses the compact=True option for complex integrals.
    # TODO: Explore the compact option for the complex integrals.
    compact = False 

    # Intermediate temp file
    fxpp = lib.H5TmpFile()
    grp = fxpp.require_group("xpp")
    grp2 = fxpp.require_group("xpp_sign")
    
    # Step-1: Compute the Lpq integrals and save them on disk. 
    # We will read them later to construct the ppaa and papa integrals.
    # for k1 in range(nkpts):
    #     for k2 in range(nkpts):
    for k1, k2 in itertools.product(range(nkpts), repeat=2):
        mo_coeffs_tmp = _mo_as_complex([mo_kpts[k1], mo_kpts[k2]])
        moij, ijslice = _conc_mos(mo_coeffs_tmp[0], mo_coeffs_tmp[1])[2:]
        kptij = np.vstack((kpts[k1], kpts[k2]))
        naux = nauxlist[(k1, k2)]
        zij = None
        
        # Looping over blockdim.
        dset = grp.create_dataset(f"{k1}_{k2}", shape=(naux, nmo, nmo), dtype=np.complex128)
        p0 = 0
        last_sign = None
        for LpqR, LpqI, sign in mydf.sr_loop(kptij, mem_now, compact):
            nblk = LpqR.shape[0]
            assert nblk == LpqI.shape[0]
            tao = []
            ao_loc = None
            zij_blk = _ao2mo.r_e2(LpqR + 1j * LpqI, moij, ijslice, tao, ao_loc, out=zij)
            dset[p0:p0+nblk] = zij_blk.reshape(nblk, nmo, nmo)
            p0 += nblk
            last_sign = sign
            # TODO: Learn whether I should store the 2D or 3D array for better I/O performance.
            #grp.create_dataset(f"{k1}_{k2}", data=zij.reshape(naux, nmo, nmo))
        grp2.create_dataset(f"{k1}_{k2}", data=last_sign)

    t1 = log.timer('density fitting ao2mo Lpq', *t0)
    # TODO: use the blksize to loop over the nmo to reduce the memory footprint.           
    # Step-2: Construct the papa integrals:
    kconserv = kpts_helper.get_kconserv(cell, kpts)
    # for k1 in range(nkpts):
    #     for k2 in range(nkpts):
    #         for k3 in range(nkpts):
    for k1, k2, k3 in kpts_helper.loop_kkk(nkpts):
        k4 = kconserv[k1, k2, k3]
        papa = np.zeros((nmo*ncas, nmo*ncas), dtype=dtype)
        zij_12 = grp[f"{k1}_{k2}"][:, :, ncore:ncore+ncas][()]
        zkl_34 = grp[f"{k3}_{k4}"][:, :, ncore:ncore+ncas][()]
        zij_12 = zij_12.reshape(-1, nmo*ncas)
        zkl_34 = zkl_34.reshape(-1, nmo*ncas)
        sign = grp2[f"{k1}_{k2}"][()]
        lib.dot(zij_12.T, zkl_34, sign, papa, 1)
        erifile[f"papa/{k1}_{k2}_{k3}"] = papa.reshape(nmo, ncas, nmo, ncas)

    papa = zij_12 = zkl_34 = None
    t2 = log.timer('density fitting ao2mo papa', *t1)

    # Step-3: Construct the ppaa integrals:
    for k1, k2, k3 in kpts_helper.loop_kkk(nkpts):
        k4 = kconserv[k1, k2, k3]
        ppaa = np.zeros((nmo*nmo, ncas*ncas), dtype=dtype)
        zij_12 = grp[f"{k1}_{k2}"][()]
        zkl_34 = grp[f"{k3}_{k4}"][:, ncore:ncore+ncas, ncore:ncore+ncas][()]
        zij_12 = zij_12.reshape(-1, nmo*nmo)
        zkl_34 = zkl_34.reshape(-1, ncas*ncas)
        sign = grp2[f"{k1}_{k2}"][()]
        lib.dot(zij_12.T, zkl_34, sign, ppaa, 1)
        erifile[f"ppaa/{k1}_{k2}_{k3}"] = ppaa.reshape(nmo, nmo, ncas, ncas)

    ppaa = zij_12 = zkl_34 = None
    t2 = log.timer('density fitting ao2mo ppaa', *t2)

    # Step-4: Construct the paap integrals
    for k1, k2, k3 in kpts_helper.loop_kkk(nkpts):
        k4 = kconserv[k1, k2, k3]
        paap = np.zeros((nmo*ncas, nmo*ncas), dtype=dtype)
        zij_12 = grp[f"{k1}_{k2}"][:, :, ncore:ncore+ncas][()] # pa
        zkl_34 = grp[f"{k3}_{k4}"][:, ncore:ncore+ncas, :][()] # ap
        zij_12 = zij_12.reshape(-1, nmo*ncas)
        zkl_34 = zkl_34.reshape(-1, ncas*nmo)
        sign = grp2[f"{k1}_{k2}"][()]
        lib.dot(zij_12.T, zkl_34, sign, paap, 1)
        erifile[f"paap/{k1}_{k2}_{k3}"] = paap.reshape(nmo, ncas, ncas, nmo)
    paap = zij_12 = zkl_34 = None
    t3 = log.timer('density fitting ao2mo paap', *t2)

    # Step-5: Construct the paap_ppmm integrals:
    for k1, k2, k3 in kpts_helper.loop_kkk(nkpts):
        k4 = kconserv[k1, k3, k2]
        paap_ppmm = np.zeros((nmo*ncas, nmo*ncas), dtype=dtype)
        zij_12 = grp[f"{k1}_{k3}"][:, :, ncore:ncore+ncas][()] # pa
        zkl_34 = grp[f"{k2}_{k4}"][:, ncore:ncore+ncas, :][()] # ap
        zij_12 = zij_12.reshape(-1, nmo*ncas)
        zkl_34 = zkl_34.reshape(-1, ncas*nmo)
        sign = grp2[f"{k1}_{k3}"][()]
        lib.dot(zij_12.T, zkl_34, sign, paap_ppmm, 1)
        erifile[f"paap_ppmm/{k1}_{k2}_{k3}"] = paap_ppmm.reshape(nmo, ncas, ncas, nmo).transpose(0, 2, 1, 3)
    paap_ppmm = zij_12 = zkl_34 = None
    t3 = log.timer('density fitting ao2mo paap_ppmm', *t2)

    if level == 1:
        j_pc_kpts = np.zeros((nkpts, nmo, ncore), dtype=dtype)
        k_pc_kpts = np.zeros((nkpts, nmo, ncore), dtype=dtype)
        for k in range(nkpts):
            zij = grp[f"{k}_{k}"][()]
            bufd = np.einsum('pii->pi', zij)
            j_pc_kpts[k] = np.einsum('pi,pj->ij', bufd, bufd[:,:ncore])
            k_cp = np.einsum('kij,kij->ij', zij[:,:ncore], zij[:,:ncore])
            k_pc_kpts[k] = k_cp.conj().T
        
        bufd = zij =  k_cp = None
    else:
        j_pc_kpts = None
        k_pc_kpts = None
    log.timer('density fitting ao2mo j_pc and k_pc', *t3)
    fxpp.close()
    return erifile, j_pc_kpts, k_pc_kpts

def _mem_usage(nkpts, ncore, ncas, nmo):
    basic = nkpts**3 * nmo**2 * ncas**2 * 16 / 1e6
    incore = basic + nkpts**3 * (ncore+ncas) * nmo**3 * 16 / 1e6
    return incore
        
class _ERIS:
    '''
    AO2MO transformation of the 2e integrals for the orbital optimization step.
    Args:
        kcasscf: instance of pbc.mcscf.CASSCF
            The K-CASSCF object.
        mo_kpts: list of numpy arrays 
            The MO coefficients for each k-point.
        method: str (direct or disk) (Default is 'direct')
            The method for the 2e integrals transformation. Basically, we require the ppaa and papa integrals, 
            each for nkpt^3 points. If method is 'direct', we will compute the required integrals on the fly. 
            If method is 'disk', we will save the required integrals on disk and read them when required.
        level: int (Default is 1)
            level-1: ppaa, papa, vhf, jpc and kpc
            level-2: Only ppaa, papa and vhf

    Saved Attributes:
        ppaa: np.array (nkpts, nkpts, nkpts, nmo, nmo, ncas, ncas)/ read from disk 
            It's a function that takes k1, k2, k3 as input and returns the ppaa integrals
        papa: np.array (nkpts, nkpts, nkpts, nmo, ncas, nmo, ncas)/ read from disk
            It's a function that takes k1, k2, k3 as input and returns the papa integrals
        j_pc: np.array (nkpts, nmo, ncore)
            potential due to core electrons
        k_pc: np.array (nkpts, nmo, ncore)
            potential due to core electrons
        vhf_c: np.array (nkpts, nmo, nmo)
            VHF matrix due to core electrons
    '''
    def __init__(self, kcasscf, mo_kpts, method='disk', level=1):
        self.erifile = None
        self.ppaa_kpts = None
        self.papa_kpts = None
        self.paap_kpts = None
        self.paap_ppmm_kpts = None

        log = lib.logger.Logger(kcasscf.stdout, kcasscf.verbose)
        cell = kcasscf._scf.cell
        kpts = kcasscf._scf.kpts
        ncore = kcasscf.ncore
        ncas = kcasscf.ncas
        nkpts = len(kpts)
        nao, nmo = mo_kpts[0].shape
        dtype = mo_kpts[0].dtype

        t1 = t0 = (logger.process_clock(), logger.perf_counter())
        dmcore_kpts = np.asarray(
            [2.0 * (mo_kpts[k][:, :ncore] @ mo_kpts[k][:, :ncore].conj().T) 
             for k in range(nkpts)], 
             dtype=dtype)
        
        vj_kpts, vk_kpts = kcasscf._scf.get_jk(cell, dmcore_kpts, kpts=kpts, hermi=1)
        self.vhf_c = np.array(
            [reduce(np.dot, (mo_kpts[k].conj().T, vj_kpts[k] - 0.5*vk_kpts[k], mo_kpts[k]))
             for k in range(nkpts)], 
             dtype=dtype)
        t1 = log.timer('vhf construction for core density', *t1)

        mem_incore = _mem_usage(nkpts, ncore, ncas, nmo)
        mem_now = lib.current_memory()[0]
        
        log.debug('Memory usage for incore ERI transformation: %.2f MB.Current memory usage: %.2f MB. Max memory: %.2f MB.',
              mem_incore, mem_now, kcasscf.max_memory)
        if (method == 'direct' and mem_now + mem_incore < 0.9 * kcasscf.max_memory):
            log.debug('Using direct ERI transformation.')
            self.ppaa_kpts, self.papa_kpts, self.paap_kpts, self.paap_ppmm_kpts, self.j_pc, self.k_pc = _do_ao2mo_direct(kcasscf, mo_kpts, nkpts, ncore, ncas, nmo, level=level)
            t1 = log.timer('direct ao2mo', *t1)
        else:
            log.debug('Using disk ERI transformation.')
            self.erifile, self.j_pc, self.k_pc = _do_ao2mo_disk(kcasscf, mo_kpts, nkpts, ncore, ncas, nmo, level=level)
            t1 = log.timer('disk ao2mo', *t1)
        # To access the ppaa and papa integrals: I am bifurcating the code based on whether 
        # we are using disk or direct method. If we are using direct method, 
        # we can directly access the ppaa and papa integrals from the attributes.
        # If we are using disk method, we need to read the integrals from the disk. 
        # To avoid writing separate code for accessing the integrals in different methods, 
        # I am defining two lambda functions that will access the integrals based on the method used.
        self.ppaa = lambda k1, k2, k3: self.get_ppaa(k1, k2, k3)
        self.papa = lambda k1, k2, k3: self.get_papa(k1, k2, k3)
        self.paap = lambda k1, k2, k3: self.get_paap(k1, k2, k3)
        self.paap_ppmm = lambda k1, k2, k3: self.get_paap_ppmm(k1, k2, k3)
        log.timer('Total ERI transformation', *t0)

    @staticmethod
    def _kkey(k1, k2, k3):
        return f"{int(k1)}_{int(k2)}_{int(k3)}"

    def _get(self, eriname, k1, k2, k3):
        # General wrapper
        arr = getattr(self, eriname + "_kpts", None)
        if arr is not None:return arr[k1, k2, k3]
        assert self.erifile is not None
        data = self.erifile[f"{eriname}/{self._kkey(k1, k2, k3)}"]
        return data[()]

    def get_ppaa(self, k1, k2, k3):
        return self._get("ppaa", k1, k2, k3)

    def get_papa(self, k1, k2, k3):
        return self._get("papa", k1, k2, k3)

    def get_paap(self, k1, k2, k3):
        return self._get("paap", k1, k2, k3)
    
    def get_paap_ppmm(self, k1, k2, k3):
        return self._get("paap_ppmm", k1, k2, k3)


if __name__ == "__main__":
    from pyscf.pbc import gto, scf
    from pyscf import lib
    # Timer level prints:
    lib.logger.TIMER_LEVEL = lib.logger.INFO

    cell = gto.Cell()
    cell.atom = '''C     0.      0.      0.    
              C     0.8917  0.8917  0.8917
              C     1.7834  1.7834  0.    
              C     2.6751  2.6751  0.8917
              C     1.7834  0.      1.7834
              C     2.6751  0.8917  2.6751
              C     0.      1.7834  1.7834
              C     0.8917  2.6751  2.6751'''
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.a = np.eye(3)*3.5668
    # cell.atom = 'C 0 0 0'
    # cell.basis = 'CC-PVDZ'
    cell.verbose = lib.logger.TIMER_LEVEL
    cell.build()

    t0 = (lib.logger.process_clock(), lib.logger.perf_counter())
    kmesh = [2, 2, 2]
    kpts = cell.make_kpts(kmesh)
    
    kmf = scf.KRHF(cell, kpts).density_fit()
    kmf.exxdiv = None
    kmf.max_cycle = 1
    kmf.kernel()
    log = lib.logger.Logger(kmf.stdout, kmf.verbose)
    log.timer("SCF calculation completed.", *t0)
    class _kCASSCF:
        # Dummy class to check the _ERIS class.
        def __init__(self, kmf, ncas, nelecas):
            self._scf = kmf
            self.ncas = ncas
            self.nelecas = nelecas
            self.ncore = int((kmf.cell.nelectron - nelecas) / 2)
            self.nkpts = len(kmf.kpts)
            self.stdout = kmf.stdout
            self.verbose = kmf.verbose
            self.max_memory = kmf.max_memory
    
    kmc = _kCASSCF(kmf, ncas=2, nelecas=2)
    mo_kpts = kmf.mo_coeff

    nmo = mo_kpts[0].shape[1]
    ncas = kmc.ncas
    ncore = kmc.ncore
    nkpts = kmc.nkpts

    eris = _ERIS(kmc, mo_kpts, method='disk', level=1)
    log.timer("Disk ERI transformation completed. ", *t0)

    eris2 = _ERIS(kmc, mo_kpts, method='direct', level=1)
    log.timer("Direct ERI transformation completed. ", *t0)

    def compare_integrals(arr1, arr2, name, shape):
        assert arr1.dtype == arr2.dtype
        assert arr1.shape == arr2.shape == shape
        assert np.allclose(arr1, arr2), f"{name} integrals do not match between direct and disk method."
        print(f"{name} integrals match between direct and disk method.")
        arr1 = arr2 = None

    # Compare the j_pc and k_pc integrals
    compare_integrals(eris.j_pc, eris2.j_pc, "j_pc", (nkpts, nmo, ncore))
    compare_integrals(eris.k_pc, eris2.k_pc, "k_pc", (nkpts, nmo, ncore))
    
    # Compare the vhf_c integrals
    compare_integrals(eris.vhf_c, eris2.vhf_c, "vhf_c", (nkpts, nmo, nmo))

    # Compare the ppaa, papa, and paap integrals
    # Gamma point (k1=k2=k3=0)
    compare_integrals(eris.ppaa(0, 0, 0), eris2.ppaa(0, 0, 0), "ppaa", (nmo, nmo, ncas, ncas))
    compare_integrals(eris.papa(0, 0, 0), eris2.papa(0, 0, 0), "papa", (nmo, ncas, nmo, ncas))
    compare_integrals(eris.paap(0, 0, 0), eris2.paap(0, 0, 0), "paap", (nmo, ncas, ncas, nmo))
    compare_integrals(eris.paap_ppmm(0, 0, 0), eris2.paap_ppmm(0, 0, 0), "paap_ppmm", (nmo, ncas, ncas, nmo))

    # Non-Gamma point (k1=0, k2=0, k3=1)
    compare_integrals(eris.ppaa(0, 0, 1), eris2.ppaa(0, 0, 1), "ppaa", (nmo, nmo, ncas, ncas))
    compare_integrals(eris.papa(0, 0, 1), eris2.papa(0, 0, 1), "papa", (nmo, ncas, nmo, ncas))
    compare_integrals(eris.paap(0, 0, 1), eris2.paap(0, 0, 1), "paap", (nmo, ncas, ncas, nmo))
    compare_integrals(eris.paap_ppmm(0, 0, 1), eris2.paap_ppmm(0, 0, 1), "paap_ppmm", (nmo, ncas, ncas, nmo))