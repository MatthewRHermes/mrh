#mini-app to debug/test the production GPU h2eff (get_h2eff_gpu_v2) path
#against the CPU reference, on a real AVAS/LASSCF molecule.
#
#The GPU driver below is synced 1:1 with
#   mrh/my_pyscf/mcscf/las_ao2mo.py -> get_h2eff_gpu_v2
#including the push/init/extract/loop/pull sequence and the blockwise
#sparsedf_array(contract1) CPU reference used for the DEBUG comparison.
import time
import numpy as np
import pyscf
import sys
from pyscf import gto, scf, mcscf, lib
from pyscf.mcscf import avas
from mrh.my_pyscf.df.sparse_df import sparsedf_array
from mrh.my_pyscf.mcscf.lasscf_async import LASSCF

gpu_run = 1
DEBUG = 1
if gpu_run:
    from mrh.my_pyscf.gpu import libgpu
    from gpu4mrh import patch_pyscf
from mrh.tests.gpu.geometry_generator import generator

np.set_printoptions(threshold=sys.maxsize)
lib.logger.TIMER_LEVEL = lib.logger.INFO

nfrags = 6
basis = 'sto3g'
outputfile = f'{nfrags}_{basis}_out.{ "gpu" if gpu_run else "cpu" }_ref.log'

mol_kwargs = dict(atom=generator(nfrags), basis=basis, verbose=4,
                  output=outputfile, max_memory=160000)
if gpu_run:
    gpu = libgpu.init()
    lib.param.use_gpu = gpu
    mol = gto.M(**mol_kwargs)
else:
    mol = gto.M(**mol_kwargs)

mf = scf.RHF(mol).density_fit()
mf.run()

las = LASSCF(mf, list((2,) * nfrags), list((2,) * nfrags), verbose=4)
frag_atom_list = [list(range(1 + 4 * frag, 3 + 4 * frag)) for frag in range(nfrags)]
ncas, nelecas, guess_mo_coeff = avas.kernel(mf, ["C 2pz"])
mo_coeff = las.set_fragments_(frag_atom_list, guess_mo_coeff)

ncore, ncas_a = las.ncore, las.ncas
nocc = ncore + ncas_a
nao, nmo = mo_coeff.shape
log = lib.logger.new_logger(las, las.verbose)
log.info(f"h2eff debug: nao={nao} nmo={nmo} ncore={ncore} ncas={ncas_a} "
         f"naux={las.with_df.get_naoaux ()} blksize={las.with_df.blockdim}")


def get_h2eff_gpu_v2(las, mo_coeff):
    gpu = las.use_gpu
    nao, nmo = mo_coeff.shape
    ncore, ncas = las.ncore, las.ncas
    nocc = ncore + ncas
    mo_cas = mo_coeff[:, ncore:nocc]
    libgpu.push_mo_coeff(gpu, mo_coeff.copy(), mo_coeff.size)
    libgpu.init_eri_h2eff(gpu, nmo, ncas)
    libgpu.extract_mo_cas(gpu, ncas, ncore, nao, nmo)
    blksize = las.with_df.blockdim
    eri = np.zeros((nmo, int(ncas * ncas * (ncas + 1) / 2)))
    eri1 = np.zeros_like(eri)
    count = 0
    if DEBUG:
        eri_cpu = np.zeros_like(eri)
    t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
    for cderi in las.with_df.loop(blksize=blksize):
        naux = cderi.shape[0]
        libgpu.get_h2eff_df_v2(gpu, cderi, nao, nmo, ncas, naux, ncore,
                               eri1, count, id(las.with_df))
        if DEBUG:
            bPmn = sparsedf_array(cderi)
            bmuP1 = bPmn.contract1(mo_cas)
            buvP = np.tensordot(mo_cas.conjugate(), bmuP1, axes=((0), (0)))
            eri2 = np.tensordot(bmuP1, buvP, axes=((2), (2)))
            eri2 = np.tensordot(mo_coeff.conjugate(), eri2, axes=((0), (0)))
            eri_cpu += lib.pack_tril(eri2.reshape(nmo * ncas, ncas, ncas)).reshape(nmo, -1)
            cderi = bPmn = bmuP1 = buvP = eri2 = None
        count += 1
    libgpu.pull_eri_h2eff(gpu, eri, nmo, ncas)
    t1 = lib.logger.timer(las, 'get_h2eff_gpu_v2', *t0)
    if DEBUG and np.allclose(eri, eri_cpu):
        log.info("h2eff_v2 GPU matches CPU reference")
    elif DEBUG:
        log.info(f"h2eff_v2 MISMATCH: max |diff| = {np.max(np.abs (eri - eri_cpu)):.3e}")
    return eri


def get_h2eff_cpu(las, mo_coeff):
    nao, nmo = mo_coeff.shape
    ncore, ncas = las.ncore, las.ncas
    nocc = ncore + ncas
    mo_cas = mo_coeff[:, ncore:nocc]
    naux = las.with_df.get_naoaux()
    blksize = las.with_df.blockdim
    eri = 0
    t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
    for cderi in las.with_df.loop(blksize=blksize):
        bPmn = sparsedf_array(cderi)
        bmuP1 = bPmn.contract1(mo_cas)
        buvP = np.tensordot(mo_cas.conjugate(), bmuP1, axes=((0), (0)))
        eri1 = np.tensordot(bmuP1, buvP, axes=((2), (2)))
        eri1 = np.tensordot(mo_coeff.conjugate(), eri1, axes=((0), (0)))
        eri += lib.pack_tril(eri1.reshape(nmo * ncas, ncas, ncas)).reshape(nmo, -1)
        cderi = bPmn = bmuP1 = buvP = eri1 = None
    lib.logger.timer(las, 'get_h2eff_cpu', *t0)
    return eri


t0 = time.time()
if gpu_run:
    eri_gpu = get_h2eff_gpu_v2(las, mo_coeff)
else:
    eri_cpu = get_h2eff_cpu(las, mo_coeff)
t1 = time.time()
log.info(f"h2eff wall time: {t1 - t0:.2f} s")
if gpu_run:
    libgpu.destroy_device(gpu)