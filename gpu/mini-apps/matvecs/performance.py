## DEPRECATED 2026-09-03: exercises a dead libgpu code path. init_new_sivecs_host,
## init_old_sivecs_host, push_sivecs_to_host, compute_sivecs, and
## pull_sivecs_from_pinned all have zero call sites under mrh/my_pyscf,
## mrh/gpu/gpu4mrh, pyscf, or pyscf-forge today -- confirmed by
## `gpu/tools/prod_usage_audit.sh push_sivecs_to_host init_new_sivecs_host
## init_old_sivecs_host compute_sivecs pull_sivecs_from_pinned`. Production
## LASSI uses the v3/pinned path instead (push_sivecs_to_device,
## compute_sivecs_full_v3, finalize_ox1_pinned -- all LIVE). At its default
## problem size (m=1000, k=400, total_n ~500k) this script also did not
## complete SCF+mol.build within 90s on a laptop CPU backend, with no
## output/error to diagnose -- never reproduced as a genuine correctness
## bug in the dead path itself, just not worth sinking more time into.
## Run gpu/tools/prod_usage_audit.sh before reviving or deep-diving a bug here.
gpu_run=1
if gpu_run:from mrh.my_pyscf.gpu import libgpu
import pyscf
import numpy as np 
if gpu_run:from gpu4mrh import patch_pyscf
from pyscf import gto, scf, lib

def test_matvecs(m, k, n_array, nruns=10):
  from mrh.my_pyscf.gpu import libgpu
  from pyscf.lib import param
  import time
  gpu = param.use_gpu
  counts = libgpu.get_num_devices(gpu)   # push_op broadcasts to this many devices

  op = np.random.random((m,k))
  ratio = 4
  nruns = nruns//ratio
  total_n = sum(n_array)
  n_len = len(n_array)
  vecs = []
  for n in n_array:
    vecs.append(np.random.random((n,k)))

  #print("m: ",m, "k:",k, "total_n",total_n, "n_array",n_array)
  #Adding these here for memory allocations because this cost will basically go to zero as the iterations proceed
  libgpu.push_op(gpu, np.ascontiguousarray(op), m, k, counts)
  libgpu.init_new_sivecs_host(gpu, m, total_n)
  libgpu.init_old_sivecs_host(gpu, k, total_n)
  vec_loc = 0
  for n, vec in zip(n_array, vecs):
    libgpu.push_sivecs_to_host(gpu, vec, vec_loc, vec.size)
    vec_loc += vec.size
  t0 = time.time()
  
  for _ in range(ratio*nruns):
    tgpu0 = time.time()
    #GPU kernel
    new_vecs_gpu = [] 
    libgpu.push_op(gpu, np.ascontiguousarray(op), m, k, counts)
    tgpu0_5 = time.time()
    libgpu.init_new_sivecs_host(gpu, m, total_n)
    libgpu.init_old_sivecs_host(gpu, k, total_n)
    tgpu1 = time.time()
    setup_time0 = tgpu0_5-tgpu0
    setup_time1 = tgpu1-tgpu0_5
    vec_loc = 0
    for n, vec in zip(n_array, vecs):
      libgpu.push_sivecs_to_host(gpu, vec, vec_loc, vec.size)
      vec_loc += vec.size
    tgpu2 = time.time()
    host_sending_time = tgpu2-tgpu1
    libgpu.compute_sivecs(gpu, m, total_n, k)
    tgpu3 = time.time()
    compute_time = tgpu3-tgpu2
    n_loc = 0

    for n in n_array:
      new_vecs_gpu.append(np.empty((n*m)))
    tgpu3_5 = time.time()
    new_allocate_time = tgpu3_5-tgpu3
    for n, new_vec in zip(n_array, new_vecs_gpu):
      libgpu.pull_sivecs_from_pinned(gpu, new_vec, n_loc, m, n)
      n_loc += n

    tgpu4 = time.time()
    pull_time = tgpu4-tgpu3_5
    total_time = tgpu4-tgpu0
    frac = (lambda x: round(x/total_time,2) if total_time > 0.0 else float('nan'))
    print("Total:",round(total_time,4))
    print("Push op cost ratio:",frac(setup_time0))
    print("Allocate pinned cost ratio:",frac(setup_time1))
    print("Pageable to Pinned cost ratio:",frac(host_sending_time))
    print("DtoH, compute, HtoD cost ratio:",frac(compute_time))
    print("Allocate pageable for results ratio:",frac(new_allocate_time))
    print("Pinned to pageable cost ratio:",frac(pull_time))

  t1=time.time()

  for _ in range(nruns): 
    #CPU kernel
    new_vecs_cpu = [] 
    for vec in vecs:
      new_vecs_cpu.append(np.dot(op, vec.T).ravel()) 
  
  t2 = time.time()
  gpu_time = t1-t0
  cpu_time = t2-t1
  speedup = round(ratio*cpu_time/gpu_time,2) if gpu_time > 0.0 else float('nan')
  print("Matvecs: CPU time:", round(cpu_time,4), "GPU time for ",ratio,"x runs:", round(gpu_time,4),
        "Speedup:", speedup)

if __name__=='__main__':
  if gpu_run:
    gpu = libgpu.init()
    from pyscf.lib import param
    param.use_gpu = gpu
  lib.logger.TIMER_LEVEL=lib.logger.INFO

  geom = ''' K 0 0 0;
           K 0 0 2;
           K 0 0 4;
           K 0 0 8;
           K 0 0 10;
           K 0 0 12;'''
  basis = 'def2tzvp'
  mol = gto.M(atom=geom, basis=basis, verbose=1)

  mol.output='test.log'
  mol.build()

  mf = scf.RHF(mol)
  mf=mf.density_fit()
  mf.with_df.auxbasis = pyscf.df.make_auxbasis(mol)
  mf.max_cycle=1
  mf.kernel()
   
  m=1000
  k=400
  n_array = np.random.randint(200, 800, size=1000)
  runs=10 
  test_matvecs(m, k, n_array, runs)
   
