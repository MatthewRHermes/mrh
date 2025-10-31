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
  libgpu.push_op(gpu, op, m, k)
  libgpu.init_new_sivecs_host(gpu, m, total_n)
  libgpu.init_old_sivecs_host(gpu, k, total_n)
  n_loc = 0
  for n, vec in zip(n_array, vecs):
    libgpu.push_sivecs_to_host(gpu, vec, n_loc, n, k)
    n_loc += n
  t0 = time.time()
  
  for _ in range(ratio*nruns):
    tgpu0 = time.time()
    #GPU kernel
    new_vecs_gpu = [] 
    libgpu.push_op(gpu, op, m, k)
    tgpu0_5 = time.time()
    libgpu.init_new_sivecs_host(gpu, m, total_n)
    libgpu.init_old_sivecs_host(gpu, k, total_n)
    tgpu1 = time.time()
    setup_time0 = round(tgpu0_5-tgpu0,2)
    setup_time1 = round(tgpu1-tgpu0_5,2)
    n_loc=0
    for n, vec in zip(n_array, vecs):
      libgpu.push_sivecs_to_host(gpu, vec, n_loc, n, k)
      n_loc += n
    tgpu2 = time.time()
    host_sending_time = round(tgpu2-tgpu1,2)
    libgpu.compute_sivecs(gpu, m, total_n, k)
    tgpu3 = time.time()
    compute_time = round(tgpu3-tgpu2,2)
    n_loc = 0

    for n in n_array:
      new_vecs_gpu.append(np.empty((n*m)))
    tgpu3_5 = time.time()
    new_allocate_time = round(tgpu3_5-tgpu3,2)
    for n, new_vec in zip(n_array, new_vecs_gpu):
      libgpu.pull_sivecs_from_pinned(gpu, new_vec, n_loc, m, n)
      n_loc += n

    tgpu4 = time.time()
    pull_time = round(tgpu4-tgpu3_5,2)
    total_time = round(tgpu4-tgpu0,2)
    print("Total:",total_time)
    print("Push op cost ratio:",round(setup_time0/total_time,2))
    print("Allocate pinned cost ratio:",round(setup_time1/total_time,2))
    print("Pageable to Pinned cost ratio:",round(host_sending_time/total_time,2))
    print("DtoH, compute, HtoD cost ratio:",round(compute_time/total_time,2))
    print("Allocate pageable for results ratio:",round(new_allocate_time/total_time,2))
    print("Pinned to pageable cost ratio:",round(pull_time/total_time,2))

  t1=time.time()

  for _ in range(nruns): 
    #CPU kernel
    new_vecs_cpu = [] 
    for vec in vecs:
      new_vecs_cpu.append(np.dot(op, vec.T).ravel()) 
  
  t2 = time.time()
  gpu_time = round(t1-t0,2)
  cpu_time = round(t2-t1,2)
  print("Matvecs: CPU time:", cpu_time, "GPU time for ",ratio,"x runs:", gpu_time, "Speedup:", round(ratio*cpu_time/gpu_time,2))

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
  if gpu_run: mol = gto.M(use_gpu = gpu, atom=geom, basis=basis, verbose=1)
  else: mol = gto.M(atom=geom, basis=basis, verbose=1)

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
   
