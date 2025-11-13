gpu_run=1
if gpu_run:from mrh.my_pyscf.gpu import libgpu
import pyscf
import numpy as np 
if gpu_run:from gpu4mrh import patch_pyscf
from pyscf import gto, scf, lib

def test_matvecs(m, k, n_array):
  from mrh.my_pyscf.gpu import libgpu
  from pyscf.lib import param
  gpu = param.use_gpu

  op = np.random.random((m,k))
  op = np.arange(m*k).reshape(m,k)+0.5
  print(op)

  total_n = sum(n_array)
  n_len = len(n_array)
  vecs = []
  for n in n_array:
    vecs.append(np.arange(n*k).reshape(n,k)-n+0.5)
    #vecs.append(np.random.random((n,k)))
  print(vecs)
  #GPU kernel
  libgpu.push_op(gpu, op, m, k)
  libgpu.init_new_sivecs_host(gpu, m, total_n)
  libgpu.init_old_sivecs_host(gpu, k, total_n)
  n_loc=0
  for n, vec in zip(n_array, vecs):
    libgpu.push_sivecs_to_host(gpu, vec, n_loc, n, k)
    n_loc += n
  libgpu.compute_sivecs(gpu, m, total_n, k)
  new_vecs_gpu = [] 
  n_loc = 0
  #for n in n_array:
  #  new_vecs_gpu.append(np.empty((n*m)))
  #for n, new_vec in zip(n_array, new_vecs_gpu):
  #  libgpu.pull_sivecs_from_pinned(gpu, new_vec, n_loc, m, n)
  #  n_loc += n

  max_n = max(n_array)
  new_vec = np.empty((max_n*m))
  for n in n_array:
    libgpu.pull_sivecs_from_pinned(gpu, new_vec, n_loc, m, n)
    print(new_vec[:n*m])
    n_loc += n
    new_vecs_gpu.append(new_vec[:n*m])
  
  #CPU kernel
  new_vecs_cpu = [] 
  for vec in vecs:
    new_vecs_cpu.append(np.dot(op, vec.T).ravel()) 
  
  print(new_vecs_gpu)
  for n, vec_cpu, vec_gpu in zip(n_array, new_vecs_cpu, new_vecs_gpu):
    correct=np.allclose(vec_cpu, vec_gpu)
    if not correct:
      print(n, m, correct)
      print("CPU",vec_cpu)
      print("GPU",vec_gpu)
    else:
      print(correct)
    

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
   
  m=2
  k=3
  n_array = [4,5]
  #n_array = np.random.randint(3, 6, size=4)
  
  test_matvecs(m, k, n_array)
   
