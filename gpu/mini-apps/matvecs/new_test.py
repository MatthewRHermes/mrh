gpu_run=1
if gpu_run:from mrh.my_pyscf.gpu import libgpu
import pyscf
import numpy as np 
if gpu_run:from gpu4mrh import patch_pyscf
from pyscf import gto, scf, lib

def test_matvecs(m, k, n_max, counts, iters= 2):
  from mrh.my_pyscf.gpu import libgpu
  from pyscf.lib import param
  gpu = param.use_gpu
  
  print(gpu)
  
  #op = np.random.random((m,k))
  op = np.arange(m*k).reshape(m,k)+0.5

  vec_total = np.arange(n_max*k*counts)+0.

  states = n_max*m*counts

  ox1_cpu = np.zeros(states)
  ox1_gpu = np.zeros(states)
  n_array = np.random.randint(1,n_max, size= counts)

  #print(n_array)
  vecs = {}

  for i,n in enumerate(n_array):
    vec_loc = i*n_max*k
    size_vec = n*k
    vecs[i] = np.ascontiguousarray((vec_total[vec_loc : vec_loc + size_vec]).reshape(n,k))
  for i,n in enumerate(n_array):
    vec_loc = (i+1)*n_max*k
    size_vec = n*k
    vecs[i] = np.ascontiguousarray((vec_total[vec_loc-size_vec : vec_loc ]).reshape(n,k)) + vecs.get(i,0)

  #print(vecs)

  #CPU kernel
  for _ in range(2):
    n_loc = 0
    for i, (n, vec) in enumerate(zip(n_array, vecs.values())):
      ox1_loc = i*n_max*m
      ox1_cpu[ox1_loc:ox1_loc+n*m] += np.dot(op, vec.T).ravel() 
      #print(np.dot(op, vec.T).ravel())
      n_loc +=n



  #GPU Kernel
  libgpu.init_ox1_pinned(gpu,states)
  total_vecsize = sum([vec.size for vec in vecs.values()])
  libgpu.init_old_sivecs_host(gpu, total_vecsize, 1)
  vec_table = {}
  vec_loc = 0
  for key, vec in vecs.items():
    vec_c = np.ascontiguousarray(vec) 
    vec_table[key] = (vec_loc, vec_c.size)
    libgpu.push_sivecs_to_host(gpu, vec_c, vec_loc, vec_c.size)
    vec_loc += vec_c.size

  for _ in range(2):
    libgpu.push_op(gpu, np.ascontiguousarray(op), m, k)
    instruction_list = np.empty(( len(n_array),6), dtype=np.int32)
    for i, n in enumerate(n_array):
      vec_loc, vec_size = vec_table[i]
      ox1_loc = i*n_max*m
      ox1_size = n*m
      #fac = (-1,1)[i%2]  
      fac = 1
      instruction_list[i] = n, vec_loc, vec_size, ox1_loc, ox1_size, fac
    #print(instruction_list)
    libgpu.push_instruction_list(gpu, instruction_list, len(n_array))
    libgpu.compute_sivecs_full(gpu, m, k, len(n_array))
    libgpu.add_ox1_pinned(gpu, ox1_gpu, states) 
  libgpu.finalize_ox1_pinned(gpu, ox1_gpu, states) 

  correct=np.allclose(ox1_cpu, ox1_gpu)
  if not correct:
    print(n, m, correct)
    print("CPU",ox1_cpu)
    print("GPU",ox1_gpu)
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
   
  m_max=10
  k=5
  n_max = 10
  counts = 4
  
  test_matvecs(m, k, n_max, counts)
   
