gpu_run=1
if gpu_run:from mrh.my_pyscf.gpu import libgpu
import pyscf
import numpy as np 
if gpu_run:from gpu4mrh import patch_pyscf
from pyscf import gto, scf, lib

def test_matvecs():
  ##checks for the transposed version
 
  z=5
  i=2
  j=3
  k=4
  l=5
  a=6
  b=5
  c=4
  d=3
  p=2
  q=3
  r=2
  s=3
  print("z,a,b,c,d,i,j,k,l,r,s")
  print(z,a,b,c,d,i,j,k,l,p,q,r,s)
  
  #op = np.random.random((p,q,r,s))
  
  #d0 = np.random.random((i,a,p))
  #d1 = np.random.random((j,b,q))
  #d2 = np.random.random((k,c,r))
  #d3 = np.random.random((l,d,s))

  #d0 = np.random.random((a,i,p))
  #d1 = np.random.random((b,j,q))
  d2 = np.random.random((c,k,r))
  d3 = np.random.random((d,l,s))

  #print(op.shape, d0.shape, d1.shape, d2.shape, d3.shape)
  #d0 = d0.transpose(1,0,2)
  #d1 = d1.transpose(1,0,2)
  #d2 = d2.transpose(1,0,2)
  #d3 = d3.transpose(1,0,2)

  #print('pqrs',op.shape,'iap', d0.shape,'jbq', d1.shape,'ckr', d2.shape,'dls', d3.shape)
  #op = np.einsum ('aip,bjq,pqrs->rsbaji', d0, d1, op)
  op = np.random.random((r,s,b,a,j,i))
   
  #op = op.transpose(0,1,4,5,2,3)

  #other = np.random.random((z,l,k,j,i))
  other = np.random.random((z,d,c,b,a))

  
  op = op.transpose(0,1,4,5,2,3)
  d2 = d2.transpose(1,0,2)
  d3 = d3.transpose(1,0,2)
  #size_op = r*s*b*a*j*i;
  #size_d2 = c*k*r;
  #size_d3 = d*l*s;
  #size_vec = z*l*k*j*i;
  #op = np.arange(size_op).reshape((r,s,b,a,j,i))+0.0
  #other = np.arange(size_vec).reshape((z,l,k,j,i))+0.0
  #d2 = np.arange(size_d2).reshape((c,k,r))+0.0
  #d3 = np.arange(size_d3).reshape((d,l,s))+0.0
  
  #print("op\n", op)
  #print("other\n", other)
  #print("d2\n", d2)
  #print("d3\n", d3)

  ox_cpu = cpu_kernel(op, other, d2, d3)
  ox_gpu = gpu_kernel(op, other, d2, d3)
  print(np.allclose(ox_cpu, ox_gpu))

def cpu_kernel(op,other,d2, d3):
  #CPU kernel
  ##remember, it's op.dot(vec.T), so this transpose is now cancelled out
  #simply a dot product now
  
  #op = op.transpose(0,1,4,5,2,3)
  #d2 = d2.transpose(1,0,2)
  #d3 = d3.transpose(1,0,2)
  #print("op")
  #print(op.ravel())
  print('rsjiba',op.shape, 'zlkji',other.shape)
  ox = lib.einsum ('rsbaji,zlkji->rsbazlk', op, other)
  #print(ox.shape, d2.shape)
  ox = lib.einsum ('ckr,rsbazlk->scbazl', d2, ox)
  #print(ox.shape, d3.shape)
  #print(np.einsum('scbazl->cbazls',ox).ravel())
  ox = lib.einsum ('dls,scbazl->dcbaz', d3, ox)
  print('dcbaz',ox.shape)
  #print("cpu ox")
  #print(ox.ravel())
  return ox.ravel()

def gpu_kernel(op,other,d2, d3) : 
  c,k,r = d2.shape
  d,l,s = d3.shape
  r,s,b,a,j,i = op.shape
  z,l,k,j,i = other.shape

  #k,c,r = d2.shape
  #l,d,s = d3.shape
  #r,s,j,i,b,a = op.shape
  #z,d,c,b,a = other.shape
  #op doesn't need to be reshaped
  size_op = r*s*b*a*j*i;
  size_req = r*s*b*a*z*l*k;
  size_d2 = c*k*r;
  size_d3 = d*l*s;
  size_vec = z*l*k*j*i;
  size_ox = d*c*b*a*z;
  size_req = 2*size_req + size_op + size_d2 + size_d3; #for storing op, d2, d3, result of op and vec, and it's transpose.
  vec_loc = 0
  ox1_loc = 0
  fac = 1
  op_t = 0
  gpu_idx = 0
  counts = 1
  libgpu.init_ox1_pinned(gpu, size_ox)
  libgpu.push_op_4frag(gpu, np.ascontiguousarray(op), size_op, size_req,counts);
  libgpu.push_d2(gpu, np.ascontiguousarray(d2), size_d2, size_op,counts); 
  libgpu.push_d3(gpu, np.ascontiguousarray(d3), size_d3, size_op + size_d2,counts);
  libgpu.push_sivecs_to_device(gpu, other, vec_loc, size_vec, counts)
  ox_gpu = np.zeros((size_ox));
  #print("gpu ox before pull")
  #print(ox_gpu)

  print("z,a,b,c,d,i,j,k,l,r,s")
  print(z,a,b,c,d,i,j,k,l,z,r,s)
  libgpu.compute_4frag_matvec(gpu, i,j,k,l,a,b,c,d,z,r,s, vec_loc, ox1_loc, fac, op_t, gpu_idx)
  libgpu.finalize_ox1_pinned(gpu, ox_gpu, size_ox) 
  #print("gpu ox")
  #print(ox_gpu)
  return ox_gpu


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
   
  test_matvecs()
 
