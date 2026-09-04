gpu_run=1
if gpu_run:from mrh.my_pyscf.gpu import libgpu
import pyscf
import numpy as np
if gpu_run:from gpu4mrh import patch_pyscf
from pyscf import gto, scf, lib

def test_matvecs():
  ##checks compute_4frag_matvec's op_t=1 path, i.e. op/d2/d3 pushed in their
  ##transposed-on-device layout (rsjiba/kcr/lds) instead of the canonical
  ##layout (rsbaji/ckr/dls) used by frag4_test.py. all dims are pairwise
  ##distinct so an axis mixup can't hide behind a square/symmetric shape.

  z=5
  i=2
  j=3
  k=4
  l=6
  a=7
  b=8
  c=9
  d=10
  r=11
  s=12

  op = np.random.random((r,s,b,a,j,i))
  other = np.random.random((z,l,k,j,i))
  d2 = np.random.random((c,k,r))
  d3 = np.random.random((d,l,s))

  ox_cpu = cpu_kernel(op, other, d2, d3)
  ox_gpu = gpu_kernel(op, other, d2, d3)
  print(np.allclose(ox_cpu, ox_gpu))

def cpu_kernel(op,other,d2, d3):
  #CPU kernel operates on the canonical (untransposed) layout, matching
  #what compute_4frag_matvec's op_t=1 branch recovers internally.
  ox = lib.einsum ('rsbaji,zlkji->rsbazlk', op, other)
  ox = lib.einsum ('ckr,rsbazlk->scbazl', d2, ox)
  ox = lib.einsum ('dls,scbazl->dcbaz', d3, ox)
  return ox.ravel()

def gpu_kernel(op,other,d2, d3):
  r,s,b,a,j,i = op.shape
  c,k,r = d2.shape
  d,l,s = d3.shape
  z,l,k,j,i = other.shape

  #op_t=1 tells compute_4frag_matvec that op/d2/d3 were pushed in their
  #transposed-on-device layout; it un-transposes them back to canonical
  #(rsbaji/ckr/dls) before running the same gemm sequence as op_t=0.
  op_pushed = np.ascontiguousarray(op.transpose(0,1,4,5,2,3))  #rsbaji -> rsjiba
  d2_pushed = np.ascontiguousarray(d2.transpose(1,0,2))        #ckr -> kcr
  d3_pushed = np.ascontiguousarray(d3.transpose(1,0,2))        #dls -> lds

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
  op_t = 1
  gpu_idx = 0
  counts = 1
  libgpu.init_ox1_pinned(gpu, size_ox)
  libgpu.push_op_4frag(gpu, op_pushed, size_op, size_req, counts);
  libgpu.push_d2(gpu, d2_pushed, size_d2, size_op, counts);
  libgpu.push_d3(gpu, d3_pushed, size_d3, size_op + size_d2, counts);
  libgpu.push_sivecs_to_device(gpu, other, vec_loc, size_vec, counts)
  ox_gpu = np.zeros((size_ox));
  libgpu.compute_4frag_matvec(gpu, i,j,k,l,a,b,c,d,z,r,s, vec_loc, ox1_loc, fac, op_t, gpu_idx)
  libgpu.finalize_ox1_pinned(gpu, ox_gpu, size_ox)
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
