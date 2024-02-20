/* -*- c++ -*- */

#if defined(_USE_CPU)

#include "device.h"

#include <stdio.h>

/* ---------------------------------------------------------------------- */

void Device::init_get_jk(py::array_t<double> _eri1, py::array_t<double> _dmtril, int _blksize, int _nset, int _nao, int _naux, int count)
{
#ifdef _SIMPLE_TIMER
  double t0 = omp_get_wtime();
#endif

  blksize = _blksize;
  nset = _nset;
  nao = _nao;
  naux = _naux;

  const int nao_pair = nao * (nao+1) / 2;
  
  py::buffer_info info_eri1 = _eri1.request(); // 2D array (232, 351)
  py::buffer_info info_dmtril = _dmtril.request(); // 2D array (nset, 351)

  // double * eri1 = static_cast<double*>(info_eri1.ptr);
  // double * dmtril = static_cast<double*>(info_dmtril.ptr);
  
  int _size_vj = nset * nao_pair;
  if(_size_vj > size_vj) {
    size_vj = _size_vj;
    //if(vj) pm->dev_free_host(vj);
    //vj = (double *) pm->dev_malloc_host(size_vj * sizeof(double));
    
    if(count > 0) printf("WARNING:: Reallocating vj with count= %i  nset= %i  nao_pair= %i\n",count, nset, nao_pair);
  }
  //for(int i=0; i<_size_vj; ++i) vj[i] = 0.0;

  int _size_vk = nset * nao * nao;
  if(_size_vk > size_vk) {
    size_vk = _size_vk;
    // if(_vktmp) pm->dev_free_host(_vktmp);
    // _vktmp = (double *) pm->dev_malloc_host(size_vk*sizeof(double));
  }
  //  for(int i=0; i<_size_vk; ++i) _vktmp[i] = 0.0;

  int _size_buf = blksize * nao * nao;
  if(_size_buf > size_buf) {
    size_buf = _size_buf;
    if(buf_tmp) pm->dev_free_host(buf_tmp);
    if(buf3) pm->dev_free_host(buf3);
    if(buf4) pm->dev_free_host(buf4);
    
    buf_tmp = (double*) pm->dev_malloc_host(2*size_buf*sizeof(double));
    buf3 = (double *) pm->dev_malloc_host(size_buf*sizeof(double)); // (nao, blksize*nao)
    buf4 = (double *) pm->dev_malloc_host(size_buf*sizeof(double)); // (blksize*nao, nao)
    
    if(count > 0) printf("WARNING:: Reallocating bufs with count= %i  blksize= %i  nao= %i\n",count, blksize, nao);
  }

  int _size_fdrv = 4 * nao * nao * num_threads;
  if(_size_fdrv > size_fdrv) {
    size_fdrv = _size_fdrv;
    if(buf_fdrv) pm->dev_free_host(buf_fdrv);
    buf_fdrv = (double *) pm->dev_malloc_host(size_fdrv*sizeof(double));

    if(count > 0) printf("WARNING:: Reallocating buf_fdrv with count= %i nao= %i  num_threads= %i\n",count, nao, num_threads);
  }
  
#ifdef _SIMPLE_TIMER
  t_array_jk[0] += omp_get_wtime() - t0;
#endif
}

/* ---------------------------------------------------------------------- */

void Device::pull_get_jk(py::array_t<double> _vj, py::array_t<double> _vk) {}

/* ---------------------------------------------------------------------- */

void Device::get_jk(int naux,
		    py::array_t<double> _eri1, py::array_t<double> _dmtril, py::list & _dms_list,
		    py::array_t<double> _vj, py::array_t<double> _vk,
		    int count)
{  
#ifdef _SIMPLE_TIMER
  double t0 = omp_get_wtime();
#endif

  const int with_j = true;
  
  py::buffer_info info_eri1 = _eri1.request(); // 2D array (naux, nao_pair)
  py::buffer_info info_dmtril = _dmtril.request(); // 2D array (nset, nao_pair)
  py::buffer_info info_vj = _vj.request(); // 2D array (nset, nao_pair)
  py::buffer_info info_vk = _vk.request(); // 3D array (nset, nao, nao)
  
  double * eri1 = static_cast<double*>(info_eri1.ptr);
  double * dmtril = static_cast<double*>(info_dmtril.ptr);
  double * vj = static_cast<double*>(info_vj.ptr);
  double * vk = static_cast<double*>(info_vk.ptr);
  
  int _size_rho = nset * naux;
  if(_size_rho > size_rho) {
    size_rho = _size_rho;
    if(rho) pm->dev_free_host(rho);
    rho = (double *) pm->dev_malloc_host(size_rho * sizeof(double));
  }

  // printf("LIBGPU:: blksize= %i  naux= %i  nao= %i  nset= %i\n",blksize,naux,nao,nset);
  // printf("LIBGPU::shape: dmtril= (%i,%i)  eri1= (%i,%i)  rho= (%i, %i)   vj= (%i,%i)  vk= (%i,%i,%i)\n",
  //   	 info_dmtril.shape[0], info_dmtril.shape[1],
  //   	 info_eri1.shape[0], info_eri1.shape[1],
  //   	 info_dmtril.shape[0], info_eri1.shape[0],
  //   	 info_dmtril.shape[0], info_eri1.shape[1],
  //   	 info_vk.shape[0],info_vk.shape[1],info_vk.shape[2]);

   int nao_pair = nao * (nao+1) / 2;
   
#ifdef _SIMPLE_TIMER
  t_array_jk[1] += omp_get_wtime() - t0;
#endif
  
  if(with_j) {

#ifdef _SIMPLE_TIMER
    double t0 = omp_get_wtime();
#endif

    DevArray2D da_rho = DevArray2D(rho, nset, naux);
    DevArray2D da_dmtril = DevArray2D(dmtril, nset, nao_pair);
    DevArray2D da_eri1 = DevArray2D(eri1, naux, nao_pair);
    
    // rho = numpy.einsum('ix,px->ip', dmtril, eri1)

#pragma omp parallel for collapse(2)
    for(int i=0; i<nset; ++i)
      for(int j=0; j<naux; ++j) {
	double val = 0.0;
	for(int k=0; k<nao_pair; ++k) val += da_dmtril(i,k) * da_eri1(j,k);
	da_rho(i,j) = val;
      }
    
#ifdef _SIMPLE_TIMER
    double t1 = omp_get_wtime();
#endif

    DevArray2D da_vj = DevArray2D(vj, nset, nao_pair);
    
    // vj += numpy.einsum('ip,px->ix', rho, eri1)

#pragma omp parallel for collapse(2)
    for(int i=0; i<nset; ++i)
      for(int j=0; j<nao_pair; ++j) {

	double val = 0.0;
	for(int k=0; k<naux; ++k) val += da_rho(i,k) * da_eri1(k,j);
	da_vj(i,j) += val;
      }

#ifdef _SIMPLE_TIMER
    t_array_jk[2] += t1 - t0;
    t_array_jk[3] += omp_get_wtime() - t1;
#endif
  }

  double * buf1 = buf_tmp;
  double * buf2 = &(buf_tmp[blksize * nao * nao]);
    
  DevArray3D da_buf1 = DevArray3D(buf1, naux, nao, nao);
  DevArray2D da_buf2 = DevArray2D(buf2, blksize * nao, nao);
  DevArray2D da_buf3 = DevArray2D(buf3, nao, naux * nao); // python swapped 1st two dimensions?
  
  for(int indxK=0; indxK<nset; ++indxK) {

    py::array_t<double> _dms = static_cast<py::array_t<double>>(_dms_list[indxK]); // element of 3D array (nset, nao, nao)
    py::buffer_info info_dms = _dms.request(); // 2D

    // rargs = (ctypes.c_int(nao), (ctypes.c_int*4)(0, nao, 0, nao), null, ctypes.c_int(0))

    int orbs_slice[4] = {0, nao, 0, nao};
    double * dms = static_cast<double*>(info_dms.ptr);

#ifdef _SIMPLE_TIMER
    t0 = omp_get_wtime();
#endif
    
    //    fmmm = _ao2mo.libao2mo.AO2MOmmm_bra_nr_s2
    //    fdrv = _ao2mo.libao2mo.AO2MOnr_e2_drv
    //    ftrans = _ao2mo.libao2mo.AO2MOtranse2_nr_s2
      
    //    fdrv(ftrans, fmmm,
    //	       buf1.ctypes.data_as(ctypes.c_void_p),
    //	       eri1.ctypes.data_as(ctypes.c_void_p),
    //	       dms[k].ctypes.data_as(ctypes.c_void_p),
    //	       ctypes.c_int(naux), *rargs)
    
    fdrv(buf1, eri1, dms, naux, nao, orbs_slice, nullptr, 0, buf_fdrv);

#ifdef _SIMPLE_TIMER
    double t1 = omp_get_wtime();
    t_array_jk[4] += t1 - t0;
#endif
    
    // buf2 = lib.unpack_tril(eri1, out=buf[1])
    
#pragma omp parallel for
    for(int i=0; i<naux; ++i) {
      
      int indx = 0;
      double * eri1_ = &(eri1[i * nao_pair]);

      // unpack lower-triangle to square
      
      for(int j=0; j<nao; ++j)
	for(int k=0; k<=j; ++k) {
	  da_buf2(i*nao+j,k) = eri1_[indx];
	  da_buf2(i*nao+k,j) = eri1_[indx];
	  indx++;
	}
      
    }

#ifdef _SIMPLE_TIMER
    double t2 = omp_get_wtime();
    t_array_jk[5] += t2 - t1;
#endif
    
    // dgemm of (nao X blksize*nao) and (blksize*nao X nao) matrices - can refactor later...
    // vk[k] += lib.dot(buf1.reshape(-1,nao).T, buf2.reshape(-1,nao))  // vk[k] is nao x nao array
    
    // buf3 = buf1.reshape(-1,nao).T
    // buf4 = buf2.reshape(-1,nao)
    
#pragma omp parallel for
    for(int i=0; i<naux; ++i) {
      for(int j=0; j<nao; ++j)
	for(int k=0; k<nao; ++k) da_buf3(k,i*nao+j) = da_buf1(i,j,k);
    }
    
    // vk[k] += lib.dot(buf3, buf4)
    // gemm(A,B,C) : C = 1.0 * A.B + 0.0 * C
    // A is (m, k) matrix
    // B is (k, n) matrix
    // C is (m, n) matrix
    // Column-ordered: (A.B)^T = B^T.A^T
    
#ifdef _SIMPLE_TIMER
    double t3 = omp_get_wtime();
    t_array_jk[6] += t3 - t2;
#endif
    
    const double alpha = 1.0;
    const double beta = (count == 0) ? 0.0 : 1.0;
    
    const int m = nao; // # of rows of first matrix buf4^T
    const int n = nao; // # of cols of second matrix buf3^T
    const int k = naux*nao; // # of cols of first matrix buf4^

    const int lda = naux * nao;
    const int ldb = nao;
    const int ldc = nao;

    const int vk_offset = indxK * nao*nao;
    
    double * vkk = vk + vk_offset;
    dgemm_((char *) "N", (char *) "N", &m, &n, &k, &alpha, buf2, &ldb, buf3, &lda, &beta, vkk, &ldc);

#ifdef _SIMPLE_TIMER
    double t4 = omp_get_wtime();
    t_array_jk[7] += t4 - t3;
    t_array_jk_count++;
#endif 
  }
  
}

/* ---------------------------------------------------------------------- */

// pyscf/pyscf/lib/ao2mo/nr_ao2mo.c::AO2MOnr_e2_drv()
#if 1
void Device::fdrv(double *vout, double *vin, double *mo_coeff,
		  int nij, int nao, int *orbs_slice, int *ao_loc, int nbas, double * _buf)
{
  struct Device::my_AO2MOEnvs envs;
  envs.bra_start = orbs_slice[0];
  envs.bra_count = orbs_slice[1] - orbs_slice[0];
  envs.ket_start = orbs_slice[2];
  envs.ket_count = orbs_slice[3] - orbs_slice[2];
  envs.nao = nao;
  envs.nbas = nbas;
  envs.ao_loc = ao_loc;
  envs.mo_coeff = mo_coeff;
  
  const int ij_pair = envs.bra_count * nao; //fmmm(NULL, NULL, buf, &envs, OUTPUTIJ);
  const int nao2 = nao * (nao + 1) / 2; //fmmm(NULL, NULL, buf, &envs, INPUT_IJ);
    
#pragma omp parallel for
  for (int i = 0; i < nij; i++) {
    const int it = omp_get_thread_num();
    double * buf = &(_buf[it * 4 * nao * nao]);

    int _i, _j, _ij;
    double * tril = vin + nao2*i;
    for (_ij = 0, _i = 0; _i < nao; _i++) 
      for (_j = 0; _j <= _i; _j++, _ij++) buf[_i*nao+_j] = tril[_ij];
    
#if 1
    const double D0 = 0;
    const double D1 = 1;
    const char SIDE_L = 'L';
    const char UPLO_U = 'U';
    int i_start = envs.bra_start;
    int i_count = envs.bra_count;

    double * _vout = vout + ij_pair*i;
    
    dsymm_(&SIDE_L, &UPLO_U, &nao, &i_count,
	   &D1, buf, &nao, mo_coeff+i_start*nao, &nao,
	   &D0, _vout, &nao);
    
#else
    ftrans(i, vout, vin, buf, &envs);
#endif
    
  }
  
}
#else
void Device::fdrv(double *vout, double *vin, double *mo_coeff,
	  int nij, int nao, int *orbs_slice, int *ao_loc, int nbas)
{
  struct Device::my_AO2MOEnvs envs;
  envs.bra_start = orbs_slice[0];
  envs.bra_count = orbs_slice[1] - orbs_slice[0];
  envs.ket_start = orbs_slice[2];
  envs.ket_count = orbs_slice[3] - orbs_slice[2];
  envs.nao = nao;
  envs.nbas = nbas;
  envs.ao_loc = ao_loc;
  envs.mo_coeff = mo_coeff;
  
#pragma omp parallel default(none)					\
  shared(vout, vin, nij, envs, nao, orbs_slice)
  {
    int i;
    int i_count = envs.bra_count;
    int j_count = envs.ket_count;
    double *buf = (double *) pm->dev_malloc_host(sizeof(double) * (nao+i_count) * (nao+j_count));
#pragma omp for schedule(dynamic)
    for (i = 0; i < nij; i++) {
      ftrans(i, vout, vin, buf, &envs);
    }
    pm->dev_free_host(buf);
  }
}
#endif

/* ---------------------------------------------------------------------- */
// pyscf/pyscf/lib/np_helper/pack_tril.c
void Device::NPdsymm_triu(int n, double *mat, int hermi)
{
  size_t i, j, j0, j1;
  
  if (hermi == HERMITIAN || hermi == SYMMETRIC) {
    TRIU_LOOP(i, j) {
      mat[i*n+j] = mat[j*n+i];
    }
  } else {
    TRIU_LOOP(i, j) {
      mat[i*n+j] = -mat[j*n+i];
    }
  }
}

/* ---------------------------------------------------------------------- */

void Device::NPdunpack_tril(int n, double *tril, double *mat, int hermi)
{
  size_t i, j, ij;
  for (ij = 0, i = 0; i < n; i++) {
    for (j = 0; j <= i; j++, ij++) {
      mat[i*n+j] = tril[ij];
    }
  }
  if (hermi) {
    NPdsymm_triu(n, mat, hermi);
  }
}

/* ---------------------------------------------------------------------- */
// pyscf/pyscf/lib/ao2mo/nr_ao2mo.c::AO2MOtranse2_nr_s2kl
void Device::ftrans(int row_id,
		    double *vout, double *vin, double *buf,
		    struct Device::my_AO2MOEnvs *envs)
{
  int nao = envs->nao;
  size_t ij_pair = fmmm(NULL, NULL, buf, envs, OUTPUTIJ);
  size_t nao2 = fmmm(NULL, NULL, buf, envs, INPUT_IJ);
  NPdunpack_tril(nao, vin+nao2*row_id, buf, 0);
  fmmm(vout+ij_pair*row_id, buf, buf+nao*nao, envs, 0);
}

/* ---------------------------------------------------------------------- */
// pyscf/pyscf/lib/ao2mo/nr_ao2mo.c::AO2MOmmm_bra_nr_s2
int Device::fmmm(double *vout, double *vin, double *buf,
		 struct my_AO2MOEnvs *envs, int seekdim)
{
  switch (seekdim) {
  case OUTPUTIJ: return envs->bra_count * envs->nao;
  case INPUT_IJ: return envs->nao * (envs->nao+1) / 2;
  }
  const double D0 = 0;
  const double D1 = 1;
  const char SIDE_L = 'L';
  const char UPLO_U = 'U';
  int nao = envs->nao;
  int i_start = envs->bra_start;
  int i_count = envs->bra_count;
  double *mo_coeff = envs->mo_coeff;
  
  dsymm_(&SIDE_L, &UPLO_U, &nao, &i_count,
         &D1, vin, &nao, mo_coeff+i_start*nao, &nao,
         &D0, vout, &nao);
  return 0;
}

/* ---------------------------------------------------------------------- */

/* ---------------------------------------------------------------------- */

void Device::hessop_get_veff(int naux, int nmo, int ncore, int nocc,
		    py::array_t<double> _bPbj, py::array_t<double> _vPpj, py::array_t<double> _vk_bj)
{
  py::buffer_info info_bPbj = _bPbj.request(); // 3D array (naux, nmo, nocc) : read-only
  py::buffer_info info_vPpj = _vPpj.request(); // 3D array (naux, nmo, nocc) : read-only 
  py::buffer_info info_vk_bj = _vk_bj.request(); // 2D array (nmo-ncore, nocc) : accumulate
  
  double * bPbj = static_cast<double*>(info_bPbj.ptr);
  double * vPpj = static_cast<double*>(info_vPpj.ptr);
  double * vk_bj = static_cast<double*>(info_vk_bj.ptr);
  
  int nvirt = nmo - ncore;

#if 0
  printf("LIBGPU:: naux= %i  nmo= %i  ncore= %i  nocc= %i  nvirt= %i\n",naux, nmo, ncore, nocc, nvirt);
  printf("LIBGPU:: shape : bPbj= (%i, %i, %i)  vPj= (%i, %i, %i)  vk_bj= (%i, %i)\n",
	 info_bPbj.shape[0],info_bPbj.shape[1],info_bPbj.shape[2],
	 info_vPpj.shape[0],info_vPpj.shape[1],info_vPpj.shape[2],
	 info_vk_bj.shape[0], info_vk_bj.shape[1]);
#endif
  
  DevArray3D da_bPbj = DevArray3D(bPbj, naux, nmo, nocc);
  DevArray3D da_vPpj = DevArray3D(vPpj, naux, nmo, nocc);
  DevArray2D da_vk_bj = DevArray2D(vk_bj, nvirt, nocc);

  // vPji = vPpj[:,:nocc,:ncore]
  // bPbi = self.bPpj[:,ncore:,:ncore]
  // vk_bj += np.tensordot (bPbi, vPji, axes=((0,2),(0,2)))

  for(int i=0; i<nvirt; ++i)
    for(int j=0; j<nocc; ++j) {
      
      double tmp = 0.0;
      for(int k=0; k<naux; ++k)
	for(int l=0; l<ncore; ++l)
	  tmp += da_bPbj(k,ncore+i,l) * da_vPpj(k,j,l);
      da_vk_bj(i,j) += tmp;
    }
  
}

#endif
