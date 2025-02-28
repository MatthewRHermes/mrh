/* -*- c++ -*- */

#if defined(_GPU_OPENMP)

#include "../device.h"

#include <stdio.h>

/* ---------------------------------------------------------------------- */

void Device::init_get_jk(py::array_t<double> _eri1, py::array_t<double> _dmtril, int _blksize, int _nset, int _nao, int count)
{
  //printf("Inside init_get_jk()\n");

#ifdef _SIMPLE_TIMER
  double t0 = omp_get_wtime();
#endif

  blksize = _blksize;
  nset = _nset;
  nao = _nao;

  const int nao_pair = nao * (nao+1) / 2;
  
  py::buffer_info info_eri1 = _eri1.request(); // 2D array (232, 351)
  py::buffer_info info_dmtril = _dmtril.request(); // 2D array (nset, 351)

  //  double * eri1 = static_cast<double*>(info_eri1.ptr);
  //  double * dmtril = static_cast<double*>(info_dmtril.ptr);
  
  int _size_vj = nset * nao_pair;
  if(_size_vj > size_vj) {
    size_vj = _size_vj;
    //if(vj) pm->dev_free_host(vj);
    //vj = (double *) pm->dev_malloc_host(size_vj * sizeof(double));
  }
  //for(int i=0; i<_size_vj; ++i) vj[i] = 0.0;

  int _size_vk = nset * nao * nao;
  if(_size_vk > size_vk) {
    size_vk = _size_vk;
    //    if(_vktmp) pm->dev_free_host(_vktmp);
    //    _vktmp = (double *) pm->dev_malloc_host(size_vk*sizeof(double));

    profile_start("Realloc");
    
    if(d_vkk) pm->dev_free(d_vkk);
    d_vkk = (double *) pm->dev_malloc(size_vk * sizeof(double));

    profile_stop();
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

    profile_start("Realloc");

    if(d_buf2) pm->dev_free(d_buf2);
    if(d_buf3) pm->dev_free(d_buf3);
    
    d_buf2 = (double *) pm->dev_malloc(size_buf * sizeof(double));
    d_buf3 = (double *) pm->dev_malloc(size_buf * sizeof(double));

    profile_stop();
  }

  int _size_fdrv = nao * nao * num_threads;
  if(_size_fdrv > size_fdrv) {
    size_fdrv = _size_fdrv;
    if(buf_fdrv) pm->dev_free_host(buf_fdrv);
    buf_fdrv = (double *) pm->dev_malloc_host(size_fdrv*sizeof(double));
  }

  // Create cuda stream

  if(stream == nullptr) {
    //printf("Creating new stream\n");
    pm->dev_stream_create(stream);
    //pm->dev_stream_wait(stream);
    //printf(" -- finished.\n");
  }

  // Create blas handle

  if(handle == nullptr) {
    profile_start("Create handle");

    cublasCreate(&handle);
    _CUDA_CHECK_ERRORS();
    cublasSetStream(handle, stream);
    _CUDA_CHECK_ERRORS();

    profile_stop();
  }

#ifdef _SIMPLE_TIMER
  double t1 = omp_get_wtime();
  t_array[0] += t1 - t0;
#endif
  
  //pm->dev_barrier();
  //printf("Leaving init_get_jk()\n");
}

/* ---------------------------------------------------------------------- */

void Device::pull_get_jk(py::array_t<double> _vj, py::array_t<double> _vk)
{
  //printf("Inside pull_get_jk()\n");
  
#ifdef _SIMPLE_TIMER
  double t0 = omp_get_wtime();
#endif
  
  //  py::buffer_info info_vj = _vj.request(); // 2D array (nset, nao_pair)
  py::buffer_info info_vk = _vk.request(); // 3D array (nset, nao, nao)
  
  //  double * vj = static_cast<double*>(info_vj.ptr);
  double * vk = static_cast<double*>(info_vk.ptr);
  //pm->dev_barrier();
  pm->dev_pull(d_vkk, vk, nset * nao * nao * sizeof(double));
  //pm->dev_barrier();

  //printf("Leaving pull_get_jk()\n");
  
#ifdef _SIMPLE_TIMER
  double t1 = omp_get_wtime();
  t_array[1] += t1 - t0;
#endif
}

/* ---------------------------------------------------------------------- */

void Device::get_jk(int naux,
		    py::array_t<double> _eri1, py::array_t<double> _dmtril, py::list & _dms_list,
		    py::array_t<double> _vj, py::array_t<double> _vk,
		    int count)
{
  //printf("Inside get_jk()\n");
  
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

  int _size_rho = info_dmtril.shape[0] * info_eri1.shape[0];
  if(_size_rho > size_rho) {
    size_rho = _size_rho;
    if(rho) pm->dev_free_host(rho);
    rho = (double *) pm->dev_malloc_host(size_rho * sizeof(double));
  }
  
  //printf("LIBGPU:: blksize= %i  naux= %i  nao= %i  nset= %i\n",blksize,naux,nao,nset);
  //printf("LIBGPU::shape: dmtril= (%i,%i)  eri1= (%i,%i)  rho= (%i, %i)   vj= (%i,%i)  vk= (%i,%i,%i)\n",
  //	 info_dmtril.shape[0], info_dmtril.shape[1],
  //      info_eri1.shape[0], info_eri1.shape[1],
  //  	 info_dmtril.shape[0], info_eri1.shape[0],
  // 	 info_dmtril.shape[0], info_eri1.shape[1],
  // 	 info_vk.shape[0],info_vk.shape[1],info_vk.shape[2]);

  int nao_pair = nao * (nao+1) / 2;
  
  if(with_j) {

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
    
    DevArray2D da_vj = DevArray2D(vj, nset, nao_pair);
    
    // vj += numpy.einsum('ip,px->ix', rho, eri1)

#pragma omp parallel for collapse(2)
    for(int i=0; i<nset; ++i)
      for(int j=0; j<nao_pair; ++j) {

	double val = 0.0;
	for(int k=0; k<naux; ++k) val += da_rho(i,k) * da_eri1(k,j);
	da_vj(i,j) += val;
      }
  }
 
  double * buf1 = buf_tmp;
  double * buf2 = &(buf_tmp[blksize * nao * nao]);
    
  DevArray3D da_buf1 = DevArray3D(buf1, naux, nao, nao);
  DevArray2D da_buf2 = DevArray2D(buf2, blksize * nao, nao);
  DevArray2D da_buf3 = DevArray2D(buf3, nao, naux * nao); // python swapped 1st two dimensions?
  
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
  
  //printf("Calling dev_push_async() for buf2\n");
  //pm->dev_push_async(d_buf2, buf2, blksize * nao * nao * sizeof(double), stream); // stream is nullptr
  pm->dev_push(d_buf2, buf2, blksize * nao * nao * sizeof(double));
  //printf(" -- finished\n");
  
  for(int indxK=0; indxK<nset; ++indxK) {
    
    py::array_t<double> _dms = static_cast<py::array_t<double>>(_dms_list[indxK]); // element of 3D array (nset, nao, nao)
    py::buffer_info info_dms = _dms.request(); // 2D

    // rargs = (ctypes.c_int(nao), (ctypes.c_int*4)(0, nao, 0, nao), null, ctypes.c_int(0))

    int orbs_slice[4] = {0, nao, 0, nao};
    double * dms = static_cast<double*>(info_dms.ptr);
    
    //    fmmm = _ao2mo.libao2mo.AO2MOmmm_bra_nr_s2
    //    fdrv = _ao2mo.libao2mo.AO2MOnr_e2_drv
    //    ftrans = _ao2mo.libao2mo.AO2MOtranse2_nr_s2
      
    //    fdrv(ftrans, fmmm,
    //	       buf1.ctypes.data_as(ctypes.c_void_p),
    //	       eri1.ctypes.data_as(ctypes.c_void_p),
    //	       dms[k].ctypes.data_as(ctypes.c_void_p),
    //	       ctypes.c_int(naux), *rargs)
    
    fdrv(buf1, eri1, dms, naux, nao, nullptr, nullptr, 0, buf_fdrv);
    
    // dgemm of (nao X blksize*nao) and (blksize*nao X nao) matrices - can refactor later...
    // vk[k] += lib.dot(buf1.reshape(-1,nao).T, buf2.reshape(-1,nao))  // vk[k] is nao x nao array
  
    // buf3 = buf1.reshape(-1,nao).T
    // buf4 = buf2.reshape(-1,nao)
    
#pragma omp parallel for collapse(3)
    for(int i=0; i<naux; ++i) {
      for(int j=0; j<nao; ++j)
	for(int k=0; k<nao; ++k) da_buf3(k,i*nao+j) = da_buf1(i,j,k);
    }

    // transfer

    profile_start("HtoD Transfer");
    
    //printf("Calling dev_push_async() for buf3\n");
    //pm->dev_barrier();
    //pm->dev_push_async(d_buf3, buf3, naux * nao * nao * sizeof(double), stream); // stream is nullptr
    pm->dev_push(d_buf3, buf3, naux * nao * nao * sizeof(double));
    //printf(" -- finished\n");
    
    //pm->dev_barrier();

    profile_stop();
    
    // vk[k] += lib.dot(buf3, buf4)
    // gemm(A,B,C) : C = 1.0 * A.B + 0.0 * C
    // A is (m, k) matrix
    // B is (k, n) matrix
    // C is (m, n) matrix
    // Column-ordered: (A.B)^T = B^T.A^T
    
    const double alpha = 1.0;
    const double beta = (count == 0) ? 0.0 : 1.0;
    
    const int m = nao; // # of rows of first matrix buf4^T
    const int n = nao; // # of cols of second matrix buf3^T
    const int k = naux*nao; // # of cols of first matrix buf4^

    const int lda = naux * nao;
    const int ldb = nao;
    const int ldc = nao;
    
    const int vk_offset = indxK * nao*nao;

    profile_start("DGEMM");

    //pm->dev_barrier();
    //printf("About to call cublasDgemm()\n");    
    
    //if(count == 0) pm->dev_push(d_vkk, vk, nset*nao*nao*sizeof(double));

#pragma omp target data //use_device_ptr(d_buf2, d_buf3, d_vkk)
    {
      //double * d_vkk_ = d_vkk+vk_offset;
      //cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_buf2, ldb, d_buf3, lda, &beta, d_vkk_, ldc);
      cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_buf2, ldb, d_buf3, lda, &beta, d_vkk+vk_offset, ldc);
    }
  
//#pragma omp target teams distribute parallel for
//    for(int i=0; i<nset*nao*nao; ++i) d_vkk[i] = i;

    //pm->dev_barrier();
    //printf(" -- finished\n");

    profile_stop();
   
#ifdef _SIMPLE_TIMER
    double t1 = omp_get_wtime();
    t_array[2] += t1 - t0;
#endif 
  }
  
  //pm->dev_pull(d_vkk, vk, nset * nao * nao * sizeof(double));
  //printf("Leaving get_jk()\n");
}
  
/* ---------------------------------------------------------------------- */

// pyscf/pyscf/lib/ao2mo/nr_ao2mo.c::AO2MOnr_e2_drv()

void Device::fdrv(double *vout, double *vin, double *mo_coeff,
		  int nij, int nao, int *orbs_slice, int *ao_loc, int nbas, double * _buf)
{
  const int ij_pair = nao * nao;
  const int nao2 = nao * (nao + 1) / 2;
    
#pragma omp parallel for
  for (int i = 0; i < nij; i++) {
    const int it = omp_get_thread_num();
    double * buf = &(_buf[it * nao * nao]);

    int _i, _j, _ij;
    double * tril = vin + nao2*i;
    for (_ij = 0, _i = 0; _i < nao; _i++) 
      for (_j = 0; _j <= _i; _j++, _ij++) buf[_i*nao+_j] = tril[_ij];
    
    const double D0 = 0;
    const double D1 = 1;
    const char SIDE_L = 'L';
    const char UPLO_U = 'U';

    double * _vout = vout + ij_pair*i;
    
    dsymm_(&SIDE_L, &UPLO_U, &nao, &nao, &D1, buf, &nao, mo_coeff, &nao, &D0, _vout, &nao);    
  }
}

#endif
