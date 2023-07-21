/* -*- c++ -*- */

#if defined(_USE_CPU)

#include "device.h"

#include <stdio.h>

extern "C" {
  void dsymm_(const char*, const char*, const int*, const int*,
	      const double*, const double*, const int*,
	      const double*, const int*,
	      const double*, double*, const int*);
  
  void dgemm_(const char * transa, const char * transb, const int * m, const int * n,
	      const int * k, const double * alpha, const double * a, const int * lda,
	      const double * b, const int * ldb, const double * beta, double * c,
	      const int * ldc);
}

/* ---------------------------------------------------------------------- */

double Device::compute(double * data)
{ 
  // do something useful
  
  double sum = 0.0;
#pragma omp parallel for reduction(+:sum)
  for(int i=0; i<n; ++i) {
    sum += data[i];
    data[i] += 1.0;
  }
    
  printf(" C-Kernel : n= %i  sum= %f\n",n, sum);
  
  return sum;
}

/* ---------------------------------------------------------------------- */

void Device::init_get_jk(py::array_t<double> _eri1, py::array_t<double> _dmtril, int blksize, int nset, int nao)
{
#ifdef _SIMPLE_TIMER
  double t0 = omp_get_wtime();
#endif
  
  py::buffer_info info_eri1 = _eri1.request(); // 2D array (232, 351)
  py::buffer_info info_dmtril = _dmtril.request(); // 2D array (nset, 351)

  double * eri1 = static_cast<double*>(info_eri1.ptr);
  double * dmtril = static_cast<double*>(info_dmtril.ptr);
  
  int _size_vj = info_dmtril.shape[0] * info_eri1.shape[1];
  if(_size_vj > size_vj) {
    size_vj = _size_vj;
    if(vj) free(vj);
    vj = (double *) malloc(size_vj * sizeof(double));
  }
  for(int i=0; i<_size_vj; ++i) vj[i] = 0.0;

  int _size_vk = nset * nao * nao;
  if(_size_vk > size_vk) {
    size_vk = _size_vk;
    if(_vktmp) free(_vktmp);
    _vktmp = (double *) malloc(size_vk*sizeof(double));
  }
  for(int i=0; i<_size_vk; ++i) _vktmp[i] = 0.0;

  int _size_buf = blksize * nao * nao;
  if(_size_buf > size_buf) {
    size_buf = _size_buf;
    if(buf_tmp) free(buf_tmp);
    if(buf3) free(buf3);
    if(buf4) free(buf4);
    
    buf_tmp = (double*) malloc(2*size_buf*sizeof(double));
    buf3 = (double *) malloc(size_buf*sizeof(double)); // (nao, blksize*nao)
    buf4 = (double *) malloc(size_buf*sizeof(double)); // (blksize*nao, nao)
  }
  
#ifdef _SIMPLE_TIMER
  t_array_jk[0] += omp_get_wtime() - t0;
#endif
}

/* ---------------------------------------------------------------------- */

void Device::free_get_jk()
{
}

/* ---------------------------------------------------------------------- */

void Device::get_jk(py::array_t<double> _eri1, py::array_t<double> _dmtril, py::array_t<double> _vjtmp,
		    py::array_t<double> _buftmp,
		    py::list & _dms_list, py::array_t<double> _vk,
		    int with_j, int blksize, int nset, int naux, int nao)
{  
  int num_threads = 1;
#pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }

#ifdef _SIMPLE_TIMER
  double t0 = omp_get_wtime();
#endif

  py::buffer_info info_eri1 = _eri1.request(); // 2D array (232, 351)
  py::buffer_info info_dmtril = _dmtril.request(); // 2D array (nset, 351)
  py::buffer_info info_buf = _buftmp.request();
  py::buffer_info info_vj = _vjtmp.request(); // 2D array (1, 351)
  py::buffer_info info_vk = _vk.request(); // 3D array (nset, 26, 26)

  double * eri1 = static_cast<double*>(info_eri1.ptr);
  double * dmtril = static_cast<double*>(info_dmtril.ptr);
  double * _buf = static_cast<double*>(info_buf.ptr);
  double * _vj = static_cast<double*>(info_vj.ptr);
  
  //double * vj; // = static_cast<double*>(info_vj.ptr);

  double * vk = static_cast<double*>(info_vk.ptr);

  int eri1_size_1d = info_eri1.shape[1];
  int dmtril_size_1d = info_dmtril.shape[1];

  int _size_rho = info_dmtril.shape[0] * info_eri1.shape[0];
  if(_size_rho > size_rho) {
    size_rho = _size_rho;
    if(rho) free(rho);
    rho = (double *) malloc(size_rho * sizeof(double));
  }
  
  int _size_vj = info_dmtril.shape[0] * info_eri1.shape[1];
  //vj = (double *) malloc(_size_vj * sizeof(double));
  //for(int i=0; i<_size_vj; ++i) vj[i] = 0.0;

  // printf("LIBGPU:: blksize= %i  naux= %i  nao= %i  nset= %i\n",blksize,naux,nao,nset);
  // printf("LIBGPU::shape: dmtril= (%i,%i)  eri1= (%i,%i)  rho= (%i, %i)   vj= (%i,%i)  vk= (%i,%i,%i)\n",
  // 	 info_dmtril.shape[0], info_dmtril.shape[1],
  // 	 info_eri1.shape[0], info_eri1.shape[1],
  // 	 info_dmtril.shape[0], info_eri1.shape[0],
  // 	 info_dmtril.shape[0], info_eri1.shape[1],
  // 	 info_vk.shape[0],info_vk.shape[1],info_vk.shape[2]);

#ifdef _SIMPLE_TIMER
  t_array_jk[1] += omp_get_wtime() - t0;
#endif
  
  if(with_j) {

#ifdef _SIMPLE_TIMER
    double t0 = omp_get_wtime();
#endif
    
    // rho = numpy.einsum('ix,px->ip', dmtril, eri1)

#pragma omp parallel for collapse(2)
    for(int i=0; i<info_dmtril.shape[0]; ++i)
      for(int j=0; j<info_eri1.shape[0]; ++j) {

	double val = 0.0;
	for(int k=0; k<dmtril_size_1d; ++k) {	  
	  int indx1 = i * info_dmtril.shape[1] + k;
	  int indx2 = j * info_eri1.shape[1] + k;
	  
	  // rho(i,j) += dmtril(i,k) * eri1(j,k)
	  val += dmtril[indx1] * eri1[indx2];
	}
	rho[i * info_eri1.shape[0] + j] = val;
      }
    
#ifdef _SIMPLE_TIMER
    double t1 = omp_get_wtime();
#endif
    
    // vj += numpy.einsum('ip,px->ix', rho, eri1)

#pragma omp parallel for collapse(2)
    for(int i=0; i<info_dmtril.shape[0]; ++i)
      for(int j=0; j<info_eri1.shape[1]; ++j) {

	double val = 0.0;
	for(int k=0; k<info_eri1.shape[0]; ++k) {
	  int indx1 = i * info_eri1.shape[0] + k;
	  int indx2 = k * info_eri1.shape[1] + j;

	  // vj(i,j) += rho(i,k) * eri1(k,j)
	  val += rho[indx1] * eri1[indx2];
	  //printf("ijk= %i %i %i  indx= %i %i %i  rho= %f  eri1= %f  vj= %f\n",i,j,k,indx,indx1,indx2,rho[indx1],eri1[indx2],vj[indx]);
	}
	//	vj[i * info_eri1.shape[1] + j] += val;
	_vj[i * info_eri1.shape[1] + j] += val;
      }

#ifdef _SIMPLE_TIMER
    t_array_jk[2] += t1 - t0;
    t_array_jk[3] += omp_get_wtime() - t1;
#endif
  }

#if 0
  double err = 0.0;
  for(int i=0; i<_size_vj; ++i) {
    err += (vj[i] - _vj[i]) * (vj[i] - _vj[i]);
    //    printf("LIBGPU:: i= %i  vj= %f\n",i,vj[i]);
  }
  printf("vj_error= %f\n",err);
  if(err > 0.1) {
    for(int i=0; i<_size_vj; ++i) printf("i= %i  vj_gpu= %f  vj_ref= %f\n",i,vj[i],_vj[i]);
    abort();
  }
#endif

#if 0
  for(int i=0; i<_size_vj; ++i) _vj[i] = vj[i];
#endif
 
  double * buf1 = buf_tmp;
  
  for(int indxK=0; indxK<nset; ++indxK) {

    py::array_t<double> _dms = static_cast<py::array_t<double>>(_dms_list[indxK]); // element of 3D array (nset, 26, 26)
    py::buffer_info info_dms = _dms.request(); // 2D
  
    // printf("  -- dms[k]: ndim= %i\n",info_dms.ndim);
    // printf("  --        shape=");
    // for(int i=0; i<info_dms.ndim; ++i) printf(" %i", info_dms.shape[i]);
    // printf("\n");

    // rargs = (ctypes.c_int(nao), (ctypes.c_int*4)(0, nao, 0, nao), null, ctypes.c_int(0))

    //    fmmm = _ao2mo.libao2mo.AO2MOmmm_bra_nr_s2
    //    fdrv = _ao2mo.libao2mo.AO2MOnr_e2_drv
    //    ftrans = _ao2mo.libao2mo.AO2MOtranse2_nr_s2
      
    //    fdrv(ftrans, fmmm,
    //	       buf1.ctypes.data_as(ctypes.c_void_p),
    //	       eri1.ctypes.data_as(ctypes.c_void_p),
    //	       dms[k].ctypes.data_as(ctypes.c_void_p),
    //	       ctypes.c_int(naux), *rargs)

    int orbs_slice[4] = {0, nao, 0, nao};
    double * dms = static_cast<double*>(info_dms.ptr);

#ifdef _SIMPLE_TIMER
    t0 = omp_get_wtime();
#endif
    
    fdrv(buf1, eri1, dms, naux, nao, orbs_slice, nullptr, 0);

#ifdef _SIMPLE_TIMER
    double t1 = omp_get_wtime();
    t_array_jk[4] += t1 - t0;
#endif
    
#if 0
    double err = 0.0;
    for(int i=0; i<naux*nao*nao; ++i) {
      err += (buf1[i]-_buf[i]) * (buf1[i]-_buf[i]);
      //      if(i < 2*nao) printf("i= %i  buf1_gpu= %f  buf1_ref= %f\n",i,buf1[i],_buf[i]);
    }
    printf("buf1_err= %f\n",err);
#endif
    
    // buf2 = lib.unpack_tril(eri1, out=buf[1])

#pragma omp parallel for
    for(int i=0; i<naux; ++i) {
      // unpack lower-triangle to square
      
      int indx = 0;
      double * buf2 = &(buf_tmp[(blksize + i) * nao * nao]);
      double * eri1_ = &(eri1[i * info_eri1.shape[1]]);
      for(int j=0; j<nao; ++j)
	for(int k=0; k<=j; ++k) {
	  int indx1 = j * nao + k;
	  int indx2 = k * nao + j;
	  buf2[indx1] = eri1_[indx];
	  buf2[indx2] = eri1_[indx];
	  indx++;
	}

      // if(i < 2) {
      // 	for(int j=0; j<2; ++j) {
      // 	  printf("\nLIBGPU::buf[%i,%i]= \n",i,j);
      // 	  for(int k=0; k<nao; ++k) printf(" %f",buf2[j*nao+k]);
      // 	  printf("\n");
      // 	}
      // }
      
    }

#ifdef _SIMPLE_TIMER
    double t2 = omp_get_wtime();
    t_array_jk[5] += t2 - t1;
#endif
    
#if 0
    err = 0.0;
    for(int i=0; i<naux*nao*nao; ++i) {
      int indx = blksize*nao*nao + i;
      //      if(i < 2*nao) printf("i= %i  buf2_gpu= %f  buf2_ref= %f\n",i,buf_tmp[indx],_buf[indx]);
      err += (buf_tmp[indx]-_buf[indx]) * (buf_tmp[indx]-_buf[indx]);
    }
    printf("buf2_err= %f\n",err);
#endif
    
    // dgemm of (nao X blksize*nao) and (blksize*nao X nao) matrices - can refactor later...
    // vk[k] += lib.dot(buf1.reshape(-1,nao).T, buf2.reshape(-1,nao))  // vk[k] is nao x nao array

#if 1
// tmp1 (26, 6032)
// tmp1[0,:]=  [ 0.37899273  0.44554206 -0.07293677  0.22942478  0.20389119 -0.13458347
//   0.0780132   0.         -0.0750426   0.04131208]
// tmp1[1,:]=  [ 0.2371684   0.28718115 -0.0409389   0.2390464   0.22400913 -0.09478144
//   0.05468187  0.         -0.05242745  0.02926458]

// tmp2 (6032, 26)
// tmp2[0,:]=  [ 0.91930255  0.44332476  0.01165171  0.16834878  0.24337649 -0.22437162
//   0.12954101  0.         -0.31094366  0.1795234 ]
// tmp2[1,:]=  [ 0.44332476  0.40159051  0.0295411   0.17964825  0.25563511 -0.14255808
//   0.08230595  0.         -0.23917498  0.13808774]
  
    // buf3 = buf1.reshape(-1,nao).T
    // buf4 = buf2.reshape(-1,nao)
    
    double * buf2 = &(buf_tmp[blksize * nao * nao]);
    
#pragma omp parallel for
    for(int i=0; i<naux; ++i)
      for(int j=0; j<nao; ++j)
	for(int k=0; k<nao; ++k) {

	  const int indx1 = i*nao*nao + j*nao + k;
	  const int indx3 = k * naux*nao + (i*nao+j);
	  
	  buf3[indx3] = buf1[indx1];
	  
	  const int indx2 = i*nao*nao + j*nao + k;
	  const int indx4 = (i*nao+j)*nao + k;
	  
	  buf4[indx4] = buf2[indx2];
	}

    // for(int i=0; i<2; ++i)
    //   for(int j=0; j<10; ++j) {
    // 	const int indx = i * nao + j;
    // 	printf("buf4: (%i,%i) indx= %i  val= %f\n",i,j,indx,buf4[indx]);
    //   }
    
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
    const double beta = 1.0; // 0 when count == 0
    
    const int m = nao; // # of rows of first matrix buf4^T
    const int n = nao; // # of cols of second matrix buf3^T
    const int k = naux*nao; // # of cols of first matrix buf4^

    const int lda = naux * nao;
    const int ldb = nao;
    const int ldc = nset * nao;

    //    double * vkk = &(_vktmp[indxK * nao]);
    double * vkk = &(vk[indxK * nao]);
    
    dgemm_((char *) "N", (char *) "N", &m, &n, &k, &alpha, buf4, &ldb, buf3, &lda, &beta, vkk, &ldc);    
#endif

#if 0
    //    double * buf2 = &(buf_tmp[blksize * nao * nao]);
    buf2 = &(buf_tmp[blksize * nao * nao]);
    
    for(int i=0; i<naux; ++i) {
      int offset = i * nao * nao;

      for(int irow=0; irow<nao; ++irow)
	for(int icol=0; icol<nao; ++icol) {
	  double val = 0.0;
	  for(int k=0; k<nao; ++k) {
	    int indx1T = offset + k*nao + irow;
	    int indx2  = offset + k*nao + icol;

	    val += buf1[indx1T] * buf2[indx2];
	  }
	  //	  _vktmp[indxK*nao*nao + irow*nao + icol] += val;

	  // seems like python has swapped order of first two indices in vk[i,j,k] to vk[j,i,k]

	  //	  _vktmp[irow*nset*nao + indxK*nao + icol] += val;
	  vk[irow*nset*nao + indxK*nao + icol] += val;
	  
	  // if(irow == 0 && icol == 0)
	  //   printf("i= %i  ijk= %i %i val= %f\n",i,irow,icol,val);
	}

      // printf("LIBGPU::buf1= \n");
      // for(int k=0; k<nao; ++k) printf("%i: %f\n",k,buf1[k*nao + 0]);
      
      // printf("\nLIBGPU::buf2= \n");
      // for(int k=0; k<nao; ++k) printf("%i: %f\n",k,buf2[k*nao + 0]);
    }
#endif
    
#if 0
    // for(int irow=0; irow<1; ++irow)
    //   for(int icol=0; icol<nao; ++icol) {
    // 	double val = 0.0;
    // 	for(int k=0; k<nao*naux; ++k) {
    // 	  int indx3 = irow*nao + k;
    // 	  int indx4 = k*nao + icol;
	  
    // 	  val += buf3[indx3] * buf4[indx4];
    // 	}
    // 	printf("vk :: irow= %i  icol= %i  vk_ref= %f  vk_gemm= %f  vk_val= %f\n",irow,icol,vk[irow*nao+icol],_vktmp[irow*nao+icol], val);
    //   }
    // abort();
    
    double vk_err = 0.0;
    int count = 0;
    printf("indxK= %i  nset= %i\n",indxK,nset);
    for(int i=0; i<nao; ++i)
      for(int j=0; j<nao; ++j) {
	int indx1 = indxK*nao*nao + i*nao + j;
	int indx2 = indxK*nao*nao + i*nao + j;
	
	const double diff = (vk[indx1] - _vktmp[indx2]) * (vk[indx1] - _vktmp[indx2]);
	
	if(diff > 1e-8 && count < 2*nao) {
	  printf("ij= %i %i  indx= %i %i  vk= %f  _vktmp= %f  diff= %f\n",
		 i,j,indx1,indx2,vk[indx1],_vktmp[indx2],diff);
	  count++;
	}
	vk_err += diff;
      }
    // printf("indxK= %i  vk_err= %f\n",indxK,vk_err);
    
    printf("nset= %i  vk_err= %f\n",nset, vk_err);
    if(vk_err > 1e-8) abort();
#endif
   
#ifdef _SIMPLE_TIMER
    double t4 = omp_get_wtime();
    t_array_jk[7] += t4 - t3;
#endif 
  }
  
  // printf("LIBGPU::vk[0,0,:]= \n");
  // for(int k=0; k<nao; ++k) printf("%i: %f\n",k,vk[k]);
  
  // printf("LIBGPU::vk[0,1,:]= \n");
  // for(int k=0; k<nao; ++k) printf("%i: %f\n",k,vk[nao + k]);
  
  // printf("LIBGPU::vk[1,0,:]= \n");
  // for(int k=0; k<nao; ++k) printf("%i: %f\n",k,vk[nao*nao + k]);
  
  // printf("LIBGPU::vk[1,1,:]= \n");
  // for(int k=0; k<nao; ++k) printf("%i: %f\n",k,vk[nao*nao + nao + k]);

#if 0
  double vk_err = 0.0;
  for(int indxK=0; indxK<nset; ++indxK) {
    //    printf("indxK= %i  nset= %i\n",indxK,nset);
    for(int i=0; i<nao; ++i)
      for(int j=0; j<nao; ++j) {
	int indx1 = indxK*nao*nao + i*nao + j;
	int indx2 = indxK*nao*nao + i*nao + j;
	vk_err += (vk[indx1] - _vktmp[indx2]) * (vk[indx1] - _vktmp[indx2]);
	if(vk_err > 1e-8) printf("indxK= %i  ij= %i %i  indx= %i %i  vk= %f  _vktmp= %f  vk_err= %f\n",
				 indxK,i,j,indx1,indx2,vk[indx1],_vktmp[indx2],vk_err);
      }
    // printf("indxK= %i  vk_err= %f\n",indxK,vk_err);
  }
  
  printf("nset= %i  vk_err= %f\n",nset, vk_err);
  if(vk_err > 1e-8) abort();
#endif

#if 0
  for(int i=0; i<nset*nao*nao; ++i) vk[i] = _vktmp[i];
#endif
}
  
/* ---------------------------------------------------------------------- */

// pyscf/pyscf/lib/ao2mo/nr_ao2mo.c::AO2MOnr_e2_drv()
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
    double *buf = (double *) malloc(sizeof(double) * (nao+i_count) * (nao+j_count));
#pragma omp for schedule(dynamic)
    for (i = 0; i < nij; i++) {
      ftrans(i, vout, vin, buf, &envs);
    }
    free(buf);
  }
}

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

// Is both _ocm2 in/out as it get over-written and resized?

void Device::orbital_response(py::array_t<double> _f1_prime,
			      py::array_t<double> _ppaa, py::array_t<double> _papa, py::array_t<double> _eri_paaa,
			      py::array_t<double> _ocm2, py::array_t<double> _tcm2, py::array_t<double> _gorb,
			      int ncore, int nocc, int nmo)
{
  int num_threads = 1;
#pragma omp parallel
  num_threads = omp_get_num_threads();
  
  py::buffer_info info_ppaa = _ppaa.request(); // 4D array (26, 26, 2, 2)
  py::buffer_info info_papa = _papa.request(); // 4D array (26, 2, 26, 2)
  py::buffer_info info_paaa = _eri_paaa.request();
  py::buffer_info info_ocm2 = _ocm2.request();
  py::buffer_info info_tcm2 = _tcm2.request();
  py::buffer_info info_gorb = _gorb.request();
  
  double * ppaa = static_cast<double*>(info_ppaa.ptr);
  double * papa = static_cast<double*>(info_papa.ptr);
  double * paaa = static_cast<double*>(info_paaa.ptr);
  double * ocm2 = static_cast<double*>(info_ocm2.ptr);
  double * tcm2 = static_cast<double*>(info_tcm2.ptr);
  double * gorb = static_cast<double*>(info_gorb.ptr);
  
  int ocm2_size_2d = info_ocm2.shape[2] * info_ocm2.shape[3];
  int ocm2_size_3d = info_ocm2.shape[1] * ocm2_size_2d;
    
  // printf("LIBGPU: Inside libgpu_orbital_response()\n");
  // printf("  -- ncore= %i\n",ncore); // 7
  // printf("  -- nocc=  %i\n",nocc); // 9
  // printf("  -- nmo=   %i\n",nmo); // 26
  // printf("  -- ppaa: ndim= %i\n",info_ppaa.ndim);
  // printf("  --       shape=");
  // for(int i=0; i<info_ppaa.ndim; ++i) printf(" %i", info_ppaa.shape[i]);
  // printf("\n");

  double * f1_prime = (double *) malloc(nmo*nmo*sizeof(double));
  
  // loop over f1 (i<nmo)
#pragma omp parallel for
  for(int p=0; p<nmo; ++p) {

    int it = omp_get_thread_num();
    
    double * f1 = &(f1_prime[p * nmo]);
  
    // pointers to slices of data
    
    int ppaa_size_2d = info_ppaa.shape[2] * info_ppaa.shape[3];
    int ppaa_size_3d = info_ppaa.shape[1] * ppaa_size_2d;
    double * praa = &(ppaa[p * ppaa_size_3d]); // ppaa.shape[1] x ppaa.shape[2] x ppaa.shape[3]
    
    int papa_size_2d = info_papa.shape[2] * info_papa.shape[3];
    int papa_size_3d = info_papa.shape[1] * papa_size_2d;
    double * para = &(papa[p * papa_size_3d]); // papa.shape[1] x papa.shape[2] x papa.shape[3]
    
    double * paaa = &(ppaa[p * ppaa_size_3d + ncore * ppaa_size_2d]); // (nocc-ncore) x ppaa.shape[2] x ppaa.shape[3]
    
    // ====================================================================
    // iteration (0, ncore)
    // ====================================================================
    
    // construct ra, ar, cm
    
    double * ra = praa; // (ncore,2,2)
    
    int indx = 0;
   
    double * cm = ocm2; // (2, 2, 2, ncore)
    
    for(int i=0; i<nmo; ++i) f1[i] = 0.0;
    
    // tensordot(paaa, cm, axes=((0,1,2), (2,1,0)))
    
    // printf("f1 += paaa{%i, %i, %i} X cm{%i, %i, %i, %i}\n",
    // 	   nocc-ncore,info_ppaa.shape[2],info_ppaa.shape[3],
    // 	   info_ocm2.shape[0],info_ocm2.shape[1],info_ocm2.shape[2],ncore);

#ifdef _SIMPLE_TIMER
    double t0 = omp_get_wtime();
#endif
    
    for(int i=0; i<ncore; ++i) {
      
      double val = 0.0;
      for(int k1=0; k1<info_ppaa.shape[3]; ++k1)
	for(int j1=0; j1<info_ppaa.shape[2]; ++j1)
	  for(int i1=0; i1<nocc-ncore; ++i1)
	    {
	      int indx1 = i1 * ppaa_size_2d + j1 * info_ppaa.shape[3] + k1;
	      int indx2 = k1 * ocm2_size_3d + j1 * ocm2_size_2d + i1 * info_ocm2.shape[3] + i;
	      val += paaa[indx1] * cm[indx2];
	    }
      
      f1[i] += val;
    }
    
#ifdef _SIMPLE_TIMER
    double t1 = omp_get_wtime();
#endif
    
    // tensordot(ra, cm, axes=((0,1,2), (3,0,1)))
    
    // printf("f1 += ra{%i, %i, %i} X cm{%i, %i, %i, %i}\n",
    // 	   ncore,info_ppaa.shape[2],info_ppaa.shape[3],
    // 	   info_ocm2.shape[0],info_ocm2.shape[1],info_ocm2.shape[2],ncore);
    
    for(int i=0; i<info_ocm2.shape[2]; ++i) {
      
      double val = 0.0;
      for(int k1=0; k1<info_ppaa.shape[3]; ++k1)
	for(int j1=0; j1<info_ppaa.shape[2]; ++j1)
	  for(int i1=0; i1<ncore; ++i1)
	    {
	      int indx1 = i1 * ppaa_size_2d + j1 * info_ppaa.shape[3] + k1;
	      int indx2 = j1 * ocm2_size_3d + k1 * ocm2_size_2d + i * info_ocm2.shape[3] + i1;
	      val += ra[indx1] * cm[indx2];
	    }
      
      f1[ncore+i] += val;
    }
    
#ifdef _SIMPLE_TIMER
    double t2 = omp_get_wtime();
#endif
    
    // tensordot(ar, cm, axes=((0,1,2), (0,3,2)))
    
    // printf("f1 += ar{%i, %i, %i} X cm{%i, %i, %i, %i}\n",
    // 	   info_papa.shape[1], ncore, info_papa.shape[3],
    // 	   info_ocm2.shape[0],info_ocm2.shape[1],info_ocm2.shape[2],ncore);

    for(int i=0; i<info_ocm2.shape[1]; ++i) {
      
      double val = 0.0;
      for(int k1=0; k1<info_ppaa.shape[3]; ++k1)
	for(int j1=0; j1<ncore; ++j1)
	  for(int i1=0; i1<info_papa.shape[1]; ++i1)
	    {
	      int indx1 = i1 * papa_size_2d + j1 * info_papa.shape[3] + k1;
	      int indx2 = i1 * ocm2_size_3d + i * ocm2_size_2d + k1 * info_ocm2.shape[3] + j1;
	      val += para[indx1] * cm[indx2];
	    }
      
      f1[ncore+i] += val;
    }
    
#ifdef _SIMPLE_TIMER
    double t3 = omp_get_wtime();
#endif
    
    // tensordot(ar, cm, axes=((0,1,2), (1,3,2)))
    
    // printf("f1 += ar{%i, %i, %i} X cm{%i, %i, %i, %i}\n",
    // 	   info_papa.shape[1], ncore, info_papa.shape[3],
    // 	   info_ocm2.shape[0],info_ocm2.shape[1],info_ocm2.shape[2],ncore);

    for(int i=0; i<info_ocm2.shape[0]; ++i) {
      
      double val = 0.0;
      for(int k1=0; k1<info_ppaa.shape[3]; ++k1)
	for(int j1=0; j1<ncore; ++j1)
	  for(int i1=0; i1<info_papa.shape[1]; ++i1)
	    {
	      int indx1 = i1 * papa_size_2d + j1 * info_papa.shape[3] + k1;
	      int indx2 = i * ocm2_size_3d + i1 * ocm2_size_2d + k1 * info_ocm2.shape[3] + j1;
	      val += para[indx1] * cm[indx2];
	    }
      
      f1[ncore+i] += val;
    }
    
#ifdef _SIMPLE_TIMER
    double t4 = omp_get_wtime();
#endif
    
    // ====================================================================
    // iteration (nocc, nmo)
    // ====================================================================
    
    // paaa = praa[ncore:nocc, :, :] = ppaa[p, ncore:nocc, :, :]
    // ra = praa[i:j] = ppaa[p, nocc:nmo, :, :]
    // ar = para[:, i:j] = papa[p, :, nocc:nmo, :]
    // cm = ocm2[:, :, :, i:j] = ocm2[:, :, :, nocc:nmo]

    // tensordot(paaa, cm, axes=((0,1,2), (2,1,0)))
    
    // printf("f1 += paaa{%i, %i, %i} X cm{%i, %i, %i, %i}\n",
    // 	   nmo-nocc,info_ppaa.shape[2],info_ppaa.shape[3],
    // 	   info_ocm2.shape[0],info_ocm2.shape[1],info_ocm2.shape[2],nmo-nocc);
    
    for(int i=nocc; i<nmo; ++i) {
      
      double val = 0.0;
      //int indx = 0;
      for(int k1=0; k1<info_ppaa.shape[3]; ++k1)
	for(int j1=0; j1<info_ppaa.shape[2]; ++j1)
	  for(int i1=0; i1<nocc-ncore; ++i1)
	    {
	      int indx1 = i1 * ppaa_size_2d + j1 * info_ppaa.shape[3] + k1;
	      int indx2 = k1 * ocm2_size_3d + j1 * ocm2_size_2d + i1 * info_ocm2.shape[3] + i;
	      val += paaa[indx1] * cm[indx2];
	    }
      
      f1[i] += val;
    }
    
#ifdef _SIMPLE_TIMER
    double t5 = omp_get_wtime();
#endif
    
    // tensordot(ra, cm, axes=((0,1,2), (3,0,1)))
    
    // printf("f1 += ra{%i, %i, %i} X cm{%i, %i, %i, %i}\n",
    // 	   nmo-nocc,info_ppaa.shape[2],info_ppaa.shape[3],
    // 	   info_ocm2.shape[0],info_ocm2.shape[1],info_ocm2.shape[2],nmo-nocc);
    
    for(int i=0; i<info_ocm2.shape[2]; ++i) {
      
      double val = 0.0;
      for(int k1=0; k1<info_ppaa.shape[3]; ++k1)
	for(int j1=0; j1<info_ppaa.shape[2]; ++j1)
	  for(int i1=0; i1<nmo-nocc; ++i1)
	    {
	      int indx1 = (nocc+i1) * ppaa_size_2d + j1 * info_ppaa.shape[3] + k1;
	      int indx2 = j1 * ocm2_size_3d + k1 * ocm2_size_2d + i * info_ocm2.shape[3] + (nocc+i1);
	      val += ra[indx1] * cm[indx2];
	    }
      
      f1[ncore+i] += val;
    }
    
#ifdef _SIMPLE_TIMER
    double t6 = omp_get_wtime();
#endif
    
    // tensordot(ar, cm, axes=((0,1,2), (0,3,2)))
    
    // printf("f1 += ar{%i, %i, %i} X cm{%i, %i, %i, %i}\n",
    // 	   info_papa.shape[1], nmo-nocc, info_papa.shape[3],
    // 	   info_ocm2.shape[0],info_ocm2.shape[1],info_ocm2.shape[2],nmo-nocc);

    for(int i=0; i<info_ocm2.shape[1]; ++i) {
      
      double val = 0.0;
      for(int k1=0; k1<info_ppaa.shape[3]; ++k1)
	for(int j1=0; j1<nmo-nocc; ++j1)
	  for(int i1=0; i1<info_papa.shape[1]; ++i1)
	    {
	      int indx1 = i1 * papa_size_2d + (nocc+j1) * info_papa.shape[3] + k1;
	      int indx2 = i1 * ocm2_size_3d + i * ocm2_size_2d + k1 * info_ocm2.shape[3] + (nocc+j1);
	      val += para[indx1] * cm[indx2];
	    }
      
      f1[ncore+i] += val;
    }
    
#ifdef _SIMPLE_TIMER
    double t7 = omp_get_wtime();
#endif
    
    // tensordot(ar, cm, axes=((0,1,2), (1,3,2)))
    
    // printf("f1 += ar{%i, %i, %i} X cm{%i, %i, %i, %i}\n",
    // 	   info_papa.shape[1], nmo-nocc, info_papa.shape[3],
    // 	   info_ocm2.shape[0],info_ocm2.shape[1],info_ocm2.shape[2],nmo-nocc);

    for(int i=0; i<info_ocm2.shape[0]; ++i) {
      
      double val = 0.0;
      for(int k1=0; k1<info_ppaa.shape[3]; ++k1)
	for(int j1=0; j1<nmo-nocc; ++j1)
	  for(int i1=0; i1<info_papa.shape[1]; ++i1)
	    {
	      int indx1 = i1 * papa_size_2d + (nocc+j1) * info_papa.shape[3] + k1;
	      int indx2 = i * ocm2_size_3d + i1 * ocm2_size_2d + k1 * info_ocm2.shape[3] + (nocc+j1);
	      val += para[indx1] * cm[indx2];
	    }
      
      f1[ncore+i] += val;
    }
#ifdef _SIMPLE_TIMER
    double t8 = omp_get_wtime();
    
    t_array[0] += t1  - t0;
    t_array[1] += t2  - t1;
    t_array[2] += t3  - t2;
    t_array[3] += t4  - t3;
    t_array[4] += t5  - t4;
    t_array[5] += t6  - t5;
    t_array[6] += t7  - t6;
    t_array[7] += t8  - t7;
#endif
  } // for(p<nmo)

  // # (H.x_aa)_va, (H.x_aa)_ac

  int _ocm2_size_1d = nocc - ncore;
  int _ocm2_size_2d = info_ocm2.shape[2] * _ocm2_size_1d;
  int _ocm2_size_3d = info_ocm2.shape[1] * ocm2_size_2d;
  int size_ecm = info_ocm2.shape[0] * info_ocm2.shape[1] * info_ocm2.shape[2] * (nocc-ncore);
  
  double * _ocm2t = (double *) malloc(size_ecm * sizeof(double));
  double * ecm2 = (double *) malloc(size_ecm * sizeof(double)); // tmp space and ecm2
  
  // ocm2 = ocm2[:,:,:,ncore:nocc] + ocm2[:,:,:,ncore:nocc].transpose (1,0,3,2)

#ifdef _SIMPLE_TIMER
  double t8 = omp_get_wtime();
#endif
  
  int indx = 0;
  double * _ocm2_tmp = ecm2;
  for(int i=0; i<info_ocm2.shape[0]; ++i)
    for(int j=0; j<info_ocm2.shape[1]; ++j)
      for(int k=0; k<info_ocm2.shape[2]; ++k)
	for(int l=0; l<(nocc-ncore); ++l)
	  {
	    int indx1 = i * ocm2_size_3d + j * ocm2_size_2d + k * info_ocm2.shape[3] + (ncore+l);
	    int indx2 = j * ocm2_size_3d + i * ocm2_size_2d + l * info_ocm2.shape[3] + (ncore+k);
	    _ocm2_tmp[indx++] = ocm2[indx1] + ocm2[indx2];
	  }
  
#ifdef _SIMPLE_TIMER
  double t9 = omp_get_wtime();
#endif
  
  // ocm2 += ocm2.transpose (2,3,0,1)

  _ocm2_size_3d = info_ocm2.shape[1] * _ocm2_size_2d;
  
  indx = 0;
  for(int i=0; i<info_ocm2.shape[0]; ++i)
    for(int j=0; j<info_ocm2.shape[1]; ++j)
      for(int k=0; k<info_ocm2.shape[2]; ++k)
	for(int l=0; l<(nocc-ncore); ++l)
	  {
	    int indx1 = i * _ocm2_size_3d + j * _ocm2_size_2d + k * _ocm2_size_1d + l;
	    int indx2 = k * _ocm2_size_3d + l * _ocm2_size_2d + i * _ocm2_size_1d + j;
	    _ocm2t[indx] = _ocm2_tmp[indx1] + _ocm2_tmp[indx2];
	    indx++;
	  }
  
#ifdef _SIMPLE_TIMER
  double t10 = omp_get_wtime();
#endif
    
  // ecm2 = ocm2 + tcm2
  
  for(int i=0; i<size_ecm; ++i) ecm2[i] = _ocm2t[i] + tcm2[i];

#ifdef _SIMPLE_TIMER
  double t11 = omp_get_wtime();
#endif
  
  // f1_prime[:ncore,ncore:nocc] += np.tensordot (self.eri_paaa[:ncore], ecm2, axes=((1,2,3),(1,2,3)))
  
  int paaa_size_1d = info_paaa.shape[3];
  int paaa_size_2d = info_paaa.shape[2] * paaa_size_1d;
  int paaa_size_3d = info_paaa.shape[1] * paaa_size_2d;
  
  for(int i=0; i<ncore; ++i) 
    for(int j=0; j<(nocc-ncore); ++j) {
      
      double val = 0.0;
      for(int k1=0; k1<info_ppaa.shape[3]; ++k1)
	for(int j1=0; j1<info_paaa.shape[2]; ++j1)
	  for(int i1=0; i1<info_paaa.shape[1]; ++i1)
	    {
	      int indx1 = i * paaa_size_3d + i1 * paaa_size_2d + j1 * paaa_size_1d + k1;
	      int indx2 = j * _ocm2_size_3d + i1 * _ocm2_size_2d + j1 * _ocm2_size_1d + k1;
	      val += paaa[indx1] * ecm2[indx2];
	    }
      
      f1_prime[i*nmo+ncore+j] += val;
    }

#ifdef _SIMPLE_TIMER
  double t12 = omp_get_wtime();
#endif
    
  // f1_prime[nocc:,ncore:nocc] += np.tensordot (self.eri_paaa[nocc:], ecm2, axes=((1,2,3),(1,2,3)))

  for(int i=nocc; i<nmo; ++i) 
    for(int j=0; j<(nocc-ncore); ++j) {
      
      double val = 0.0;
      for(int k1=0; k1<info_ppaa.shape[3]; ++k1)
	for(int j1=0; j1<info_paaa.shape[2]; ++j1)
	  for(int i1=0; i1<info_paaa.shape[1]; ++i1)
	    {
	      int indx1 = i * paaa_size_3d + i1 * paaa_size_2d + j1 * paaa_size_1d + k1;
	      int indx2 = j * _ocm2_size_3d + i1 * _ocm2_size_2d + j1 * _ocm2_size_1d + k1;
	      val += paaa[indx1] * ecm2[indx2];
	    }
      
      f1_prime[i*nmo+ncore+j] += val;
    }
  
#ifdef _SIMPLE_TIMER
  double t13 = omp_get_wtime();
#endif
    
  // return gorb + (f1_prime - f1_prime.T)

  double * g_f1_prime = (double *) malloc(nmo*nmo*sizeof(double));
  
  indx = 0;
  for(int i=0; i<nmo; ++i)
    for(int j=0; j<nmo; ++j) {
      int indx1 = i * nmo + j;
      int indx2 = j * nmo + i;
      g_f1_prime[indx] = gorb[indx] + f1_prime[indx1] - f1_prime[indx2];
      indx++;
    }
  
  py::buffer_info info_f1_prime = _f1_prime.request();
  double * res = static_cast<double*>(info_f1_prime.ptr);

  for(int i=0; i<nmo*nmo; ++i) res[i] = g_f1_prime[i];

#ifdef _SIMPLE_TIMER
  double t14 = omp_get_wtime();

  t_array[8]  += t9  - t8;
  t_array[9]  += t10 - t9;
  t_array[10] += t11 - t10;
  t_array[11] += t12 - t11;
  t_array[12] += t13 - t12;
  t_array[13] += t14 - t13;
#endif
  
#if 0
  free(ar_global);
#endif
  
  free(g_f1_prime);
  free(ecm2);
  free(_ocm2t);
  free(f1_prime);
}

#endif
