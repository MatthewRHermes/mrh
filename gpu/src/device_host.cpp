/* -*- c++ -*- */

#if defined(_USE_CPU)

#include "device.h"

#include <stdio.h>

/* ---------------------------------------------------------------------- */

void Device::init_get_jk(py::array_t<double> _eri1, py::array_t<double> _dmtril, int _blksize, int nset, int nao)
{
#ifdef _SIMPLE_TIMER
  double t0 = omp_get_wtime();
#endif

  blksize = _blksize;
  
  py::buffer_info info_eri1 = _eri1.request(); // 2D array (232, 351)
  py::buffer_info info_dmtril = _dmtril.request(); // 2D array (nset, 351)

  // double * eri1 = static_cast<double*>(info_eri1.ptr);
  // double * dmtril = static_cast<double*>(info_dmtril.ptr);
  
  int _size_vj = info_dmtril.shape[0] * info_eri1.shape[1];
  if(_size_vj > size_vj) {
    size_vj = _size_vj;
    if(vj) pm->dev_free_host(vj);
    vj = (double *) pm->dev_malloc_host(size_vj * sizeof(double));
  }
  for(int i=0; i<_size_vj; ++i) vj[i] = 0.0;

  int _size_vk = nset * nao * nao;
  if(_size_vk > size_vk) {
    size_vk = _size_vk;
    if(_vktmp) pm->dev_free_host(_vktmp);
    _vktmp = (double *) pm->dev_malloc_host(size_vk*sizeof(double));
  }
  for(int i=0; i<_size_vk; ++i) _vktmp[i] = 0.0;

  int _size_buf = blksize * nao * nao;
  if(_size_buf > size_buf) {
    size_buf = _size_buf;
    if(buf_tmp) pm->dev_free_host(buf_tmp);
    if(buf3) pm->dev_free_host(buf3);
    if(buf4) pm->dev_free_host(buf4);
    
    buf_tmp = (double*) pm->dev_malloc_host(2*size_buf*sizeof(double));
    buf3 = (double *) pm->dev_malloc_host(size_buf*sizeof(double)); // (nao, blksize*nao)
    buf4 = (double *) pm->dev_malloc_host(size_buf*sizeof(double)); // (blksize*nao, nao)
  }

  int _size_fdrv = 4 * nao * nao * num_threads;
  if(_size_fdrv > size_fdrv) {
    size_fdrv = _size_fdrv;
    if(buf_fdrv) pm->dev_free_host(buf_fdrv);
    buf_fdrv = (double *) pm->dev_malloc_host(size_fdrv*sizeof(double));
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

void Device::get_jk(int naux, int nao, int nset, 
		    py::array_t<double> _eri1, py::array_t<double> _dmtril, py::list & _dms_list,
		    py::array_t<double> _vj, py::array_t<double> _vk,
		    int count)
{  
#ifdef _SIMPLE_TIMER
  double t0 = omp_get_wtime();
#endif

  const int with_j = true;
  
  py::buffer_info info_eri1 = _eri1.request(); // 2D array (232, 351)
  py::buffer_info info_dmtril = _dmtril.request(); // 2D array (nset, 351)
  py::buffer_info info_vj = _vj.request(); // 2D array (1, 351)
  py::buffer_info info_vk = _vk.request(); // 3D array (nset, 26, 26)

  double * eri1 = static_cast<double*>(info_eri1.ptr);
  double * dmtril = static_cast<double*>(info_dmtril.ptr);
  double * vj = static_cast<double*>(info_vj.ptr);
  double * vk = static_cast<double*>(info_vk.ptr);

  int eri1_size_1d = info_eri1.shape[1];
  int dmtril_size_1d = info_dmtril.shape[1];

  int _size_rho = info_dmtril.shape[0] * info_eri1.shape[0];
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
  
  for(int indxK=0; indxK<nset; ++indxK) {

    py::array_t<double> _dms = static_cast<py::array_t<double>>(_dms_list[indxK]); // element of 3D array (nset, 26, 26)
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

    double * buf2 = &(buf_tmp[blksize * nao * nao]);
    
    DevArray3D da_buf2 = DevArray3D(buf2, blksize, nao, nao);
    
#pragma omp parallel for
    for(int i=0; i<naux; ++i) {
      
      int indx = 0;
      double * eri1_ = &(eri1[i * nao_pair]);

      // unpack lower-triangle to square
      
      for(int j=0; j<nao; ++j)
	for(int k=0; k<=j; ++k) {	  
	  da_buf2(i,j,k) = eri1_[indx];
	  da_buf2(i,k,j) = eri1_[indx];
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

    DevArray3D da_buf1 = DevArray3D(buf_tmp, naux, nao, nao);
    DevArray3D da_buf3 = DevArray3D(buf3, nao, naux, nao); // python swapped 1st two dimensions?
    
#pragma omp parallel for
    for(int i=0; i<naux; ++i) {
      for(int j=0; j<nao; ++j)
	for(int k=0; k<nao; ++k) da_buf3(k,i,j) = da_buf1(i,j,k);
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
    const double beta = (count == 0) ? 0.0 : 1.0; // 0 when count == 0
    
    const int m = nao; // # of rows of first matrix buf4^T
    const int n = nao; // # of cols of second matrix buf3^T
    const int k = naux*nao; // # of cols of first matrix buf4^

    const int lda = naux * nao;
    const int ldb = nao;
    const int ldc = nset * nao;

    double * vkk = &(vk[indxK * nao]);
    
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

// Is both _ocm2 in/out as it get over-written and resized?

void Device::orbital_response(py::array_t<double> _f1_prime,
			      py::array_t<double> _ppaa, py::array_t<double> _papa, py::array_t<double> _eri_paaa,
			      py::array_t<double> _ocm2, py::array_t<double> _tcm2, py::array_t<double> _gorb,
			      int ncore, int nocc, int nmo)
{  
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

  double * f1_prime = (double *) pm->dev_malloc_host(nmo*nmo*sizeof(double));
  
  // loop over f1 (i<nmo)
#pragma omp parallel for
  for(int p=0; p<nmo; ++p) {
    
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
  
  double * _ocm2t = (double *) pm->dev_malloc_host(size_ecm * sizeof(double));
  double * ecm2 = (double *) pm->dev_malloc_host(size_ecm * sizeof(double)); // tmp space and ecm2
  
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

  double * g_f1_prime = (double *) pm->dev_malloc_host(nmo*nmo*sizeof(double));
  
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

  t_array_count++;
#endif
  
#if 0
  pm->dev_free_host(ar_global);
#endif
  
  pm->dev_free_host(g_f1_prime);
  pm->dev_free_host(ecm2);
  pm->dev_free_host(_ocm2t);
  pm->dev_free_host(f1_prime);
}

#endif
