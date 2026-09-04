/* -*- c++ -*- */

#include <stdio.h>

#include "device.h"

#include <unistd.h>
#include <string.h>
#include <sched.h>
#define _MIN(A,B) (A<B)?A:B
#define _MAX(A,B) (A>B)?A:B
#define _SIZE_FCI_BATCHES 6

/* ---------------------------------------------------------------------- */

DeviceImpham::DeviceImpham(DeviceContext & ctx) : ctx(ctx)
{
  size_eri_impham = 0;
  pin_eri_impham = nullptr;
}

DeviceImpham::~DeviceImpham()
{
  if(pin_eri_impham) ctx.pm->dev_free_host(pin_eri_impham);
  pin_eri_impham = nullptr;
}

/* ---------------------------------------------------------------------- */

void DeviceImpham::init_eri_impham(int naoaux, int nao_f, int return_4c2eeri)
{
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Inside Device::init_eri_impham()  return_4c2eeri= %i\n",return_4c2eeri);
  if (return_4c2eeri) printf("LIBGPU :: -- returning 4c2e\n");
  else printf("LIBGPU :: -- returning 3c2e\n");
#endif
  
  double t0 = omp_get_wtime();
  
  ctx.pm->dev_profile_start("init_eri_impham");

  int nao_f_pair = nao_f*(nao_f+1)/2;
  int _size_eri_impham = 0;
  
  if (return_4c2eeri) _size_eri_impham = ctx.num_devices * nao_f_pair*nao_f_pair;  //when used like this, answer accumulates on gpu
  else _size_eri_impham = naoaux*nao_f_pair;  // answer accumulates on cpu
  
  if (_size_eri_impham > size_eri_impham) {
    size_eri_impham = _size_eri_impham;
    
#ifdef _DEBUG_DEVICE
    printf("resizing eri_impham in init\n");
    printf("size_eri %d\n",size_eri_impham );
#endif
    
    if (pin_eri_impham) ctx.pm->dev_free_host(pin_eri_impham);
    pin_eri_impham = (double *) ctx.pm->dev_malloc_host(size_eri_impham*sizeof(double));
  }
  
  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1 - t0);
  
  ctx.pm->dev_profile_stop();
  
  // counts in pull eri_impham
  
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Leaving Device::init_eri_impah()\n");
#endif
}

/* ---------------------------------------------------------------------- */

void DeviceImpham::compute_eri_impham(int nao_s, int nao_f, int blksize, int naux, int count, size_t addr_dfobj, int return_4c2eeri)
{
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Inside Device::comute_eri_impham()\n");
  printf("LIBGPU :: -- from gpu: %i %i %i %i %i\n",nao_s, nao_f, blksize, naux, count);
#endif
  
  ctx.pm->dev_profile_start("compute_eri_impham");
  
  double t0 = omp_get_wtime();

  const int device_id = count % ctx.num_devices;
  
  ctx.pm->dev_set_device(device_id);
  
  my_device_data * dd = &(ctx.device_data[device_id]);

  dd->active = 1;

  // using fetch_eri, assume it's already there

  int nao_s_pair = nao_s * (nao_s + 1)/2;

  double * d_cderi = ctx.cache->dd_fetch_eri(dd, nullptr, naux, nao_s_pair, addr_dfobj, count);

  double * d_cderi_unpacked = dd->jk.d_buf1;

  int * d_my_unpack_map_ptr = ctx.cache->dd_fetch_pumap(dd, nao_s, _PUMAP_2D_UNPACK);

  ctx.utils->getjk_unpack_buf2(d_cderi_unpacked, d_cderi, d_my_unpack_map_ptr, naux, nao_s, nao_s_pair);

  const double alpha = 1.0;
  const double beta = 0.0;
  int zero = 0;
  int nao_s2 = nao_s * nao_s;
  int nao_sf = nao_s * nao_f;
  int nao_f2 = nao_f * nao_f;
  int nao_f_pair = nao_f * (nao_f+1)/2;
  
  double * d_bPeu = dd->jk.d_buf2;

  // b^P_ue = b^P_uu * M_ue
  
  ctx.ml->set_handle();
  ctx.ml->gemm_batch((char *) "N", (char *) "T", 
               &nao_s, &nao_f, &nao_s,
               &alpha, 
               d_cderi_unpacked, &nao_s, &nao_s2, 
               dd->d_mo_coeff, &nao_f, &zero, 
               &beta, 
               d_bPeu, &nao_s, &nao_sf, 
               &naux);

  // b^P_ee = b^P_ue * M_ue
  
  double * d_bPee = dd->jk.d_buf1;
  
  ctx.ml->gemm_batch((char *) "N", (char *) "N", 
               &nao_f, &nao_f, &nao_s,
               &alpha, 
               dd->d_mo_coeff, &nao_f, &zero, 
               d_bPeu, &nao_s, &nao_sf, 
               &beta, 
               d_bPee, &nao_f, &nao_f2, 
               &naux);

  //do packing
 
  d_my_unpack_map_ptr = ctx.cache->dd_fetch_pumap(dd, nao_f, _PUMAP_2D_UNPACK);

  double * d_eri_unpacked = dd->jk.d_buf2;

  pack_eri(d_eri_unpacked, d_bPee, d_my_unpack_map_ptr, naux, nao_f, nao_f_pair);

  if (return_4c2eeri){
    double beta_ = (count < ctx.num_devices) ? 0.0 : 1.0;
#ifdef _DEBUG_DEVICE
    printf("returning 4c2e\n");
    printf("beta %f\n",beta_);
#endif
    
    ctx.ml->gemm((char *) "N", (char *) "T", &nao_f_pair, &nao_f_pair, &naux,
	     &alpha, d_eri_unpacked, &nao_f_pair, d_eri_unpacked, &nao_f_pair, &beta_, dd->jk.d_buf3, &nao_f_pair);

  } else {
#ifdef _DEBUG_DEVICE
    printf("returning 3c2e\n");
#endif
    
    double * eri_impham = &(pin_eri_impham[count*blksize * nao_f_pair]);

    ctx.pm->dev_pull_async(d_eri_unpacked, eri_impham, naux*nao_f_pair*sizeof(double));
  }
  
#if 0
  double * h_eri_impham = (double *)ctx.pm->dev_malloc_host(nao_f_pair*nao_f_pair*sizeof(double));
  ctx.pm->dev_pull_async(dd->jk.d_buf3, h_eri_impham, nao_f_pair*nao_f_pair*sizeof(double));
  ctx.pm->dev_stream_wait();
  for (int i =0;i<nao_f_pair;++i){ for (int j=0;j<nao_f_pair;++j){printf("%f\t",h_eri_impham[i*nao_f_pair+j]); }printf("\n");}
  ctx.pm->dev_free_host(h_eri_impham);
#endif
  
  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1 - t0);
  
  ctx.pm->dev_profile_stop();
  
  // counts in pull eri_impham
}

/* ---------------------------------------------------------------------- */

void DeviceImpham::compute_eri_impham_v2(int nao_s, int nao_f, int blksize, int naux, int count, size_t addr_dfobj_in, size_t addr_dfobj_out)
{
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Inside Device::comute_eri_impham()\n");
#endif

  ctx.pm->dev_profile_start("compute_eri_impham");
  double t0 = omp_get_wtime();

  const int device_id = count % ctx.num_devices;
  ctx.pm->dev_set_device(device_id);
  my_device_data * dd = &(ctx.device_data[device_id]);
  
  dd->active = 1;

  double * d_cderi = nullptr;
  // using fetch_eri, assume it's already there
  int nao_s_pair = nao_s * (nao_s + 1)/2;
  d_cderi = ctx.cache->dd_fetch_eri(dd, nullptr, naux, nao_s_pair, addr_dfobj_in, count);
  
  double * d_cderi_unpacked = dd->jk.d_buf1;

  int * d_my_unpack_map_ptr = ctx.cache->dd_fetch_pumap(dd, nao_s, _PUMAP_2D_UNPACK);

  ctx.utils->getjk_unpack_buf2(d_cderi_unpacked,d_cderi,d_my_unpack_map_ptr,naux, nao_s, nao_s_pair);

  const double alpha = 1.0;
  const double beta = 0.0;
  int zero = 0;
  int nao_s2 = nao_s * nao_s;
  int nao_sf = nao_s * nao_f;
  int nao_f2 = nao_f * nao_f;
  int nao_f_pair = nao_f * (nao_f+1)/2;
  
  double * d_bPeu = dd->jk.d_buf2;
  
  // b^P_ue = b^P_uu * M_ue
  
  ctx.ml->set_handle();
  ctx.ml->gemm_batch((char *) "N", (char *) "T", 
               &nao_s, &nao_f, &nao_s,
               &alpha, 
               d_cderi_unpacked, &nao_s, &nao_s2, 
               dd->d_mo_coeff, &nao_f, &zero, 
               &beta, 
               d_bPeu, &nao_s, &nao_sf, 
               &naux);
  
  // b^P_ee = b^P_ue * M_ue
  
  double * d_bPee = dd->jk.d_buf1;
  
  ctx.ml->gemm_batch((char *) "N", (char *) "N", 
               &nao_f, &nao_f, &nao_s,
               &alpha, 
               dd->d_mo_coeff, &nao_f, &zero, 
               d_bPeu, &nao_s, &nao_sf, 
               &beta, 
               d_bPee, &nao_f, &nao_f2, 
               &naux);

  //do packing
  
  d_my_unpack_map_ptr = ctx.cache->dd_fetch_pumap(dd, nao_f, _PUMAP_2D_UNPACK);
  // new (transfer to exisiting smaller cholesky vector)
  double * d_cderi_out = ctx.cache->dd_fetch_eri(dd, nullptr, naux, nao_f_pair, addr_dfobj_out, count);
  //TODO: add growing logic 
  //ctx.ml->gemm((char *) "T", (char *) "N", &nao_f_pair, &nao_f_pair, &naux, &alpha, dd->jk.d_buf2, &ldb, dd->jk.d_buf3, &lda, &beta, (dd->jk.d_vkk)+vk_offset, &ldc);

  pack_eri(d_cderi_out, d_bPee,d_my_unpack_map_ptr, naux, nao_f, nao_f_pair);
  ctx.pm->dev_profile_stop();
  
  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1 - t0); // just doing this addition in pull, not in init or compute
  // counts in pull eri_impham

}

/* ---------------------------------------------------------------------- */

#if defined(_ENABLE_P2P)
void DeviceImpham::pull_eri_impham(py::array_t<double> _eri, int naoaux, int nao_f, int return_4c2eeri)
{
  //This should be obsolete in a production version. We want this calculation to not exist, and the impurity eri should directly get transferred from gpu to gpu in it's corresponding location. 

  // if not possible, then the cpu version should be refactored to allow pull to happen async (i think it's pageable right now and it will negate all performance when you pull bPee to cpu (and then transfer it back again)) 
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: -- Inside Device::pull_eri_impham()\n");
#endif
  
  ctx.pm->dev_profile_start("pull_eri_impham");
  
  double t0 = omp_get_wtime();
  
  int nao_f_pair = nao_f * (nao_f+1)/2;

  int N = nao_f_pair * nao_f_pair;
  
  py::buffer_info info_eri = _eri.request(); 
  double * eri = static_cast<double*>(info_eri.ptr);

  if (return_4c2eeri){

    std::vector<double *> e_vec(ctx.num_devices);
    std::vector<double *> buf_vec(ctx.num_devices);
    std::vector<int> active(ctx.num_devices);
  
    for(int i=0; i<ctx.num_devices; ++i) {
      my_device_data * dd = &(ctx.device_data[i]);
      e_vec[i] = dd->jk.d_buf3; // this has the result
      buf_vec[i] = dd->jk.d_buf2; // this is a temp buffer
      active[i] = dd->active;
    }

    ctx.comm->mgpu_reduce(e_vec, pin_eri_impham, N, true, buf_vec, active);

#pragma omp parallel for
    for(int i=0; i<N; ++i) eri[i] += pin_eri_impham[i];

  } else {

    for(int i=0; i<ctx.num_devices; ++i) {
      ctx.pm->dev_set_device(i);
      ctx.pm->dev_barrier();
    }

#pragma omp parallel for
    for(int i=0; i<naoaux*nao_f_pair; ++i) eri[i] = pin_eri_impham[i];
  }

  ctx.pm->dev_profile_stop();
  
  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1 - t0); // just doing this addition in pull, not in init or compute
    
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: -- Leaving Device::pull_eri_impham()\n");
#endif

}

#else

void DeviceImpham::pull_eri_impham(py::array_t<double> _eri, int naoaux, int nao_f, int return_4c2eeri)
{
  //This should be obsolete in a production version. We want this calculation to not exist, and the impurity eri should directly get transferred from gpu to gpu in it's corresponding location. 

  // if not possible, then the cpu version should be refactored to allow pull to happen async (i think it's pageable right now and it will negate all performance when you pull bPee to cpu (and then transfer it back again)) 
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: -- Inside Device::pull_eri_impham()\n");
#endif
  
  ctx.pm->dev_profile_start("pull_eri_impham");
  
  double t0 = omp_get_wtime();
  
  int nao_f_pair = nao_f * (nao_f+1)/2;
  py::buffer_info info_eri = _eri.request(); 
  double * eri = static_cast<double*>(info_eri.ptr);
  
#if 0
  printf("starting pull\n");
  for (int i=0;i<nao_f_pair*nao_f_pair; ++i){printf("%f\t",eri[i]);}printf("\n");
#endif

  if (return_4c2eeri){
  
    for (int i=0; i<ctx.num_devices; ++i){
      ctx.pm->dev_set_device(i); 
      my_device_data * dd = &(ctx.device_data[i]);
      double * eri_impham =&pin_eri_impham[i * nao_f_pair*nao_f_pair];
      if (dd->active) ctx.pm->dev_pull_async(dd->jk.d_buf3, eri_impham, nao_f_pair*nao_f_pair*sizeof(double));
    }

#ifdef _DEBUG_DEVICE
    printf("returning 4c2e\n");
    for (int i=0;i<ctx.num_devices;++i){
      ctx.pm->dev_set_device(i); 
      my_device_data * dd = &(ctx.device_data[i]);
      ctx.pm->dev_stream_wait();
      if (dd->jk.d_buf3){
	for (int j=0;j <nao_f_pair;++j){
	  for (int k=0;k <nao_f_pair;++k){
	    printf("%f\t",pin_eri_impham[i*nao_f_pair*nao_f_pair+j*nao_f_pair+k]);
	  } printf("\n");
	}} printf("\n");
    }
#endif

    for(int i=0; i<ctx.num_devices; ++i) {
      ctx.pm->dev_set_device(i);
      my_device_data * dd = &(ctx.device_data[i]);
      ctx.pm->dev_stream_wait();
      
      if(dd->active) {
	double * tmp = &(pin_eri_impham[i * nao_f_pair*nao_f_pair]);
#pragma omp parallel for
	for(int j=0; j<nao_f_pair*nao_f_pair; ++j) eri[j] += tmp[j];
      }
    }

  } else {
    
#ifdef _DEBUG_DEVICE
    printf("returning 3c2e\n");
#endif

    for(int i=0; i<ctx.num_devices; ++i) {
      ctx.pm->dev_set_device(i);
      ctx.pm->dev_barrier();
    }
    
    std::memcpy(eri, pin_eri_impham, naoaux*nao_f_pair*sizeof(double));
  }

  ctx.pm->dev_profile_stop();
  
  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1 - t0); // just doing this addition in pull, not in init or compute
    
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: -- Leaving Device::pull_eri_impham()\n");
#endif

}

#endif

/* ---------------------------------------------------------------------- */
/* Device facade forwarders (Phase 5 Option A: keep the flat Python API)    */
/* ---------------------------------------------------------------------- */

void Device::init_eri_impham(int naoaux, int nao_f, int return_4c2eeri)
{ _impham->init_eri_impham(naoaux, nao_f, return_4c2eeri); }

void Device::compute_eri_impham(int nao_s, int nao_f, int blksize, int naux, int count, size_t addr_dfobj, int return_4c2eeri)
{ _impham->compute_eri_impham(nao_s, nao_f, blksize, naux, count, addr_dfobj, return_4c2eeri); }

void Device::pull_eri_impham(py::array_t<double> _eri, int naoaux, int nao_f, int return_4c2eeri)
{ _impham->pull_eri_impham(_eri, naoaux, nao_f, return_4c2eeri); }

void Device::compute_eri_impham_v2(int nao_s, int nao_f, int blksize, int naux, int count, size_t addr_dfobj_in, size_t addr_dfobj_out)
{ _impham->compute_eri_impham_v2(nao_s, nao_f, blksize, naux, count, addr_dfobj_in, addr_dfobj_out); }
