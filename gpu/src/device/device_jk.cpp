/* -*- c++ -*- */

#include <stdio.h>

#include "device.h"

/* ---------------------------------------------------------------------- */

DeviceJk::DeviceJk(DeviceContext & _ctx)
  : ctx(_ctx)
{
  buf_vj = nullptr;
  buf_vk = nullptr;
  size_buf_vj = 0;
  size_buf_vk = 0;
}

DeviceJk::~DeviceJk()
{
  if(buf_vj) ctx.pm->dev_free_host(buf_vj);
  if(buf_vk) ctx.pm->dev_free_host(buf_vk);
  buf_vj = nullptr;
  buf_vk = nullptr;
}

/* ---------------------------------------------------------------------- */

void DeviceJk::init_get_jk(py::array_t<double> _eri1, py::array_t<double> _dmtril, int blksize, int nset, int nao, int naux, int count)
{
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Inside Device::init_get_jk() :: blksize= %i  nset= %i  nao= %i  naux= %i  count= %i\n",blksize,nset,nao,naux,count);
#endif

  ctx.pm->dev_profile_start("init_get_jk");
  
  double t0 = omp_get_wtime();

  const int device_id = count % ctx.num_devices;
  
  ctx.pm->dev_set_device(device_id);

  my_device_data * dd = &(ctx.device_data[device_id]);
  
  //  if(dd->stream == nullptr) dd->stream = pm->dev_get_queue();
  
  int nao_pair = nao * (nao+1) / 2;
  
  int _size_vj = nset * nao_pair;

  grow_array(ctx.pm, dd->jk.d_vj, _size_vj, dd->jk.size_vj, "vj", FLERR);
  
  int _size_vk = nset * nao * nao;

  grow_array(ctx.pm, dd->jk.d_vkk, _size_vk, dd->jk.size_vk, "vkk", FLERR);

  int _size_buf = blksize * nao * nao;
  if(_size_vj > _size_buf) _size_buf = _size_vj;
  if(_size_vk > _size_buf) _size_buf = _size_vk;
  
  grow_array(ctx.pm, dd->jk.d_buf1, _size_buf, dd->jk.size_buf1, "buf1", FLERR);
  grow_array(ctx.pm, dd->jk.d_buf2, _size_buf, dd->jk.size_buf2, "buf2", FLERR);
  grow_array(ctx.pm, dd->jk.d_buf3, _size_buf, dd->jk.size_buf3, "buf3", FLERR);
  
  int _size_dms = nset * nao * nao;
  grow_array(ctx.pm, dd->jk.d_dms, _size_dms, dd->jk.size_dms, "dms", FLERR);

  int _size_dmtril = nset * nao_pair;
  grow_array(ctx.pm, dd->jk.d_dmtril, _size_dmtril, dd->jk.size_dmtril, "dmtril", FLERR);

  int _size_buf_vj = ctx.num_devices * nset * nao_pair;
  grow_array_host(ctx.pm, buf_vj, _size_buf_vj, size_buf_vj, "h:buf_vj");

  int _size_buf_vk = ctx.num_devices * nset * nao * nao;
  grow_array_host(ctx.pm, buf_vk, _size_buf_vk, size_buf_vk, "h:buf_vk");

  // 1-time initialization
  
  ctx.cache->dd_fetch_pumap(dd, nao, _PUMAP_2D_UNPACK);
  
  // Create blas handle

  // if(dd->handle == nullptr) {
  //   ml->create_handle();
  //   //    dd->handle = ml->get_handle();
  // }
 
  // do all devices participate in calculation?
  
  if(count == 0) 
    for(int i=0; i<ctx.num_devices; ++i) ctx.device_data[i].active = 0;
  
  ctx.pm->dev_profile_stop();
  
  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1 - t0);
 //counts in pull_get_jk

#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: -- Leaving Device::init_get_jk()\n");
#endif
}

/* ---------------------------------------------------------------------- */

// The _vj and _vk arguements aren't actually used anymore and could be removed.

void DeviceJk::get_jk(int naux, int nao, int nset,
		    py::array_t<double> _eri1, py::array_t<double> _dmtril, py::list & _dms_list,
		    py::array_t<double> _vj, py::array_t<double> _vk,
		    int with_k, int count, size_t addr_dfobj)
{
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Inside Device::get_jk() w/ with_k= %i\n",with_k);
#endif
  
  double t0 = omp_get_wtime();

  ctx.pm->dev_profile_start("get_jk :: init");

  const int device_id = count % ctx.num_devices;
  
  ctx.pm->dev_set_device(device_id);

  my_device_data * dd = &(ctx.device_data[device_id]);

  dd->active = 1;

  const int with_j = 1;
  
  py::buffer_info info_eri1 = _eri1.request(); // 2D array (naux, nao_pair)
  py::buffer_info info_dmtril = _dmtril.request(); // 2D array (nset, nao_pair)

  double * eri1 = static_cast<double*>(info_eri1.ptr);
  double * dmtril = static_cast<double*>(info_dmtril.ptr);
  
  int nao_pair = nao * (nao+1) / 2;

  // Bcast() from master device ; make sure devices arrays allocated
  
#if defined(_ENABLE_P2P)
  if(count == 0) {
    size_t size = nset * nao_pair * sizeof(double);

    std::vector<double *> dmtril_vec(ctx.num_devices); // array of device addresses 

    dmtril_vec[0] = dd->jk.d_dmtril;
    
    for(int i=1; i<ctx.num_devices; ++i) {
      my_device_data * dest = &(ctx.device_data[i]);

      // ensure memory allocated ; duplicating what's in init_get_jk()
      
      if(size > dest->jk.size_dmtril) {
	dest->jk.size_dmtril = size;

	ctx.pm->dev_set_device(i);
	if(dest->jk.d_dmtril) ctx.pm->dev_free(dest->jk.d_dmtril, "dmtril");
	dest->jk.d_dmtril = (double *) ctx.pm->dev_malloc(size * sizeof(double), "dmtril", FLERR); // why is this not async?
      }
      
      dmtril_vec[i] = dest->jk.d_dmtril;
    }
    
    ctx.comm->mgpu_bcast(dmtril_vec, dmtril, size);  // host -> gpu 0, then Bcast to all gpu
  }
#else
  if(count < ctx.num_devices) {
    int err = ctx.pm->dev_push_async(dd->jk.d_dmtril, dmtril, nset * nao_pair * sizeof(double));
    if(err) {
      printf("LIBGPU:: dev_push_async(d_dmtril) failed on count= %i\n",count);
      exit(1);
    }
  }
#endif
    
  int _size_rho = nset * naux;
  grow_array(ctx.pm, dd->jk.d_rho, _size_rho, dd->jk.size_rho, "rho", FLERR);
    
#if 0
  py::buffer_info info_vj = _vj.request(); // 2D array (nset, nao_pair)
  py::buffer_info info_vk = _vk.request(); // 3D array (nset, nao, nao)
  
  printf("LIBGPU:: device= %i  naux= %i  nao= %i  nset= %i  nao_pair= %i  count= %i\n",device_id,naux,nao,nset,nao_pair,count);
  printf("LIBGPU::shape: dmtril= (%i,%i)  eri1= (%i,%i)  rho= (%i, %i)   vj= (%i,%i)  vk= (%i,%i,%i)\n",
  	 info_dmtril.shape[0], info_dmtril.shape[1],
  	 info_eri1.shape[0], info_eri1.shape[1],
  	 info_dmtril.shape[0], info_eri1.shape[0],
  	 info_vj.shape[0], info_vj.shape[1],
  	 info_vk.shape[0],info_vk.shape[1],info_vk.shape[2]);
  
  DevArray2D da_eri1 = DevArray2D(eri1, naux, nao_pair, ctx.pm, DA_HOST);
  //  printf("LIBGPU:: eri1= %p  dfobj= %lu  count= %i  combined= %lu\n",eri1,addr_dfobj,count,addr_dfobj+count);
  printf("LIBGPU:: dfobj= %#012x  count= %i  combined= %#012x  update_dfobj= %i\n",addr_dfobj,count,addr_dfobj+count, ctx.cache->update_dfobj);
  printf("LIBGPU::     0:      %f %f %f %f\n",da_eri1(0,0), da_eri1(0,1), da_eri1(0,nao_pair-2), da_eri1(0,nao_pair-1));
  printf("LIBGPU::     1:      %f %f %f %f\n",da_eri1(1,0), da_eri1(1,1), da_eri1(1,nao_pair-2), da_eri1(1,nao_pair-1));
  printf("LIBGPU::     naux-2: %f %f %f %f\n",da_eri1(naux-2,0), da_eri1(naux-2,1), da_eri1(naux-2,nao_pair-2), da_eri1(naux-2,nao_pair-1));
  printf("LIBGPU::     naux-1: %f %f %f %f\n",da_eri1(naux-1,0), da_eri1(naux-1,1), da_eri1(naux-1,nao_pair-2), da_eri1(naux-1,nao_pair-1));
#endif
  
  double * d_eri = ctx.cache->dd_fetch_eri(dd, eri1, naux, nao_pair, addr_dfobj, count);

  ctx.pm->dev_profile_stop();
  
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Starting with_j calculation\n");
#endif

  if (with_j){
    
    ctx.pm->dev_profile_start("get_jk :: with_j");
    
    // rho = numpy.einsum('ix,px->ip', dmtril, eri1)
    
    getjk_rho(dd->jk.d_rho, dd->jk.d_dmtril, d_eri, nset, naux, nao_pair);
    
    // vj += numpy.einsum('ip,px->ix', rho, eri1)
   
    int init = (count < ctx.num_devices) ? 1 : 0;
  
    getjk_vj(dd->jk.d_vj, dd->jk.d_rho, d_eri, nset, nao_pair, naux, init);
    
    ctx.pm->dev_profile_stop();
  }
    
  if(!with_k) {
    
    double t1 = omp_get_wtime();
    LIBGPU_PROFILE(ctx, t1 - t0);
// counts in pull_jk
    
#ifdef _DEBUG_DEVICE
    printf("LIBGPU ::  -- Leaving Device::get_jk()\n");
#endif
    
    return;
  }
  
  // buf2 = lib.unpack_tril(eri1, out=buf[1])
    
  ctx.pm->dev_profile_start("get_jk :: with_k");

  ctx.utils->getjk_unpack_buf2(dd->jk.d_buf2, d_eri, dd->fci.d_pumap_ptr, naux, nao, nao_pair);

#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- finished\n");
  printf("LIBGPU :: Starting with_k calculation\n");
#endif
    
  for(int indxK=0; indxK<nset; ++indxK) {

    double t4 = omp_get_wtime();
    
    py::array_t<double> _dms = static_cast<py::array_t<double>>(_dms_list[indxK]); // element of 3D array (nset, nao, nao)
    py::buffer_info info_dms = _dms.request(); // 2D

    double * dms = static_cast<double*>(info_dms.ptr);

    double * d_dms = &(dd->jk.d_dms[indxK*nao*nao]);

    if(count < ctx.num_devices) {
#ifdef _DEBUG_DEVICE
      printf("LIBGPU ::  -- calling dev_push_async(dms) for indxK= %i  nset= %i\n",indxK,nset);
#endif
    
      int err = ctx.pm->dev_push_async(d_dms, dms, nao*nao*sizeof(double));
      if(err) {
	printf("LIBGPU:: dev_push_async(d_dms) on indxK= %i\n",indxK);
	printf("LIBGPU:: d_dms= %p  dms= %p  nao= %i  device= %i\n",(void*) d_dms, (void*) dms,nao,device_id);
	exit(1);
      }
    }

    {
      const double alpha = 1.0;
      const double beta = 0.0;
      const int nao2 = nao * nao;
      const int zero = 0;

      ctx.ml->set_handle();
      ctx.ml->gemm_batch((char *) "T", (char *) "T", &nao, &nao, &nao,
		     &alpha, dd->jk.d_buf2, &nao, &nao2, d_dms, &nao, &zero, &beta, dd->jk.d_buf1, &nao, &nao2, &naux);
    }
    
    // dgemm of (nao X blksize*nao) and (blksize*nao X nao) matrices - can refactor later...
    // vk[k] += lib.dot(buf1.reshape(-1,nao).T, buf2.reshape(-1,nao))  // vk[k] is nao x nao array
  
    // buf3 = buf1.reshape(-1,nao).T
    // buf4 = buf2.reshape(-1,nao)
    
    ctx.utils->transpose(dd->jk.d_buf3, dd->jk.d_buf1, naux*nao, nao);
    
    // vk[k] += lib.dot(buf3, buf4)
    // gemm(A,B,C) : C = alpha * A.B + beta * C
    // A is (m, k) matrix
    // B is (k, n) matrix
    // C is (m, n) matrix
    // Column-ordered: (A.B)^T = B^T.A^T

    {
      const double alpha = 1.0;
      const double beta = (count < ctx.num_devices) ? 0.0 : 1.0; // first pass by each device initializes array, otherwise accumulate
      
      const int m = nao; // # of rows of first matrix buf4^T
      const int n = nao; // # of cols of second matrix buf3^T
      const int k = naux*nao; // # of cols of first matrix buf4^
      
      const int lda = naux * nao;
      const int ldb = nao;
      const int ldc = nao;
      
      const int vk_offset = indxK * nao*nao;
      
      ctx.ml->set_handle();
      ctx.ml->gemm((char *) "N", (char *) "N", &m, &n, &k, &alpha, dd->jk.d_buf2, &ldb, dd->jk.d_buf3, &lda, &beta, (dd->jk.d_vkk)+vk_offset, &ldc);
    }
  
  } // for(nset)
    
  ctx.pm->dev_profile_stop();
    
  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1 - t0);
  // counts in pull jk
    
#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- finished\n");
  printf("LIBGPU :: -- Leaving Device::get_jk()\n");
#endif
}
  
/* ---------------------------------------------------------------------- */

#if defined(_ENABLE_P2P)
void DeviceJk::pull_get_jk(py::array_t<double> _vj, py::array_t<double> _vk, int nao, int nset, int with_k)
{
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: -- Inside Device::pull_get_jk()\n");
#endif
 
  double t0 = omp_get_wtime();
  
  ctx.pm->dev_profile_start("pull_get_jk");
  
  py::buffer_info info_vj = _vj.request(); // 2D array (nset, nao_pair)
  
  double * vj = static_cast<double*>(info_vj.ptr);
  
  int nao_pair = nao * (nao+1) / 2;
  
  int N = nset * nao_pair;
  
  std::vector<double *> v_vec(ctx.num_devices);
  std::vector<double *> buf_vec(ctx.num_devices);
  std::vector<int> active(ctx.num_devices);
  
  for(int i=0; i<ctx.num_devices; ++i) {
    my_device_data * dd = &(ctx.device_data[i]);
    v_vec[i] = dd->jk.d_vj;
    buf_vec[i] = dd->jk.d_buf3;
    active[i] = dd->active;
  }
  
  if(v_vec[0]) {
    ctx.comm->mgpu_reduce(v_vec, buf_vj, N, true, buf_vec, active);
    
#pragma omp parallel for
    for(int j=0; j<N; ++j) vj[j] += buf_vj[j];
  }
  
  ctx.cache->update_dfobj = 0;
  
  if(!with_k) {
    ctx.pm->dev_profile_stop();
    
#ifdef _DEBUG_DEVICE
    printf("LIBGPU :: -- Leaving Device::pull_get_jk()\n");
#endif
    
    return;
  }
  
  py::buffer_info info_vk = _vk.request(); // 3D array (nset, nao, nao)
  
  double * vk = static_cast<double*>(info_vk.ptr);
  
  N = nset * nao * nao;
  
  for(int i=0; i<ctx.num_devices; ++i) {
    my_device_data * dd = &(ctx.device_data[i]);
    v_vec[i] = dd->jk.d_vkk;
  }
  
  if(v_vec[0]) {
    ctx.comm->mgpu_reduce(v_vec, buf_vk, N, true, buf_vec, active);
    
#pragma omp parallel for
    for(int j=0; j<N; ++j) vk[j] += buf_vk[j];
  }
  
  ctx.pm->dev_profile_stop();
  
  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1 - t0); // just doing this addition in pull, not in init or compute
  
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: -- Leaving Device::pull_get_jk()\n");
#endif
}

#else

void DeviceJk::pull_get_jk(py::array_t<double> _vj, py::array_t<double> _vk, int nao, int nset, int with_k)
{
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: -- Inside Device::pull_get_jk()\n");
#endif

  double t0 = omp_get_wtime();
    
  ctx.pm->dev_profile_start("pull_get_jk");
  
  py::buffer_info info_vj = _vj.request(); // 2D array (nset, nao_pair)
  
  double * vj = static_cast<double*>(info_vj.ptr);

  int nao_pair = nao * (nao+1) / 2;
  
  int size = nset * nao_pair * sizeof(double);

  double * tmp;
  
  for(int i=0; i<ctx.num_devices; ++i) {
    ctx.pm->dev_set_device(i);
    
    my_device_data * dd = &(ctx.device_data[i]);

    if(i == 0) tmp = vj;
    else tmp = &(buf_vj[i * nset * nao_pair]);
    
    if(dd->active) ctx.pm->dev_pull_async(dd->jk.d_vj, tmp, size);
  }
  
  for(int i=0; i<ctx.num_devices; ++i) {
    ctx.pm->dev_set_device(i);
    
    my_device_data * dd = &(ctx.device_data[i]);
    
    ctx.pm->dev_stream_wait();

    if(i > 0 && dd->active) {
      
      tmp = &(buf_vj[i * nset * nao_pair]);
#pragma omp parallel for
      for(int j=0; j<nset*nao_pair; ++j) vj[j] += tmp[j];
      
    }
  }
  
  ctx.cache->update_dfobj = 0;
  
  if(!with_k) {
    ctx.pm->dev_profile_stop();
    
#ifdef _DEBUG_DEVICE
    printf("LIBGPU :: -- Leaving Device::pull_get_jk()\n");
#endif
    
    return;
  }
    
  py::buffer_info info_vk = _vk.request(); // 3D array (nset, nao, nao)
    
  double * vk = static_cast<double*>(info_vk.ptr);

  size = nset * nao * nao * sizeof(double);

  for(int i=0; i<ctx.num_devices; ++i) {
    ctx.pm->dev_set_device(i);
      
    my_device_data * dd = &(ctx.device_data[i]);

    if(i == 0) tmp = vk;
    else tmp = &(buf_vk[i * nset * nao * nao]);

    if(dd->active) ctx.pm->dev_pull_async(dd->jk.d_vkk, tmp, size);
  }

  for(int i=0; i<ctx.num_devices; ++i) {
    ctx.pm->dev_set_device(i);
    
    my_device_data * dd = &(ctx.device_data[i]);
    
    ctx.pm->dev_stream_wait();

    if(i > 0 && dd->active) {
      
      tmp = &(buf_vk[i * nset * nao * nao]);
#pragma omp parallel for
      for(int j=0; j<nset*nao*nao; ++j) vk[j] += tmp[j];
    
    }

  }

  ctx.pm->dev_profile_stop();
  
  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1 - t0); // just doing this addition in pull, not in init or compute
    
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: -- Leaving Device::pull_get_jk()\n");
#endif
}
#endif

/* ---------------------------------------------------------------------- */
/* Device facade forwarders (Phase 5 Option A: keep the flat Python API)    */
/* ---------------------------------------------------------------------- */

void Device::init_get_jk(py::array_t<double> _eri1, py::array_t<double> _dmtril, int blksize, int nset, int nao, int naux, int count)
{ _jk->init_get_jk(_eri1, _dmtril, blksize, nset, nao, naux, count); }

void Device::get_jk(int naux, int nao, int nset,
		    py::array_t<double> _eri1, py::array_t<double> _dmtril, py::list & _dms_list,
		    py::array_t<double> _vj, py::array_t<double> _vk,
		    int with_k, int count, size_t addr_dfobj)
{ _jk->get_jk(naux, nao, nset, _eri1, _dmtril, _dms_list, _vj, _vk, with_k, count, addr_dfobj); }

void Device::pull_get_jk(py::array_t<double> _vj, py::array_t<double> _vk, int nao, int nset, int with_k)
{ _jk->pull_get_jk(_vj, _vk, nao, nset, with_k); }
