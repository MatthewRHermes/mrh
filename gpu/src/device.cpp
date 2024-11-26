/* -*- c++ -*- */

#include <stdio.h>

#include "device.h"

#define _NUM_SIMPLE_TIMER 11

#define _DEBUG_OPENMP

#ifdef _DEBUG_OPENMP
#include <unistd.h>
#include <string.h>
#include <sched.h>
#endif

/* ---------------------------------------------------------------------- */

Device::Device()
{
  printf("LIBGPU: created device\n");
  
  pm = new PM();

  ml = new MATHLIB(pm);

  update_dfobj = 0;
  
  rho = nullptr;
  //vj = nullptr;
  _vktmp = nullptr;

  buf_tmp = nullptr;
  buf3 = nullptr;
  buf4 = nullptr;

  buf_fdrv = nullptr;

  size_buf_vj = 0;
  size_buf_vk = 0;
  
  buf_vj = nullptr;
  buf_vk = nullptr;
  
  //ao2mo
  size_buf_j_pc = 0;
  size_buf_k_pc = 0;
  size_fxpp = 0;
  size_bufpa = 0;

  buf_j_pc = nullptr;
  buf_k_pc = nullptr;
  pin_fxpp = nullptr;
  pin_bufpa = nullptr;

#if defined(_USE_GPU)
  use_eri_cache = true;
#endif
  
  num_threads = 1;
#pragma omp parallel
  num_threads = omp_get_num_threads();

  num_devices = pm->dev_num_devices();
  
  device_data = (my_device_data*) pm->dev_malloc_host(num_devices * sizeof(my_device_data));

  for(int i=0; i<num_devices; ++i) {
    device_data[i].device_id = i;
    
    device_data[i].size_rho = 0;
    device_data[i].size_vj = 0;
    device_data[i].size_vk = 0;
    device_data[i].size_buf = 0;
    device_data[i].size_dms = 0;
    device_data[i].size_dmtril = 0;
    device_data[i].size_eri1 = 0;
    device_data[i].size_ucas = 0;
    device_data[i].size_umat = 0;
    device_data[i].size_h2eff = 0;
    device_data[i].size_mo_coeff = 0;
    device_data[i].size_j_pc = 0;
    device_data[i].size_k_pc = 0;
    device_data[i].size_bufd = 0;
    device_data[i].size_bufpa = 0;
    
    device_data[i].d_rho = nullptr;
    device_data[i].d_vj = nullptr;
    device_data[i].d_buf1 = nullptr;
    device_data[i].d_buf2 = nullptr;
    device_data[i].d_buf3 = nullptr;
    device_data[i].d_vkk = nullptr;
    device_data[i].d_dms = nullptr;
    device_data[i].d_mo_coeff=nullptr;
    device_data[i].d_dmtril = nullptr;
    device_data[i].d_eri1 = nullptr; // when not using eri cache
    device_data[i].d_ucas = nullptr;
    device_data[i].d_umat = nullptr;
    device_data[i].d_h2eff = nullptr;
    
    device_data[i].d_pumap_ptr = nullptr;
    
    device_data[i].d_j_pc = nullptr;
    device_data[i].d_k_pc = nullptr;
    device_data[i].d_bufd = nullptr;
    device_data[i].d_bufpa = nullptr;

#if defined (_USE_GPU)
    device_data[i].handle = nullptr;
    device_data[i].stream = nullptr;
#endif
  }

#ifdef _DEBUG_OPENMP
  char nname[16];
  gethostname(nname, 16);
  int rnk = 0;
  
#pragma omp parallel for ordered
  for(int it=0; it<omp_get_num_threads(); ++it) {
    char list_cores[7*CPU_SETSIZE];
    get_cores(list_cores);
#pragma omp ordered
    printf("LIBGPU: To affinity and beyond!! nname= %s  rnk= %d  tid= %d: list_cores= (%s)\n",
	   nname, rnk, omp_get_thread_num(), list_cores);
  }

#endif
  
#ifdef _SIMPLE_TIMER
  t_array = (double* ) malloc(_NUM_SIMPLE_TIMER * sizeof(double));
  for(int i=0; i<_NUM_SIMPLE_TIMER; ++i) t_array[i] = 0.0;
#endif
}

/* ---------------------------------------------------------------------- */

Device::~Device()
{
  printf("LIBGPU: destroying device\n");

  pm->dev_free_host(rho);
  //pm->dev_free_host(vj);
  pm->dev_free_host(_vktmp);

  pm->dev_free_host(buf_tmp);
  pm->dev_free_host(buf3);
  pm->dev_free_host(buf4);

  pm->dev_free_host(buf_vj);
  pm->dev_free_host(buf_vk);
  
  pm->dev_free_host(buf_fdrv);
  
  pm->dev_free_host(buf_j_pc);
  pm->dev_free_host(buf_k_pc);
  pm->dev_free_host(pin_fxpp);
  pm->dev_free_host(pin_bufpa);


#ifdef _SIMPLE_TIMER
  double total = 0.0;
  for(int i=0; i<_NUM_SIMPLE_TIMER; ++i) total += t_array[i];
  
  printf("\nLIBGPU :: SIMPLE_TIMER\n");
  printf("\nLIBGPU :: SIMPLE_TIMER :: get_jk\n");
  printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= init_get_jk()      time= %f s\n",0,t_array[0]);
  printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= pull_get_jk()      time= %f s\n",1,t_array[1]);
  printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= get_jk()           time= %f s\n",2,t_array[2]);
    
  printf("\nLIBGPU :: SIMPLE_TIMER :: hessop\n");
  printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= hessop_get_veff()  time= %f s\n",3,t_array[3]);
  
  printf("\nLIBGPU :: SIMPLE_TIMER :: orbital_response\n");
  printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= orbital_response() time= %f s\n",4,t_array[4]);

  
  printf("\nLIBGPU :: SIMPLE_TIMER :: _update_h2eff\n");
  printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= update_h2eff_sub() time= %f s\n",5,t_array[5]);

  printf("\nLIBGPU :: SIMPLE_TIMER :: _h2eff_df \n");
  printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= h2eff_df() time= %f s\n",6,t_array[6]);
 
  printf("\nLIBGPU :: SIMPLE_TIMER :: transfer_mo_coeff \n");
  printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= transfer_mo_coeff() time= %f s\n",7,t_array[7]);
  
  printf("\nLIBGPU :: SIMPLE_TIMER :: df_ao2mo_pass1\n");
  printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= init_ints_and_jkpc() time= %f s\n",8,t_array[8]);
  printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= compute_ints_and_jkpc() time= %f s\n",9,t_array[9]);
  printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= pull_ints_and_jkpc() time= %f s\n",10,t_array[10]);
  printf("LIBGPU :: SIMPLE_TIMER :: total= %f s\n",total);
  free(t_array);
#endif

  // print summary of cached eri blocks

  if(use_eri_cache) {
    printf("\nLIBGPU :: eri cache statistics :: count= %i\n",eri_list.size());
    for(int i=0; i<eri_list.size(); ++i)
      printf("LIBGPU :: %i : eri= %p  Mbytes= %f  count= %i  update= %i device= %i\n", i, eri_list[i],
	     eri_size[i]*sizeof(double)/1024./1024., eri_count[i], eri_update[i], eri_device[i]);
    
    eri_count.clear();
    eri_size.clear();
#ifdef _DEBUG_ERI_CACHE
    for(int i=0; i<d_eri_host.size(); ++i) pm->dev_free_host( d_eri_host[i] );
#endif
    for(int i=0; i<d_eri_cache.size(); ++i) pm->dev_free( d_eri_cache[i] );
    eri_list.clear();
  }
  
#if defined(_USE_GPU)
  for(int i=0; i<num_devices; ++i) {
  
    pm->dev_set_device(i);
    my_device_data * dd = &(device_data[i]);
    
    pm->dev_free(dd->d_rho);
    pm->dev_free(dd->d_vj);
    pm->dev_free(dd->d_buf1);
    pm->dev_free(dd->d_buf2);
    pm->dev_free(dd->d_buf3);
    pm->dev_free(dd->d_vkk);
    pm->dev_free(dd->d_dms);
    pm->dev_free(dd->d_dmtril);
    pm->dev_free(dd->d_eri1);
    pm->dev_free(dd->d_h2eff);
    pm->dev_free(dd->d_mo_coeff);
    pm->dev_free(dd->d_j_pc);
    pm->dev_free(dd->d_k_pc);
    for(int i=0; i<dd->size_pumap.size(); ++i) {
      pm->dev_free_host(dd->pumap[i]);
      pm->dev_free(dd->d_pumap[i]);
    }
    dd->type_pumap.clear();
    dd->size_pumap.clear();
    dd->pumap.clear();
    dd->d_pumap.clear();

#if defined (_USE_GPU)
    if(dd->handle) cublasDestroy(dd->handle);
    
    if(dd->stream) pm->dev_stream_destroy(dd->stream);
#endif
  }

  printf("LIBGPU :: Finished\n");
#endif

  delete ml;
  
  delete pm;
}

/* ---------------------------------------------------------------------- */

// xthi.c from http://docs.cray.com/books/S-2496-4101/html-S-2496-4101/cnlexamples.html

// util-linux-2.13-pre7/schedutils/taskset.c
void Device::get_cores(char *str)
{
  cpu_set_t mask;
  sched_getaffinity(0, sizeof(cpu_set_t), &mask);

  char *ptr = str;
  int i, j, entry_made = 0;
  for (i = 0; i < CPU_SETSIZE; i++) {
    if (CPU_ISSET(i, &mask)) {
      int run = 0;
      entry_made = 1;
      for (j = i + 1; j < CPU_SETSIZE; j++) {
        if (CPU_ISSET(j, &mask)) run++;
        else break;
      }
      if (!run)
        sprintf(ptr, "%d,", i);
      else if (run == 1) {
        sprintf(ptr, "%d,%d,", i, i + 1);
        i++;
      } else {
        sprintf(ptr, "%d-%d,", i, i + run);
        i += run;
      }
      while (*ptr != 0) ptr++;
    }
  }
  ptr -= entry_made;
  *ptr = 0;
}

/* ---------------------------------------------------------------------- */

int Device::get_num_devices()
{
  printf("LIBGPU: getting number of devices\n");
  return pm->dev_num_devices();
}

/* ---------------------------------------------------------------------- */
    
void Device::get_dev_properties(int N)
{
  printf("LIBGPU: reporting device properties N= %i\n",N);
  pm->dev_properties(N);
}

/* ---------------------------------------------------------------------- */
    
void Device::set_device(int id)
{
  printf("LIBGPU: setting device id= %i\n",id);
  pm->dev_set_device(id);
}

/* ---------------------------------------------------------------------- */
    
void Device::set_update_dfobj_(int _val)
{
  update_dfobj = _val; // this is reset to zero in Device::pull_get_jk
}

/* ---------------------------------------------------------------------- */
    
void Device::disable_eri_cache_()
{
  use_eri_cache = false;
  printf("LIBGPU :: Error : Not able to disable eri caching as additional support needs to be added to track eri_extra array.");
  exit(1);
}

/* ---------------------------------------------------------------------- */

// return stored values for Python side to make decisions
// update_dfobj == true :: nothing useful to return if need to update eri blocks on device
// count_ == -1 :: return # of blocks cached for dfobj
// count_ >= 0 :: return extra data for cached block

void Device::get_dfobj_status(size_t addr_dfobj, py::array_t<int> _arg)
{
  py::buffer_info info_arg = _arg.request();
  int * arg = static_cast<int*>(info_arg.ptr);
  
  int naux_ = arg[0];
  int nao_pair_ = arg[1];
  int count_ = arg[2];
  int update_dfobj_ = arg[3];
  
  // printf("Inside get_dfobj_status(): addr_dfobj= %#012x  naux_= %i  nao_pair_= %i  count_= %i  update_dfobj_= %i\n",
  // 	 addr_dfobj, naux_, nao_pair_, count_, update_dfobj_);
  
  update_dfobj_ = update_dfobj;

  // nothing useful to return if need to update eri blocks on device
  
  if(update_dfobj) { 
    // printf("Leaving get_dfobj_status(): addr_dfobj= %#012x  update_dfobj_= %i\n", addr_dfobj, update_dfobj_);
    
    arg[3] = update_dfobj_;
    return;
  }
  
  // return # of blocks cached for dfobj

  if(count_ == -1) {
    int id = eri_list.size();
    for(int i=0; i<eri_list.size(); ++i)
      if(eri_list[i] == addr_dfobj) {
	id = i;
	break;
      }

    if(id < eri_list.size()) count_ = eri_num_blocks[id];
    
    // printf("Leaving get_dfobj_status(): addr_dfobj= %#012x  count_= %i  update_dfobj_= %i\n", addr_dfobj, count_, update_dfobj_);

    arg[2] = count_;
    arg[3] = update_dfobj_;
    return;
  }

  // return extra data for cached block
  
  int id = eri_list.size();
  for(int i=0; i<eri_list.size(); ++i)
    if(eri_list[i] == addr_dfobj+count_) {
      id = i;
      break;
    }

  // printf("eri_list.size()= %i  id= %i\n",eri_list.size(), id);
  
  naux_ = -1;
  nao_pair_ = -1;
  
  if(id < eri_list.size()) {
  
    naux_     = eri_extra[id * _ERI_CACHE_EXTRA    ];
    nao_pair_ = eri_extra[id * _ERI_CACHE_EXTRA + 1];

  }

  arg[0] = naux_;
  arg[1] = nao_pair_;
  arg[2] = count_;
  arg[3] = update_dfobj_;
  
  // printf("Leaving get_dfobj_status(): addr_dfobj= %#012x  id= %i  naux_= %i  nao_pair_= %i  count_= %i  update_dfobj_= %i\n",
  // 	 addr_dfobj, id, naux_, nao_pair_, count_, update_dfobj_);
  
  // printf("Leaving get_dfobj_status(): addr_dfobj= %#012x  id= %i  arg= %i %i %i %i\n",
  // 	 addr_dfobj, id, arg[0], arg[1], arg[2], arg[3]);
}


/* ---------------------------------------------------------------------- */

// The _vj and _vk arguements aren't actually used anymore and could be removed.

void Device::get_jk(int naux, int nao, int nset,
		    py::array_t<double> _eri1, py::array_t<double> _dmtril, py::list & _dms_list,
		    py::array_t<double> _vj, py::array_t<double> _vk,
		    int with_k, int count, size_t addr_dfobj)
{
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Inside Device::get_jk() w/ with_k= %i\n",with_k);
#endif
  
#ifdef _SIMPLE_TIMER
  double t0 = omp_get_wtime();
#endif

  profile_start("get_jk :: init");

  const int device_id = count % num_devices;
  
  pm->dev_set_device(device_id);

  my_device_data * dd = &(device_data[device_id]);
    
  const int with_j = 1;
  
  py::buffer_info info_eri1 = _eri1.request(); // 2D array (naux, nao_pair)
  py::buffer_info info_dmtril = _dmtril.request(); // 2D array (nset, nao_pair)

  double * eri1 = static_cast<double*>(info_eri1.ptr);
  double * dmtril = static_cast<double*>(info_dmtril.ptr);
  
  int nao_pair = nao * (nao+1) / 2;

  double * d_eri;
  if(!use_eri_cache) {
    // if not caching, then eri block always transferred
    
    pm->dev_push_async(dd->d_eri1, eri1, naux * nao_pair * sizeof(double));
    d_eri = dd->d_eri1;
  }

  if(count < num_devices) {
    int err = pm->dev_push_async(dd->d_dmtril, dmtril, nset * nao_pair * sizeof(double));
    if(err) {
      printf("LIBGPU:: dev_push_async(d_dmtril) failed on count= %i\n",count);
      exit(1);
    }
  }
  
  int _size_rho = nset * naux;
  if(_size_rho > dd->size_rho) {
    dd->size_rho = _size_rho;
    if(dd->d_rho) pm->dev_free_async(dd->d_rho);
    dd->d_rho = (double *) pm->dev_malloc_async(_size_rho * sizeof(double));
  }

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
  
  DevArray2D da_eri1 = DevArray2D(eri1, naux, nao_pair, pm, DA_HOST);
  //  printf("LIBGPU:: eri1= %p  dfobj= %lu  count= %i  combined= %lu\n",eri1,addr_dfobj,count,addr_dfobj+count);
  printf("LIBGPU:: dfobj= %#012x  count= %i  combined= %#012x  update_dfobj= %i\n",addr_dfobj,count,addr_dfobj+count, update_dfobj);
  printf("LIBGPU::     0:      %f %f %f %f\n",da_eri1(0,0), da_eri1(0,1), da_eri1(0,nao_pair-2), da_eri1(0,nao_pair-1));
  printf("LIBGPU::     1:      %f %f %f %f\n",da_eri1(1,0), da_eri1(1,1), da_eri1(1,nao_pair-2), da_eri1(1,nao_pair-1));
  printf("LIBGPU::     naux-2: %f %f %f %f\n",da_eri1(naux-2,0), da_eri1(naux-2,1), da_eri1(naux-2,nao_pair-2), da_eri1(naux-2,nao_pair-1));
  printf("LIBGPU::     naux-1: %f %f %f %f\n",da_eri1(naux-1,0), da_eri1(naux-1,1), da_eri1(naux-1,nao_pair-2), da_eri1(naux-1,nao_pair-1));
#endif
  
  if(use_eri_cache)
    d_eri = dd_fetch_eri(dd, eri1, naux, nao_pair, addr_dfobj, count);
  
  profile_stop();
  
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Starting with_j calculation\n");
#endif
  
  if (with_j){
    
    profile_start("get_jk :: with_j");
    
    // rho = numpy.einsum('ix,px->ip', dmtril, eri1)

    getjk_rho(dd->d_rho, dd->d_dmtril, d_eri, nset, naux, nao_pair);
    
    // vj += numpy.einsum('ip,px->ix', rho, eri1)
   
    int init = (count < num_devices) ? 1 : 0;
  
    getjk_vj(dd->d_vj, dd->d_rho, d_eri, nset, nao_pair, naux, init);

    profile_stop();
  }
    
  if(!with_k) {
    
#ifdef _SIMPLE_TIMER
    double t1 = omp_get_wtime();
    t_array[2] += t1 - t0;
#endif
    
#ifdef _DEBUG_DEVICE
    printf("LIBGPU ::  -- Leaving Device::get_jk()\n");
#endif
    
    return;
  }
  
  // buf2 = lib.unpack_tril(eri1, out=buf[1])
    
  profile_start("get_jk :: with_k");

  getjk_unpack_buf2(dd->d_buf2, d_eri, dd->d_pumap_ptr, naux, nao, nao_pair);

#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- finished\n");
  printf("LIBGPU :: Starting with_k calculation\n");
#endif
  
  for(int indxK=0; indxK<nset; ++indxK) {

#ifdef _SIMPLE_TIMER
    double t4 = omp_get_wtime();
#endif
    
    py::array_t<double> _dms = static_cast<py::array_t<double>>(_dms_list[indxK]); // element of 3D array (nset, nao, nao)
    py::buffer_info info_dms = _dms.request(); // 2D

    double * dms = static_cast<double*>(info_dms.ptr);

    double * d_dms = &(dd->d_dms[indxK*nao*nao]);

    if(count < num_devices) {
#ifdef _DEBUG_DEVICE
      printf("LIBGPU ::  -- calling dev_push_async(dms) for indxK= %i  nset= %i\n",indxK,nset);
#endif
    
      int err = pm->dev_push_async(d_dms, dms, nao*nao*sizeof(double));
      if(err) {
	printf("LIBGPU:: dev_push_async(d_dms) on indxK= %i\n",indxK);
	printf("LIBGPU:: d_dms= %#012x  dms= %#012x  nao= %i  stream= %#012x\n",d_dms,dms,nao,dd->stream);
	exit(1);
      }
    }

    {
      const double alpha = 1.0;
      const double beta = 0.0;
      const int nao2 = nao * nao;
      const int zero = 0;

      ml->set_handle();
      ml->gemm_batch((char *) "T", (char *) "T", &nao, &nao, &nao,
		     &alpha, dd->d_buf2, &nao, &nao2, d_dms, &nao, &zero, &beta, dd->d_buf1, &nao, &nao2, &naux);
    }
    
    // dgemm of (nao X blksize*nao) and (blksize*nao X nao) matrices - can refactor later...
    // vk[k] += lib.dot(buf1.reshape(-1,nao).T, buf2.reshape(-1,nao))  // vk[k] is nao x nao array
  
    // buf3 = buf1.reshape(-1,nao).T
    // buf4 = buf2.reshape(-1,nao)
    
    transpose(dd->d_buf3, dd->d_buf1, naux*nao, nao);
    
    // vk[k] += lib.dot(buf3, buf4)
    // gemm(A,B,C) : C = alpha * A.B + beta * C
    // A is (m, k) matrix
    // B is (k, n) matrix
    // C is (m, n) matrix
    // Column-ordered: (A.B)^T = B^T.A^T

    {
      const double alpha = 1.0;
      const double beta = (count < num_devices) ? 0.0 : 1.0; // first pass by each device initializes array, otherwise accumulate
      
      const int m = nao; // # of rows of first matrix buf4^T
      const int n = nao; // # of cols of second matrix buf3^T
      const int k = naux*nao; // # of cols of first matrix buf4^
      
      const int lda = naux * nao;
      const int ldb = nao;
      const int ldc = nao;
      
      const int vk_offset = indxK * nao*nao;
      
      ml->set_handle();
      ml->gemm((char *) "N", (char *) "N", &m, &n, &k, &alpha, dd->d_buf2, &ldb, dd->d_buf3, &lda, &beta, (dd->d_vkk)+vk_offset, &ldc);
    }
    
  } // for(nset)
  
  profile_stop();
    
#ifdef _SIMPLE_TIMER
  double t1 = omp_get_wtime();
  t_array[2] += t1 - t0;
#endif
    
#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- finished\n");
  printf("LIBGPU :: -- Leaving Device::get_jk()\n");
#endif
}
  
/* ---------------------------------------------------------------------- */

void Device::pull_get_jk(py::array_t<double> _vj, py::array_t<double> _vk, int nao, int nset, int with_k)
{
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: -- Inside Device::pull_get_jk()\n");
#endif

#ifdef _SIMPLE_TIMER
  double t0 = omp_get_wtime();
#endif
    
  profile_start("pull_get_jk");
  
  py::buffer_info info_vj = _vj.request(); // 2D array (nset, nao_pair)
  
  double * vj = static_cast<double*>(info_vj.ptr);

  int nao_pair = nao * (nao+1) / 2;
  
  int size = nset * nao_pair * sizeof(double);

  double * tmp;
  
  for(int i=0; i<num_devices; ++i) {
    pm->dev_set_device(i);
    
    my_device_data * dd = &(device_data[i]);

    if(i == 0) tmp = vj;
    else tmp = &(buf_vj[i * nset * nao_pair]);
    
    if(dd->d_vj) pm->dev_pull_async(dd->d_vj, tmp, size);
  }
  
  for(int i=0; i<num_devices; ++i) {
    pm->dev_set_device(i);
    
    my_device_data * dd = &(device_data[i]);
    
    pm->dev_stream_wait();

    if(i > 0 && dd->d_vj) {
      
      tmp = &(buf_vj[i * nset * nao_pair]);
#pragma omp parallel for
      for(int j=0; j<nset*nao_pair; ++j) vj[j] += tmp[j];
      
    }
  }
  
  update_dfobj = 0;
  
  if(!with_k) {
    profile_stop();
    
#ifdef _DEBUG_DEVICE
    printf("LIBGPU :: -- Leaving Device::pull_get_jk()\n");
#endif
    
    return;
  }
    
  py::buffer_info info_vk = _vk.request(); // 3D array (nset, nao, nao)
    
  double * vk = static_cast<double*>(info_vk.ptr);

  size = nset * nao * nao * sizeof(double);

  for(int i=0; i<num_devices; ++i) {
    pm->dev_set_device(i);
      
    my_device_data * dd = &(device_data[i]);

    if(i == 0) tmp = vk;
    else tmp = &(buf_vk[i * nset * nao * nao]);

    if(dd->d_vkk) pm->dev_pull_async(dd->d_vkk, tmp, size);
  }

  for(int i=0; i<num_devices; ++i) {
    pm->dev_set_device(i);
    
    my_device_data * dd = &(device_data[i]);
    
    pm->dev_stream_wait();

    if(i > 0 && dd->d_vkk) {
      
      tmp = &(buf_vk[i * nset * nao * nao]);
#pragma omp parallel for
      for(int j=0; j<nset*nao*nao; ++j) vk[j] += tmp[j];
    
    }

  }

  profile_stop();
  
#ifdef _SIMPLE_TIMER
  double t1 = omp_get_wtime();
  t_array[1] += t1 - t0;
#endif
    
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: -- Leaving Device::pull_get_jk()\n");
#endif
}

/* ---------------------------------------------------------------------- */

int * Device::dd_fetch_pumap(my_device_data * dd, int size_pumap_, int type_pumap)
{
  // search if pack/unpack map already created

  int indx = -1;
  for(int i=0; i<dd->size_pumap.size(); ++i)
    if(dd->type_pumap[i] == type_pumap && dd->size_pumap[i] == size_pumap_) indx = i;

  // add unpack/pack map if not found
  
  if(indx < 0) {
    dd->type_pumap.push_back(type_pumap);
    dd->size_pumap.push_back(size_pumap_);
    dd->pumap.push_back(nullptr);
    dd->d_pumap.push_back(nullptr);

    indx = dd->type_pumap.size() - 1;

    int size_pumap = -1;
    
    if(type_pumap == _PUMAP_2D_UNPACK) {
      int nao = size_pumap_;
      size_pumap = nao * nao;
      
      dd->pumap[indx] = (int *) pm->dev_malloc_host(size_pumap * sizeof(int));
      dd->d_pumap[indx] = (int *) pm->dev_malloc_async(size_pumap * sizeof(int));
      
      int _i, _j, _ij;
      int * tm = dd->pumap[indx];
      for(_ij = 0, _i = 0; _i < nao; _i++)
	for(_j = 0; _j<=_i; _j++, _ij++) {
	  tm[_i*nao + _j] = _ij;
	  tm[_i + nao*_j] = _ij;
	}
      
    } else if(type_pumap == _PUMAP_H2EFF_UNPACK) {

#if 1
      int ncas = size_pumap_;
      size_pumap = ncas * ncas;

      dd->pumap[indx] = (int *) pm->dev_malloc_host(size_pumap * sizeof(int));
      dd->d_pumap[indx] = (int *) pm->dev_malloc_async(size_pumap * sizeof(int));

      int * tm = dd->pumap[indx];
      int _ij, _i, _j;
      for(_ij = 0, _i = 0; _i < ncas; _i++)
	for(_j = 0; _j<=_i; _j++, _ij++) {
	  tm[_i*ncas + _j] = _ij;
	  tm[_i + ncas*_j] = _ij;
	}
#else
      int ncas = size_pumap_;
      int ncas_pair = ncas * (ncas+1)/2;
      size_pumap = ncas * ncas * ncas;

      dd->pumap[indx] = (int *) pm->dev_malloc_host(size_pumap * sizeof(int));
      dd->d_pumap[indx] = (int *) pm->dev_malloc_async(size_pumap * sizeof(int));

      int * tm = dd->pumap[indx];
      for (int _i=0; _i<ncas;++_i) {
	for (int _j=0, _jk=0; _j<ncas; ++_j) {
	  for (int _k=0;_k<=_j;++_k,++_jk) {
	    tm[_i*ncas*ncas + _j*ncas+_k]=_i*ncas_pair+_jk;
	    tm[_i*ncas*ncas + _k*ncas+_j]=_i*ncas_pair+_jk;
	  }
	}
      }
#endif
    } else if(type_pumap == _PUMAP_H2EFF_PACK) {
#if 1
      int ncas = size_pumap_;
      int ncas_pair = ncas * (ncas+1)/2;
      size_pumap = ncas_pair;

      dd->pumap[indx] = (int *) pm->dev_malloc_host(size_pumap * sizeof(int));
      dd->d_pumap[indx] = (int *) pm->dev_malloc_async(size_pumap * sizeof(int));

      int * tm = dd->pumap[indx];
      int _i, _j, _ij;
      for (_i=0, _ij=0; _i<ncas; ++_i) {
	for (_j=0; _j<=_i; ++_j, ++_ij) {
	  tm[_ij] = _i*ncas + _j;
	}
      }
#else
      int ncas = size_pumap_;
      int ncas_pair = ncas * (ncas+1)/2;
      size_pumap = ncas * ncas_pair;

      dd->pumap[indx] = (int *) pm->dev_malloc_host(size_pumap * sizeof(int));
      dd->d_pumap[indx] = (int *) pm->dev_malloc_async(size_pumap * sizeof(int));

      int * tm = dd->pumap[indx];
      int _i, _j, _k, _ijk;
      for (_ijk=0, _i=0; _i<ncas;++_i){
	for (_j=0; _j<ncas; ++_j){
	  for (_k=0;_k<=_j;++_k,++_ijk){
	    tm[_ijk] = _i*ncas*ncas + _j*ncas+_k;
	  }
	}
      }
#endif
      
    } // if(type_pumap)
    
    pm->dev_push_async(dd->d_pumap[indx], dd->pumap[indx], size_pumap*sizeof(int));
  } // if(map_not_found)
  
  // set pointers to current map

  dd->pumap_ptr = dd->pumap[indx];
  dd->d_pumap_ptr = dd->d_pumap[indx];

  return dd->d_pumap_ptr;
}

/* ---------------------------------------------------------------------- */

// Is both _ocm2 in/out as it get over-written and resized?

void Device::orbital_response(py::array_t<double> _f1_prime,
			      py::array_t<double> _ppaa, py::array_t<double> _papa, py::array_t<double> _eri_paaa,
			      py::array_t<double> _ocm2, py::array_t<double> _tcm2, py::array_t<double> _gorb,
			      int ncore, int nocc, int nmo)
{
  
#ifdef _SIMPLE_TIMER
  double t0 = omp_get_wtime();
#endif
    
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
    
    //int indx = 0;
   
    double * cm = ocm2; // (2, 2, 2, ncore)
    
    for(int i=0; i<nmo; ++i) f1[i] = 0.0;
    
    // tensordot(paaa, cm, axes=((0,1,2), (2,1,0)))
    
    // printf("f1 += paaa{%i, %i, %i} X cm{%i, %i, %i, %i}\n",
    // 	   nocc-ncore,info_ppaa.shape[2],info_ppaa.shape[3],
    // 	   info_ocm2.shape[0],info_ocm2.shape[1],info_ocm2.shape[2],ncore);
    
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
  } // for(p<nmo)

  // # (H.x_aa)_va, (H.x_aa)_ac

  int _ocm2_size_1d = nocc - ncore;
  int _ocm2_size_2d = info_ocm2.shape[2] * _ocm2_size_1d;
  int _ocm2_size_3d = info_ocm2.shape[1] * ocm2_size_2d;
  int size_ecm = info_ocm2.shape[0] * info_ocm2.shape[1] * info_ocm2.shape[2] * (nocc-ncore);
  
  double * _ocm2t = (double *) pm->dev_malloc_host(size_ecm * sizeof(double));
  double * ecm2 = (double *) pm->dev_malloc_host(size_ecm * sizeof(double)); // tmp space and ecm2
  
  // ocm2 = ocm2[:,:,:,ncore:nocc] + ocm2[:,:,:,ncore:nocc].transpose (1,0,3,2)

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
    
  // ecm2 = ocm2 + tcm2
  
  for(int i=0; i<size_ecm; ++i) ecm2[i] = _ocm2t[i] + tcm2[i];
  
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
  double t1 = omp_get_wtime();
  t_array[4]  += t1  - t0;
#endif
  
#if 0
  pm->dev_free_host(ar_global);
#endif
  
  pm->dev_free_host(g_f1_prime);
  pm->dev_free_host(ecm2);
  pm->dev_free_host(_ocm2t);
  pm->dev_free_host(f1_prime);
}

/* ---------------------------------------------------------------------- */

void Device::profile_start(const char * label)
{
#ifdef _USE_NVTX
  nvtxRangePushA(label);
#endif
}

/* ---------------------------------------------------------------------- */

void Device::profile_stop()
{
#ifdef _USE_NVTX
  nvtxRangePop();
#endif
}

/* ---------------------------------------------------------------------- */

void Device::profile_next(const char * label)
{
#ifdef _USE_NVTX
  nvtxRangePop();
  nvtxRangePushA(label);
#endif
}

/* ---------------------------------------------------------------------- */


