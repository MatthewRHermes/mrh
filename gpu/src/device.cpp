/* -*- c++ -*- */

#include <stdio.h>

#include "device.h"

#define _NUM_SIMPLE_TIMER 11
#define _NUM_SIMPLE_COUNTER 7

#include <unistd.h>
#include <string.h>
#include <sched.h>

/* ---------------------------------------------------------------------- */

Device::Device()
{  
  pm = new PM();

  ml = new MATHLIB(pm);

  verbose_level = 0;
  
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
  size_buf_ppaa = 0;
  size_fxpp = 0;//remove when ao2mo_v3 is running
  size_bufpa = 0;

  buf_j_pc = nullptr;
  buf_k_pc = nullptr;
  buf_ppaa = nullptr;
  pin_fxpp = nullptr;//remove when ao2mo_v3 is running
  pin_bufpa = nullptr;
  // h2eff_df
  size_buf_eri_h2eff=0;
  buf_eri_h2eff=nullptr;

#if defined(_USE_GPU)
  use_eri_cache = true;
#endif
  
  num_threads = 1;
#pragma omp parallel
  num_threads = omp_get_num_threads();

  num_devices = pm->dev_num_devices();
  
  //  device_data = (my_device_data*) pm->dev_malloc_host(num_devices * sizeof(my_device_data));
  device_data = new my_device_data[num_devices];

  for(int i=0; i<num_devices; ++i) {
    pm->dev_set_device(i);
    
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
    device_data[i].size_mo_cas = 0;
    device_data[i].size_eri_unpacked = 0; // this variable is not needed
    device_data[i].size_j_pc = 0;
    device_data[i].size_k_pc = 0;
    device_data[i].size_bufd = 0;
    device_data[i].size_bufpa = 0;
    device_data[i].size_bufaa = 0;
    device_data[i].size_eri_h2eff=0;
    
    device_data[i].d_rho = nullptr;
    device_data[i].d_vj = nullptr;
    device_data[i].d_buf1 = nullptr;
    device_data[i].d_buf2 = nullptr;
    device_data[i].d_buf3 = nullptr;
    device_data[i].d_vkk = nullptr;
    device_data[i].d_dms = nullptr;
    device_data[i].d_mo_coeff=nullptr;
    device_data[i].d_mo_cas=nullptr;
    device_data[i].d_dmtril = nullptr;
    device_data[i].d_eri1 = nullptr; // when not using eri cache
    device_data[i].d_ucas = nullptr;
    device_data[i].d_umat = nullptr;
    device_data[i].d_h2eff = nullptr;
    device_data[i].d_eri_h2eff = nullptr;//for h2eff_df_v2
    
    device_data[i].d_pumap_ptr = nullptr;
    
    device_data[i].d_j_pc = nullptr;
    device_data[i].d_k_pc = nullptr;
    device_data[i].d_bufd = nullptr;
    device_data[i].d_bufpa = nullptr;
    device_data[i].d_bufaa = nullptr;
    device_data[i].d_ppaa = nullptr;//initialized, but not allocated (used dd->d_buf3)

#if defined (_USE_GPU)
    device_data[i].handle = nullptr;
    device_data[i].stream = nullptr;
#endif

    ml->create_handle();
  }

  t_array = (double* ) malloc(_NUM_SIMPLE_TIMER * sizeof(double));
  for(int i=0; i<_NUM_SIMPLE_TIMER; ++i) t_array[i] = 0.0;
  count_array = (int* ) malloc(_NUM_SIMPLE_COUNTER * sizeof(int));
  for(int i=0; i<_NUM_SIMPLE_COUNTER; ++i) count_array[i] = 0;
}

/* ---------------------------------------------------------------------- */

Device::~Device()
{
  if(verbose_level) printf("LIBGPU: destroying device\n");

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
  pm->dev_free_host(buf_ppaa);
  pm->dev_free_host(pin_fxpp);
  pm->dev_free_host(pin_bufpa);//remove when ao2mo_v3 is running

  if(verbose_level) get_dev_properties(num_devices);

  if(verbose_level) { // this needs to be cleaned up and generalized...
    double total = 0.0;
    for(int i=0; i<_NUM_SIMPLE_TIMER; ++i) total += t_array[i];
  
    printf("\nLIBGPU :: SIMPLE_TIMER\n");
    printf("\nLIBGPU :: SIMPLE_TIMER :: get_jk\n");
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= init_get_jk()            time= %f s\n",0,t_array[0]);
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= pull_get_jk()            time= %f s\n",1,t_array[1]);
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= get_jk()                 time= %f s\n",2,t_array[2]);
    
    printf("\nLIBGPU :: SIMPLE_TIMER :: hessop\n");
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= hessop_get_veff()        time= %f s\n",3,t_array[3]);
    
    printf("\nLIBGPU :: SIMPLE_TIMER :: orbital_response\n");
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= orbital_response()       time= %f s\n",4,t_array[4]);
    
    
    printf("\nLIBGPU :: SIMPLE_TIMER :: _update_h2eff\n");
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= update_h2eff_sub()       time= %f s\n",5,t_array[5]);
    
    printf("\nLIBGPU :: SIMPLE_TIMER :: _h2eff_df \n");
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= h2eff_df()               time= %f s\n",6,t_array[6]);
    
    printf("\nLIBGPU :: SIMPLE_TIMER :: transfer_mo_coeff \n");
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= transfer_mo_coeff()      time= %f s\n",7,t_array[7]);
    
    printf("\nLIBGPU :: SIMPLE_TIMER :: df_ao2mo_pass1\n");
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= init_ints_and_jkpc()     time= %f s\n",8,t_array[8]);
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= compute_ints_and_jkpc()  time= %f s\n",9,t_array[9]);
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= pull_ints_and_jkpc()     time= %f s\n",10,t_array[10]);
    printf("LIBGPU :: SIMPLE_TIMER :: total= %f s\n",total);
    free(t_array);
    
    
    printf("\nLIBGPU :: SIMPLE_COUNTER\n");
    printf("\nLIBGPU :: SIMPLE_COUNTER :: get_jk\n");
    printf("LIBGPU :: SIMPLE_COUNTER :: i= %i  name= get_jk()             counts= %i \n",0,count_array[0]);
    
    printf("\nLIBGPU :: SIMPLE_COUNTER :: hessop\n");
    printf("LIBGPU :: SIMPLE_COUNTER :: i= %i  name= hessop_get_veff()    counts= %i \n",1,count_array[1]);
    
    printf("\nLIBGPU :: SIMPLE_COUNTER :: orbital_response\n");
    printf("LIBGPU :: SIMPLE_COUNTER :: i= %i  name= orbital_response()   counts= %i \n",2,count_array[2]);
    
    printf("\nLIBGPU :: SIMPLE_COUNTER :: update_h2eff_sub\n");
    printf("LIBGPU :: SIMPLE_COUNTER :: i= %i  name= update_h2eff_sub()   counts= %i \n",3,count_array[3]);
    
    printf("\nLIBGPU :: SIMPLE_COUNTER :: _h2eff_df\n");
    printf("LIBGPU :: SIMPLE_COUNTER :: i= %i  name= h2eff_df()           counts= %i \n",4,count_array[4]);
    
    printf("\nLIBGPU :: SIMPLE_COUNTER :: transfer_mo_coeff\n");
    printf("LIBGPU :: SIMPLE_COUNTER :: i= %i  name= transfer_mo_coeff()  counts= %i \n",5,count_array[5]);
    
    printf("\nLIBGPU :: SIMPLE_COUNTER :: ao2mo\n");
    printf("LIBGPU :: SIMPLE_COUNTER :: i= %i  name=ao2mo_pass_v3()       counts= %i \n",6,count_array[6]);
    
    free(count_array);
  }

  // print summary of cached eri blocks

  if(use_eri_cache) {
    if(verbose_level) {
      printf("\nLIBGPU :: eri cache statistics :: count= %zu\n",eri_list.size());
      for(int i=0; i<eri_list.size(); ++i)
	printf("LIBGPU :: %i : eri= %p  Mbytes= %f  count= %i  update= %i device= %i\n", i, eri_list[i],
	       eri_size[i]*sizeof(double)/1024./1024., eri_count[i], eri_update[i], eri_device[i]);
    }
    
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
    pm->dev_free(dd->d_mo_coeff);
    pm->dev_free(dd->d_mo_cas);
    pm->dev_free(dd->d_dmtril);
    pm->dev_free(dd->d_eri1);
    pm->dev_free(dd->d_ucas);
    pm->dev_free(dd->d_umat);
    pm->dev_free(dd->d_h2eff);
    pm->dev_free(dd->d_eri_h2eff);
    
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
  }

  if(verbose_level) printf("LIBGPU :: Finished\n");
#endif

  delete [] device_data;
  
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
  if(verbose_level) printf("LIBGPU: getting number of devices\n");
  return pm->dev_num_devices();
}

/* ---------------------------------------------------------------------- */
    
void Device::get_dev_properties(int N)
{
  printf("LIBGPU: reporting device properties N= %i\n",N);
  
  char nname[16];
  gethostname(nname, 16);
  int rnk = 0;
  
#pragma omp parallel for ordered
  for(int it=0; it<num_threads; ++it) {
    char list_cores[7*CPU_SETSIZE];
    get_cores(list_cores);
#pragma omp ordered
    printf("LIBGPU: To affinity and beyond!! nname= %s  rnk= %d  tid= %d: list_cores= (%s)\n",
	   nname, rnk, omp_get_thread_num(), list_cores);
  }
  
  pm->dev_properties(N);
}

/* ---------------------------------------------------------------------- */
    
void Device::set_device(int id)
{
  if(verbose_level) printf("LIBGPU: setting device id= %i\n",id);
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
    
void Device::set_verbose_(int _verbose)
{
  verbose_level = _verbose; // setting nonzero prints affinity + timing info
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

void Device::init_get_jk(py::array_t<double> _eri1, py::array_t<double> _dmtril, int blksize, int nset, int nao, int naux, int count)
{
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Inside Device::init_get_jk()\n");
#endif

  profile_start("init_get_jk");
  
  double t0 = omp_get_wtime();

  const int device_id = count % num_devices;
  
  pm->dev_set_device(device_id);

  my_device_data * dd = &(device_data[device_id]);
  
  //  if(dd->stream == nullptr) dd->stream = pm->dev_get_queue();
  
  int nao_pair = nao * (nao+1) / 2;
  
  int _size_vj = nset * nao_pair;
  if(_size_vj > dd->size_vj) {
    dd->size_vj = _size_vj;
    if(dd->d_vj) pm->dev_free_async(dd->d_vj);
    dd->d_vj = (double *) pm->dev_malloc_async(_size_vj * sizeof(double));
  }
  
  int _size_vk = nset * nao * nao;
  if(_size_vk > dd->size_vk) {
    dd->size_vk = _size_vk;
    
    if(dd->d_vkk) pm->dev_free_async(dd->d_vkk);
    dd->d_vkk = (double *) pm->dev_malloc_async(_size_vk * sizeof(double));
  }

  int _size_buf = blksize * nao * nao;
  if(_size_buf > dd->size_buf) {
    dd->size_buf = _size_buf;
    if(buf_tmp) pm->dev_free_host(buf_tmp);
    if(buf3) pm->dev_free_host(buf3);
    if(buf4) pm->dev_free_host(buf4);
    
    buf_tmp = (double *) pm->dev_malloc_host(2*_size_buf*sizeof(double));
    buf3 = (double *) pm->dev_malloc_host(_size_buf*sizeof(double)); // (nao, blksize*nao)
    buf4 = (double *) pm->dev_malloc_host(_size_buf*sizeof(double)); // (blksize*nao, nao)

    if(dd->d_buf1) pm->dev_free_async(dd->d_buf1);
    if(dd->d_buf2) pm->dev_free_async(dd->d_buf2);
    if(dd->d_buf3) pm->dev_free_async(dd->d_buf3);
    
    dd->d_buf1 = (double *) pm->dev_malloc_async(_size_buf * sizeof(double));
    dd->d_buf2 = (double *) pm->dev_malloc_async(_size_buf * sizeof(double));
    dd->d_buf3 = (double *) pm->dev_malloc_async(_size_buf * sizeof(double));
  }
  
  int _size_dms = nset * nao * nao;
  if(_size_dms > dd->size_dms) {
    dd->size_dms = _size_dms;
    if(dd->d_dms) pm->dev_free_async(dd->d_dms);
    dd->d_dms = (double *) pm->dev_malloc_async(_size_dms * sizeof(double));
  }

  int _size_dmtril = nset * nao_pair;
  if(_size_dmtril > dd->size_dmtril) {
    dd->size_dmtril = _size_dmtril;
    if(dd->d_dmtril) pm->dev_free_async(dd->d_dmtril);
    dd->d_dmtril = (double *) pm->dev_malloc_async(_size_dmtril * sizeof(double));
  }

  if(!use_eri_cache) {
    int _size_eri1 = naux * nao_pair;
    if(_size_eri1 > dd->size_eri1) {
      dd->size_eri1 = _size_eri1;
      if(dd->d_eri1) pm->dev_free_async(dd->d_eri1);
      dd->d_eri1 = (double *) pm->dev_malloc_async(_size_eri1 * sizeof(double));
    }
  }
  
  int _size_buf_vj = num_devices * nset * nao_pair;
  if(_size_buf_vj > size_buf_vj) {
    size_buf_vj = _size_buf_vj;
    if(buf_vj) pm->dev_free_host(buf_vj);
    buf_vj = (double *) pm->dev_malloc_host(_size_buf_vj*sizeof(double));
  }

  int _size_buf_vk = num_devices * nset * nao * nao;
  if(_size_buf_vk > size_buf_vk) {
    size_buf_vk = _size_buf_vk;
    if(buf_vk) pm->dev_free_host(buf_vk);
    buf_vk = (double *) pm->dev_malloc_host(_size_buf_vk*sizeof(double));
  }

  // 1-time initialization
  
  dd_fetch_pumap(dd, nao, _PUMAP_2D_UNPACK);
  
  // Create blas handle

  // if(dd->handle == nullptr) {
  //   ml->create_handle();
  //   //    dd->handle = ml->get_handle();
  // }
  
  profile_stop();
    
  double t1 = omp_get_wtime();
  t_array[0] += t1 - t0;
 //counts in pull_get_jk

#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: -- Leaving Device::init_get_jk()\n");
#endif
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
  
  double t0 = omp_get_wtime();

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
    
    double t1 = omp_get_wtime();
    t_array[2] += t1 - t0;
// counts in pull_jk
    
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

    double t4 = omp_get_wtime();
    
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
	printf("LIBGPU:: d_dms= %#012x  dms= %#012x  nao= %i  device= %i\n",d_dms,dms,nao,device_id);
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
    
  double t1 = omp_get_wtime();
  t_array[2] += t1 - t0;
  // counts in pull jk
    
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

  double t0 = omp_get_wtime();
    
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
  
  double t1 = omp_get_wtime();
  t_array[1] += t1 - t0;
  count_array[0]+=1; // just doing this addition in pull, not in init or compute
    
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

double * Device::dd_fetch_eri(my_device_data * dd, double * eri1, int naux, int nao_pair, size_t addr_dfobj, int count)
{
#if defined(_DEBUG_DEVICE) || defined(_DEBUG_ERI_CACHE)
  if(eri1 != nullptr) return dd_fetch_eri_debug(dd, eri1, naux, nao_pair, addr_dfobj, count);
#endif

  double * d_eri;
  
  // retrieve id of cached eri block
  
  int id = eri_list.size();
  for(int i=0; i<eri_list.size(); ++i)
    if(eri_list[i] == addr_dfobj+count) {
      id = i;
      break;
    }
  
  // grab/update cached data
  
  if(id < eri_list.size()) {
    
    eri_count[id]++;
    d_eri = d_eri_cache[id];

    bool full_molecule = id-count == 0;
    
    if(!full_molecule && update_dfobj) {
      eri_update[id]++;
      int err = pm->dev_push_async(d_eri, eri1, naux * nao_pair * sizeof(double));
      if(err) {
	printf("LIBGPU:: dev_push_async(d_eri) updating eri block\n");
	exit(1);
      }
    }
    
  } else {
    
    eri_list.push_back(addr_dfobj+count);
    eri_count.push_back(1);
    eri_update.push_back(0);
    eri_size.push_back(naux * nao_pair);
    eri_device.push_back(dd->device_id);
    
    eri_num_blocks.push_back(0); // grow array
    eri_num_blocks[id-count]++;  // increment # of blocks for this dfobj
    
    eri_extra.push_back(naux);
    eri_extra.push_back(nao_pair);
    
    int id = d_eri_cache.size();
    
    d_eri_cache.push_back( (double *) pm->dev_malloc_async(naux * nao_pair * sizeof(double)));
    d_eri = d_eri_cache[ id ];
    
    int err = pm->dev_push_async(d_eri, eri1, naux * nao_pair * sizeof(double));
    if(err) {
      printf("LIBGPU:: dev_push_async(d_eri) initializing new eri block\n");
      exit(1);
    }

#ifdef _DEBUG_DEVICE
    printf("LIBGPU:: dd_fetch_eri :: addr= %p  count= %i  naux= %i  nao_pair= %i\n",addr_dfobj+count, count, naux, nao_pair);
#endif
    
  }

  return d_eri;
}

/* ---------------------------------------------------------------------- */

double * Device::dd_fetch_eri_debug(my_device_data * dd, double * eri1, int naux, int nao_pair, size_t addr_dfobj, int count)
{   
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Starting eri_cache lookup for ERI %p\n",addr_dfobj+count);
#endif

  double * d_eri;
  
  // retrieve id of cached eri block
  
  int id = eri_list.size();
  for(int i=0; i<eri_list.size(); ++i)
    if(eri_list[i] == addr_dfobj+count) {
      id = i;
      break;
    }
  
  // grab/update cached data
  
  if(id < eri_list.size()) {
#ifdef _DEBUG_DEVICE
    printf("LIBGPU :: -- eri block found: id= %i\n",id);
#endif
    
    eri_count[id]++;
    d_eri = d_eri_cache[id];
    
#ifdef _DEBUG_ERI_CACHE
    int diff_size = eri_size[id] - naux * nao_pair;
    if(diff_size != 0) {
      printf("LIBGPU:: Error: eri_cache size != 0  diff_size= %i\n",diff_size);
      exit(1);
    }
    
    double * eri_host = d_eri_host[id];
    double diff_eri = 0.0;
    for(int i=0; i<naux*nao_pair; ++i) diff_eri += (eri_host[i] - eri1[i]) * (eri_host[i] - eri1[i]);
    
    if(diff_eri > 1e-10) {
      for(int i=0; i<naux*nao_pair; ++i) eri_host[i] = eri1[i];
      pm->dev_push_async(d_eri, eri1, naux * nao_pair * sizeof(double));
      eri_update[id]++;
      
      // update_dfobj fails to correctly update device ; this is an error
      if(!update_dfobj) {
	printf("LIBGPU :: Warning: ERI %p updated on device w/ diff_eri= %.10e, but update_dfobj= %i\n",addr_dfobj+count,diff_eri,update_dfobj);
	//count = -1;
	//return;
	exit(1);
      }
    } else {
      
      // update_dfobj falsely updates device ; this is loss of performance
      if(update_dfobj) {
	printf("LIBGPU :: Warning: ERI %p not updated on device w/ diff_eri= %.10e, but update_dfobj= %i\n",addr_dfobj+count,diff_eri,update_dfobj);
	//count = -1;
	//return;
	//exit(1);
      }
    }
#else
    if(update_dfobj) {
#ifdef _DEBUG_DEVICE
      printf("LIBGPU :: -- updating eri block: id= %i\n",id);
#endif
      eri_update[id]++;
      int err = pm->dev_push_async(d_eri, eri1, naux * nao_pair * sizeof(double));
      if(err) {
	printf("LIBGPU:: dev_push_async(d_eri) updating eri block\n");
	exit(1);
      }
    }
#endif
    
  } else {
    eri_list.push_back(addr_dfobj+count);
    eri_count.push_back(1);
    eri_update.push_back(0);
    eri_size.push_back(naux * nao_pair);
    eri_device.push_back(dd->device_id);
    
    eri_num_blocks.push_back(0); // grow array
    eri_num_blocks[id-count]++;  // increment # of blocks for this dfobj
    
    eri_extra.push_back(naux);
    eri_extra.push_back(nao_pair);
    
    int id = d_eri_cache.size();
#ifdef _DEBUG_DEVICE
    printf("LIBGPU :: -- allocating new eri block: %i\n",id);
#endif
    
    d_eri_cache.push_back( (double *) pm->dev_malloc_async(naux * nao_pair * sizeof(double)));
    d_eri = d_eri_cache[ id ];
    
#ifdef _DEBUG_DEVICE
    printf("LIBGPU :: -- initializing eri block\n");
#endif
    int err = pm->dev_push_async(d_eri, eri1, naux * nao_pair * sizeof(double));
    if(err) {
      printf("LIBGPU:: dev_push_async(d_eri) initializing new eri block\n");
      exit(1);
    }
    
#ifdef _DEBUG_ERI_CACHE
    d_eri_host.push_back( (double *) pm->dev_malloc_host(naux*nao_pair * sizeof(double)) );
    double * d_eri_host_ = d_eri_host[id];
    for(int i=0; i<naux*nao_pair; ++i) d_eri_host_[i] = eri1[i];
#endif
    
#ifdef _DEBUG_DEVICE
    printf("LIBGPU:: dd_fetch_eri_debug :: addr= %p  count= %i  naux= %i  nao_pair= %i\n",addr_dfobj+count, count, naux, nao_pair);
#endif
  }

  return d_eri;
}

/* ---------------------------------------------------------------------- */

void Device::push_mo_coeff(py::array_t<double> _mo_coeff, int _size_mo_coeff)
{
  double t0 = omp_get_wtime();
  
  py::buffer_info info_mo_coeff = _mo_coeff.request(); // 2D array (naux, nao_pair)

  double * mo_coeff = static_cast<double*>(info_mo_coeff.ptr);

  // host pushes to each device; optimize later host->device0 plus device-device transfers (i.e. bcast)
  
  for(int id=0; id<num_devices; ++id) {
    
    pm->dev_set_device(id);
  
    my_device_data * dd = &(device_data[id]);
  
    if (_size_mo_coeff > dd->size_mo_coeff){
      dd->size_mo_coeff = _size_mo_coeff;
      if (dd->d_mo_coeff) pm->dev_free_async(dd->d_mo_coeff);
      dd->d_mo_coeff = (double *) pm->dev_malloc_async(_size_mo_coeff*sizeof(double));
    }
    
    pm->dev_push_async(dd->d_mo_coeff, mo_coeff, _size_mo_coeff*sizeof(double));
  }
  
  double t1 = omp_get_wtime();
  t_array[7] += t1 - t0;
  count_array[5] +=1;
}

/* ---------------------------------------------------------------------- */

void Device::init_jk_ao2mo(int ncore, int nmo)
{
  double t0 = omp_get_wtime();

  // host initializes on each device
  
  for(int id=0; id<num_devices; ++id) {
    pm->dev_set_device(id);
    
    my_device_data * dd = &(device_data[id]);
    
    int size_j_pc = ncore*nmo;
    int size_k_pc = ncore*nmo;
    
    if (size_j_pc > dd->size_j_pc){
      dd->size_j_pc = size_j_pc;
      if (dd->d_j_pc) pm->dev_free_async(dd->d_j_pc);
      dd->d_j_pc = (double *) pm->dev_malloc_async(size_j_pc*sizeof(double));
    }

    if (size_k_pc > dd->size_k_pc){
      dd->size_k_pc = size_k_pc;
      if (dd->d_k_pc) pm->dev_free_async(dd->d_k_pc);
      dd->d_k_pc = (double *) pm->dev_malloc_async(size_k_pc*sizeof(double));
    }
  }
  
  int _size_buf_j_pc = num_devices*nmo*ncore;
  if(_size_buf_j_pc > size_buf_j_pc) {
    size_buf_j_pc = _size_buf_j_pc;
    if(buf_j_pc) pm->dev_free_host(buf_j_pc);
    buf_j_pc = (double *) pm->dev_malloc_host(_size_buf_j_pc*sizeof(double));
  }
  
  int _size_buf_k_pc = num_devices*nmo*ncore;
  if(_size_buf_k_pc > size_buf_k_pc) {
    size_buf_k_pc = _size_buf_k_pc;
    if(buf_k_pc) pm->dev_free_host(buf_k_pc);
    buf_k_pc = (double *) pm->dev_malloc_host(_size_buf_k_pc*sizeof(double));
    }
  
  double t1 = omp_get_wtime();
  t_array[8] += t1 - t0;
  // counts in pull ppaa
}

/* ---------------------------------------------------------------------- */

void Device::init_ints_ao2mo(int naoaux, int nmo, int ncas)
{
  double t0 = omp_get_wtime();
  
  int _size_fxpp = naoaux*nmo*nmo;
  if (_size_fxpp > size_fxpp){
    size_fxpp = _size_fxpp;
    if (pin_fxpp) pm->dev_free_host(pin_fxpp);
    pin_fxpp = (double *) pm->dev_malloc_host(_size_fxpp*sizeof(double));
  }
  
  int _size_bufpa = naoaux*nmo*ncas;
  if (_size_bufpa > size_bufpa){
    size_bufpa = _size_bufpa;
    if (pin_bufpa) pm->dev_free_host(pin_bufpa);
    pin_bufpa = (double *) pm->dev_malloc_host(_size_bufpa*sizeof(double));
  }
  
  double t1 = omp_get_wtime();
  t_array[8] += t1 - t0;
  // counts in pull ppaa
}

/* ---------------------------------------------------------------------- */

void Device::init_ints_ao2mo_v3(int naoaux, int nmo, int ncas)
{
  double t0 = omp_get_wtime();
  
  int _size_bufpa = naoaux*nmo*ncas;
  if (_size_bufpa > size_bufpa){
    size_bufpa = _size_bufpa;
    if (pin_bufpa) pm->dev_free_host(pin_bufpa);
    pin_bufpa = (double *) pm->dev_malloc_host(_size_bufpa*sizeof(double));
  }
  
  double t1 = omp_get_wtime();
  t_array[8] += t1 - t0;
  // counts in pull ppaa
}
/* ---------------------------------------------------------------------- */
void Device::init_ppaa_ao2mo( int nmo, int ncas)
{
  double t0 = omp_get_wtime();

  // initializing only cpu side, gpu ppaa will be a buffer array (dd->d_buf3) 
  int _size_buf_ppaa = num_devices*nmo*nmo*ncas*ncas;
  if(_size_buf_ppaa > size_buf_ppaa) {
    size_buf_ppaa = _size_buf_ppaa;
    if(buf_ppaa) pm->dev_free_host(buf_ppaa);
    buf_ppaa = (double *) pm->dev_malloc_host(_size_buf_ppaa*sizeof(double));
  }
  
  double t1 = omp_get_wtime();
  t_array[8] += t1 - t0;
  // counts in pull ppaa
}
/* ---------------------------------------------------------------------- */

void Device::init_eri_h2eff(int nmo, int ncas)
{
  double t0 = omp_get_wtime();
  
  // host initializes on each device 

  int ncas_pair = ncas*(ncas+1)/2;
  int size_eri_h2eff = nmo*ncas*ncas_pair;

  for(int id=0; id<num_devices; ++id) {
    pm->dev_set_device(id);

    my_device_data * dd = &(device_data[id]);

    if (size_eri_h2eff > dd->size_eri_h2eff){
      //      printf("setting size\n");
      dd->size_eri_h2eff = size_eri_h2eff;
      if (dd->d_eri_h2eff) pm->dev_free_async(dd->d_eri_h2eff);
      dd->d_eri_h2eff = (double *) pm->dev_malloc_async(dd->size_eri_h2eff*sizeof(double));
    }

  }
  int _size_buf_eri_h2eff = num_devices*nmo*ncas*ncas_pair;
  if(_size_buf_eri_h2eff > size_buf_eri_h2eff) {
    size_buf_eri_h2eff = _size_buf_eri_h2eff;
    if(buf_eri_h2eff) pm->dev_free_host(buf_eri_h2eff);
    buf_eri_h2eff = (double *) pm->dev_malloc_host(size_buf_eri_h2eff*sizeof(double));
    }
  
  double t1 = omp_get_wtime();
  t_array[8] += t1 - t0;
  // counts in pull ppaa
}
/* ---------------------------------------------------------------------- */
void Device::extract_mo_cas(int ncas, int ncore, int nao)
{
  double t0 = omp_get_wtime();
  
  const int _size_mo_cas = ncas*nao; 
  for(int id=0; id<num_devices; ++id) {
    pm->dev_set_device(id);
    my_device_data * dd = &(device_data[id]);
    if (_size_mo_cas > dd->size_mo_cas){
      dd->size_mo_cas = _size_mo_cas;
      if (dd->d_mo_cas) pm->dev_free_async(dd->d_mo_cas);
      dd->d_mo_cas = (double *) pm->dev_malloc_async(_size_mo_cas*sizeof(double));
    }
    #if 0 
    dim3 block_size(1,1,1);
    dim3 grid_size(_TILE(ncas, block_size.x), _TILE(nao, block_size.y));
    get_mo_cas<<<grid_size, block_size, 0, dd->stream>>>(dd->d_mo_coeff, dd->d_mo_cas, ncas, ncore, nao);
    #else
    get_mo_cas(dd->d_mo_coeff,dd->d_mo_cas, ncas, ncore, nao);
    #endif
  }
  
  double t1 = omp_get_wtime();
  t_array[7] += t1 - t0;
}

/* ---------------------------------------------------------------------- */

void Device::pull_jk_ao2mo(py::array_t<double> _j_pc, py::array_t<double> _k_pc, int nmo, int ncore)
{
  double t0 = omp_get_wtime();

  py::buffer_info info_j_pc = _j_pc.request(); //2D array (nmo*ncore)
  double * j_pc = static_cast<double*>(info_j_pc.ptr);
  double * tmp;
  
  py::buffer_info info_k_pc = _k_pc.request(); //2D array (nmo*ncore)
  double * k_pc = static_cast<double*>(info_k_pc.ptr);
  int size = nmo*ncore;//*sizeof(double);

  // Pulling j_pc from all devices
  
  for (int i=0; i<num_devices; ++i){
    pm->dev_set_device(i);

    my_device_data * dd = &(device_data[i]);

    if (i==0) tmp = j_pc;
    else tmp = &(buf_j_pc[i*nmo*ncore]);
    
    if (dd->d_j_pc) pm->dev_pull_async(dd->d_j_pc, tmp, size*sizeof(double));
  }
  
  // Adding j_pc from all devices
  
  for(int i=0; i<num_devices; ++i) {
    pm->dev_set_device(i);

    my_device_data * dd = &(device_data[i]);
    
    pm->dev_stream_wait();

    if(i > 0 && dd->d_j_pc) {
      
      tmp = &(buf_j_pc[i * nmo* ncore]);
//#pragma omp parallel for
      for(int j=0; j<ncore*nmo; ++j) j_pc[j] += tmp[j];
    }
  }
  
  // Pulling k_pc from all devices
  
  for (int i=0; i<num_devices; ++i){
    pm->dev_set_device(i);
    
    my_device_data * dd = &(device_data[i]);

    if (i==0) tmp = k_pc;
    else tmp = &(buf_k_pc[i*nmo*ncore]);
    
    if (dd->d_k_pc) pm->dev_pull_async(dd->d_k_pc, tmp, size*sizeof(double));
  }
  
  // Adding k_pc from all devices
  
  for(int i=0; i<num_devices; ++i) {
    pm->dev_set_device(i);
    
    my_device_data * dd = &(device_data[i]);
    
    pm->dev_stream_wait();

    if(i > 0 && dd->d_k_pc) {
      
      tmp = &(buf_k_pc[i * nmo* ncore]);
//#pragma omp parallel for
      for(int j=0; j<ncore*nmo; ++j) k_pc[j] += tmp[j];
    }
  }
    
  double t1 = omp_get_wtime();
  t_array[10] += t1 - t0;
  // counts in pull ppaa
}

/* ---------------------------------------------------------------------- */

void Device::pull_ints_ao2mo(py::array_t<double> _fxpp, py::array_t<double> _bufpa, int blksize, int naoaux, int nmo, int ncas)
{
  double t0 = omp_get_wtime();
  
  py::buffer_info info_fxpp = _fxpp.request(); //3D array (nmo*nmo*naoaux)
  double * fxpp = static_cast<double*>(info_fxpp.ptr);
  //printf("size_fxpp %i\n", size_fxpp);
    
  int count = 0;
  int k = 0;
  
  // naive version to start; we can make this faster
  while(k < naoaux) {
    int size_vector = (naoaux-k > blksize) ? blksize : naoaux-k; // transfer whole blksize or last subset?
    
    //printf("k= %i  size_vector= %i\n",k,size_vector);
    for (int i=0; i<nmo; ++i)
      for (int j=0; j<nmo; ++j) {
	int indx_in = count * nmo * nmo * blksize + i * nmo * size_vector + j * size_vector;
	int indx_out = i * nmo * naoaux + j * naoaux + k;
	
	std::memcpy(&(fxpp[indx_out]), &(pin_fxpp[indx_in]), size_vector*sizeof(double));
      }
    
    k += blksize;
    count++;
  }
  
  py::buffer_info info_bufpa = _bufpa.request(); //3D array (naoaux*nmo*ncas)
  double * bufpa = static_cast<double*>(info_bufpa.ptr);
  //printf("size_bufpa %i\n", size_bufpa);
  std::memcpy(bufpa, pin_bufpa, size_bufpa*sizeof(double));
  
  double t1 = omp_get_wtime();
  t_array[10] += t1 - t0;
  // counts in pull ppaa
}

/* ---------------------------------------------------------------------- */
void Device::pull_ints_ao2mo_v3(py::array_t<double> _bufpa, int blksize, int naoaux, int nmo, int ncas)
{
  double t0 = omp_get_wtime();
  
  py::buffer_info info_bufpa = _bufpa.request(); //3D array (naoaux*nmo*ncas)
  double * bufpa = static_cast<double*>(info_bufpa.ptr);
  //printf("size_bufpa %i\n", size_bufpa);
  std::memcpy(bufpa, pin_bufpa, size_bufpa*sizeof(double));
  
  double t1 = omp_get_wtime();
  t_array[10] += t1 - t0;
  // counts in pull ppaa
}

/* ---------------------------------------------------------------------- */
void Device::pull_ppaa_ao2mo(py::array_t<double> _ppaa, int nmo, int ncas)
{
  double t0 = omp_get_wtime();

  py::buffer_info info_ppaa = _ppaa.request(); //2D array (nmo*ncore)
  double * ppaa = static_cast<double*>(info_ppaa.ptr);
  double * tmp;
  const int _size_ppaa = nmo*nmo*ncas*ncas;
  // Pulling ppaa from all devices
  
  for (int i=0; i<num_devices; ++i){
    pm->dev_set_device(i);

    my_device_data * dd = &(device_data[i]);

    if (i==0) tmp = ppaa;
    else tmp = &(buf_ppaa[i*_size_ppaa]);
    
    if (dd->d_ppaa) pm->dev_pull_async(dd->d_ppaa, tmp, _size_ppaa*sizeof(double));
  }
  
  // Adding ppaa from all devices
  
  for(int i=0; i<num_devices; ++i) {
    pm->dev_set_device(i);

    my_device_data * dd = &(device_data[i]);
    
    pm->dev_stream_wait();

    if(i > 0 && dd->d_ppaa) {
      
      tmp = &(buf_ppaa[i * _size_ppaa]);
//#pragma omp parallel for
      for(int j=0; j<_size_ppaa; ++j) ppaa[j] += tmp[j];
    }
  }
  
  double t1 = omp_get_wtime();
  t_array[10] += t1 - t0;
  count_array[6] += 1; //doing this in ppaa pull, not in any inits or computes
}

/* ---------------------------------------------------------------------- */



void Device::df_ao2mo_pass1_v2 (int blksize, int nmo, int nao, int ncore, int ncas, int naux, 
				  py::array_t<double> _eri1,
				  int count, size_t addr_dfobj)
{
  double t0 = omp_get_wtime();
  
  profile_start("AO2MO v2");

  const int device_id = count % num_devices;

  pm->dev_set_device(device_id);

  my_device_data * dd = &(device_data[device_id]);

  //printf(" naux %i blksize %i\n", naux, blksize);
#ifdef _DEBUG_DEVICE
  printf("LIBGPU:: Inside Device::df_ao2mo_pass1_fdrv()\n");
  printf("LIBGPU:: dfobj= %#012x  count= %i  combined= %#012x %p update_dfobj= %i\n",addr_dfobj,count,addr_dfobj+count,addr_dfobj+count,update_dfobj);
  printf("LIBGPU:: blksize= %i  nmo= %i  nao= %i  ncore= %i  ncas= %i  naux= %i  count= %i\n",blksize, nmo, nao, ncore, ncas, naux, count);
#endif

  //  py::buffer_info info_eri1 = _eri1.request(); // 2D array (naux, nao_pair) nao_pair= nao*(nao+1)/2
  const int nao_pair = nao*(nao+1)/2;
  //  double * eri = static_cast<double*>(info_eri1.ptr);
  
  int _size_eri = naux * nao_pair;
  int _size_eri_unpacked = naux * nao * nao; 
  
#ifdef _DEBUG_DEVICE
#if defined (_GPU_CUDA)
  size_t freeMem;size_t totalMem;
  freeMem=0;totalMem=0;
  cudaMemGetInfo(&freeMem, &totalMem);
  printf("Starting ao2mo Free memory %lu bytes, total memory %lu bytes\n",freeMem,totalMem);
#endif
#endif

  if(_size_eri_unpacked > dd->size_buf) {
    dd->size_buf = _size_eri_unpacked;
    
    if(dd->d_buf1) pm->dev_free_async(dd->d_buf1);
    if(dd->d_buf2) pm->dev_free_async(dd->d_buf2);
    if(dd->d_buf3) pm->dev_free_async(dd->d_buf3);
    
    dd->d_buf1 = (double *) pm->dev_malloc_async(dd->size_buf * sizeof(double));// use for (eri@mo)
    dd->d_buf2 = (double *) pm->dev_malloc_async(dd->size_buf * sizeof(double));//use for eri_unpacked, then for bufpp_t
    dd->d_buf3 = (double *) pm->dev_malloc_async(dd->size_buf * sizeof(double));
  }
  
  double * d_buf = dd->d_buf1; //for eri*mo_coeff (don't pull or push) 
  double * d_eri_unpacked = dd->d_buf2; //set memory for the entire eri array on GPU
  
  //unpack 2D eri of size naux * nao(nao+1)/2 to a full naux*nao*nao 3D matrix
  
  double * d_eri = nullptr;
  
  if(use_eri_cache) {
    //    d_eri = dd_fetch_eri(dd, eri, naux, nao_pair, addr_dfobj, count);
    d_eri = dd_fetch_eri(dd, nullptr, naux, nao_pair, addr_dfobj, count);
  } else {
    if(_size_eri > dd->size_eri1) {
      dd->size_eri1 = _size_eri;
      if(dd->d_eri1) pm->dev_free_async(dd->d_eri1);
      dd->d_eri1 = (double *) pm->dev_malloc_async(_size_eri * sizeof(double));
    }
    d_eri = dd->d_eri1;
    
    //    printf("d_cderi= %p  cderi= %p  _size_eri= %i  naux= %i  nao_pair= %i\n",d_eri, eri, _size_eri, naux, nao_pair); // naux is negative because eri_extra not correctly initialized; and eri is nullptr in this call; use_eri_cache must be used
    //    pm->dev_push_async(d_eri, eri, _size_eri * sizeof(double));
  }
  
  int * my_d_tril_map_ptr = dd_fetch_pumap(dd, nao, _PUMAP_2D_UNPACK);

  getjk_unpack_buf2(d_eri_unpacked, d_eri, my_d_tril_map_ptr, naux, nao, nao_pair);
  
  //bufpp = mo.T @ eri @ mo
  //buf = np.einsum('ijk,kl->ijl',eri_unpacked,mo_coeff),i=naux,j=nao,l=nao
  
  const double alpha = 1.0;
  const double beta = 0.0;
  const int nao2 = nao * nao;
  const int zero = 0;
  
  ml->set_handle();
  ml->gemm_batch((char *) "N", (char *) "N", &nao, &nao, &nao,
		 &alpha, d_eri_unpacked, &nao, &nao2, dd->d_mo_coeff, &nao, &zero, &beta, d_buf, &nao, &nao2, &naux);
  
  //bufpp = np.einsum('jk,ikl->ijl',mo_coeff.T,buf),i=naux,j=nao,l=nao
  
  double * d_bufpp = dd->d_buf2;//set memory for the entire bufpp array, no pushing needed

  ml->gemm_batch((char *) "T", (char *) "N", &nao, &nao, &nao,
		 &alpha, dd->d_mo_coeff, &nao, &zero, d_buf, &nao, &nao2, &beta, d_bufpp, &nao, &nao2, &naux);

  int _size_bufpa = naux*nmo*ncas;
  if(_size_bufpa > dd->size_bufpa) {
    dd->size_bufpa = _size_bufpa;
    
    if(dd->d_bufpa) pm->dev_free_async(dd->d_bufpa);
    dd->d_bufpa = (double *) pm->dev_malloc_async(dd->size_bufpa * sizeof(double));
  }
  
  double * d_bufpa = dd->d_bufpa;

  get_bufpa(d_bufpp, d_bufpa, naux, nmo, ncore, ncas);

  double * bufpa = &(pin_bufpa[count*blksize*nmo*ncas]);

  pm->dev_pull_async(d_bufpa, bufpa, naux*nmo*ncas*sizeof(double));

  double * d_fxpp = dd->d_buf1;
  
  // fxpp[str(k)] =bufpp.transpose(1,2,0);

  transpose_120(d_bufpp, d_fxpp, naux, nmo, nmo);

// calculate j_pc
  
  // k_cp += numpy.einsum('kij,kij->ij', bufpp[:,:ncore], bufpp[:,:ncore])

  int one = 1;
  int nmo_ncore = nmo * ncore;
  double beta_ = (count < num_devices) ? 0.0 : 1.0;
  
  ml->gemm_batch((char *) "N", (char *) "T", &one, &one, &naux,
		 &alpha, d_fxpp, &one, &naux, d_fxpp, &one, &naux, &beta_, dd->d_k_pc, &one, &one, &nmo_ncore);
  
  double * fxpp = &(pin_fxpp[count*blksize*nmo*nmo]);

  pm->dev_pull_async(d_fxpp, fxpp, naux*nmo*nmo *sizeof(double));
  
  //bufd work

  int _size_bufd = naux*nmo;
  if(_size_bufd > dd->size_bufd) {
    dd->size_bufd = _size_bufd;
    
    if(dd->d_bufd) pm->dev_free_async(dd->d_bufd);
    dd->d_bufd = (double *) pm->dev_malloc_async(dd->size_bufd * sizeof(double));
  }
  
  double * d_bufd = dd->d_bufd;

  get_bufd(d_bufpp, d_bufd, naux, nmo);
  
// calculate j_pc
  
  // self.j_pc += numpy.einsum('ki,kj->ij', bufd, bufd[:,:ncore])

  ml->gemm((char *) "N", (char *) "T", &ncore, &nmo, &naux,
  	   &alpha, d_bufd, &nmo, d_bufd, &nmo, &beta_, dd->d_j_pc, &ncore);
  
#ifdef _DEBUG_DEVICE
#if defined (_GPU_CUDA)
  printf("LIBGPU :: Leaving Device::df_ao2mo_pass1_fdrv()\n"); 
  cudaMemGetInfo(&freeMem, &totalMem);
  printf("Ending ao2mo fdrv Free memory %lu bytes, total memory %lu bytes\n",freeMem,totalMem);
#endif
#endif
  
  profile_stop();
  
  double t1 = omp_get_wtime();
  t_array[9] += t1 - t0;
  // counts in pull ppaa
}
/* ---------------------------------------------------------------------- */

void Device::df_ao2mo_v3 (int blksize, int nmo, int nao, int ncore, int ncas, int naux, 
				  py::array_t<double> _eri1,
				  int count, size_t addr_dfobj)
{
  double t0 = omp_get_wtime();
  
  profile_start("AO2MO v3");

  const int device_id = count % num_devices;

  pm->dev_set_device(device_id);

  my_device_data * dd = &(device_data[device_id]);


  //  py::buffer_info info_eri1 = _eri1.request(); // 2D array (naux, nao_pair) nao_pair= nao*(nao+1)/2
  const int nao_pair = nao*(nao+1)/2;
  //  double * eri = static_cast<double*>(info_eri1.ptr);
  
  int _size_eri = naux * nao_pair;
  int _size_eri_unpacked = naux * nao * nao; 
  int _size_ppaa = nmo * nmo * ncas * ncas;
  if (_size_eri_unpacked < _size_ppaa) {
  _size_eri_unpacked = _size_ppaa;
  }

#ifdef _DEBUG_DEVICE
#if defined (_GPU_CUDA)
  size_t freeMem;size_t totalMem;
  freeMem=0;totalMem=0;
  cudaMemGetInfo(&freeMem, &totalMem);
  printf("Starting ao2mo Free memory %lu bytes, total memory %lu bytes\n",freeMem,totalMem);
#endif
#endif

  if(_size_eri_unpacked > dd->size_buf) {
    dd->size_buf = _size_eri_unpacked;
    
    if(dd->d_buf1) pm->dev_free_async(dd->d_buf1);
    if(dd->d_buf2) pm->dev_free_async(dd->d_buf2);
    if(dd->d_buf3) pm->dev_free_async(dd->d_buf3);
    
    dd->d_buf1 = (double *) pm->dev_malloc_async(dd->size_buf * sizeof(double));// use for (eri@mo)
    dd->d_buf2 = (double *) pm->dev_malloc_async(dd->size_buf * sizeof(double));//use for eri_unpacked, then for bufpp_t
    dd->d_buf3 = (double *) pm->dev_malloc_async(dd->size_buf * sizeof(double));//for ppaa
  }
  // my guess is blksize*nao_s*nao_s > nmo_f * nmo_f * ncas_f * ncas_f (dd->size_eri_unpacked is for the entire system. Therefore nao_s > nao_f. Since blksize = 240, ncas_f must be less than 15)
  
  double * d_buf = dd->d_buf1; //for eri*mo_coeff (don't pull or push) 
  double * d_eri_unpacked = dd->d_buf2; //set memory for the entire eri array on GPU
  
  //unpack 2D eri of size naux * nao(nao+1)/2 to a full naux*nao*nao 3D matrix
  
  double * d_eri = nullptr;
  
  if(use_eri_cache) {
    //    d_eri = dd_fetch_eri(dd, eri, naux, nao_pair, addr_dfobj, count);
    d_eri = dd_fetch_eri(dd, nullptr, naux, nao_pair, addr_dfobj, count);
  } else {
    if(_size_eri > dd->size_eri1) {
      dd->size_eri1 = _size_eri;
      if(dd->d_eri1) pm->dev_free_async(dd->d_eri1);
      dd->d_eri1 = (double *) pm->dev_malloc_async(_size_eri * sizeof(double));
    }
    d_eri = dd->d_eri1;
    
    //    printf("d_cderi= %p  cderi= %p  _size_eri= %i  naux= %i  nao_pair= %i\n",d_eri, eri, _size_eri, naux, nao_pair); // naux is negative because eri_extra not correctly initialized; and eri is nullptr in this call; use_eri_cache must be used
    //    pm->dev_push_async(d_eri, eri, _size_eri * sizeof(double));
  }
  
  int * my_d_tril_map_ptr = dd_fetch_pumap(dd, nao, _PUMAP_2D_UNPACK);

  getjk_unpack_buf2(d_eri_unpacked, d_eri, my_d_tril_map_ptr, naux, nao, nao_pair);
  
  //bufpp = mo.T @ eri @ mo
  //buf = np.einsum('ijk,kl->ijl',eri_unpacked,mo_coeff),i=naux,j=nao,l=nao
  
  const double alpha = 1.0;
  const double beta = 0.0;
  const int nao2 = nao * nao;
  const int zero = 0;
  
  ml->set_handle();
  ml->gemm_batch((char *) "N", (char *) "N", &nao, &nao, &nao,
		 &alpha, d_eri_unpacked, &nao, &nao2, dd->d_mo_coeff, &nao, &zero, &beta, d_buf, &nao, &nao2, &naux);
  
  //bufpp = np.einsum('jk,ikl->ijl',mo_coeff.T,buf),i=naux,j=nao,l=nao
  
  double * d_bufpp = dd->d_buf2;//set memory for the entire bufpp array, no pushing needed

  ml->gemm_batch((char *) "T", (char *) "N", &nao, &nao, &nao,
		 &alpha, dd->d_mo_coeff, &nao, &zero, d_buf, &nao, &nao2, &beta, d_bufpp, &nao, &nao2, &naux);

  int _size_bufpa = naux*nmo*ncas;
  if(_size_bufpa > dd->size_bufpa) {
    dd->size_bufpa = _size_bufpa;
    
    if(dd->d_bufpa) pm->dev_free_async(dd->d_bufpa);
    dd->d_bufpa = (double *) pm->dev_malloc_async(dd->size_bufpa * sizeof(double));
  }
  
  double * d_bufpa = dd->d_bufpa;

  get_bufpa(d_bufpp, d_bufpa, naux, nmo, ncore, ncas);

  double * bufpa = &(pin_bufpa[count*blksize*nmo*ncas]);

  pm->dev_pull_async(d_bufpa, bufpa, naux*nmo*ncas*sizeof(double));

  double * d_fxpp = dd->d_buf1;
  
  // fxpp[str(k)] =bufpp.transpose(1,2,0);

  transpose_120(d_bufpp, d_fxpp, naux, nmo, nmo);

// calculate j_pc
  
  // k_cp += numpy.einsum('kij,kij->ij', bufpp[:,:ncore], bufpp[:,:ncore])

  int one = 1;
  int nmo_ncore = nmo * ncore;
  double beta_ = (count < num_devices) ? 0.0 : 1.0;
  
  ml->gemm_batch((char *) "N", (char *) "T", &one, &one, &naux,
		 &alpha, d_fxpp, &one, &naux, d_fxpp, &one, &naux, &beta_, dd->d_k_pc, &one, &one, &nmo_ncore);

  //removing because ppaa consumes fxpp
  #if 0 
  double * fxpp = &(pin_fxpp[count*blksize*nmo*nmo]);

  pm->dev_pull_async(d_fxpp, fxpp, naux*nmo*nmo *sizeof(double));
  #else
  #endif
  
  //bufd work

  int _size_bufd = naux*nmo;
  if(_size_bufd > dd->size_bufd) {
    dd->size_bufd = _size_bufd;
    
    if(dd->d_bufd) pm->dev_free_async(dd->d_bufd);
    dd->d_bufd = (double *) pm->dev_malloc_async(dd->size_bufd * sizeof(double));
  }
  
  double * d_bufd = dd->d_bufd;

  get_bufd(d_bufpp, d_bufd, naux, nmo);
  
// calculate j_pc
  
  // self.j_pc += numpy.einsum('ki,kj->ij', bufd, bufd[:,:ncore])

  ml->gemm((char *) "N", (char *) "T", &ncore, &nmo, &naux,
	   &alpha, d_bufd, &nmo, d_bufd, &nmo, &beta_, dd->d_j_pc, &ncore);

  // new work
  int _size_bufaa = naux*ncas*ncas;
  if(_size_bufaa > dd->size_bufaa) {
    dd->size_bufaa = _size_bufaa;
    
    if(dd->d_bufaa) pm->dev_free_async(dd->d_bufaa);
    dd->d_bufaa = (double *) pm->dev_malloc_async(dd->size_bufaa * sizeof(double));
  }
  double * d_bufaa = dd->d_bufaa;

  get_bufaa(d_bufpp, d_bufaa, naux, nmo, ncore, ncas);
#if 0 
  double * h_bufaa  = (double*) pm->dev_malloc_host(_size_bufaa*sizeof(double));
  pm->dev_pull(d_bufaa, h_bufaa, _size_bufaa*sizeof(double));
  //for (int i = 0; i<naux; ++i){
  //for (int k = 0; k<ncas; ++k){
  //for (int l = 0; l<ncas; ++l){
  //printf("%f\t",h_bufaa[(i*ncas+k)*ncas+l]);}}printf("\n");}
#endif
#if 0 
  double * h_fxpp  = (double*) pm->dev_malloc_host(nao*nao*naux*sizeof(double));
  pm->dev_pull(d_fxpp, h_fxpp, nao*nao*naux*sizeof(double));
  //for (int k = 0; k<nao; ++k){
  //for (int l = 0; l<nao; ++l){
  //for (int i = 0; i<naux; ++i){
  //printf("%f\t",h_fxpp[(k*nao+l)*naux+i]);}printf("\n");}}
#endif



  const int ncas2 = ncas*ncas;

  // calculate ppaa
  dd->d_ppaa = dd->d_buf3;
  ml->gemm ((char *) "N", (char *) "N", &ncas2, &nao2, &naux,  
                   &alpha,  d_bufaa, &ncas2, d_fxpp, &naux, &beta_, dd->d_ppaa, &ncas2);                  
#if 0
  for (int i = 0; i<nao2; ++i){
  for (int j = 0; j<ncas2; ++j){
  h_ppaa[i*ncas2+j]=0;
  for (int k = 0; k<naux; ++k){
  h_ppaa[i*ncas2+j] = h_ppaa[i*ncas2+j] + h_fxpp[i*naux+k]*h_bufaa[k*ncas2+j];
   }}}
#endif
#if 0
  double * h_ppaa  = (double*) pm->dev_malloc_host(_size_ppaa*sizeof(double));
  printf("ppaa from gpu\n"); 
  pm->dev_pull(d_ppaa, h_ppaa, _size_ppaa*sizeof(double));
  for (int i = 0; i<nmo; ++i){
  for (int j = 0; j<nmo; ++j){
  for (int k = 0; k<ncas; ++k){
  for (int l = 0; l<ncas; ++l){
  printf("%f\t",h_ppaa[((i*nmo+j)*ncas+k)*ncas+l]);}}printf("\n");}}
#endif

#ifdef _DEBUG_DEVICE
#if defined (_GPU_CUDA)
  printf("LIBGPU :: Leaving Device::df_ao2mo_pass1_fdrv()\n"); 
  cudaMemGetInfo(&freeMem, &totalMem);
  printf("Ending ao2mo fdrv Free memory %lu bytes, total memory %lu bytes\n",freeMem,totalMem);
#endif
#endif
  
  profile_stop();
  
  double t1 = omp_get_wtime();
  t_array[9] += t1 - t0;
  // counts in pull ppaa
}


/* ---------------------------------------------------------------------- */

void Device::update_h2eff_sub(int ncore, int ncas, int nocc, int nmo,
                              py::array_t<double> _umat, py::array_t<double> _h2eff_sub)
{
  double t0 = omp_get_wtime();

#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Inside Device :: Starting update_h2eff_sub function\n");
#endif

  profile_start("Setup initial h2eff_sub");
  
  py::buffer_info info_umat = _umat.request(); // 2d array nmo*nmo
  py::buffer_info info_h2eff_sub = _h2eff_sub.request();// 2d array (nmo * ncas) x (ncas*(ncas+1)/2)

  const int device_id = 0;//count % num_devices;

  pm->dev_set_device(device_id);

  my_device_data * dd = &(device_data[device_id]);

  const int ncas_pair = ncas * (ncas+1)/2;
  
  double * umat = static_cast<double*>(info_umat.ptr);
  double * h2eff_sub = static_cast<double*>(info_h2eff_sub.ptr);

#ifdef _DEBUG_DEVICE
#if defined (_GPU_CUDA)
  size_t freeMem;size_t totalMem;
  freeMem=0;totalMem=0;
  cudaMemGetInfo(&freeMem, &totalMem);
  printf("Starting h2eff_update Free memory %lu bytes, total memory %lu bytes\n",freeMem,totalMem);
#endif
#endif
  
  int _size_h2eff_unpacked = nmo*ncas*ncas*ncas;
  int _size_h2eff_packed = nmo*ncas*ncas_pair;

  if(_size_h2eff_unpacked > dd->size_buf) {
    dd->size_buf = _size_h2eff_unpacked;

    if(dd->d_buf1) pm->dev_free_async(dd->d_buf1);
    if(dd->d_buf2) pm->dev_free_async(dd->d_buf2);
    if(dd->d_buf3) pm->dev_free_async(dd->d_buf3);

    dd->d_buf1 = (double *) pm->dev_malloc_async(dd->size_buf * sizeof(double));
    dd->d_buf2 = (double *) pm->dev_malloc_async(dd->size_buf * sizeof(double));
    dd->d_buf3 = (double *) pm->dev_malloc_async(dd->size_buf * sizeof(double));
  }

  double * d_h2eff_unpacked = dd->d_buf1;

  if(ncas*ncas > dd->size_ucas) {
    dd->size_ucas = ncas * ncas;
    if(dd->d_ucas) pm->dev_free_async(dd->d_ucas);
    dd->d_ucas = (double *) pm->dev_malloc_async(dd->size_ucas * sizeof(double));
  }
  
  if(nmo*nmo > dd->size_umat) {
    dd->size_umat = nmo * nmo;
    if(dd->d_umat) pm->dev_free_async(dd->d_umat);
    dd->d_umat = (double *) pm->dev_malloc_async(dd->size_umat * sizeof(double));
  }
  
  pm->dev_push_async(dd->d_umat, umat, nmo*nmo*sizeof(double));

#ifdef _DEBUG_H2EFF
  printf("LIBGPU :: Inside Device :: -- Setup update function\n");
#endif
  
  profile_next("extraction");
  
  //ucas = umat[ncore:nocc, ncore:nocc]

  extract_submatrix(dd->d_umat, dd->d_ucas, ncas, ncore, nmo);
  
  //h2eff_sub = h2eff_sub.reshape (nmo*ncas, ncas*(ncas+1)//2)
  //h2eff_sub = lib.numpy_helper.unpack_tril (h2eff_sub)
  //h2eff_sub = h2eff_sub.reshape (nmo, ncas, ncas, ncas)

  if(_size_h2eff_packed > dd->size_h2eff) {
    dd->size_h2eff = _size_h2eff_packed;
    if(dd->d_h2eff) pm->dev_free_async(dd->d_h2eff);
    dd->d_h2eff = (double *) pm->dev_malloc_async(dd->size_h2eff * sizeof(double));
  }
  
  double * d_h2eff_sub = dd->d_h2eff;
  
  pm->dev_push_async(d_h2eff_sub, h2eff_sub, _size_h2eff_packed * sizeof(double));

  profile_next("map creation and pushed");
  
  int * d_my_unpack_map_ptr = dd_fetch_pumap(dd, ncas, _PUMAP_H2EFF_UNPACK);

  profile_next("unpacking");

#ifdef _DEBUG_H2EFF
  printf("LIBGPU :: Inside Device :: -- created and pushed unpacking map\n");
#endif

  unpack_h2eff_2d(d_h2eff_sub, d_h2eff_unpacked, d_my_unpack_map_ptr, nmo, ncas, ncas_pair);
  
  profile_next("2 dgemms");
  
#ifdef _DEBUG_H2EFF
  printf("LIBGPU :: Inside Device :: -- unpacked h2eff_sub \n");
#endif
  
  //1. h2eff_sub = np.tensordot (ucas, h2eff_sub, axes=((0),(1))) # bpaa
  //2. h2eff_sub = np.tensordot (umat, h2eff_sub, axes=((0),(1))) # qbaa
  //3. h2eff_sub = np.tensordot (h2eff_sub, ucas, axes=((2),(0))) # qbab
  //4. h2eff_sub = np.tensordot (h2eff_sub, ucas, axes=((2),(0))) # qbbb
  // doing 3,4,tranpose, 1,2, tranpose
  
  const double alpha=1.0;
  const double beta=0.0;
  
  //h2eff_step1=([pi]jk,jJ->[pi]kJ)

  double * d_h2eff_step1 = dd->d_buf2;

  int zero = 0;
  int ncas2 = ncas * ncas;
  int ncas_nmo = ncas * nmo;
  
  ml->set_handle();
  ml->gemm_batch((char *) "N", (char *) "N", &ncas, &ncas, &ncas,
		 &alpha, dd->d_ucas, &ncas, &zero, d_h2eff_unpacked, &ncas, &ncas2, &beta, d_h2eff_step1, &ncas, &ncas2, &ncas_nmo);
  
  //h2eff_step2=([pi]kJ,kK->[pi]JK
  
  double * d_h2eff_step2 = dd->d_buf1;

  ml->gemm_batch((char *) "N", (char *) "T", &ncas, &ncas, &ncas,
		 &alpha, d_h2eff_step1, &ncas, &ncas2, dd->d_ucas, &ncas, &zero, &beta, d_h2eff_step2, &ncas, &ncas2, &ncas_nmo);
  
  profile_next("transpose");
  
#ifdef _DEBUG_H2EFF
  printf("LIBGPU :: Inside Device :: -- Finished first 2 cublasDgemmStridedBatched Functions \n");
#endif
  
  //h2eff_tranposed=(piJK->JKpi)
  
  double * d_h2eff_transposed = dd->d_buf2;

  transpose_2310(d_h2eff_step2, d_h2eff_transposed, nmo, ncas);
  
  profile_next("last 2 dgemm");
  
#ifdef _DEBUG_H2EFF
  printf("LIBGPU :: Inside Device :: -- Finished transposing\n");
#endif
  
  double * d_h2eff_step3 = dd->d_buf1;

  //h2eff_sub=np.einsum('iI,JKip->JKIp',ucas,h2eff_sub) h2eff=ncas,ncas,ncas,nmo; ucas=ncas,ncas

  ml->gemm_batch((char *) "N", (char *) "T", &nmo, &ncas, &ncas,
		 &alpha, d_h2eff_transposed, &nmo, &ncas_nmo, dd->d_ucas, &ncas, &zero, &beta, d_h2eff_step3, &nmo, &ncas_nmo, &ncas2);
  
  //h2eff_step4=([JK]Ip,pP->[JK]IP)

  double * d_h2eff_step4 = dd->d_buf2;

  ml->gemm_batch((char *) "N", (char *) "N", &nmo, &ncas, &nmo,
		 &alpha, dd->d_umat, &nmo, &zero, d_h2eff_step3, &nmo, &ncas_nmo, &beta, d_h2eff_step4, &nmo, &ncas_nmo, &ncas2);
  
  profile_next("2nd transpose");

#ifdef _DEBUG_H2EFF
  printf("LIBGPU :: Inside Device :: -- Finished last 2 cublasDgemmStridedBatched Functions \n");
#endif

  double * d_h2eff_transpose2 = dd->d_buf1;
  
  //h2eff_tranposed=(JKIP->PIJK) 3201

  transpose_3210(d_h2eff_step4, d_h2eff_transpose2, nmo, ncas);
  
#ifdef _DEBUG_H2EFF
  printf("LIBGPU :: Inside Device :: -- finished transposing back\n");
#endif
  
  //ix_i, ix_j = np.tril_indices (ncas)
  //h2eff_sub = h2eff_sub.reshape (nmo, ncas, ncas*ncas)
  //h2eff_sub = h2eff_sub[:,:,(ix_i*ncas)+ix_j]
  //h2eff_sub = h2eff_sub.reshape (nmo, -1)

  profile_next("second map and packing");
  
  int * d_my_pack_map_ptr = dd_fetch_pumap(dd, ncas, _PUMAP_H2EFF_PACK);

  pack_h2eff_2d(d_h2eff_transpose2, d_h2eff_sub, d_my_pack_map_ptr, nmo, ncas, ncas_pair);
  
#ifdef _DEBUG_H2EFF
  printf("LIBGPU :: Inside Device :: -- Freed map\n");
#endif
  
  pm->dev_pull_async(d_h2eff_sub, h2eff_sub, _size_h2eff_packed*sizeof(double));

  pm->dev_stream_wait(); // is this required or can we delay waiting?
  
  profile_stop();
  
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Inside Device :: Leaving update function\n");
#if defined (_GPU_CUDA)
  cudaMemGetInfo(&freeMem, &totalMem);
  
  printf("Ending h2eff_sub_update Free memory %lu bytes, total memory %lu bytes\n",freeMem,totalMem);
#endif
#endif
  
  double t1 = omp_get_wtime();
  t_array[5] += t1 - t0;
  count_array[3] += 1;
}

/* ---------------------------------------------------------------------- */

void Device::get_h2eff_df(py::array_t<double> _cderi, 
                                int nao, int nmo, int ncas, int naux, int ncore, 
                                py::array_t<double> _eri, int count, size_t addr_dfobj) 
{
  double t0 = omp_get_wtime();

#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Inside Device :: Starting h2eff_df_contract1 function");
  printf("LIBGPU:: dfobj= %#012x count= %i combined= %#012x %p update_dfobj= %i\n",addr_dfobj,count,addr_dfobj+count,addr_dfobj+count,update_dfobj);
#endif 
  
  profile_start("h2eff df setup");
  
  py::buffer_info info_eri = _eri.request(); //2D array nao * ncas * ncas_pair
  
  const int device_id = count % num_devices;
  
  pm->dev_set_device(device_id);
  
  my_device_data * dd = &(device_data[device_id]);
  
  const int nao_pair = nao * (nao+1)/2;
  const int ncas_pair = ncas * (ncas+1)/2;
  const int _size_eri = nmo*ncas*ncas_pair;
  const int _size_cderi = naux*nao_pair;
  const int _size_cderi_unpacked = naux*nao*nao;
  const int _size_mo_cas = nao*ncas;
  
  double * eri = static_cast<double*>(info_eri.ptr);
  double * d_mo_coeff = dd->d_mo_coeff;
  
  double * d_mo_cas = (double*) pm->dev_malloc_async(_size_mo_cas*sizeof(double));
  
  py::buffer_info info_cderi = _cderi.request(); // 2D array blksize * nao_pair
  double * cderi = static_cast<double*>(info_cderi.ptr);

  // d_mo_cas

  get_mo_cas(d_mo_coeff, d_mo_cas, ncas, ncore, nao);

  double * d_cderi = nullptr;
  
  if(use_eri_cache) {
    d_cderi = dd_fetch_eri(dd, cderi, naux, nao_pair, addr_dfobj, count);
  } else {
    if(_size_cderi > dd->size_eri1) {
      dd->size_eri1 = _size_cderi;
      if(dd->d_eri1) pm->dev_free_async(dd->d_eri1);
      dd->d_eri1 = (double *) pm->dev_malloc_async(_size_cderi * sizeof(double));
    }
    d_cderi = dd->d_eri1;

    pm->dev_push_async(d_cderi, cderi, _size_cderi * sizeof(double));
  }

  double * d_cderi_unpacked = (double*) pm->dev_malloc_async( _size_cderi_unpacked * sizeof(double));

  int * d_my_unpack_map_ptr = dd_fetch_pumap(dd, nao, _PUMAP_2D_UNPACK);

  getjk_unpack_buf2(d_cderi_unpacked,d_cderi,d_my_unpack_map_ptr,naux, nao, nao_pair);
  
  //bPmu = np.einsum('Pmn,nu->Pmu',cderi,mo_cas)
  
  const double alpha = 1.0;
  const double beta = 0.0;
  const int _size_bPmu = naux*ncas*nao;
  
  double * d_bPmu = (double*) pm->dev_malloc_async(_size_bPmu *sizeof(double));

  int zero = 0;
  int nao2 = nao * nao;
  int ncas_nao = ncas * nao;
  
  ml->set_handle();
  ml->gemm_batch((char *) "N", (char *) "N", &nao, &ncas, &nao,
		 &alpha, d_cderi_unpacked, &nao, &nao2, d_mo_cas, &nao, &zero, &beta, d_bPmu, &nao, &ncas_nao, &naux);
  
  pm->dev_free_async(d_cderi_unpacked);

  //bPvu = np.einsum('mv,Pmu->Pvu',mo_cas.conjugate(),bPmu)
  
  const int _size_bPvu = naux*ncas*ncas;
  
  double * d_bPvu = (double*) pm->dev_malloc_async(_size_bPvu *sizeof(double));
  
  int ncas2 = ncas * ncas;
  ml->gemm_batch((char *) "T", (char *) "N", &ncas, &ncas, &nao,
		 &alpha, d_mo_cas, &nao, &zero, d_bPmu, &nao, &ncas_nao, &beta, d_bPvu, &ncas, &ncas2, &naux);
  
  //eri = np.einsum('Pmw,Pvu->mwvu', bPmu, bPvu)
  //transpose bPmu
  
  double * d_bumP = (double*) pm->dev_malloc_async(_size_bPmu *sizeof(double));

  transpose_120(d_bPmu, d_bumP, naux, ncas, nao, 1); // this call distributes work items differently 
  
  pm->dev_free_async(d_bPmu);

  double * d_buvP = (double*) pm->dev_malloc_async(_size_bPvu *sizeof(double));

  //transpose bPvu

  transpose_210(d_bPvu, d_buvP, naux, ncas, ncas);

  pm->dev_free_async(d_bPvu);

  //h_vuwm[i*ncas*nao+j]+=h_bvuP[i*naux + k]*h_bumP[j*naux+k];
  //dgemm (probably just simple, not strided/batched, contracted dimension = P)

  const int _size_mwvu = nao*ncas*ncas*ncas;
  double * d_vuwm = (double*) pm ->dev_malloc_async( _size_mwvu*sizeof(double));

  ml->gemm((char *) "T", (char *) "N", &ncas_nao, &ncas2, &naux,
	   &alpha, d_bumP, &naux, d_buvP, &naux, &beta, d_vuwm, &ncas_nao);
  
  pm->dev_free_async(d_bumP);
  pm->dev_free_async(d_buvP);

  //eri = np.einsum('mM,mwvu->Mwvu', mo_coeff.conjugate(),eri)
  //gemm_batch(batch = v*u, contracted dimenion = m)
  
  double * d_vuwM = (double*) pm ->dev_malloc_async(_size_mwvu*sizeof(double));
  
  ml->gemm_batch((char *) "T", (char *) "T", &ncas, &nao, &nao,
		 &alpha, d_vuwm, &nao, &ncas_nao, d_mo_coeff, &nao, &zero, &beta, d_vuwM, &ncas, &ncas_nao, &ncas2);
  
  pm->dev_free_async(d_vuwm);

  double * d_eri = (double*) pm->dev_malloc_async(_size_eri*sizeof(double));

  int * my_d_tril_map_ptr = dd_fetch_pumap(dd, ncas, _PUMAP_2D_UNPACK);

  pack_d_vuwM(d_vuwM, d_eri, my_d_tril_map_ptr, nmo, ncas, ncas_pair);
  
  pm->dev_free_async(d_vuwM);

  pm->dev_pull_async(d_eri, eri, _size_eri*sizeof(double));

  pm->dev_free_async(d_eri);

  pm->dev_stream_wait();
  
  profile_stop();
  
  double t1 = omp_get_wtime();
  t_array[6] += t1 - t0;//TODO: add the array size
}
/* ---------------------------------------------------------------------- */

void Device::get_h2eff_df_v1(py::array_t<double> _cderi, 
                                int nao, int nmo, int ncas, int naux, int ncore, 
                                py::array_t<double> _eri, int count, size_t addr_dfobj) 
{
  double t0 = omp_get_wtime();

#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Inside Device :: Starting h2eff_df_contract1 function");
  printf("LIBGPU:: dfobj= %#012x count= %i combined= %#012x %p update_dfobj= %i\n",addr_dfobj,count,addr_dfobj+count,addr_dfobj+count,update_dfobj);
#endif 
  
  profile_start("h2eff df setup");
  
  py::buffer_info info_eri = _eri.request(); //2D array nao * ncas * ncas_pair
  
  const int device_id = count % num_devices;
  
  pm->dev_set_device(device_id);
  
  my_device_data * dd = &(device_data[device_id]);
  
  const int nao_pair = nao * (nao+1)/2;
  const int ncas_pair = ncas * (ncas+1)/2;
  const int _size_eri = nmo*ncas*ncas_pair;
  const int _size_cderi = naux*nao_pair;
  const int _size_mo_cas = nao*ncas;

  const int eri_size = naux * nao * nao;
  const int bump_buvp = naux * ncas * (ncas + nao); 
  const int size_vuwm = ncas * ncas * ncas * nao;
  
#if 1

#if 1
  // assume nao>ncas
  
  int _size_cderi_unpacked = eri_size;
  if(bump_buvp > _size_cderi_unpacked) _size_cderi_unpacked = bump_buvp;
  if(size_vuwm > _size_cderi_unpacked) _size_cderi_unpacked = size_vuwm;
  
  // the above exercise is done so as to avoid new memory allocations during the calculations and allocate the largest needed arrays.
  // bump and buvp is done together because they need to exist simultaneously. naux*nao^2 vs naux*ncas*(nao+ncas) vs nmo*ncas^3 
#else
  // doing this because naux*nao**2 > nao*ncas**3 and naux*nao**2 > naux*ncas*(ncas+nao)
  const int _size_cderi_unpacked = naux * nao * nao; 
#endif
  
  if(_size_cderi_unpacked > dd->size_buf) {
  //printf("Size ERI in h2eff v2: %i", dd->size_eri_unpacked);
    
    dd->size_eri_unpacked = _size_cderi_unpacked;
    dd->size_buf = _size_cderi_unpacked;
  
    if (dd->d_buf1) pm->dev_free_async(dd->d_buf1);
    if (dd->d_buf2) pm->dev_free_async(dd->d_buf2);
    if (dd->d_buf3) pm->dev_free_async(dd->d_buf3);
    
    dd->d_buf1 = (double *) pm->dev_malloc_async ( dd->size_buf * sizeof(double));
    dd->d_buf2 = (double *) pm->dev_malloc_async ( dd->size_buf * sizeof(double));
    dd->d_buf3 = (double *) pm->dev_malloc_async ( dd->size_buf * sizeof(double));
  }
#endif
  
  double * eri = static_cast<double*>(info_eri.ptr);
  double * d_mo_coeff = dd->d_mo_coeff;
  double * d_mo_cas = dd->d_mo_cas; 
  
  py::buffer_info info_cderi = _cderi.request(); // 2D array blksize * nao_pair
  double * cderi = static_cast<double*>(info_cderi.ptr);

  double * d_cderi = nullptr;
  
  if(use_eri_cache) {
    d_cderi = dd_fetch_eri(dd, cderi, naux, nao_pair, addr_dfobj, count);
  } else {
    if(_size_cderi > dd->size_eri1) {
      dd->size_eri1 = _size_cderi;
      if(dd->d_eri1) pm->dev_free_async(dd->d_eri1);
      dd->d_eri1 = (double *) pm->dev_malloc_async(_size_cderi * sizeof(double));
    }
    d_cderi = dd->d_eri1;

    pm->dev_push_async(d_cderi, cderi, _size_cderi * sizeof(double));
  }

  double * d_cderi_unpacked = dd->d_buf1;

  int * d_my_unpack_map_ptr = dd_fetch_pumap(dd, nao, _PUMAP_2D_UNPACK);

  getjk_unpack_buf2(d_cderi_unpacked,d_cderi,d_my_unpack_map_ptr,naux, nao, nao_pair);
  
  //bPmu = np.einsum('Pmn,nu->Pmu',cderi,mo_cas)
  
  const double alpha = 1.0;
  const double beta = 0.0;
  int zero = 0;
  int nao2 = nao * nao;
  int ncas_nao = ncas * nao;
  int ncas2 = ncas * ncas;
  const int _size_bPmu = naux*ncas*nao;

  double * d_bPmu = dd->d_buf2;
  
  ml->set_handle();
  ml->gemm_batch((char *) "N", (char *) "N", &nao, &ncas, &nao,
		 &alpha, d_cderi_unpacked, &nao, &nao2, d_mo_cas, &nao, &zero, &beta, d_bPmu, &nao, &ncas_nao, &naux);
  const int _size_bPvu = naux*ncas*ncas;
  
  //bPvu = np.einsum('mv,Pmu->Pvu',mo_cas.conjugate(),bPmu)

  double * d_bPvu= dd->d_buf2 + naux*ncas*nao;

  ml->set_handle();
  ml->gemm_batch((char *) "T", (char *) "N", &ncas, &ncas, &nao,
		 &alpha, d_mo_cas, &nao, &zero, d_bPmu, &nao, &ncas_nao, &beta, d_bPvu, &ncas, &ncas2, &naux);
  
  //eri = np.einsum('Pmw,Pvu->mwvu', bPmu, bPvu)

  //transpose bPmu
  double * d_bumP = dd->d_buf1;

  transpose_120(d_bPmu, d_bumP, naux, ncas, nao, 1); // this call distributes work items differently 

  double * d_buvP = dd->d_buf1+naux*ncas*nao;

  //transpose bPvu

  transpose_210(d_bPvu, d_buvP, naux, ncas, ncas);

  const int _size_mwvu = nao*ncas*ncas*ncas;
  
  double * d_vuwm = dd->d_buf2;
  
  ml->gemm((char *) "T", (char *) "N", &ncas_nao, &ncas2, &naux,
	   &alpha, d_bumP, &naux, d_buvP, &naux, &beta, d_vuwm, &ncas_nao);
  
  double * d_vuwM = dd->d_buf1;
  
  ml->gemm_batch((char *) "T", (char *) "T", &ncas, &nao, &nao,
		 &alpha, d_vuwm, &nao, &ncas_nao, d_mo_coeff, &nao, &zero, &beta, d_vuwM, &ncas, &ncas_nao, &ncas2);
  
  double * d_eri = dd->d_buf2;
  
  int * my_d_tril_map_ptr = dd_fetch_pumap(dd, ncas, _PUMAP_2D_UNPACK);

  pack_d_vuwM(d_vuwM, d_eri, my_d_tril_map_ptr, nmo, ncas, ncas_pair);
  
  pm->dev_pull_async(d_eri, eri, _size_eri*sizeof(double));

  pm->dev_stream_wait(); // this is required because 1) eri immediately consumed on python side and 2) all devices would write to same array
  
  profile_stop();
  
  double t1 = omp_get_wtime();
  t_array[6] += t1 - t0;//TODO: add the array size
  count_array[4] += 1; // doing this in compute instead of pull because v2 is not complete
}


/* ---------------------------------------------------------------------- */

void Device::get_h2eff_df_v2(py::array_t<double> _cderi, 
                                int nao, int nmo, int ncas, int naux, int ncore, 
                                py::array_t<double> _eri, int count, size_t addr_dfobj) 
{
  double t0 = omp_get_wtime();

#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Inside Device :: Starting h2eff_df_contract1 function");
  printf("LIBGPU:: dfobj= %#012x count= %i combined= %#012x %p update_dfobj= %i\n",addr_dfobj,count,addr_dfobj+count,addr_dfobj+count,update_dfobj);
#endif 
  
  profile_start("h2eff df setup");
  
  py::buffer_info info_eri = _eri.request(); //2D array nao * ncas * ncas_pair
  
  const int device_id = count % num_devices;
  
  pm->dev_set_device(device_id);
  
  my_device_data * dd = &(device_data[device_id]);
  
  const int nao_pair = nao * (nao+1)/2;
  const int ncas_pair = ncas * (ncas+1)/2;
  const int _size_eri = nmo*ncas*ncas_pair;
  const int _size_cderi = naux*nao_pair;
  const int _size_mo_cas = nao*ncas;

  const int eri_size = naux * nao * nao;
  const int bump_buvp = naux * ncas * (ncas + nao); 
  const int size_vuwm = ncas * ncas * ncas * nao;
#if 1
  #if 1
  // assume nao>ncas
  int _size_cderi_unpacked = 0;
  if (eri_size>=bump_buvp)
     {
     if (eri_size>=size_vuwm){_size_cderi_unpacked = eri_size;}
     else {_size_cderi_unpacked = size_vuwm;}
     }
  else 
     {
     if (bump_buvp>=size_vuwm){ _size_cderi_unpacked = bump_buvp;}
     else {_size_cderi_unpacked = size_vuwm;}
     }
  // the above exercise is done so as to avoid new memory allocations during the calculations and allocate the largest needed arrays. bump and buvp is done together because they need to exist simultaneously.     naux*nao^2 vs naux*ncas*(nao+ncas) vs nmo*ncas^3 
  #else
  // doing this because naux*nao**2 > nao*ncas**3 and naux*nao**2 > naux*ncas*(ncas+nao)
  const int _size_cderi_unpacked = naux * nao * nao; 
  #endif
  if (_size_cderi_unpacked > dd->size_eri_unpacked){
  //printf("Size ERI in h2eff v2: %i", dd->size_eri_unpacked);
  dd->size_eri_unpacked = _size_cderi_unpacked;
  if (dd->d_buf1) pm->dev_free_async(dd->d_buf1);
  if (dd->d_buf2) pm->dev_free_async(dd->d_buf2);
  dd->d_buf1 = (double *) pm->dev_malloc_async ( dd->size_eri_unpacked * sizeof(double));
  dd->d_buf2 = (double *) pm->dev_malloc_async ( dd->size_eri_unpacked * sizeof(double));
  }
#endif
  double * eri = static_cast<double*>(info_eri.ptr);
  double * d_mo_coeff = dd->d_mo_coeff;
  double * d_mo_cas = dd->d_mo_cas; 
  
  py::buffer_info info_cderi = _cderi.request(); // 2D array blksize * nao_pair
  double * cderi = static_cast<double*>(info_cderi.ptr);

  double * d_cderi = nullptr;
  
  if(use_eri_cache) {
    d_cderi = dd_fetch_eri(dd, cderi, naux, nao_pair, addr_dfobj, count);
  } else {
    if(_size_cderi > dd->size_eri1) {
      dd->size_eri1 = _size_cderi;
      if(dd->d_eri1) pm->dev_free_async(dd->d_eri1);
      dd->d_eri1 = (double *) pm->dev_malloc_async(_size_cderi * sizeof(double));
    }
    d_cderi = dd->d_eri1;

    pm->dev_push_async(d_cderi, cderi, _size_cderi * sizeof(double));
  }

  double * d_cderi_unpacked = dd->d_buf1;

  int * d_my_unpack_map_ptr = dd_fetch_pumap(dd, nao, _PUMAP_2D_UNPACK);

  getjk_unpack_buf2(d_cderi_unpacked,d_cderi,d_my_unpack_map_ptr,naux, nao, nao_pair);
  
  //bPmu = np.einsum('Pmn,nu->Pmu',cderi,mo_cas)
  
  const double alpha = 1.0;
  const double beta = 0.0;
  int zero = 0;
  int nao2 = nao * nao;
  int ncas_nao = ncas * nao;
  int ncas2 = ncas * ncas;
  const int _size_bPmu = naux*ncas*nao;

  double * d_bPmu = dd->d_buf2;
  
  ml->set_handle();
  ml->gemm_batch((char *) "N", (char *) "N", &nao, &ncas, &nao,
		 &alpha, d_cderi_unpacked, &nao, &nao2, d_mo_cas, &nao, &zero, &beta, d_bPmu, &nao, &ncas_nao, &naux);
  const int _size_bPvu = naux*ncas*ncas;
  //bPvu = np.einsum('mv,Pmu->Pvu',mo_cas.conjugate(),bPmu)

  double * d_bPvu= dd->d_buf2 + naux*ncas*nao;

  ml->set_handle();
  ml->gemm_batch((char *) "T", (char *) "N", &ncas, &ncas, &nao,
		 &alpha, d_mo_cas, &nao, &zero, d_bPmu, &nao, &ncas_nao, &beta, d_bPvu, &ncas, &ncas2, &naux);
  
  //eri = np.einsum('Pmw,Pvu->mwvu', bPmu, bPvu)
  //transpose bPmu
  double * d_bumP = dd->d_buf1;
  transpose_120(d_bPmu, d_bumP, naux, ncas, nao, 1); // this call distributes work items differently 
  double * d_buvP = dd->d_buf1+naux*ncas*nao;
  //transpose bPvu
  transpose_210(d_bPvu, d_buvP, naux, ncas, ncas);

  const int _size_mwvu = nao*ncas*ncas*ncas;
  double * d_vuwm = dd->d_buf2; 
  ml->gemm((char *) "T", (char *) "N", &ncas_nao, &ncas2, &naux,
	   &alpha, d_bumP, &naux, d_buvP, &naux, &beta, d_vuwm, &ncas_nao);
  double * d_vuwM = dd->d_buf1;
  ml->gemm_batch((char *) "T", (char *) "T", &ncas, &nao, &nao,
		 &alpha, d_vuwm, &nao, &ncas_nao, d_mo_coeff, &nao, &zero, &beta, d_vuwM, &ncas, &ncas_nao, &ncas2);
  double * d_eri = dd->d_buf2;
  int * my_d_tril_map_ptr = dd_fetch_pumap(dd, ncas, _PUMAP_2D_UNPACK);
  
  int init = (count < num_devices) ? 1 : 0;
  if (count < num_devices){
  pack_d_vuwM(d_vuwM, d_eri, my_d_tril_map_ptr, nmo, ncas, ncas_pair);
  } else{
  printf("num_devices: %i",num_devices);
  pack_d_vuwM_add(d_vuwM, d_eri, my_d_tril_map_ptr, nmo, ncas, ncas_pair);
  }

  profile_stop();
  
  double t1 = omp_get_wtime();
  t_array[6] += t1 - t0;//TODO: add the array size
  count_array[4] += 1; // see v1 comment
}


/* ---------------------------------------------------------------------- */
void Device::pull_eri_h2eff(py::array_t<double> _eri, int nmo, int ncas)
{
  py::buffer_info info_eri = _eri.request(); //2D array (nmo * (ncas*ncas_pair))
  double * eri = static_cast<double*>(info_eri.ptr);
  double * tmp;

  const int size_eri_h2eff = nmo*ncas*ncas*(ncas+1)/2;
  //printf("pulling eri\n");
  // Pulling eri from all devices
  for (int i=0; i<num_devices; ++i){
    pm->dev_set_device(i);

    my_device_data * dd = &(device_data[i]);

    if (i==0) tmp = eri;
    else tmp = &(buf_eri_h2eff[i*size_eri_h2eff]);
    
    if (dd->d_eri_h2eff) pm->dev_pull_async(dd->d_eri_h2eff, tmp, size_eri_h2eff*sizeof(double));
  }
  // Adding eri from all devices
  for(int i=0; i<num_devices; ++i) {
    pm->dev_set_device(i);

    my_device_data * dd = &(device_data[i]);

    pm->dev_stream_wait();

    if(i > 0 && dd->d_eri_h2eff) {

      tmp = &(buf_eri_h2eff[i * size_eri_h2eff]);
//#pragma omp parallel for
      for(int j=0; j< size_eri_h2eff; ++j) eri[j] += tmp[j];
    }
#if 0
for (int i = 0; i<nmo; ++i){
for (int j = 0; j<ncas*ncas*(ncas+1)/2; ++j){
 printf("%i \t", eri[i*ncas*ncas*(ncas+1)/2 + j]);}printf("\n");
}printf("\n\n\n\n\n\n\n\n\n\n\n\n");
#else
#endif
  }
}
/* ---------------------------------------------------------------------- */


// Is both _ocm2 in/out as it get over-written and resized?

void Device::orbital_response(py::array_t<double> _f1_prime,
			      py::array_t<double> _ppaa, py::array_t<double> _papa, py::array_t<double> _eri_paaa,
			      py::array_t<double> _ocm2, py::array_t<double> _tcm2, py::array_t<double> _gorb,
			      int ncore, int nocc, int nmo)
{
  double t0 = omp_get_wtime();
    
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

  double t1 = omp_get_wtime();
  t_array[4]  += t1  - t0;
  count_array[2] += 1; 
  
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


