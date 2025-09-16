/* -*- c++ -*- */

#include <stdio.h>

#include "device.h"

#define _NUM_SIMPLE_TIMER 32
#define _NUM_SIMPLE_COUNTER 23
#include <unistd.h>
#include <string.h>
#include <sched.h>
#define _MIN(A,B) (A<B)?A:B
#define _MAX(A,B) (A>B)?A:B

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

  buf_fdrv = nullptr;

  size_buf_vj = 0;
  size_buf_vk = 0;
  
  buf_vj = nullptr;
  buf_vk = nullptr;
  
  //ao2mo
  size_buf_j_pc = 0;
  size_buf_k_pc = 0;
  size_buf_ppaa = 0;
  size_buf_papa = 0;
  size_fxpp = 0;//remove when ao2mo_v3 is running
  size_bufpa = 0;//remove when ao2mo_v4 is running

  buf_j_pc = nullptr;
  buf_k_pc = nullptr;
  buf_ppaa = nullptr;
  buf_papa = nullptr;
  pin_fxpp = nullptr;//remove when ao2mo_v3 is running
  pin_bufpa = nullptr;//remove when ao2mo_v4 is running
  // h2eff_df
  size_buf_eri_h2eff = 0;
  buf_eri_h2eff = nullptr;

  // eri_impham

  size_eri_impham = 0;
  pin_eri_impham = nullptr;
  
#if defined(_USE_GPU)
  use_eri_cache = true;
#endif
  
  num_threads = 1;
#pragma omp parallel
  num_threads = omp_get_num_threads();

  num_devices = pm->dev_num_devices();
  
  device_data = new my_device_data[num_devices];

  for(int i=0; i<num_devices; ++i) {
    pm->dev_set_device(i);
    
    device_data[i].device_id = i;
    device_data[i].active = 0;
    
    device_data[i].size_rho = 0;
    device_data[i].size_vj = 0;
    device_data[i].size_vk = 0;
    device_data[i].size_buf1 = 0;
    device_data[i].size_buf2 = 0;
    device_data[i].size_buf3 = 0;
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
    //pdft
    device_data[i].size_mo_grid=0;
    device_data[i].size_ao_grid=0;
    device_data[i].size_buf_pdft=0;
    device_data[i].size_cascm2=0;
    device_data[i].size_Pi=0;
    device_data[i].size_rho=0;
    //fci
    device_data[i].size_clinka=0;
    device_data[i].size_clinkb=0;
    device_data[i].size_cibra=0;
    device_data[i].size_ciket=0;
    device_data[i].size_tdm1=0;
    device_data[i].size_tdm2=0;
    device_data[i].size_tdm2_p=0;
    
    
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
    device_data[i].d_eri_h2eff = nullptr; //for h2eff_df_v2
    
    device_data[i].d_pumap_ptr = nullptr;
    
    device_data[i].d_j_pc = nullptr;
    device_data[i].d_k_pc = nullptr;
    device_data[i].d_bufd = nullptr;
    device_data[i].d_bufpa = nullptr;
    device_data[i].d_bufaa = nullptr;
    device_data[i].d_papa = nullptr;//initialized, but not allocated (used dd->d_buf3)

    // pdft
    device_data[i].d_ao_grid=nullptr;
    device_data[i].d_mo_grid=nullptr;
    device_data[i].d_cascm2=nullptr;
    device_data[i].d_Pi=nullptr;
    device_data[i].d_buf_pdft1=nullptr;
    device_data[i].d_buf_pdft2=nullptr;
    //fci
    device_data[i].d_clinka=nullptr;
    device_data[i].d_clinkb=nullptr;
    device_data[i].d_cibra=nullptr;
    device_data[i].d_ciket=nullptr;
    device_data[i].d_tdm1=nullptr;
    device_data[i].d_tdm2=nullptr;
    device_data[i].d_tdm2_p=nullptr;
    device_data[i].d_tdm1h=nullptr;
    device_data[i].d_tdm3ha=nullptr;
    device_data[i].d_tdm3hb=nullptr;
    device_data[i].d_pdm1=nullptr;
    device_data[i].d_pdm2=nullptr;

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

  // check device connectivity

  int rank = 0;
  int peer_error = pm->dev_check_peer(rank, num_devices);
  if(!peer_error) pm->dev_enable_peer(rank, num_devices);
}

/* ---------------------------------------------------------------------- */

Device::~Device()
{
  if(verbose_level) printf("LIBGPU: destroying device\n");

  pm->dev_free_host(rho);
  //pm->dev_free_host(vj);
  pm->dev_free_host(_vktmp);

  pm->dev_free_host(buf_vj);
  pm->dev_free_host(buf_vk);
  
  pm->dev_free_host(buf_fdrv);
  
  pm->dev_free_host(buf_j_pc);
  pm->dev_free_host(buf_k_pc);
  pm->dev_free_host(buf_ppaa);
  pm->dev_free_host(buf_papa);
  pm->dev_free_host(pin_fxpp);//remove 
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

    printf("\nLIBGPU :: SIMPLE_TIMER :: eri_impham\n");
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= init_eri_impham()     time= %f s\n",11,t_array[11]);
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= compute_eri_impham()  time= %f s\n",12,t_array[12]);
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= pull_eri_impham()     time= %f s\n",13,t_array[13]);

    printf("\nLIBGPU :: SIMPLE_TIMER :: fci_related\n");
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= init_tdm1()               time= %f s\n",14,t_array[14]);
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= init_tdm2()               time= %f s\n",15,t_array[15]);
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= push_ci()                 time= %f s\n",16,t_array[16]);
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= push_link_index()         time= %f s\n",17,t_array[17]);
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= trans_rdm1a()             time= %f s\n",18,t_array[18]);
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= trans_rdm1b()             time= %f s\n",19,t_array[19]);
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= make_rdm1a()              time= %f s\n",20,t_array[20]);
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= make_rdm1b()              time= %f s\n",21,t_array[21]);
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= tdm12kern_a()             time= %f s\n",22,t_array[22]);
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= tdm12kern_b()             time= %f s\n",23,t_array[23]);
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= tdm12kern_ab()            time= %f s\n",24,t_array[24]);
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= rdm12kern_sf()            time= %f s\n",25,t_array[25]);
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= tdm13h_spin()             time= %f s\n",26,t_array[26]);
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= tdm1h_spin()              time= %f s\n",27,t_array[27]);
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= sfudm_spin()              time= %f s\n",28,t_array[28]);
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= pptdm_spin()              time= %f s\n",29,t_array[29]);
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= pull_tdm1()               time= %f s\n",30,t_array[30]);
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= pull_tdm2()               time= %f s\n",31,t_array[31]);
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= pull_tdm13h()             time= %f s\n",32,t_array[32]);
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
    
    printf("\nLIBGPU :: SIMPLE_COUNTER :: eri_impham\n");
    printf("LIBGPU :: SIMPLE_COUNTER :: i= %i  name=eri_impham()       counts= %i \n",7,count_array[7]);

    
    printf("\nLIBGPU :: SIMPLE_COUNTER :: fci_kernels\n");
    printf("LIBGPU :: SIMPLE_COUNTER :: i= %i  name= trans_rdm1a()        counts= %i \n",8,count_array[8]);
    printf("LIBGPU :: SIMPLE_COUNTER :: i= %i  name= trans_rdm1b()        counts= %i \n",9,count_array[9]);
    printf("LIBGPU :: SIMPLE_COUNTER :: i= %i  name= make_rdm1a()         counts= %i \n",10,count_array[10]);
    printf("LIBGPU :: SIMPLE_COUNTER :: i= %i  name= make_rdm1b()         counts= %i \n",11,count_array[11]);
    printf("LIBGPU :: SIMPLE_COUNTER :: i= %i  name= tdm12kern_a()        counts= %i \n",12,count_array[12]);
    printf("LIBGPU :: SIMPLE_COUNTER :: i= %i  name= tdm12kern_b()        counts= %i \n",13,count_array[13]);
    printf("LIBGPU :: SIMPLE_COUNTER :: i= %i  name= tdm12kern_ab()       counts= %i \n",14,count_array[14]);
    printf("LIBGPU :: SIMPLE_COUNTER :: i= %i  name= rdm12kern_sf()       counts= %i \n",15,count_array[15]);
    printf("LIBGPU :: SIMPLE_COUNTER :: i= %i  name= tdm13h_spin()        counts= %i \n",16,count_array[16]);
    printf("LIBGPU :: SIMPLE_COUNTER :: i= %i  name= tdm1h_spin()         counts= %i \n",17,count_array[17]);
    printf("LIBGPU :: SIMPLE_COUNTER :: i= %i  name= sfudm_spin()         counts= %i \n",18,count_array[18]);
    printf("LIBGPU :: SIMPLE_COUNTER :: i= %i  name= ppdm_spin()          counts= %i \n",19,count_array[19]);
    printf("LIBGPU :: SIMPLE_COUNTER :: i= %i  name= pull_tdm1()          counts= %i \n",20,count_array[20]);
    printf("LIBGPU :: SIMPLE_COUNTER :: i= %i  name= pull_tdm2()          counts= %i \n",21,count_array[21]);
    printf("LIBGPU :: SIMPLE_COUNTER :: i= %i  name= pull_tdm13h()        counts= %i \n",22,count_array[22]);
    free(count_array);
  }

  // print summary of cached eri blocks

  if(use_eri_cache) {
    if(verbose_level) {
      printf("\nLIBGPU :: eri cache statistics :: count= %zu\n",eri_list.size());
      for(int i=0; i<eri_list.size(); ++i)
	printf("LIBGPU :: %i : eri= %p  Mbytes= %f  count= %i  update= %i device= %i\n", i, (void*) eri_list[i],
	       eri_size[i]*sizeof(double)/1024./1024., eri_count[i], eri_update[i], eri_device[i]);
    }
    
    eri_count.clear();
    eri_size.clear();
#ifdef _DEBUG_ERI_CACHE
    for(int i=0; i<d_eri_host.size(); ++i) pm->dev_free_host( d_eri_host[i] );
#endif
    for(int i=0; i<d_eri_cache.size(); ++i) {
      int id = eri_device[i];
      pm->dev_set_device(id);
      pm->dev_free(d_eri_cache[i], "eri_cache");
    }
    eri_list.clear();
  }
  
#if defined(_USE_GPU)
  for(int i=0; i<num_devices; ++i) {
  
    pm->dev_set_device(i);
    
    my_device_data * dd = &(device_data[i]);
    
    pm->dev_free(dd->d_rho, "rho");
    pm->dev_free(dd->d_vj, "vj");
    pm->dev_free(dd->d_buf1, "buf1");
    pm->dev_free(dd->d_buf2, "buf2");
    pm->dev_free(dd->d_buf3, "buf3");
    pm->dev_free(dd->d_vkk, "vkk");
    pm->dev_free(dd->d_dms, "dms");
    pm->dev_free(dd->d_mo_coeff, "mo_coeff");
    pm->dev_free(dd->d_mo_cas, "mo_cas");
    pm->dev_free(dd->d_dmtril, "dmtril");
    pm->dev_free(dd->d_eri1, "eri1");
    pm->dev_free(dd->d_ucas, "ucas");
    pm->dev_free(dd->d_umat, "umat");
    pm->dev_free(dd->d_h2eff, "h2eff");
    pm->dev_free(dd->d_eri_h2eff, "eri_h2eff");
    
    pm->dev_free(dd->d_j_pc, "j_pc");
    pm->dev_free(dd->d_k_pc, "k_pc");

    pm->dev_free(dd->d_ao_grid, "ao_grid");
    pm->dev_free(dd->d_mo_grid, "ao_grid");
    pm->dev_free(dd->d_cascm2, "cascm2");
    pm->dev_free(dd->d_Pi, "Pi");
    pm->dev_free(dd->d_buf_pdft1, "buf_pdft1");
    pm->dev_free(dd->d_buf_pdft2, "buf_pdft2");

    pm->dev_free(dd->d_bufpa, "bufpa");
    pm->dev_free(dd->d_bufd, "bufd");
    pm->dev_free(dd->d_bufaa, "bufaa");
    pm->dev_free(dd->d_clinka, "clinka");
    pm->dev_free(dd->d_clinkb, "clinkb");
    pm->dev_free(dd->d_cibra, "cibra");
    pm->dev_free(dd->d_ciket, "ciket");
    pm->dev_free(dd->d_tdm1, "tdm1");
    pm->dev_free(dd->d_tdm2, "tdm2");
    pm->dev_free(dd->d_tdm2_p, "tdm2_p");
    //pm->dev_free(dd->d_tdm1h, "tdm1h");
    //pm->dev_free(dd->d_tdm3ha, "tdm3ha");
    //pm->dev_free(dd->d_tdm3hb, "tdm3hb");
    //pm->dev_free(dd->d_pdm1, "pdm1");
    //pm->dev_free(dd->d_pdm2, "pdm2");

    for(int i=0; i<dd->size_pumap.size(); ++i) {
      pm->dev_free_host(dd->pumap[i]);
      
      std::string name = "pumap-" + std::to_string(i);
      pm->dev_free(dd->d_pumap[i], name);
    }
    dd->type_pumap.clear();
    dd->size_pumap.clear();
    dd->pumap.clear();
    dd->d_pumap.clear();
  }

  if(verbose_level) {
    pm->print_mem_summary();
    
    printf("LIBGPU :: Finished\n");
  }
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
  printf("LIBGPU :: Inside Device::init_get_jk() :: blksize= %i  nset= %i  nao= %i  naux= %i  count= %i\n",blksize,nset,nao,naux,count);
#endif

  pm->dev_profile_start("init_get_jk");
  
  double t0 = omp_get_wtime();

  const int device_id = count % num_devices;
  
  pm->dev_set_device(device_id);

  my_device_data * dd = &(device_data[device_id]);
  
  //  if(dd->stream == nullptr) dd->stream = pm->dev_get_queue();
  
  int nao_pair = nao * (nao+1) / 2;
  
  int _size_vj = nset * nao_pair;

  grow_array(dd->d_vj, _size_vj, dd->size_vj, "vj", FLERR);
  
  int _size_vk = nset * nao * nao;

  grow_array(dd->d_vkk, _size_vk, dd->size_vk, "vkk", FLERR);

  int _size_buf = blksize * nao * nao;
  if(_size_vj > _size_buf) _size_buf = _size_vj;
  if(_size_vk > _size_buf) _size_buf = _size_vk;
  
  grow_array(dd->d_buf1, _size_buf, dd->size_buf1, "buf1", FLERR);
  grow_array(dd->d_buf2, _size_buf, dd->size_buf2, "buf2", FLERR);
  grow_array(dd->d_buf3, _size_buf, dd->size_buf3, "buf3", FLERR);
  
  int _size_dms = nset * nao * nao;
  grow_array(dd->d_dms, _size_dms, dd->size_dms, "dms", FLERR);

  int _size_dmtril = nset * nao_pair;
  grow_array(dd->d_dmtril, _size_dmtril, dd->size_dmtril, "dmtril", FLERR);

  if(!use_eri_cache) {
    int _size_eri1 = naux * nao_pair;
    grow_array(dd->d_eri1, _size_eri1, dd->size_eri1, "eri1", FLERR);
  }
  
  int _size_buf_vj = num_devices * nset * nao_pair;
  grow_array_host(buf_vj, _size_buf_vj, size_buf_vj, "h:buf_vj");

  int _size_buf_vk = num_devices * nset * nao * nao;
  grow_array_host(buf_vk, _size_buf_vk, size_buf_vk, "h:buf_vk");

  // 1-time initialization
  
  dd_fetch_pumap(dd, nao, _PUMAP_2D_UNPACK);
  
  // Create blas handle

  // if(dd->handle == nullptr) {
  //   ml->create_handle();
  //   //    dd->handle = ml->get_handle();
  // }
 
  // do all devices participate in calculation?
  
  if(count == 0) 
    for(int i=0; i<num_devices; ++i) device_data[i].active = 0;
  
  pm->dev_profile_stop();
  
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

  pm->dev_profile_start("get_jk :: init");

  const int device_id = count % num_devices;
  
  pm->dev_set_device(device_id);

  my_device_data * dd = &(device_data[device_id]);

  dd->active = 1;

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

  // Bcast() from master device ; make sure devices arrays allocated
  
#if defined(_ENABLE_P2P)
  if(count == 0) {
    size_t size = nset * nao_pair * sizeof(double);

    std::vector<double *> dmtril_vec(num_devices); // array of device addresses 

    dmtril_vec[0] = dd->d_dmtril;
    
    for(int i=1; i<num_devices; ++i) {
      my_device_data * dest = &(device_data[i]);

      // ensure memory allocated ; duplicating what's in init_get_jk()
      
      if(size > dest->size_dmtril) {
	dest->size_dmtril = size;

	pm->dev_set_device(i);
	if(dest->d_dmtril) pm->dev_free(dest->d_dmtril, "dmtril");
	dest->d_dmtril = (double *) pm->dev_malloc(size * sizeof(double), "dmtril", FLERR); // why is this not async?
      }
      
      dmtril_vec[i] = dest->d_dmtril;
    }
    
    mgpu_bcast(dmtril_vec, dmtril, size);  // host -> gpu 0, then Bcast to all gpu
  }
#else
  if(count < num_devices) {
    int err = pm->dev_push_async(dd->d_dmtril, dmtril, nset * nao_pair * sizeof(double));
    if(err) {
      printf("LIBGPU:: dev_push_async(d_dmtril) failed on count= %i\n",count);
      exit(1);
    }
  }
#endif
    
  int _size_rho = nset * naux;
  grow_array(dd->d_rho, _size_rho, dd->size_rho, "rho", FLERR);
    
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

  pm->dev_profile_stop();
  
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Starting with_j calculation\n");
#endif

  if (with_j){
    
    pm->dev_profile_start("get_jk :: with_j");
    
    // rho = numpy.einsum('ix,px->ip', dmtril, eri1)
    
    getjk_rho(dd->d_rho, dd->d_dmtril, d_eri, nset, naux, nao_pair);
    
    // vj += numpy.einsum('ip,px->ix', rho, eri1)
   
    int init = (count < num_devices) ? 1 : 0;
  
    getjk_vj(dd->d_vj, dd->d_rho, d_eri, nset, nao_pair, naux, init);
    
    pm->dev_profile_stop();
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
    
  pm->dev_profile_start("get_jk :: with_k");

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
	printf("LIBGPU:: d_dms= %p  dms= %p  nao= %i  device= %i\n",(void*) d_dms, (void*) dms,nao,device_id);
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
    
  pm->dev_profile_stop();
    
  double t1 = omp_get_wtime();
  t_array[2] += t1 - t0;
  // counts in pull jk
    
#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- finished\n");
  printf("LIBGPU :: -- Leaving Device::get_jk()\n");
#endif
}
  
/* ---------------------------------------------------------------------- */

#if defined(_ENABLE_P2P)
void Device::pull_get_jk(py::array_t<double> _vj, py::array_t<double> _vk, int nao, int nset, int with_k)
{
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: -- Inside Device::pull_get_jk()\n");
#endif
 
  double t0 = omp_get_wtime();
  
  pm->dev_profile_start("pull_get_jk");
  
  py::buffer_info info_vj = _vj.request(); // 2D array (nset, nao_pair)
  
  double * vj = static_cast<double*>(info_vj.ptr);
  
  int nao_pair = nao * (nao+1) / 2;
  
  int N = nset * nao_pair;
  
  std::vector<double *> v_vec(num_devices);
  std::vector<double *> buf_vec(num_devices);
  std::vector<int> active(num_devices);
  
  for(int i=0; i<num_devices; ++i) {
    my_device_data * dd = &(device_data[i]);
    v_vec[i] = dd->d_vj;
    buf_vec[i] = dd->d_buf3;
    active[i] = dd->active;
  }
  
  if(v_vec[0]) {
    mgpu_reduce(v_vec, buf_vj, N, true, buf_vec, active);
    
#pragma omp parallel for
    for(int j=0; j<N; ++j) vj[j] += buf_vj[j];
  }
  
  update_dfobj = 0;
  
  if(!with_k) {
    pm->dev_profile_stop();
    
#ifdef _DEBUG_DEVICE
    printf("LIBGPU :: -- Leaving Device::pull_get_jk()\n");
#endif
    
    return;
  }
  
  py::buffer_info info_vk = _vk.request(); // 3D array (nset, nao, nao)
  
  double * vk = static_cast<double*>(info_vk.ptr);
  
  N = nset * nao * nao;
  
  for(int i=0; i<num_devices; ++i) {
    my_device_data * dd = &(device_data[i]);
    v_vec[i] = dd->d_vkk;
  }
  
  if(v_vec[0]) {
    mgpu_reduce(v_vec, buf_vk, N, true, buf_vec, active);
    
#pragma omp parallel for
    for(int j=0; j<N; ++j) vk[j] += buf_vk[j];
  }
  
  pm->dev_profile_stop();
  
  double t1 = omp_get_wtime();
  t_array[1] += t1 - t0;
  count_array[0]+=1; // just doing this addition in pull, not in init or compute
  
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: -- Leaving Device::pull_get_jk()\n");
#endif
}

#else

void Device::pull_get_jk(py::array_t<double> _vj, py::array_t<double> _vk, int nao, int nset, int with_k)
{
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: -- Inside Device::pull_get_jk()\n");
#endif

  double t0 = omp_get_wtime();
    
  pm->dev_profile_start("pull_get_jk");
  
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
    
    if(dd->active) pm->dev_pull_async(dd->d_vj, tmp, size);
  }
  
  for(int i=0; i<num_devices; ++i) {
    pm->dev_set_device(i);
    
    my_device_data * dd = &(device_data[i]);
    
    pm->dev_stream_wait();

    if(i > 0 && dd->active) {
      
      tmp = &(buf_vj[i * nset * nao_pair]);
#pragma omp parallel for
      for(int j=0; j<nset*nao_pair; ++j) vj[j] += tmp[j];
      
    }
  }
  
  update_dfobj = 0;
  
  if(!with_k) {
    pm->dev_profile_stop();
    
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

    if(dd->active) pm->dev_pull_async(dd->d_vkk, tmp, size);
  }

  for(int i=0; i<num_devices; ++i) {
    pm->dev_set_device(i);
    
    my_device_data * dd = &(device_data[i]);
    
    pm->dev_stream_wait();

    if(i > 0 && dd->active) {
      
      tmp = &(buf_vk[i * nset * nao * nao]);
#pragma omp parallel for
      for(int j=0; j<nset*nao*nao; ++j) vk[j] += tmp[j];
    
    }

  }

  pm->dev_profile_stop();
  
  double t1 = omp_get_wtime();
  t_array[1] += t1 - t0;
  count_array[0]+=1; // just doing this addition in pull, not in init or compute
    
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: -- Leaving Device::pull_get_jk()\n");
#endif
}
#endif

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

      std::string name = "pumap-" + std::to_string(indx);
      dd->d_pumap[indx] = (int *) pm->dev_malloc_async(size_pumap * sizeof(int), name, FLERR);
      
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
      
      std::string name = "pumap-" + std::to_string(indx);
      dd->d_pumap[indx] = (int *) pm->dev_malloc_async(size_pumap * sizeof(int), name, FLERR);

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
      
      std::string name = "pumap-" + std::to_string(indx);
      dd->d_pumap[indx] = (int *) pm->dev_malloc_async(size_pumap * sizeof(int), name, FLERR);

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
      
      std::string name = "pumap-" + std::to_string(indx);
      dd->d_pumap[indx] = (int *) pm->dev_malloc_async(size_pumap * sizeof(int), name, FLERR);

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
      
      std::string name = "pumap-" + std::to_string(indx);
      dd->d_pumap[indx] = (int *) pm->dev_malloc_async(size_pumap * sizeof(int), name, FLERR);

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

    if(naux != eri_extra[id*2] || nao_pair != eri_extra[id*2+1]) {
      printf("LIBGPU :: dd_fetch_eri() has inconsistent naux= {%i, %i} and nao_pair= {%i, %i} for block id= %i\n",naux, eri_extra[id*2], nao_pair, eri_extra[id*2+1], id);
      exit(1);
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

    d_eri = (double *) pm->dev_malloc_async(naux * nao_pair * sizeof(double), "eri_cache", FLERR);
    d_eri_cache.push_back(d_eri);
    
    int err = pm->dev_push_async(d_eri, eri1, naux * nao_pair * sizeof(double));
    if(err) {
      printf("LIBGPU:: dev_push_async(d_eri) initializing new eri block\n");
      exit(1);
    }
    
#ifdef _DEBUG_DEVICE
    printf("LIBGPU:: dd_fetch_eri :: addr= %p  count= %i  naux= %i  nao_pair= %i\n",(void*)(addr_dfobj+count), count, naux, nao_pair);
#endif    
  }

  return d_eri;
}

/* ---------------------------------------------------------------------- */

double * Device::dd_fetch_eri_debug(my_device_data * dd, double * eri1, int naux, int nao_pair, size_t addr_dfobj, int count)
{   
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Starting eri_cache lookup for ERI %p\n",(void*)(addr_dfobj+count));
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
	printf("LIBGPU :: Warning: ERI %p updated on device w/ diff_eri= %.10e, but update_dfobj= %i\n",(void*)(addr_dfobj+count),diff_eri,update_dfobj);
	//count = -1;
	//return;
	exit(1);
      }
    } else {
      
      // update_dfobj falsely updates device ; this is loss of performance
      if(update_dfobj) {
	printf("LIBGPU :: Warning: ERI %p not updated on device w/ diff_eri= %.10e, but update_dfobj= %i\n",(void*)(addr_dfobj+count)//,diff_eri,update_dfobj);
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
    
    int id_ = d_eri_cache.size();
#ifdef _DEBUG_DEVICE
    printf("LIBGPU :: -- allocating new eri block: %i\n",id);
#endif
    
    d_eri = (double *) pm->dev_malloc_async(naux * nao_pair * sizeof(double), "eri_cache", FLERR);
    d_eri_cache.push_back(d_eri);
    
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
    double * d_eri_host_ = d_eri_host[id_];
    for(int i=0; i<naux*nao_pair; ++i) d_eri_host_[i] = eri1[i];
#endif
    
#ifdef _DEBUG_DEVICE
    printf("LIBGPU:: dd_fetch_eri_debug :: addr= %p  count= %i  naux= %i  nao_pair= %i\n",(void*)(addr_dfobj+count), count, naux, nao_pair);
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

#if defined(_ENABLE_P2P)
  std::vector<double *> mo_vec(num_devices); // array of device addresses 
    
  for(int id=0; id<num_devices; ++id) {
    pm->dev_set_device(id);
    
    my_device_data * dd = &(device_data[id]);

    grow_array(dd->d_mo_coeff, _size_mo_coeff, dd->size_mo_coeff, "mo_coeff", FLERR);
    
    mo_vec[id] = dd->d_mo_coeff;
  }
    
  mgpu_bcast(mo_vec, mo_coeff, _size_mo_coeff*sizeof(double)); // host -> gpu 0, then Bcast to all gpu

#else
  for(int id=0; id<num_devices; ++id) {
    
    pm->dev_set_device(id);
  
    my_device_data * dd = &(device_data[id]);
    
    grow_array(dd->d_mo_coeff, _size_mo_coeff, dd->size_mo_coeff, "mo_coeff", FLERR);
    
    pm->dev_push_async(dd->d_mo_coeff, mo_coeff, _size_mo_coeff*sizeof(double));
  }
#endif
  
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

    grow_array(dd->d_j_pc, size_j_pc, dd->size_j_pc, "j_pc", FLERR);
    grow_array(dd->d_k_pc, size_k_pc, dd->size_k_pc, "k_pc", FLERR);

    dd->active = 0;
  }
  
  int _size_buf_j_pc = num_devices*nmo*ncore;
  
  grow_array_host(buf_j_pc, _size_buf_j_pc, size_buf_j_pc, "h:buf_j_pc");
  
  int _size_buf_k_pc = num_devices*nmo*ncore;

  grow_array_host(buf_k_pc, _size_buf_k_pc, size_buf_k_pc, "h:buf_k_pc");
  
  double t1 = omp_get_wtime();
  t_array[8] += t1 - t0;
  // counts in pull ppaa
}

/* ---------------------------------------------------------------------- */

void Device::init_ppaa_papa_ao2mo( int nmo, int ncas)
{
  double t0 = omp_get_wtime();

  // initializing only cpu side, gpu ppaa will be a buffer array (dd->d_buf3) 

  int _size_buf_ppaa = num_devices*nmo*nmo*ncas*ncas;
  grow_array_host(buf_ppaa, _size_buf_ppaa, size_buf_ppaa, "h:buf_ppaa");

  int _size_buf_papa = num_devices*nmo*ncas*nmo*ncas;
  grow_array_host(buf_papa, _size_buf_papa, size_buf_papa, "h:buf_papa");

  double t1 = omp_get_wtime();
  t_array[8] += t1 - t0;
  // counts in pull ppaa_papa
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

    dd->active = 0;

    grow_array(dd->d_eri_h2eff, size_eri_h2eff, dd->size_eri_h2eff, "eri_h2eff", FLERR);
  }
  
  int _size_buf_eri_h2eff = num_devices * size_eri_h2eff;

  grow_array_host(buf_eri_h2eff, _size_buf_eri_h2eff, size_buf_eri_h2eff, "h:buf_eri_h2eff");
  
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

    grow_array(dd->d_mo_cas, _size_mo_cas, dd->size_mo_cas, "mo_cas", FLERR);

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

#if defined(_ENABLE_P2P)
void Device::pull_jk_ao2mo_v4(py::array_t<double> _j_pc, py::array_t<double> _k_pc, int nmo, int ncore)
{
  double t0 = omp_get_wtime();

  py::buffer_info info_j_pc = _j_pc.request(); //2D array (nmo*ncore)
  double * j_pc = static_cast<double*>(info_j_pc.ptr);
  
  py::buffer_info info_k_pc = _k_pc.request(); //2D array (nmo*ncore)
  double * k_pc = static_cast<double*>(info_k_pc.ptr);
  
  int N = nmo * ncore;

  std::vector<double *> pc_vec(num_devices);
  std::vector<double *> buf_vec(num_devices);
  std::vector<int> active(num_devices);
  
  for(int i=0; i<num_devices; ++i) {
    my_device_data * dd = &(device_data[i]);
    pc_vec[i] = dd->d_j_pc;
    buf_vec[i] = dd->d_buf1;
    active[i] = dd->active;
  }

  mgpu_reduce(pc_vec, buf_j_pc, N, true, buf_vec, active);

#pragma omp parallel for
  for(int i=0; i<nmo*ncore; ++i) j_pc[i] = buf_j_pc[i];

  // Pulling k_pc from all devices

  for(int i=0; i<num_devices; ++i) {
    my_device_data * dd = &(device_data[i]);
    pc_vec[i] = dd->d_k_pc;
    buf_vec[i] = dd->d_buf1;
  }
  
  mgpu_reduce(pc_vec, buf_k_pc, N, true, buf_vec, active);

#pragma omp parallel for
  for(int i=0; i<nmo*ncore; ++i) k_pc[i] = buf_k_pc[i];
  
  double t1 = omp_get_wtime();
  t_array[10] += t1 - t0;
  // counts in pull ppaa
}

#else

void Device::pull_jk_ao2mo_v4(py::array_t<double> _j_pc, py::array_t<double> _k_pc, int nmo, int ncore)
{
  double t0 = omp_get_wtime();

  py::buffer_info info_j_pc = _j_pc.request(); //2D array (nmo*ncore)
  double * j_pc = static_cast<double*>(info_j_pc.ptr);
  double * tmp;
  
  py::buffer_info info_k_pc = _k_pc.request(); //2D array (nmo*ncore)
  double * k_pc = static_cast<double*>(info_k_pc.ptr);
  
  int size = nmo*ncore;//*sizeof(double);

  printf("nmo= %i  ncore= %i\n",nmo, ncore);
  
  // Pulling j_pc from all devices
  
  for (int i=0; i<num_devices; ++i){
    pm->dev_set_device(i);
    my_device_data * dd = &(device_data[i]);
    
    tmp = &(buf_j_pc[i*nmo*ncore]);
    
    if(dd->active) pm->dev_pull_async(dd->d_j_pc, tmp, size*sizeof(double));
  }
  
  // Adding j_pc from all devices

  for(int i=0; i<num_devices; ++i) {
    pm->dev_set_device(i);

    my_device_data * dd = &(device_data[i]);
    
    pm->dev_stream_wait();

    if(i > 0 && dd->active) {
      
      tmp = &(buf_j_pc[i * nmo* ncore]);
//#pragma omp parallel for
      for(int j=0; j<ncore*nmo; ++j) buf_j_pc[j] += tmp[j];
    }
  }
#ifdef _DEBUG_DEVICE
  for (int i=0; i<num_devices;++i){
      for (int j=0; j<nmo;++j){
          for (int k=0; k<ncore;++k){
              printf("%f\t",buf_j_pc[i*nmo*ncore +j*ncore+k]);
          } printf("\n");
      } printf("\n");
  } 
#endif
  //copy buf_j_pc[first nmo*ncore] to j_pc
  std::memcpy(j_pc,buf_j_pc,nmo*ncore*sizeof(double));

  // Pulling k_pc from all devices
  
  for (int i=0; i<num_devices; ++i){
    pm->dev_set_device(i);
    
    my_device_data * dd = &(device_data[i]);

    tmp = &(buf_k_pc[i*nmo*ncore]);
    
    if(dd->active) pm->dev_pull_async(dd->d_k_pc, tmp, size*sizeof(double));
  }
  
  // Adding k_pc from all devices
  
  for(int i=0; i<num_devices; ++i) {
    pm->dev_set_device(i);
    
    my_device_data * dd = &(device_data[i]);
    
    pm->dev_stream_wait();

    if(i > 0 && dd->active) {
      
      tmp = &(buf_k_pc[i * nmo* ncore]);
//#pragma omp parallel for
      for(int j=0; j<ncore*nmo; ++j) buf_k_pc[j] += tmp[j];
    }
  }
    
  //copy buf_k_pc[first nmo*ncore] to k_pc
  std::memcpy(k_pc,buf_k_pc,nmo*ncore*sizeof(double));
  double t1 = omp_get_wtime();
  t_array[10] += t1 - t0;
  // counts in pull ppaa
}
#endif

/* ---------------------------------------------------------------------- */

#if defined(_ENABLE_P2P)
void Device::pull_ppaa_papa_ao2mo_v4(py::array_t<double> _ppaa, py::array_t<double> _papa, int nmo, int ncas)
{
  double t0 = omp_get_wtime();

  py::buffer_info info_ppaa = _ppaa.request(); //2D array (nmo*ncore)
  py::buffer_info info_papa = _papa.request(); //2D array (nmo*ncore)
  double * ppaa = static_cast<double*>(info_ppaa.ptr);
  double * papa = static_cast<double*>(info_papa.ptr);

  int N = nmo*nmo*ncas*ncas;
  
  // Pulling ppaa from all devices

  //  printf("nmo= %i  ncas= %i  N= %i\n",nmo, ncas, N);

  std::vector<double *> p_vec(num_devices);
  std::vector<double *> buf_vec(num_devices);
  std::vector<int> active(num_devices);
  
  for(int i=0; i<num_devices; ++i) {
    my_device_data * dd = &(device_data[i]);
    p_vec[i] = dd->d_ppaa; // pointing at d_buf3
    buf_vec[i] = dd->d_buf2;
    active[i] = dd->active;
  }

  mgpu_reduce(p_vec, buf_ppaa, N, true, buf_vec, active);

#pragma omp parallel for
  for(int i=0; i<N; ++i) ppaa[i] = buf_ppaa[i];

  // Pulling papa from all devices
  
  for(int i=0; i<num_devices; ++i) {
    my_device_data * dd = &(device_data[i]);
    p_vec[i] = dd->d_papa; // pointing at d_buf3
  }

  mgpu_reduce(p_vec, buf_papa, N, true, buf_vec, active);

#pragma omp parallel for
  for(int i=0; i<N; ++i) papa[i] = buf_papa[i];

  double t1 = omp_get_wtime();
  t_array[10] += t1 - t0;
  count_array[6] += 1; //doing this in ppaa pull, not in any inits or computes
}

#else
void Device::pull_ppaa_papa_ao2mo_v4(py::array_t<double> _ppaa, py::array_t<double> _papa, int nmo, int ncas)
{
  double t0 = omp_get_wtime();

  py::buffer_info info_ppaa = _ppaa.request(); //2D array (nmo*ncore)
  py::buffer_info info_papa = _papa.request(); //2D array (nmo*ncore)
  double * ppaa = static_cast<double*>(info_ppaa.ptr);
  double * papa = static_cast<double*>(info_papa.ptr);
  double * tmp;
  const int _size_ppaa = nmo*nmo*ncas*ncas;
  const int _size_papa = nmo*nmo*ncas*ncas;
  // Pulling ppaa from all devices
  
  for (int i=0; i<num_devices; ++i){
    pm->dev_set_device(i);

    my_device_data * dd = &(device_data[i]);

    tmp = &(buf_ppaa[i*_size_ppaa]);
    
    if (dd->active) pm->dev_pull_async(dd->d_ppaa, tmp, _size_ppaa*sizeof(double));
  }
  
  // Adding ppaa from all devices
  
  for(int i=0; i<num_devices; ++i) {
    pm->dev_set_device(i);

    my_device_data * dd = &(device_data[i]);
    
    pm->dev_stream_wait();

    if(i > 0 && dd->active) {
      
      tmp = &(buf_ppaa[i * _size_ppaa]);
//#pragma omp parallel for
      for(int j=0; j<_size_ppaa; ++j) buf_ppaa[j] += tmp[j];
    }
  }
  //copy buf_ppaa[first nmo*nmo*ncas*ncas] to ppaa
  std::memcpy(ppaa,buf_ppaa,_size_ppaa*sizeof(double));

  // Pulling papa from all devices
  for (int i=0; i<num_devices; ++i){
    pm->dev_set_device(i);

    my_device_data * dd = &(device_data[i]);

    tmp = &(buf_papa[i*_size_papa]);
    
    if (dd->d_papa) pm->dev_pull_async(dd->d_papa, tmp, _size_papa*sizeof(double));
  }
  
  // Adding papa from all devices
  
  for(int i=0; i<num_devices; ++i) {
    pm->dev_set_device(i);

    my_device_data * dd = &(device_data[i]);
    
    pm->dev_stream_wait();

    if(i > 0 && dd->active) {
      
      tmp = &(buf_papa[i * _size_papa]);
//#pragma omp parallel for
      for(int j=0; j<_size_papa; ++j) buf_papa[j] += tmp[j];
    }
  }
  //copy buf_papa[first nmo*nmo*ncas*ncas] to papa
  std::memcpy(papa,buf_papa,_size_papa*sizeof(double));
  double t1 = omp_get_wtime();
  t_array[10] += t1 - t0;
  count_array[6] += 1; //doing this in ppaa pull, not in any inits or computes
}
#endif

/* ---------------------------------------------------------------------- */

void Device::df_ao2mo_v4 (int blksize, int nmo, int nao, int ncore, int ncas, int naux, 
				  int count, size_t addr_dfobj)
{
  printf("using ao2mo v4\n");
  double t0 = omp_get_wtime();
  
  pm->dev_profile_start("AO2MO v4");

  const int device_id = count % num_devices;

  pm->dev_set_device(device_id);

  my_device_data * dd = &(device_data[device_id]);

  dd->active = 1;

  //  py::buffer_info info_eri1 = _eri1.request(); // 2D array (naux, nao_pair) nao_pair= nao*(nao+1)/2
  const int nao_pair = nao*(nao+1)/2;
  //  double * eri = static_cast<double*>(info_eri1.ptr);
  
  int _size_eri = naux * nao_pair;
  int _size_eri_unpacked = naux * nao * nao; 
  int _size_ppaa = nmo * nmo * ncas * ncas;

#ifdef _DEBUG_DEVICE
#if defined (_GPU_CUDA)
  size_t freeMem;size_t totalMem;
  freeMem=0;totalMem=0;
  cudaMemGetInfo(&freeMem, &totalMem);
  printf("Starting ao2mo Free memory %lu bytes, total memory %lu bytes\n",freeMem,totalMem);
#endif
#endif

  int max_size_buf = 2 * _size_ppaa;
  if(_size_eri_unpacked > max_size_buf) max_size_buf = _size_eri_unpacked;
  
  grow_array(dd->d_buf1, max_size_buf, dd->size_buf1, "buf1", FLERR);
  grow_array(dd->d_buf2, max_size_buf, dd->size_buf2, "buf2", FLERR);
  grow_array(dd->d_buf3, max_size_buf, dd->size_buf3, "buf3", FLERR);
  
  // I want to fit both ppaa and papa inside buf3 to remove it from cpu side
  // my guess is blksize*nao_s*nao_s > 2 * nmo_f * nmo_f * ncas_f * ncas_f (dd->size_eri_unpacked is for the entire system. Usually nao_s > sqrt(2)*nao_f, blksize = 240, ncas_f must be less than 15)
  double * d_buf = dd->d_buf1; 
  double * d_eri_unpacked = dd->d_buf2; 
  
  double * d_eri = nullptr;
  
  if(use_eri_cache) {
    //    d_eri = dd_fetch_eri(dd, eri, naux, nao_pair, addr_dfobj, count);
    d_eri = dd_fetch_eri(dd, nullptr, naux, nao_pair, addr_dfobj, count);
  } else {
    grow_array(dd->d_eri1, _size_eri, dd->size_eri1, "eri1", FLERR);
    d_eri = dd->d_eri1;
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
  
  double * d_bufpp = dd->d_buf2;

  ml->gemm_batch((char *) "T", (char *) "N", &nao, &nao, &nao,
		 &alpha, dd->d_mo_coeff, &nao, &zero, d_buf, &nao, &nao2, &beta, d_bufpp, &nao, &nao2, &naux);

  int _size_bufpa = naux*nmo*ncas;
  grow_array(dd->d_bufpa, _size_bufpa, dd->size_bufpa, "bufpa", FLERR);
  
  double * d_bufpa = dd->d_bufpa;

  get_bufpa(d_bufpp, d_bufpa, naux, nmo, ncore, ncas);

  // making papa on device, so no longer need to pull
  //double * bufpa = &(pin_bufpa[count*blksize*nmo*ncas]);
  //pm->dev_pull_async(d_bufpa, bufpa, naux*nmo*ncas*sizeof(double));

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

  //bufd work

  int _size_bufd = naux*nmo;
  grow_array(dd->d_bufd, _size_bufd, dd->size_bufd, "bufd", FLERR);
  
  double * d_bufd = dd->d_bufd;

  get_bufd(d_bufpp, d_bufd, naux, nmo);
  
// calculate j_pc
  
  // self.j_pc += numpy.einsum('ki,kj->ij', bufd, bufd[:,:ncore])

  ml->gemm((char *) "N", (char *) "T", &ncore, &nmo, &naux,
	   &alpha, d_bufd, &nmo, d_bufd, &nmo, &beta_, dd->d_j_pc, &ncore);

  int _size_bufaa = naux*ncas*ncas;
  grow_array(dd->d_bufaa, _size_bufaa, dd->size_bufaa, "bufaa", FLERR);

  double * d_bufaa = dd->d_bufaa;

  get_bufaa(d_bufpp, d_bufaa, naux, nmo, ncore, ncas);

  const int ncas2 = ncas*ncas;
  const int nmo_ncas = nmo*ncas;

  // calculate ppaa
  dd->d_ppaa = dd->d_buf3;
  ml->gemm ((char *) "N", (char *) "N", &ncas2, &nao2, &naux,  
                   &alpha,  d_bufaa, &ncas2, d_fxpp, &naux, &beta_, dd->d_ppaa, &ncas2);                  
  
  // calculate papa
  //dd->d_papa = dd->d_buf3 + _size_ppaa*sizeof(double);
  dd->d_papa = dd->d_buf3 + _size_ppaa;
  ml->gemm ((char *) "N", (char *) "T", &nmo_ncas, &nmo_ncas, &naux, 
                  &alpha, d_bufpa, &nmo_ncas, d_bufpa, &nmo_ncas, &beta_, dd->d_papa, &nmo_ncas); 
#ifdef _DEBUG_DEVICE
#if defined (_GPU_CUDA)
  printf("LIBGPU :: Leaving Device::df_ao2mo_pass1_fdrv()\n"); 
  cudaMemGetInfo(&freeMem, &totalMem);
  printf("Ending ao2mo fdrv Free memory %lu bytes, total memory %lu bytes\n",freeMem,totalMem);
#endif
#endif
  
  pm->dev_profile_stop();
  
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

  pm->dev_profile_start("Setup initial h2eff_sub");
  
  py::buffer_info info_umat = _umat.request(); // 2d array nmo*nmo
  py::buffer_info info_h2eff_sub = _h2eff_sub.request();// 2d array (nmo * ncas) x (ncas*(ncas+1)/2)

  const int device_id = 0; //count % num_devices;

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

  grow_array(dd->d_buf1, _size_h2eff_unpacked, dd->size_buf1, "buf1", FLERR);
  //  grow_array(dd->d_buf2, _size_h2eff_unpacked, dd->size_buf2, "buf2", FLERR);
  //  grow_array(dd->d_buf3, _size_h2eff_unpacked, dd->size_buf3, "buf3", FLERR);
  
  double * d_h2eff_unpacked = dd->d_buf1;

  grow_array(dd->d_ucas, ncas*ncas, dd->size_ucas, "ucas", FLERR);

  grow_array(dd->d_umat, nmo*nmo, dd->size_umat, "umat", FLERR);
  
  pm->dev_push_async(dd->d_umat, umat, nmo*nmo*sizeof(double));

#ifdef _DEBUG_H2EFF
  printf("LIBGPU :: Inside Device :: -- Setup update function\n");
#endif
  
  pm->dev_profile_next("extraction");
  
  //ucas = umat[ncore:nocc, ncore:nocc]

  extract_submatrix(dd->d_umat, dd->d_ucas, ncas, ncore, nmo);
  
  //h2eff_sub = h2eff_sub.reshape (nmo*ncas, ncas*(ncas+1)//2)
  //h2eff_sub = lib.numpy_helper.unpack_tril (h2eff_sub)
  //h2eff_sub = h2eff_sub.reshape (nmo, ncas, ncas, ncas)

  grow_array(dd->d_h2eff, _size_h2eff_packed, dd->size_h2eff, "h2eff", FLERR);
  
  double * d_h2eff_sub = dd->d_h2eff;
  
  pm->dev_push_async(d_h2eff_sub, h2eff_sub, _size_h2eff_packed * sizeof(double));

  pm->dev_profile_next("map creation and pushed");
  
  int * d_my_unpack_map_ptr = dd_fetch_pumap(dd, ncas, _PUMAP_H2EFF_UNPACK);

  pm->dev_profile_next("unpacking");

#ifdef _DEBUG_H2EFF
  printf("LIBGPU :: Inside Device :: -- created and pushed unpacking map\n");
#endif

  unpack_h2eff_2d(d_h2eff_sub, d_h2eff_unpacked, d_my_unpack_map_ptr, nmo, ncas, ncas_pair);
  
  pm->dev_profile_next("2 dgemms");
  
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
  
  pm->dev_profile_next("transpose");
  
#ifdef _DEBUG_H2EFF
  printf("LIBGPU :: Inside Device :: -- Finished first 2 cublasDgemmStridedBatched Functions \n");
#endif
  
  //h2eff_tranposed=(piJK->JKpi)
  
  double * d_h2eff_transposed = dd->d_buf2;

  transpose_2310(d_h2eff_step2, d_h2eff_transposed, nmo, ncas);
  
  pm->dev_profile_next("last 2 dgemm");
  
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
  
  pm->dev_profile_next("2nd transpose");

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

  pm->dev_profile_next("second map and packing");
  
  int * d_my_pack_map_ptr = dd_fetch_pumap(dd, ncas, _PUMAP_H2EFF_PACK);

  pack_h2eff_2d(d_h2eff_transpose2, d_h2eff_sub, d_my_pack_map_ptr, nmo, ncas, ncas_pair);
  
#ifdef _DEBUG_H2EFF
  printf("LIBGPU :: Inside Device :: -- Freed map\n");
#endif
  
  pm->dev_pull_async(d_h2eff_sub, h2eff_sub, _size_h2eff_packed*sizeof(double));

  pm->dev_stream_wait(); // is this required or can we delay waiting?
  
  pm->dev_profile_stop();
  
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
void Device::get_h2eff_df_v2(py::array_t<double> _cderi, 
                                int nao, int nmo, int ncas, int naux, int ncore, 
                                py::array_t<double> _eri, int count, size_t addr_dfobj) 
{
  double t0 = omp_get_wtime();

#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Inside Device::get_h2eff_df_v2()\n");
  printf("LIBGPU:: dfobj= %p count= %i combined= %lu %p update_dfobj= %i\n",(void*)(addr_dfobj), count, addr_dfobj+count, (void*)(addr_dfobj+count),update_dfobj);
#endif 
  
  pm->dev_profile_start("h2eff df setup");
  
  py::buffer_info info_eri = _eri.request(); //2D array nao * ncas * ncas_pair
  
  const int device_id = count % num_devices;
  
  pm->dev_set_device(device_id);
  
  my_device_data * dd = &(device_data[device_id]);

  dd->active = 1;

  const int nao_pair = nao * (nao+1)/2;
  const int ncas_pair = ncas * (ncas+1)/2;
  const int _size_eri_h2eff = nmo*ncas*ncas_pair;
  const int _size_eri = naux*nao_pair;
  const int _size_mo_cas = nao*ncas;

  const int _size_eri_unpacked = naux * nao * nao;
  const int bump_buvp = naux * ncas * (ncas + nao);
  
  // buf2 will hold vuwm
  
  const int size_vuwm = ncas * ncas * ncas * nao;

  // buf1 will hold 1) cderi_unpacked 2) both bumP & buvP 3) vuwM

  const int size_cderi_unpacked = naux * nao * nao;
  
  const int _size_bPmu = naux*ncas*nao;
  const int _size_bPvu = naux*ncas*ncas;
  
  const int size_bumP_buvP = _size_bPmu + _size_bPvu;
  const int size_vuwM = nmo * ncas * ncas_pair;
  
  // int max_size_buf = (_size_eri_unpacked > _size_eri_h2eff) ? _size_eri_unpacked : _size_eri_h2eff;
  // if(size_vuwm > max_size_buf) max_size_buf = size_vuwm;
  // if(size_cderi_unpacked > max_size_buf) max_size_buf = size_cderi_unpacked;
  // if(size_bumP_buvP > max_size_buf) max_size_buf = size_bumP_buvP;
  // if(size_vuwM > max_size_buf) max_size_buf = size_vuwM;

  // if(device_id == 0)
  // printf("get_h2eff_df_v2 :: device_id= %i  naux= %i nmo= %i ncas= %i ncas_pair= %i nao= %i nao_pair= %i  _size_eri_unpacked= %i  _size_eri_h2eff= %i  size_vuwm= %i  size_cderi_unpacked= %i  size_bumP_buvP= %i\n",device_id,naux,nmo,ncas,ncas_pair,nao,nao_pair,_size_eri_unpacked, _size_eri_h2eff, size_vuwm, size_cderi_unpacked, size_bumP_buvP);

  int max_size_buf = size_cderi_unpacked;
  if(size_bumP_buvP > max_size_buf) max_size_buf = size_bumP_buvP;
  if(size_vuwM > max_size_buf) max_size_buf = size_vuwM; 
  if(size_vuwm > max_size_buf) max_size_buf = size_vuwm; 
  
  grow_array(dd->d_buf1, max_size_buf, dd->size_buf1, "buf1", FLERR); // holds cderi_unpacked and bumP+buvP and vuwM

  max_size_buf = size_bumP_buvP;
  if(size_vuwm > max_size_buf) max_size_buf = size_vuwm;
  
  grow_array(dd->d_buf2, max_size_buf, dd->size_buf2, "buf2", FLERR); // holds bPmu+bPvu and vuwm

  max_size_buf = _size_eri_h2eff;
  if(size_vuwM > max_size_buf) max_size_buf = size_vuwM;
  
  grow_array(dd->d_buf3, max_size_buf, dd->size_buf3, "buf3", FLERR); // holds eri_h2eff
  
  double * eri = static_cast<double*>(info_eri.ptr);
  double * d_mo_coeff = dd->d_mo_coeff;
  double * d_mo_cas = dd->d_mo_cas; 
  
  py::buffer_info info_cderi = _cderi.request(); // 2D array blksize * nao_pair
  double * cderi = static_cast<double*>(info_cderi.ptr);

  double * d_cderi = nullptr;
  
  if(use_eri_cache) {
    d_cderi = dd_fetch_eri(dd, cderi, naux, nao_pair, addr_dfobj, count);
  } else {
    grow_array(dd->d_eri1, _size_eri, dd->size_eri1, "eri1", FLERR);
    d_cderi = dd->d_eri1;

    pm->dev_push_async(d_cderi, cderi, _size_eri * sizeof(double));
  }

  double * d_cderi_unpacked = dd->d_buf1;

  int * d_my_unpack_map_ptr = dd_fetch_pumap(dd, nao, _PUMAP_2D_UNPACK);

  // CHRIS :: Start chunking w/r naux
  
  getjk_unpack_buf2(d_cderi_unpacked, d_cderi, d_my_unpack_map_ptr, naux, nao, nao_pair);
  
  //bPmu = np.einsum('Pmn,nu->Pmu',cderi,mo_cas)
  
  const double alpha = 1.0;
  const double beta = 0.0;
  int zero = 0;
  int nao2 = nao * nao;
  int ncas_nao = ncas * nao;
  int ncas2 = ncas * ncas;

  double * d_bPmu = dd->d_buf2;
  
  ml->set_handle();
  ml->gemm_batch((char *) "N", (char *) "N", &nao, &ncas, &nao,
		 &alpha, d_cderi_unpacked, &nao, &nao2, d_mo_cas, &nao, &zero, &beta, d_bPmu, &nao, &ncas_nao, &naux);
  
  //bPvu = np.einsum('mv,Pmu->Pvu',mo_cas.conjugate(),bPmu)

  double * d_bPvu= dd->d_buf2 + naux*ncas*nao;

  ml->set_handle();
  ml->gemm_batch((char *) "T", (char *) "N", &ncas, &ncas, &nao,
		 &alpha, d_mo_cas, &nao, &zero, d_bPmu, &nao, &ncas_nao, &beta, d_bPvu, &ncas, &ncas2, &naux);
  
  //eri = np.einsum('Pmw,Pvu->mwvu', bPmu, bPvu)

  //transpose bPmu

  double * d_bumP = dd->d_buf1;

  transpose_120(d_bPmu, d_bumP, naux, ncas, nao, 1); // this call distributes work items differently 

  double * d_buvP = dd->d_buf1 + naux*ncas*nao;

  // printf("size_buf1= %i  size_bumP= %i  size_buvP= %i  sum= %i\n",
  // 	 dd->size_buf, naux*ncas*nao, naux*ncas*ncas, naux*ncas*nao + naux*ncas*ncas);
  
  //transpose bPvu

  transpose_210(d_bPvu, d_buvP, naux, ncas, ncas);

  // printf("size_buf2= %i  _size_mwvu= %i\n",dd->size_buf, size_vuwm);
  
  double * d_vuwm = dd->d_buf2; 

  ml->set_handle();
  ml->gemm((char *) "T", (char *) "N", &ncas_nao, &ncas2, &naux,
	   &alpha, d_bumP, &naux, d_buvP, &naux, &beta, d_vuwm, &ncas_nao);

  // CHRIS :: Stop chunking w/r naux
  
  double * d_vuwM = dd->d_buf1;

  ml->set_handle();
  ml->gemm_batch((char *) "T", (char *) "T", &ncas, &nao, &nao,
		 &alpha, d_vuwm, &nao, &ncas_nao, d_mo_coeff, &nao, &zero, &beta, d_vuwM, &ncas, &ncas_nao, &ncas2);

  int * my_d_tril_map_ptr = dd_fetch_pumap(dd, ncas, _PUMAP_2D_UNPACK);
  
  if (count < num_devices) {
    pack_d_vuwM(d_vuwM, dd->d_eri_h2eff, my_d_tril_map_ptr, nmo, ncas, ncas_pair);
  } else {
    pack_d_vuwM(d_vuwM, dd->d_buf3, my_d_tril_map_ptr, nmo, ncas, ncas_pair);
    vecadd(dd->d_buf3, dd->d_eri_h2eff, _size_eri_h2eff);
  }

  pm->dev_profile_stop();
  
  double t1 = omp_get_wtime();
  t_array[6] += t1 - t0;//TODO: add the array size
  count_array[4] += 1; // see v1 comment
  
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Leaving Device::get_h2eff_df_v2()\n");
#endif  
}

/* ---------------------------------------------------------------------- */

#if defined(_ENABLE_P2P)
void Device::pull_eri_h2eff(py::array_t<double> _eri, int nmo, int ncas)
{
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: -- Inside Device::pull_eri_h2eff()\n");
#endif
  
  py::buffer_info info_eri = _eri.request(); //2D array (nmo * (ncas*ncas_pair))
  double * eri = static_cast<double*>(info_eri.ptr);

  const int ncas_pair = ncas*(ncas+1)/2;
  
  const int N = nmo*ncas * ncas_pair;

  std::vector<double *> e_vec(num_devices);
  std::vector<double *> buf_vec(num_devices);
  std::vector<int> active(num_devices);
  
  for(int i=0; i<num_devices; ++i) {
    my_device_data * dd = &(device_data[i]);
    e_vec[i] = dd->d_eri_h2eff;
    buf_vec[i] = dd->d_buf3;
    active[i] = dd->active;
  }

  mgpu_reduce(e_vec, buf_eri_h2eff, N, true, buf_vec, active);

#pragma omp parallel for
  for(int i=0; i<N; ++i) eri[i] = buf_eri_h2eff[i];
  
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Leaving Device::get_h2eff_df_v2()\n");
#endif
}
#else
void Device::pull_eri_h2eff(py::array_t<double> _eri, int nmo, int ncas)
{
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: -- Inside Device::pull_eri_h2eff()\n");
#endif
  
  py::buffer_info info_eri = _eri.request(); //2D array (nmo * (ncas*ncas_pair))
  double * eri = static_cast<double*>(info_eri.ptr);
  double * tmp;

  const int ncas_pair = ncas*(ncas+1)/2;
  const int size_eri_h2eff = nmo*ncas*ncas_pair;
  
  // Pulling eri from all devices
  
  for (int i=0; i<num_devices; ++i){
    pm->dev_set_device(i);

    my_device_data * dd = &(device_data[i]);

    tmp = &(buf_eri_h2eff[i*size_eri_h2eff]);
    
    if(dd->active) pm->dev_pull_async(dd->d_eri_h2eff, tmp, size_eri_h2eff*sizeof(double));
  }
  
  // Adding eri from all devices
  
  for(int i=0; i<num_devices; ++i) {
    pm->dev_set_device(i);

    my_device_data * dd = &(device_data[i]);

    pm->dev_stream_wait();

    if(i > 0 && dd->active) {

      tmp = &(buf_eri_h2eff[i * size_eri_h2eff]);
//#pragma omp parallel for
      for(int j=0; j< size_eri_h2eff; ++j) buf_eri_h2eff[j] += tmp[j];
    }
  }
  
  std::memcpy(eri, buf_eri_h2eff, size_eri_h2eff*sizeof(double));
  
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Leaving Device::get_h2eff_df_v2()\n");
#endif
}
#endif

/* ---------------------------------------------------------------------- */

void Device::init_eri_impham(int naoaux, int nao_f, int return_4c2eeri)
{
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Inside Device::init_eri_impham()  return_4c2eeri= %i\n",return_4c2eeri);
  if (return_4c2eeri) printf("LIBGPU :: -- returning 4c2e\n");
  else printf("LIBGPU :: -- returning 3c2e\n");
#endif
  
  double t0 = omp_get_wtime();
  
  pm->dev_profile_start("init_eri_impham");

  int nao_f_pair = nao_f*(nao_f+1)/2;
  int _size_eri_impham = 0;
  
  if (return_4c2eeri) _size_eri_impham = num_devices * nao_f_pair*nao_f_pair;  //when used like this, answer accumulates on gpu
  else _size_eri_impham = naoaux*nao_f_pair;  // answer accumulates on cpu
  
  if (_size_eri_impham > size_eri_impham) {
    size_eri_impham = _size_eri_impham;
    
#ifdef _DEBUG_DEVICE
    printf("resizing eri_impham in init\n");
    printf("size_eri %d\n",size_eri_impham );
#endif
    
    if (pin_eri_impham) pm->dev_free_host(pin_eri_impham);
    pin_eri_impham = (double *) pm->dev_malloc_host(size_eri_impham*sizeof(double));
  }
  
  double t1 = omp_get_wtime();
  t_array[11] += t1 - t0;
  
  pm->dev_profile_stop();
  
  // counts in pull eri_impham
  
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Leaving Device::init_eri_impah()\n");
#endif
}

/* ---------------------------------------------------------------------- */

void Device::compute_eri_impham(int nao_s, int nao_f, int blksize, int naux, int count, size_t addr_dfobj, int return_4c2eeri)
{
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Inside Device::comute_eri_impham()\n");
  printf("LIBGPU :: -- from gpu: %i %i %i %i %i\n",nao_s, nao_f, blksize, naux, count);
#endif
  
  pm->dev_profile_start("compute_eri_impham");
  
  double t0 = omp_get_wtime();

  const int device_id = count % num_devices;
  
  pm->dev_set_device(device_id);
  
  my_device_data * dd = &(device_data[device_id]);

  dd->active = 1;

  // using fetch_eri, assume it's already there

  int nao_s_pair = nao_s * (nao_s + 1)/2;

  double * d_cderi = dd_fetch_eri(dd, nullptr, naux, nao_s_pair, addr_dfobj, count);

  double * d_cderi_unpacked = dd->d_buf1;

  int * d_my_unpack_map_ptr = dd_fetch_pumap(dd, nao_s, _PUMAP_2D_UNPACK);

  getjk_unpack_buf2(d_cderi_unpacked, d_cderi, d_my_unpack_map_ptr, naux, nao_s, nao_s_pair);

  const double alpha = 1.0;
  const double beta = 0.0;
  int zero = 0;
  int nao_s2 = nao_s * nao_s;
  int nao_sf = nao_s * nao_f;
  int nao_f2 = nao_f * nao_f;
  int nao_f_pair = nao_f * (nao_f+1)/2;
  
  double * d_bPeu = dd->d_buf2;

  // b^P_ue = b^P_uu * M_ue
  
  ml->set_handle();
  ml->gemm_batch((char *) "N", (char *) "T", 
               &nao_s, &nao_f, &nao_s,
               &alpha, 
               d_cderi_unpacked, &nao_s, &nao_s2, 
               dd->d_mo_coeff, &nao_f, &zero, 
               &beta, 
               d_bPeu, &nao_s, &nao_sf, 
               &naux);

  // b^P_ee = b^P_ue * M_ue
  
  double * d_bPee = dd->d_buf1;
  
  ml->gemm_batch((char *) "N", (char *) "N", 
               &nao_f, &nao_f, &nao_s,
               &alpha, 
               dd->d_mo_coeff, &nao_f, &zero, 
               d_bPeu, &nao_s, &nao_sf, 
               &beta, 
               d_bPee, &nao_f, &nao_f2, 
               &naux);

  //do packing
 
  d_my_unpack_map_ptr = dd_fetch_pumap(dd, nao_f, _PUMAP_2D_UNPACK);

  double * d_eri_unpacked = dd->d_buf2;

  pack_eri(d_eri_unpacked, d_bPee, d_my_unpack_map_ptr, naux, nao_f, nao_f_pair);

  if (return_4c2eeri){
    double beta_ = (count < num_devices) ? 0.0 : 1.0;
#ifdef _DEBUG_DEVICE
    printf("returning 4c2e\n");
    printf("beta %f\n",beta_);
#endif
    
    ml->gemm((char *) "N", (char *) "T", &nao_f_pair, &nao_f_pair, &naux,
	     &alpha, d_eri_unpacked, &nao_f_pair, d_eri_unpacked, &nao_f_pair, &beta_, dd->d_buf3, &nao_f_pair);

  } else {
#ifdef _DEBUG_DEVICE
    printf("returning 3c2e\n");
#endif
    
    double * eri_impham = &(pin_eri_impham[count*blksize * nao_f_pair]);

    pm->dev_pull_async(d_eri_unpacked, eri_impham, naux*nao_f_pair*sizeof(double));
  }
  
#if 0
  double * h_eri_impham = (double *)pm->dev_malloc_host(nao_f_pair*nao_f_pair*sizeof(double));
  pm->dev_pull_async(dd->d_buf3, h_eri_impham, nao_f_pair*nao_f_pair*sizeof(double));
  pm->dev_stream_wait();
  for (int i =0;i<nao_f_pair;++i){ for (int j=0;j<nao_f_pair;++j){printf("%f\t",h_eri_impham[i*nao_f_pair+j]); }printf("\n");}
  pm->dev_free_host(h_eri_impham);
#endif
  
  double t1 = omp_get_wtime();
  t_array[12] += t1 - t0;
  
  pm->dev_profile_stop();
  
  // counts in pull eri_impham
}

/* ---------------------------------------------------------------------- */

void Device::compute_eri_impham_v2(int nao_s, int nao_f, int blksize, int naux, int count, size_t addr_dfobj_in, size_t addr_dfobj_out)
{
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Inside Device::comute_eri_impham()\n");
#endif

  pm->dev_profile_start("compute_eri_impham");
  double t0 = omp_get_wtime();

  const int device_id = count % num_devices;
  pm->dev_set_device(device_id);
  my_device_data * dd = &(device_data[device_id]);
  
  dd->active = 1;

  double * d_cderi = nullptr;
  // using fetch_eri, assume it's already there
  int nao_s_pair = nao_s * (nao_s + 1)/2;
  d_cderi = dd_fetch_eri(dd, nullptr, naux, nao_s_pair, addr_dfobj_in, count);
  
  double * d_cderi_unpacked = dd->d_buf1;

  int * d_my_unpack_map_ptr = dd_fetch_pumap(dd, nao_s, _PUMAP_2D_UNPACK);

  getjk_unpack_buf2(d_cderi_unpacked,d_cderi,d_my_unpack_map_ptr,naux, nao_s, nao_s_pair);

  const double alpha = 1.0;
  const double beta = 0.0;
  int zero = 0;
  int nao_s2 = nao_s * nao_s;
  int nao_sf = nao_s * nao_f;
  int nao_f2 = nao_f * nao_f;
  int nao_f_pair = nao_f * (nao_f+1)/2;
  
  double * d_bPeu = dd->d_buf2;
  
  // b^P_ue = b^P_uu * M_ue
  
  ml->set_handle();
  ml->gemm_batch((char *) "N", (char *) "T", 
               &nao_s, &nao_f, &nao_s,
               &alpha, 
               d_cderi_unpacked, &nao_s, &nao_s2, 
               dd->d_mo_coeff, &nao_f, &zero, 
               &beta, 
               d_bPeu, &nao_s, &nao_sf, 
               &naux);
  
  // b^P_ee = b^P_ue * M_ue
  
  double * d_bPee = dd->d_buf1;
  
  ml->gemm_batch((char *) "N", (char *) "N", 
               &nao_f, &nao_f, &nao_s,
               &alpha, 
               dd->d_mo_coeff, &nao_f, &zero, 
               d_bPeu, &nao_s, &nao_sf, 
               &beta, 
               d_bPee, &nao_f, &nao_f2, 
               &naux);

  //do packing
  
  d_my_unpack_map_ptr = dd_fetch_pumap(dd, nao_f, _PUMAP_2D_UNPACK);
  // new (transfer to exisiting smaller cholesky vector)
  double * d_cderi_out = dd_fetch_eri(dd, nullptr, naux, nao_f_pair, addr_dfobj_out, count);
  //TODO: add growing logic 
  //ml->gemm((char *) "T", (char *) "N", &nao_f_pair, &nao_f_pair, &naux, &alpha, dd->d_buf2, &ldb, dd->d_buf3, &lda, &beta, (dd->d_vkk)+vk_offset, &ldc);

  pack_eri(d_cderi_out, d_bPee,d_my_unpack_map_ptr, naux, nao_f, nao_f_pair);
  pm->dev_profile_stop();
  
  double t1 = omp_get_wtime();
  t_array[12] += t1 - t0;
  count_array[7]+=1; // just doing this addition in pull, not in init or compute
  // counts in pull eri_impham

}

/* ---------------------------------------------------------------------- */

#if defined(_ENABLE_P2P)
void Device::pull_eri_impham(py::array_t<double> _eri, int naoaux, int nao_f, int return_4c2eeri)
{
  //This should be obsolete in a production version. We want this calculation to not exist, and the impurity eri should directly get transferred from gpu to gpu in it's corresponding location. 

  // if not possible, then the cpu version should be refactored to allow pull to happen async (i think it's pageable right now and it will negate all performance when you pull bPee to cpu (and then transfer it back again)) 
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: -- Inside Device::pull_eri_impham()\n");
#endif
  
  pm->dev_profile_start("pull_eri_impham");
  
  double t0 = omp_get_wtime();
  
  int nao_f_pair = nao_f * (nao_f+1)/2;

  int N = nao_f_pair * nao_f_pair;
  
  py::buffer_info info_eri = _eri.request(); 
  double * eri = static_cast<double*>(info_eri.ptr);

  if (return_4c2eeri){

    std::vector<double *> e_vec(num_devices);
    std::vector<double *> buf_vec(num_devices);
    std::vector<int> active(num_devices);
  
    for(int i=0; i<num_devices; ++i) {
      my_device_data * dd = &(device_data[i]);
      e_vec[i] = dd->d_buf3; // this has the result
      buf_vec[i] = dd->d_buf2; // this is a temp buffer
      active[i] = dd->active;
    }

    mgpu_reduce(e_vec, pin_eri_impham, N, true, buf_vec, active);

#pragma omp parallel for
    for(int i=0; i<N; ++i) eri[i] += pin_eri_impham[i];

  } else {

    for(int i=0; i<num_devices; ++i) {
      pm->dev_set_device(i);
      pm->dev_barrier();
    }

#pragma omp parallel for
    for(int i=0; i<naoaux*nao_f_pair; ++i) eri[i] = pin_eri_impham[i];
  }

  pm->dev_profile_stop();
  
  double t1 = omp_get_wtime();
  t_array[13] += t1 - t0;
  count_array[7]+=1; // just doing this addition in pull, not in init or compute
    
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: -- Leaving Device::pull_eri_impham()\n");
#endif

}

#else

void Device::pull_eri_impham(py::array_t<double> _eri, int naoaux, int nao_f, int return_4c2eeri)
{
  //This should be obsolete in a production version. We want this calculation to not exist, and the impurity eri should directly get transferred from gpu to gpu in it's corresponding location. 

  // if not possible, then the cpu version should be refactored to allow pull to happen async (i think it's pageable right now and it will negate all performance when you pull bPee to cpu (and then transfer it back again)) 
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: -- Inside Device::pull_eri_impham()\n");
#endif
  
  pm->dev_profile_start("pull_eri_impham");
  
  double t0 = omp_get_wtime();
  
  int nao_f_pair = nao_f * (nao_f+1)/2;
  py::buffer_info info_eri = _eri.request(); 
  double * eri = static_cast<double*>(info_eri.ptr);
  
#if 0
  printf("starting pull\n");
  for (int i=0;i<nao_f_pair*nao_f_pair; ++i){printf("%f\t",eri[i]);}printf("\n");
#endif

  if (return_4c2eeri){
  
    for (int i=0; i<num_devices; ++i){
      pm->dev_set_device(i); 
      my_device_data * dd = &(device_data[i]);
      double * eri_impham =&pin_eri_impham[i * nao_f_pair*nao_f_pair];
      if (dd->active) pm->dev_pull_async(dd->d_buf3, eri_impham, nao_f_pair*nao_f_pair*sizeof(double));
    }

#ifdef _DEBUG_DEVICE
    printf("returning 4c2e\n");
    for (int i=0;i<num_devices;++i){
      pm->dev_set_device(i); 
      my_device_data * dd = &(device_data[i]);
      pm->dev_stream_wait();
      if (dd->d_buf3){
	for (int j=0;j <nao_f_pair;++j){
	  for (int k=0;k <nao_f_pair;++k){
	    printf("%f\t",pin_eri_impham[i*nao_f_pair*nao_f_pair+j*nao_f_pair+k]);
	  } printf("\n");
	}} printf("\n");
    }
#endif

    for(int i=0; i<num_devices; ++i) {
      pm->dev_set_device(i);
      my_device_data * dd = &(device_data[i]);
      pm->dev_stream_wait();
      
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

    for(int i=0; i<num_devices; ++i) {
      pm->dev_set_device(i);
      pm->dev_barrier();
    }
    
    std::memcpy(eri, pin_eri_impham, naoaux*nao_f_pair*sizeof(double));
  }

  pm->dev_profile_stop();
  
  double t1 = omp_get_wtime();
  t_array[13] += t1 - t0;
  count_array[7]+=1; // just doing this addition in pull, not in init or compute
    
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: -- Leaving Device::pull_eri_impham()\n");
#endif

}

#endif

/* ---------------------------------------------------------------------- */

void Device::init_mo_grid(int ngrid, int nmo)
{
  printf("starting init mo_grid\n");
  double t0 = omp_get_wtime();
  
  for(int id=0; id<num_devices; ++id) {
    pm->dev_set_device(id);

    my_device_data * dd = &(device_data[id]);

    int size_mo_grid = ngrid*nmo;

    grow_array(dd->d_mo_grid, size_mo_grid, dd->size_mo_grid, "mo_grid", FLERR);

    dd->active = 0;
  }
  
  double t1 = omp_get_wtime();
  
  //TODO:t_array[] += t1 - t0;
  // counts in pull Pi
}

/* ---------------------------------------------------------------------- */

void Device::push_ao_grid(py::array_t<double> _ao, int ngrid, int nao, int count)
{
  printf("starting push_mo_grid\n");
  double t0 = omp_get_wtime();
  
  py::buffer_info info_ao = _ao.request(); // 2D array (ngrid, nao)
  double * ao = static_cast<double*>(info_ao.ptr);
  
  int id = count%num_devices;

  pm->dev_set_device(id);

  my_device_data * dd = &(device_data[id]);

  int size_ao_grid = ngrid*nao;

  grow_array(dd->d_ao_grid, size_ao_grid, dd->size_ao_grid, "ao_grid", FLERR);
  
  pm->dev_push_async(dd->d_ao_grid, ao, size_ao_grid*sizeof(double));
  
  double t1 = omp_get_wtime();
  
  //TODO:t_array[] += t1 - t0;
  // counts in pull Pi
}

/* ---------------------------------------------------------------------- */

void Device::compute_mo_grid(int ngrid, int nao, int nmo)
{
  printf("starting compute\n");
  double t0 = omp_get_wtime();
  const int device_id =0;// count % num_devices;
  pm->dev_set_device(device_id);
  my_device_data * dd = &(device_data[device_id]);
  
  dd->active = 1;

  const double alpha = 1.0;
  const double beta = 0.0;
  #if 0
  double * h_mo_coeff = (double *)pm->dev_malloc_host(nao*nmo*sizeof(double));
  pm->dev_pull_async(dd->d_mo_coeff, h_mo_coeff, nao*nmo*sizeof(double));
  pm->dev_stream_wait();
  for (int i =0;i<nao;++i){for (int j=0;j<nmo;++j){printf("%f\t",h_mo_coeff[i*nmo+j]);}printf("\n");}
  #endif
  ml->set_handle();
  ml->gemm((char *) "N", (char *) "N", 
             &nmo, &ngrid, &nao,
             &alpha, 
             dd->d_mo_coeff, &nmo, 
             dd->d_ao_grid, &nao, 
             &beta, 
             dd->d_mo_grid, &nmo
             );
  double t1 = omp_get_wtime();  
  //TODO:t_array[] += t1 - t0;
  // counts in pull Pi
}

/* ---------------------------------------------------------------------- */

void Device::pull_mo_grid(py::array_t<double>_mo, int ngrid, int nmo)
{
double t0 = omp_get_wtime();

py::buffer_info info_mo = _mo.request(); // 2D array (ngrid, nao)
double * mo = static_cast<double*>(info_mo.ptr);

for(int id=0; id<1; ++id) {
  pm->dev_set_device(id);
  my_device_data * dd = &(device_data[id]);
  int size_mo_grid = ngrid*nmo;
  
  if(dd->active) {pm->dev_pull_async(dd->d_mo_grid, mo, size_mo_grid*sizeof(double));
    
  pm->dev_stream_wait();}
  pm->dev_barrier(); 
  
}
double t1 = omp_get_wtime();
}

/* ---------------------------------------------------------------------- */
void Device::push_cascm2 (py::array_t<double> _cascm2, int ncas) 
{
  double t0 = omp_get_wtime();
   
  py::buffer_info info_cascm2 = _cascm2.request(); // 4D array (ncas, ncas, ncas, ncas)
  double * cascm2 = static_cast<double*>(info_cascm2.ptr);

  for(int id=0; id<1; ++id) {
    pm->dev_set_device(id);
    my_device_data * dd = &(device_data[id]);
    
    int size_cascm2 = ncas*ncas*ncas*ncas;

    grow_array(dd->d_cascm2, size_cascm2, dd->size_cascm2, "cascm2", FLERR);

    pm->dev_push_async(dd->d_cascm2, cascm2, size_cascm2*sizeof(double));
  }
  
  double t1 = omp_get_wtime();
  
  //TODO:t_array[] += t1 - t0;
  // counts in pull Pi
}

/* ---------------------------------------------------------------------- */


void Device::init_Pi(int ngrid)
{
  double t0 = omp_get_wtime();
  
  for(int id=0; id<1; ++id) {
    pm->dev_set_device(id);

    my_device_data * dd = &(device_data[id]);

    int size_Pi = ngrid;

    grow_array(dd->d_Pi, size_Pi, dd->size_Pi, "Pi", FLERR);
  }
  
  double t1 = omp_get_wtime();
  
  //TODO:t_array[] += t1 - t0;
  // counts in pull Pi
}

/* ---------------------------------------------------------------------- */
void Device::compute_rho_to_Pi(py::array_t<double> _rho, int ngrid, int count)
{
  double t0 = omp_get_wtime();
  const int device_id = count % num_devices;
  pm->dev_set_device(device_id);
  my_device_data * dd = &(device_data[device_id]);
  
  py::buffer_info info_rho = _rho.request(); // 1D array (ngrid)
  double * cascm2 = static_cast<double*>(info_rho.ptr);
  grow_array(dd->d_rho, ngrid, dd->size_rho, "rho", FLERR);
  pm->dev_push_async(dd->d_rho, rho, ngrid*sizeof(double));
  get_rho_to_Pi(dd->d_rho, dd->d_Pi, ngrid);
}
/* ---------------------------------------------------------------------- */

void Device::compute_Pi (int ngrid, int ncas, int nao, int count) 
{
  double t0 = omp_get_wtime();
  const int device_id = count % num_devices;
  pm->dev_set_device(device_id);
  my_device_data * dd = &(device_data[device_id]);
  const double alpha = 1.0;
  const double beta = 0.0;
  const int one = 1; 
  ml->set_handle();
  
  int _size_buf_pdft = ngrid*ncas*ncas;

  int _size_orig = dd->size_buf_pdft; // because grow_array() updates dd->size_buf_pdft on first call

  grow_array(dd->d_buf_pdft1, _size_buf_pdft, dd->size_buf_pdft, "buf_pdft1", FLERR);
  grow_array(dd->d_buf_pdft2, _size_buf_pdft, _size_orig,        "buf_pdft2", FLERR);

  int ncas2 = ncas*ncas;
  //make mo_grid to ngrid*ncas*ncas (ai,aj->aij)
  double * d_mo_grid = dd->d_buf_pdft1;  //mo grid is only ngrid*ncas, using buf_pdft1 because efficient to not allot more
  ml->set_handle();
  ml->gemm((char *) "N", (char *) "N", 
             &ncas, &ngrid, &nao,
             &alpha, 
             dd->d_mo_coeff, &ncas, 
             dd->d_ao_grid, &nao, 
             &beta, 
             d_mo_grid, &ncas
             );

  // do buf1 = aij, ijkl->akl, mo, cascm2
  double * d_gridkern = dd->d_buf_pdft2; //trying to make it close to pyscf-forge mcpdft
  #if 0
  ml->gemm ((char *) "N", (char *) "N",
             &ngrid, &ncas2, &ncas2, 
             &alpha,
             dd->d_buf_pdft1, &ngrid,
             dd->d_cascm2, &ncas2,
             &beta, 
             dd->d_buf_pdft2, &ngrid);
             
  #else
  make_buf_pdft(d_gridkern, dd->d_buf_pdft1, dd->d_cascm2, ngrid, ncas);
  #endif
  // do Pi = (akl,akl->a, buf1, mo)/2
  #if 0
  const double half=0.5;
  ml->gemm_batch ((char *) "N",(char *) "T", 
             &one, &one, &ncas2,
             &half, 
             dd->d_buf_pdft1, &ncas2, &ncas2, 
             dd->d_buf_pdft2, &ncas2, &ncas2, 
             &beta, 
             dd->d_Pi, &one, &one, 
             &ngrid);
  #else
  make_Pi_final(d_gridkern, dd->d_buf_pdft1, dd->d_Pi, ngrid, ncas);
  #endif
             
}

/* ---------------------------------------------------------------------- */

void Device::pull_Pi (py::array_t<double> _Pi, int ngrid, int count)
{
  double t0 = omp_get_wtime();

  py::buffer_info info_Pi = _Pi.request(); //1D array (ngrid)
  double * Pi = static_cast<double*>(info_Pi.ptr);

  int device_id = count%num_devices;

  pm->dev_set_device(device_id);
  my_device_data * dd = &(device_data[device_id]);

  if (dd->d_Pi) pm->dev_pull_async(dd->d_Pi, Pi, ngrid*sizeof(double));
  
} 

/* ---------------------------------------------------------------------- */

// Is both _ocm2 in/out as it get over-written and resized?

void Device::orbital_response(py::array_t<double> _f1_prime,
			      py::array_t<double> _ppaa, py::array_t<double> _papa, py::array_t<double> _eri_paaa,
			      py::array_t<double> _ocm2, py::array_t<double> _tcm2, py::array_t<double> _gorb,
			      int ncore, int nocc, int nmo) // obselete
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

void Device::mgpu_bcast(std::vector<double *> d_ptr, double * h_ptr, size_t size)
{
  // push data from host to first device
  
  pm->dev_set_device(0);
    
  int err = pm->dev_push_async(d_ptr[0], h_ptr, size);
  
  if(err) {
    printf("LIBGPU:: dev_push_async(d_ptr[0]) failed\n");
    exit(1);
  }

  for(int i=1; i<d_ptr.size(); ++i)
    pm->dev_memcpy_peer(d_ptr[i], i, d_ptr[0], 0, size);
}

/* ---------------------------------------------------------------------- */

void Device::mgpu_reduce(std::vector<double *> d_ptr, double * h_ptr, int N, bool blocking, std::vector<double *> buf_ptr, std::vector<int> active)
{
#if defined(_DEBUG_P2P)
  printf("LIBGPU :: -- GPU-GPU Reduction  Starting!\n");
#endif
  
  size_t size = N * sizeof(double);

  int num_active = 0;
  for(int i=0; i<num_devices; ++i) num_active += active[i];
  
  int nrecv = num_active / 2;

  int nactive = num_active;

  // accumulate result to device 0 using binary tree reduction
  
  int il = 0;
  while(nrecv > 0) {

#if defined(_DEBUG_P2P)
    printf("LIBGPU :: -- GPU-GPU Reduction  il= %i  nactive= %i  nrecv= %i\n",il,nactive,nrecv);
#endif
    
    // odd number of recievers and not last level (clean-up pre-reduction)
      
    if((nactive > 1) && (nactive % 2)) {
      
#if defined(_DEBUG_P2P)
      printf("LIBGPU :: -- GPU-GPU Reduction  pre clean-up odd reciever  nactive= %i  nrecv= %i\n",nactive,nrecv);
#endif
      
      int dest = nactive - 2;
      int src = nactive - 1;
      
      if(d_ptr[src] && active[src]) {
#if defined(_DEBUG_P2P)
	printf("LIBGPU :: -- GPU-GPU Reduction  -- src %i(%p) --> dest %i(%p, %p)\n",
	       src, d_ptr[src], dest, buf_ptr[dest], d_ptr[dest]);
#endif
	
	if(blocking) {
	  // need to ensure dest is done using buf
	  
	  pm->dev_set_device(dest);
	  
	  pm->dev_stream_wait();
	}
	
	// src initiates transfer
	
	pm->dev_set_device(src);
	
	pm->dev_memcpy_peer(buf_ptr[dest],dest, d_ptr[src], src, size);
	
	// dest launches kernel
	
	pm->dev_set_device(dest); 
	
	vecadd(buf_ptr[dest], d_ptr[dest], N);
      }
      
      nactive--;
    }
    
    // binary tree reduction
    
    if(nactive > nrecv) {

#if defined(_DEBUG_P2P)
      printf("LIBGPU :: -- GPU-GPU Reduction  binary reduction   nactive= %i  nrecv= %i\n",nactive,nrecv);
#endif
      
      int nsend = nactive - nrecv;

      for(int i=0; i<nsend; ++i) {

	int dest = i;
	int src = nrecv + i;

	if(d_ptr[src] && active[src]) {
#if defined(_DEBUG_P2P)	
	printf("LIBGPU :: -- GPU-GPU Reduction  -- src %i(%p) --> dest %i(%p, %p)\n",
	       src, d_ptr[src], dest, buf_ptr[dest], d_ptr[dest]);
#endif

	  if(blocking) {
	    // need to ensure dest is done using buf
	    
	    pm->dev_set_device(dest);
	    
	    pm->dev_stream_wait();
	  }

	  // src initiates transfer
	  
	  pm->dev_set_device(src);
	  
	  pm->dev_memcpy_peer(buf_ptr[dest], dest, d_ptr[src], src, size);

	  // dest launches kernel
	  
	  pm->dev_set_device(dest); 
	  
	  vecadd(buf_ptr[dest], d_ptr[dest], N);
	}
      }

      nactive = nrecv;

      // odd number of recievers and not last level (clean-up post-reduction)
      
      if((nrecv > 1) && (nrecv % 2)) {

#if defined(_DEBUG_P2P)
       	printf("LIBGPU :: -- GPU-GPU Reduction  post clean-up odd reciever  nactive= %i  nrecv= %i\n",nactive,nrecv);
#endif
    
	int dest = nrecv - 2;
	int src = nrecv - 1;

	if(d_ptr[src] && active[src]) {
#if defined(_DEBUG_P2P)	
	printf("LIBGPU :: -- GPU-GPU Reduction  -- src %i(%p) --> dest %i(%p, %p)\n",
	       src, d_ptr[src], dest, buf_ptr[dest], d_ptr[dest]);
#endif
	  if(blocking) {
	    // need to ensure dest is done using buf
	    pm->dev_set_device(dest);
	    
	    pm->dev_stream_wait();
	  }

	  // src initiates transfer
	  
	  pm->dev_set_device(src);
	  
	  pm->dev_memcpy_peer(buf_ptr[dest], dest, d_ptr[src], src, size);

	  // dest launches kernel
	  
	  pm->dev_set_device(dest); 
	  
	  vecadd(buf_ptr[dest], d_ptr[dest], N);
	}

	nrecv--;
	nactive--;
      }
      
    }

    nrecv /= 2;
    il++;
  }

  // accumulate result on host

#if defined(_DEBUG_P2P)
  printf("LIBGPU :: -- GPU-GPU Reduction  transferring result to host\n");
#endif
  
  pm->dev_set_device(0);
  
  pm->dev_pull(d_ptr[0], h_ptr, size);
  
#if defined(_DEBUG_P2P)
  printf("LIBGPU :: -- GPU-GPU Reduction  completed!\n");
#endif
}

/* ---------------------------------------------------------------------- */
void Device::init_tdm1(int norb)
{
  double t0 = omp_get_wtime();

  int size_tdm1 = norb*norb; 
  int id=0;
  pm->dev_set_device(id);
  my_device_data * dd = &(device_data[id]);
  pm->dev_profile_start("tdms :: init tdm1");

  grow_array(dd->d_tdm1, size_tdm1, dd->size_tdm1, "TDM1", FLERR);
  pm->dev_barrier();
  pm->dev_profile_stop();
  double t1 = omp_get_wtime();
  t_array[14] += t1 - t0;
} 
/* ---------------------------------------------------------------------- */
void Device::init_tdm2(int norb)
{
  double t0 = omp_get_wtime();
  int size_tdm2 = norb*norb*norb*norb; 
  int id=0;
  pm->dev_set_device(id);
  my_device_data * dd = &(device_data[id]);
  pm->dev_profile_start("tdms :: init tdm2");
  grow_array(dd->d_tdm2, size_tdm2, dd->size_tdm2, "TDM2", FLERR);
  pm->dev_barrier();
  pm->dev_profile_stop();
  double t1 = omp_get_wtime();
  t_array[15] += t1 - t0;
} 
/* ---------------------------------------------------------------------- */
void Device::init_tdm3hab(int norb)
{
  double t0 = omp_get_wtime();
  int size_tdm2 = norb*norb*norb*norb; 
  int id=0;
  pm->dev_set_device(id);
  my_device_data * dd = &(device_data[id]);
  pm->dev_profile_start("tdms :: init tdm2");
  grow_array(dd->d_tdm2, size_tdm2, dd->size_tdm2, "TDM2", FLERR);
  grow_array(dd->d_tdm2_p, size_tdm2, dd->size_tdm2_p, "TDM2_p", FLERR);
  //pointed to tdm3ha/b in the function itself
  pm->dev_barrier();
  pm->dev_profile_stop();
  double t1 = omp_get_wtime();
  //t_array[15] += t1 - t0;//TODO: Fix timing array position
} 

/* ---------------------------------------------------------------------- */
void Device::push_ci(py::array_t<double> _cibra, py::array_t<double> _ciket, int na, int nb)
{
  //obsolete
  double t0 = omp_get_wtime();
  int id = 0;
  pm->dev_set_device(id); 
  my_device_data * dd = &(device_data[id]);
  pm->dev_profile_start("tdms :: push ci");

  py::buffer_info info_cibra = _cibra.request(); //2D array (na, nb)
  double * cibra = static_cast<double*>(info_cibra.ptr);
  py::buffer_info info_ciket = _ciket.request(); //2D array (na, nb)
  double * ciket = static_cast<double*>(info_ciket.ptr);
  int size_cibra = na*nb;
  int size_ciket = na*nb;
  grow_array(dd->d_cibra, size_cibra, dd->size_cibra, "cibra", FLERR);
  grow_array(dd->d_ciket, size_ciket, dd->size_ciket, "ciket", FLERR);

  pm->dev_push_async(dd->d_cibra, cibra, size_cibra*sizeof(double));
  pm->dev_push_async(dd->d_ciket, ciket, size_ciket*sizeof(double));
  pm->dev_barrier();
  pm->dev_profile_stop();
  double t1 = omp_get_wtime();
  t_array[16] += t1 - t0;
  
}
/* ---------------------------------------------------------------------- */
void Device::push_cibra(py::array_t<double> _cibra, int na, int nb)
{
  double t0 = omp_get_wtime();
  int id = 0;
  pm->dev_set_device(id); 
  my_device_data * dd = &(device_data[id]);
  pm->dev_profile_start("tdms :: push cibra");

  py::buffer_info info_cibra = _cibra.request(); //2D array (na, nb)
  double * cibra = static_cast<double*>(info_cibra.ptr);
  int size_cibra = na*nb;
  grow_array(dd->d_cibra, size_cibra, dd->size_cibra, "cibra", FLERR);

  pm->dev_push_async(dd->d_cibra, cibra, size_cibra*sizeof(double));
  pm->dev_barrier();
  pm->dev_profile_stop();
  double t1 = omp_get_wtime();
  t_array[16] += t1 - t0;
  
} 
 /* ---------------------------------------------------------------------- */
void Device::push_ciket(py::array_t<double> _ciket, int na, int nb)
{
  double t0 = omp_get_wtime();
  int id = 0;
  pm->dev_set_device(id); 
  my_device_data * dd = &(device_data[id]);
  pm->dev_profile_start("tdms :: push ciket");

  py::buffer_info info_ciket = _ciket.request(); //2D array (na, nb)
  double * ciket = static_cast<double*>(info_ciket.ptr);
  int size_ciket = na*nb;
  grow_array(dd->d_ciket, size_ciket, dd->size_ciket, "ciket", FLERR);

  pm->dev_push_async(dd->d_ciket, ciket, size_ciket*sizeof(double));
  pm->dev_barrier();
  pm->dev_profile_stop();
  double t1 = omp_get_wtime();
  t_array[16] += t1 - t0;
} 
/* ---------------------------------------------------------------------- */
void Device::push_link_indexa(int na, int nlinka, py::array_t<int> _link_indexa)
{
  double t0 = omp_get_wtime();
  int id = 0;
  pm->dev_set_device(id); 
  my_device_data * dd = &(device_data[id]);
  pm->dev_profile_start("tdms :: push link index");
  py::buffer_info info_link_indexa = _link_indexa.request(); //3D array (na, nlinka, 4)
  int * link_indexa = static_cast<int*>(info_link_indexa.ptr);
  int size_clinka = na*nlinka*4; //a,i,str,sign
  grow_array(dd->d_clinka, size_clinka, dd->size_clinka, "clink", FLERR);

  pm->dev_push_async(dd->d_clinka, link_indexa, size_clinka*sizeof(int));

  pm->dev_barrier();
  pm->dev_profile_stop();
  double t1 = omp_get_wtime();
  t_array[17] += t1 - t0;
}
/* ---------------------------------------------------------------------- */
void Device::push_link_indexb(int nb, int nlinkb, py::array_t<int> _link_indexb)
{
  double t0 = omp_get_wtime();
  int id = 0;
  pm->dev_set_device(id); 
  my_device_data * dd = &(device_data[id]);
  pm->dev_profile_start("tdms :: push link index");
  py::buffer_info info_link_indexb = _link_indexb.request(); //3D array (na, nlinka, 4)
  int * link_indexb = static_cast<int*>(info_link_indexb.ptr);
  int size_clinkb = nb*nlinkb*4; //a,i,str,sign
  grow_array(dd->d_clinkb, size_clinkb, dd->size_clinkb, "clinkb", FLERR);

  pm->dev_push_async(dd->d_clinkb, link_indexb, size_clinkb*sizeof(int));

  pm->dev_barrier();
  pm->dev_profile_stop();
  double t1 = omp_get_wtime();
  t_array[17] += t1 - t0;
}
/* ---------------------------------------------------------------------- */
void Device::compute_trans_rdm1a(int na, int nb, int nlinka, int nlinkb, int norb)
{
  double t0 = omp_get_wtime();
  int id = 0;
  pm->dev_set_device(id); 
  my_device_data * dd = &(device_data[id]);
  pm->dev_profile_start("tdms :: compute_trans_rdm1a");

  int norb2 = norb*norb;
  int size_tdm1 = norb2;
  grow_array(dd->d_tdm1,size_tdm1, dd->size_tdm1, "tdm1", FLERR); //actual returned
  set_to_zero(dd->d_tdm1, size_tdm1);

  compute_FCItrans_rdm1a(dd->d_cibra, dd->d_ciket, dd->d_tdm1, norb, na, nb, nlinka, dd->d_clinka);

  pm->dev_barrier();
  pm->dev_profile_stop();
  double t1 = omp_get_wtime();
  t_array[18] += t1 - t0;
  count_array[8]++;
}
/* ---------------------------------------------------------------------- */
void Device::compute_trans_rdm1b(int na, int nb, int nlinka, int nlinkb, int norb)
{
  double t0 = omp_get_wtime();
  int id = 0;
  pm->dev_set_device(id); 
  my_device_data * dd = &(device_data[id]);
  pm->dev_profile_start("tdms :: compute_trans_rdm1b");

  int norb2 = norb*norb;
  int size_tdm1 = norb2;
  grow_array(dd->d_tdm1,size_tdm1, dd->size_tdm1, "tdm1", FLERR); //actual returned
  set_to_zero(dd->d_tdm1, size_tdm1);

  compute_FCItrans_rdm1b(dd->d_cibra, dd->d_ciket, dd->d_tdm1, norb, na, nb, nlinkb, dd->d_clinkb);
  pm->dev_barrier();
  pm->dev_profile_stop();
  double t1 = omp_get_wtime();
  t_array[19] += t1 - t0;
  count_array[9]++;
}
/* ---------------------------------------------------------------------- */
void Device::compute_make_rdm1a(int na, int nb, int nlinka, int nlinkb, int norb)
{
  double t0 = omp_get_wtime();
  int id = 0;
  pm->dev_set_device(id); 
  my_device_data * dd = &(device_data[id]);

  pm->dev_profile_start("tdms :: compute_make_rdm1a");
  int norb2 = norb*norb;
  int size_tdm1 = norb2;
  grow_array(dd->d_tdm1,size_tdm1, dd->size_tdm1, "tdm1", FLERR); //actual returned
  set_to_zero(dd->d_tdm1, size_tdm1);
  
  compute_FCImake_rdm1a(dd->d_cibra, dd->d_ciket, dd->d_tdm1, norb, na, nb, nlinka, dd->d_clinka);

  pm->dev_barrier();
  pm->dev_profile_stop();
  double t1 = omp_get_wtime();
  t_array[20] += t1 - t0;
  count_array[10]++;
}
/* ---------------------------------------------------------------------- */
void Device::compute_make_rdm1b(int na, int nb, int nlinka, int nlinkb, int norb)
{
  double t0 = omp_get_wtime();
  int id = 0;
  pm->dev_set_device(id); 
  my_device_data * dd = &(device_data[id]);
  pm->dev_profile_start("tdms :: make_rdm1b");

  int norb2 = norb*norb;
  int size_tdm1 = norb2;
  grow_array(dd->d_tdm1,size_tdm1, dd->size_tdm1, "tdm1", FLERR); //actual returned
  set_to_zero(dd->d_tdm1, size_tdm1);

  compute_FCImake_rdm1b(dd->d_cibra, dd->d_ciket, dd->d_tdm1, norb, na, nb, nlinkb, dd->d_clinkb);

  pm->dev_barrier();
  pm->dev_profile_stop();
  double t1 = omp_get_wtime();
  t_array[21] += t1 - t0;
  count_array[11]++;
}

/* ---------------------------------------------------------------------- */
void Device::compute_tdm12kern_a(int na, int nb, int nlinka, int nlinkb, int norb )
{
  double t0 = omp_get_wtime();
  int id=0;
  pm->dev_set_device(id);
  ml->set_handle(id);
  my_device_data * dd = &(device_data[id]);
  pm->dev_profile_start("tdms :: compute_tdm12kern_a");
 
  int norb2 = norb*norb;
  int size_buf = norb2*nb;
  int size_rdm2 = norb2*norb2;
  int size_tdm1 = norb2;
  int size_tdm2 = norb2*norb2;
  int zero = 0;
  int bits_buf = sizeof(double)*size_buf;
  int bits_tdm1 = sizeof(double)*size_tdm1;
  int bits_tdm2 = sizeof(double)*size_tdm2;
  grow_array(dd->d_tdm1,size_tdm1, dd->size_tdm1, "tdm1", FLERR); //actual returned
  grow_array(dd->d_tdm2,size_tdm2, dd->size_tdm2, "tdm2", FLERR); //actual returned
  grow_array(dd->d_buf1,size_buf, dd->size_buf1, "buf1", FLERR); 
  grow_array(dd->d_buf2,size_buf, dd->size_buf2, "buf2", FLERR); 
  //set buf array to zero
  //must also set tdm1/2, pdm1/2 to zero because it may have residual from previous calls
  //set_to_zero(dd->d_pdm1, size_pdm1); 
  //set_to_zero(dd->d_pdm2, size_pdm2); 
  ml->memset(dd->d_buf1, &zero, &bits_buf); 
  ml->memset(dd->d_buf2, &zero, &bits_buf); 
  ml->memset(dd->d_tdm1, &zero, &bits_tdm1);
  ml->memset(dd->d_tdm2, &zero, &bits_tdm2);

  const double alpha = 1.0;
  const double beta = 1.0;
  const int one = 1;

  for (int stra_id = 0; stra_id<na; ++stra_id) { 
    compute_FCIrdm2_a_t1ci( dd->d_cibra, dd->d_buf2, stra_id, nb, norb, nlinka, dd->d_clinka); 
    compute_FCIrdm2_a_t1ci( dd->d_ciket, dd->d_buf1, stra_id, nb, norb, nlinka, dd->d_clinka); 
    double * bravec = &(dd->d_cibra[stra_id*nb]);
    ml->gemv((char *) "N", &norb2,  &nb,
                &alpha, 
                dd->d_buf1, &norb2, 
                bravec, &one, 
                &beta, 
                dd->d_tdm1, &one); 
    ml->gemm((char *) "N",(char *) "T", &norb2, &norb2, &nb, 
                &alpha,
                dd->d_buf1, &norb2,
                dd->d_buf2, &norb2,
                &beta,
                dd->d_tdm2, &norb2); //convert to gemm_batched, edit na loops to batches
    ml->memset(dd->d_buf1, &zero, &bits_buf); 
    ml->memset(dd->d_buf2, &zero, &bits_buf); 
  }     

  transpose_jikl(dd->d_tdm2, dd->d_buf1, norb);

  pm->dev_barrier();
  pm->dev_profile_stop();
  double t1 = omp_get_wtime();
  t_array[22] += t1 - t0;
  count_array[12]++;
}

/* ---------------------------------------------------------------------- */
void Device::compute_tdm12kern_b(int na, int nb, int nlinka, int nlinkb, int norb)
{
  double t0 = omp_get_wtime();
  int id=0;
  pm->dev_set_device(id);
  ml->set_handle(id);
  my_device_data * dd = &(device_data[id]);
  pm->dev_profile_start("tdms :: compute_tdm12kern_b");
 
  int norb2 = norb*norb;
  int size_buf = norb2*nb;
  int size_tdm1 = norb2;
  int size_tdm2 = norb2*norb2;
  int bits_buf = sizeof(double)*size_buf;
  int bits_tdm1 = sizeof(double)*size_tdm1;
  int bits_tdm2 = sizeof(double)*size_tdm2;
  int zero = 0;
  grow_array(dd->d_tdm1,size_tdm1, dd->size_tdm1, "tdm1", FLERR); //actual returned
  grow_array(dd->d_tdm2,size_tdm2, dd->size_tdm2, "tdm2", FLERR); //actual returned
  grow_array(dd->d_buf1,size_buf, dd->size_buf1, "buf1", FLERR); 
  grow_array(dd->d_buf2,size_buf, dd->size_buf2, "buf2", FLERR); 
  //set buf array to zero
  //must also set tdm1/2 to zero because it may have residual from previous calls
  //think if you need to zero everything 
  ml->memset(dd->d_buf1, &zero, &bits_buf); 
  ml->memset(dd->d_buf2, &zero, &bits_buf); 
  ml->memset(dd->d_tdm1, &zero, &bits_tdm1);
  ml->memset(dd->d_tdm2, &zero, &bits_tdm2);

  const double alpha = 1.0;
  const double beta = 1.0;
  const int one = 1;
  const int two = 2;
  for (int stra_id = 0; stra_id<na; ++stra_id) { 
    //csum = FCIrdm2_a_t1ci(bra, buf1, bcount, stra_id, strb_id,norb, nb, nlinka, clink_indexa); //Decided to not do bcounts, sent full nb, reduces variables and usually have small ci space
    compute_FCIrdm2_b_t1ci( dd->d_cibra, dd->d_buf2, stra_id, nb, norb, nlinkb, dd->d_clinkb); 
    compute_FCIrdm2_b_t1ci( dd->d_ciket, dd->d_buf1, stra_id, nb, norb, nlinkb, dd->d_clinkb); 
    double * bravec = &(dd->d_cibra[stra_id*nb]);//bra+stra_id*nb+strb_id;
    ml->gemv((char *) "N", &norb2,  &nb,
                &alpha, 
                dd->d_buf1, &norb2, 
                bravec, &one, 
                &beta, 
                dd->d_tdm1, &one); //convert to gemv_batched, edit na loop to batches, may need to increase buf

    ml->gemm((char *) "N",(char *) "T", &norb2, &norb2, &nb, 
                &alpha,
                dd->d_buf1, &norb2,
                dd->d_buf2, &norb2,
                &beta,
                dd->d_tdm2, &norb2); //convert to gemm_batched, edit na loops to batches
 
    ml->memset(dd->d_buf1, &zero, &bits_buf); 
    ml->memset(dd->d_buf2, &zero, &bits_buf); 
    }     
  transpose_jikl(dd->d_tdm2, dd->d_buf1, norb);

  pm->dev_barrier();
  pm->dev_profile_stop();
  double t1 = omp_get_wtime();
  t_array[23] += t1-t0;
  count_array[13]++;
}
/* ---------------------------------------------------------------------- */
void Device::compute_tdm12kern_ab(int na, int nb, int nlinka, int nlinkb, int norb)
{
  double t0 = omp_get_wtime();
  int id=0;
  pm->dev_set_device(id);
  ml->set_handle(id);
  my_device_data * dd = &(device_data[id]);
  pm->dev_profile_start("tdms :: compute_tdm12kern_ab");
 
  int norb2 = norb*norb;
  int size_buf = norb2*nb;
  int size_rdm2 = norb2*norb2;
  int size_tdm2 = norb2*norb2;
  int zero = 0;
  int bits_buf = sizeof(double)*size_buf;
  int bits_tdm2 = sizeof(double)*size_tdm2;
  //no rdm1, tdm1, pdm1
  grow_array(dd->d_tdm2,size_tdm2, dd->size_tdm2, "tdm2", FLERR); //actual returned
  grow_array(dd->d_buf1,size_buf, dd->size_buf1, "buf1", FLERR);  
  grow_array(dd->d_buf2,size_buf, dd->size_buf2, "buf2", FLERR); 
  //set buf array to zero
  ml->memset(dd->d_buf1, &zero, &bits_buf); 
  ml->memset(dd->d_buf2, &zero, &bits_buf); 
  ml->memset(dd->d_tdm2, &zero, &bits_tdm2);
  const double alpha = 1.0;
  const double beta = 1.0;
  const int one = 1;
  for (int stra_id = 0; stra_id<na; ++stra_id) { 
    compute_FCIrdm2_a_t1ci( dd->d_cibra, dd->d_buf2, stra_id, nb, norb, nlinka, dd->d_clinka); 
    compute_FCIrdm2_b_t1ci( dd->d_ciket, dd->d_buf1, stra_id, nb, norb, nlinkb, dd->d_clinkb); 
    ml->gemm((char *) "N",(char *) "T", &norb2, &norb2, &nb, 
                &alpha,
                dd->d_buf1, &norb2,
                dd->d_buf2, &norb2,
                &beta,
                dd->d_tdm2, &norb2); //convert to gemm_batched, edit na loops to batches

    ml->memset(dd->d_buf1, &zero, &bits_buf); 
    ml->memset(dd->d_buf2, &zero, &bits_buf); 
    }     
    transpose_jikl(dd->d_tdm2, dd->d_buf1, norb);

    pm->dev_barrier();
    pm->dev_profile_stop();
    double t1 = omp_get_wtime();
    t_array[24] += t1-t0;
    count_array[14]++;
}
/* ---------------------------------------------------------------------- */
void Device::compute_rdm12kern_sf(int na, int nb, int nlinka, int nlinkb, int norb)
{
  double t0 = omp_get_wtime();
  int id=0;
  pm->dev_set_device(id);
  ml->set_handle(id);
  my_device_data * dd = &(device_data[id]);
  pm->dev_profile_start("tdms :: compute_rdm12kern_sf");
 
  int norb2 = norb*norb;
  int size_buf = norb2*nb;
  int size_tdm2 = norb2*norb2;
  int size_tdm1 = norb2;
  int zero = 0;
  int bits_buf = sizeof(double)*size_buf;
  int bits_tdm1 = sizeof(double)*size_tdm1;
  int bits_tdm2 = sizeof(double)*size_tdm2;
  grow_array(dd->d_tdm1,size_tdm1, dd->size_tdm1, "tdm1", FLERR); //actual returned
  grow_array(dd->d_tdm2,size_tdm2, dd->size_tdm2, "tdm2", FLERR); //actual returned
  grow_array(dd->d_buf1,size_buf, dd->size_buf1, "buf1", FLERR);  
  //set buf array to zero
  ml->memset(dd->d_buf1, &zero, &bits_buf); 
  ml->memset(dd->d_tdm1, &zero, &bits_tdm1);
  ml->memset(dd->d_tdm2, &zero, &bits_tdm2);
  for (int stra_id = 0; stra_id<na; ++stra_id) { 
    //these two functions constitute FCI_rdm_t1ci_sf
    compute_FCIrdm2_b_t1ci(dd->d_ciket, dd->d_buf1, stra_id, nb, norb, nlinkb, dd->d_clinkb); //rdm2_0b_t1ci is identical except where zeroing happens. since we do zeroing before at the start and at the end of each stra_id iteration , a special function is not needed. 
    compute_FCIrdm2_a_t1ci(dd->d_ciket, dd->d_buf1, stra_id, nb, norb, nlinka, dd->d_clinka); 
    const double alpha = 1.0;
    const double beta = 1.0;
    const int one = 1;
    double * ketvec = &(dd->d_ciket[stra_id*nb]);//ket+stra_id*nb+strb_id;
    ml->gemv((char *) "N", &norb2, &nb, 
                &alpha, 
                dd->d_buf1, &norb2, 
                ketvec, &one, 
                &beta, 
                dd->d_tdm1, &one); //convert to gemv_batched, edit na loop to batches, may need to increase buf

    ml->gemm((char *) "N",(char *) "T", &norb2, &norb2, &nb, 
                &alpha,
                dd->d_buf1, &norb2,
                dd->d_buf1, &norb2,
                &beta,
                dd->d_tdm2, &norb2); //convert to gemm_batched, edit na loops to batches
    ml->memset(dd->d_buf1, &zero, &bits_buf); 
      }     
    transpose_jikl(dd->d_tdm2, dd->d_buf1, norb);

    pm->dev_barrier();
    pm->dev_profile_stop();
    double t1 = omp_get_wtime();
    t_array[25] += t1-t0;
    count_array[15]++;
}
/* ---------------------------------------------------------------------- */
void Device::compute_tdm12kern_a_v2(int na, int nb, int nlinka, int nlinkb, int norb )
{
  double t0 = omp_get_wtime();
  int id = 0;
  pm->dev_set_device(id);
  ml->set_handle(id);
  my_device_data * dd = &(device_data[id]);
  pm->dev_profile_start("tdms :: compute_tdm12kern_a");
  int norb2 = norb*norb;
  int size_buf = norb2*nb;
  int size_tdm2 = norb2*norb2;
  int size_tdm1 = norb2;

  int zero = 0;
  int one = 1;
  const double alpha = 1.0;
  const double beta = 0.0;
  int bits_tdm1 = sizeof(double)*size_tdm1;
  int bits_tdm2 = sizeof(double)*size_tdm2;
  int _size_buf = _MAX(dd->size_buf1, dd->size_buf2);// (dd->size_buf1 > dd->size_buf2) ? dd->size_buf1 : dd->size_buf2;
  #ifdef _TEMP_BUFSIZING
  _size_buf = size_buf*6;
  #endif
  int final_size_buf = _MAX(_size_buf, size_buf);//(_size_buf > size_buf) ? _size_buf : size_buf;
  int buf_batch_size = final_size_buf/size_buf; //this is integer division // number of buf1/2 in a single buffer
  int gemm_batch_size = final_size_buf/(norb2*norb2); // this is integer division // number of tdm2 in a single buf
  int gemv_batch_size = final_size_buf/norb2; // this is integer division // number of tdm1 in a single buf
  int num_buf_batches; 
  int num_buf_batches_for_gemv; 
  int num_gemm_batches; 
  int num_gemv_batches; 
  grow_array(dd->d_buf1,final_size_buf, dd->size_buf1, "buf1", FLERR); 
  grow_array(dd->d_buf2,final_size_buf, dd->size_buf2, "buf2", FLERR); 
  grow_array(dd->d_buf3,final_size_buf, dd->size_buf3, "buf3", FLERR); 
  int bits_buf = sizeof(double)*buf_batch_size*size_buf;
  ml->memset(dd->d_buf1, &zero, &bits_buf); 
  ml->memset(dd->d_buf2, &zero, &bits_buf); 
  grow_array(dd->d_tdm1, size_tdm1, dd->size_tdm1, "tdm1", FLERR);
  grow_array(dd->d_tdm2, size_tdm2, dd->size_tdm2, "tdm2", FLERR); 
  ml->memset(dd->d_tdm1, &zero, &bits_tdm1);
  ml->memset(dd->d_tdm2, &zero, &bits_tdm2);
 

  for (int stra_id = 0; stra_id<na; stra_id += buf_batch_size){
  //for (int stra_id = 0; stra_id<na; ++stra_id) { 
    //compute_FCIrdm2_a_t1ci( dd->d_cibra, dd->d_buf2, stra_id, nb, norb, nlinka, dd->d_clinka); 
    //compute_FCIrdm2_a_t1ci( dd->d_ciket, dd->d_buf1, stra_id, nb, norb, nlinka, dd->d_clinka); 
    num_buf_batches = _MIN(buf_batch_size, na-stra_id);
    compute_FCIrdm2_a_t1ci_v2( dd->d_cibra, dd->d_buf2, stra_id, num_buf_batches, nb, norb, nlinka, dd->d_clinka); 
    compute_FCIrdm2_a_t1ci_v2( dd->d_ciket, dd->d_buf1, stra_id, num_buf_batches, nb, norb, nlinka, dd->d_clinka); 
    //double * bravec = &(dd->d_cibra[stra_id*nb]);
    //ml->gemv((char *) "N", &norb2,  &nb,
    //            &alpha, dd->d_buf1, &norb2, bravec, &one, 
    //            &beta, dd->d_tdm1, &one); 
    for (int i=0; i<num_buf_batches; i+=gemv_batch_size){
      double * bravec = &(dd->d_cibra[(stra_id+i)*nb]);
      num_gemv_batches = _MIN(gemv_batch_size, num_buf_batches-i);
      ml->gemv_batch((char *) "N", &norb2, &nb,
          &alpha, &(dd->d_buf1[i*size_buf]), &norb2, &size_buf,
          bravec, &one, &nb, 
          &beta, dd->d_buf3, &one, &size_tdm1,
          &num_gemv_batches);
      reduce_buf3_to_rdm(dd->d_buf3, dd->d_tdm1, size_tdm1, num_gemv_batches);
      }

    //ml->gemm((char *) "N",(char *) "T", &norb2, &norb2, &nb, 
    //            &alpha, dd->d_buf1, &norb2, dd->d_buf2, &norb2,
    //            &beta, dd->d_tdm2, &norb2); //convert to gemm_batched, edit na loops to batches
    for (int i=0; i<num_buf_batches; i+=gemm_batch_size) {
      num_gemm_batches = _MIN(gemm_batch_size, num_buf_batches-i);
      ml->gemm_batch((char *) "N",(char *) "T", &norb2, &norb2, &nb, 
        &alpha, 
        &(dd->d_buf1[i*size_buf]), &norb2, &size_buf, 
        &(dd->d_buf2[i*size_buf]), &norb2, &size_buf, 
        &beta, dd->d_buf3, &norb2, &size_tdm2, &num_gemm_batches); 
      reduce_buf3_to_rdm(dd->d_buf3, dd->d_tdm2, size_tdm2, num_gemm_batches);
      }

    ml->memset(dd->d_buf1, &zero, &bits_buf); 
    ml->memset(dd->d_buf2, &zero, &bits_buf); 
  }     

  transpose_jikl(dd->d_tdm2, dd->d_buf1, norb);

  pm->dev_barrier();
  pm->dev_profile_stop();
  double t1 = omp_get_wtime();
  t_array[22] += t1 - t0;
  count_array[12]++;
}

/* ---------------------------------------------------------------------- */
void Device::compute_tdm12kern_b_v2(int na, int nb, int nlinka, int nlinkb, int norb)
{
  double t0 = omp_get_wtime();
  int id = 0;
  pm->dev_set_device(id);
  ml->set_handle(id);
  my_device_data * dd = &(device_data[id]);
  pm->dev_profile_start("tdms :: compute_tdm12kern_b");
  int norb2 = norb*norb;
  int size_buf = norb2*nb;
  int size_tdm2 = norb2*norb2;
  int size_tdm1 = norb2;

  int zero = 0;
  int one = 1;
  const double alpha = 1.0;
  const double beta = 0.0;
  int bits_tdm1 = sizeof(double)*size_tdm1;
  int bits_tdm2 = sizeof(double)*size_tdm2;
  int _size_buf = _MAX(dd->size_buf1, dd->size_buf2);// (dd->size_buf1 > dd->size_buf2) ? dd->size_buf1 : dd->size_buf2;
  #ifdef _TEMP_BUFSIZING
  _size_buf = size_buf*6;
  #endif
  int final_size_buf = _MAX(_size_buf, size_buf);//(_size_buf > size_buf) ? _size_buf : size_buf;
  int buf_batch_size = final_size_buf/size_buf; //this is integer division // number of buf1/2 in a single buffer
  int gemm_batch_size = final_size_buf/(norb2*norb2); // this is integer division // number of tdm2 in a single buf
  int gemv_batch_size = final_size_buf/norb2; // this is integer division // number of tdm1 in a single buf
  int num_buf_batches; 
  int num_buf_batches_for_gemv; 
  int num_gemm_batches; 
  int num_gemv_batches; 
  grow_array(dd->d_buf1,final_size_buf, dd->size_buf1, "buf1", FLERR); 
  grow_array(dd->d_buf2,final_size_buf, dd->size_buf2, "buf2", FLERR); 
  grow_array(dd->d_buf3,final_size_buf, dd->size_buf3, "buf3", FLERR); 
  int bits_buf = sizeof(double)*buf_batch_size*size_buf;
  ml->memset(dd->d_buf1, &zero, &bits_buf); 
  ml->memset(dd->d_buf2, &zero, &bits_buf); 
  grow_array(dd->d_tdm1, size_tdm1, dd->size_tdm1, "tdm1", FLERR);
  grow_array(dd->d_tdm2, size_tdm2, dd->size_tdm2, "tdm2", FLERR); 
  ml->memset(dd->d_tdm1, &zero, &bits_tdm1);
  ml->memset(dd->d_tdm2, &zero, &bits_tdm2);
 

  for (int stra_id = 0; stra_id<na; stra_id += buf_batch_size){
  //for (int stra_id = 0; stra_id<na; ++stra_id) { 
    num_buf_batches = _MIN(buf_batch_size, na-stra_id);
    //compute_FCIrdm2_b_t1ci( dd->d_cibra, dd->d_buf2, stra_id, nb, norb, nlinkb, dd->d_clinkb); 
    //compute_FCIrdm2_b_t1ci( dd->d_ciket, dd->d_buf1, stra_id, nb, norb, nlinkb, dd->d_clinkb); 
    compute_FCIrdm2_b_t1ci_v2( dd->d_cibra, dd->d_buf2, stra_id, num_buf_batches, nb, norb, nlinkb, dd->d_clinkb); 
    compute_FCIrdm2_b_t1ci_v2( dd->d_ciket, dd->d_buf1, stra_id, num_buf_batches, nb, norb, nlinkb, dd->d_clinkb); 
    //double * bravec = &(dd->d_cibra[stra_id*nb]);
    //ml->gemv((char *) "N", &norb2,  &nb,
    //            &alpha, dd->d_buf1, &norb2, bravec, &one, 
    //            &beta, dd->d_tdm1, &one); 
    for (int i=0; i<num_buf_batches; i+=gemv_batch_size){
      double * bravec = &(dd->d_cibra[(stra_id+i)*nb]);
      num_gemv_batches = _MIN(gemv_batch_size, num_buf_batches-i);
      ml->gemv_batch((char *) "N", &norb2, &nb,
          &alpha, &(dd->d_buf1[i*size_buf]), &norb2, &size_buf,
          bravec, &one, &nb, 
          &beta, dd->d_buf3, &one, &size_tdm1,
          &num_gemv_batches);
      reduce_buf3_to_rdm(dd->d_buf3, dd->d_tdm1, size_tdm1, num_gemv_batches);
      }

    //ml->gemm((char *) "N",(char *) "T", &norb2, &norb2, &nb, 
    //            &alpha, dd->d_buf1, &norb2, dd->d_buf2, &norb2,
    //            &beta, dd->d_tdm2, &norb2); //convert to gemm_batched, edit na loops to batches
    for (int i=0; i<num_buf_batches; i+=gemm_batch_size) {
      num_gemm_batches = _MIN(gemm_batch_size, num_buf_batches-i);
      ml->gemm_batch((char *) "N",(char *) "T", &norb2, &norb2, &nb, 
        &alpha, 
        &(dd->d_buf1[i*size_buf]), &norb2, &size_buf, 
        &(dd->d_buf2[i*size_buf]), &norb2, &size_buf, 
        &beta, dd->d_buf3, &norb2, &size_tdm2, &num_gemm_batches); 
      reduce_buf3_to_rdm(dd->d_buf3, dd->d_tdm2, size_tdm2, num_gemm_batches);
      }

    ml->memset(dd->d_buf1, &zero, &bits_buf); 
    ml->memset(dd->d_buf2, &zero, &bits_buf); 
  }     

  transpose_jikl(dd->d_tdm2, dd->d_buf1, norb);

  pm->dev_barrier();
  pm->dev_profile_stop();
  double t1 = omp_get_wtime();
  t_array[23] += t1 - t0;
  count_array[13]++;
}

/* ---------------------------------------------------------------------- */
void Device::compute_tdm12kern_ab_v2(int na, int nb, int nlinka, int nlinkb, int norb)
{
  double t0 = omp_get_wtime();
  int id = 0;
  pm->dev_set_device(id);
  ml->set_handle(id);
  my_device_data * dd = &(device_data[id]);
  pm->dev_profile_start("tdms :: compute_tdm12kern_ab");
  int norb2 = norb*norb;
  int size_buf = norb2*nb;
  int size_tdm2 = norb2*norb2;
  int size_tdm1 = norb2;

  int zero = 0;
  int one = 1;
  const double alpha = 1.0;
  const double beta = 0.0;
  int bits_tdm1 = sizeof(double)*size_tdm1;
  int bits_tdm2 = sizeof(double)*size_tdm2;
  int _size_buf = _MAX(dd->size_buf1, dd->size_buf2);// (dd->size_buf1 > dd->size_buf2) ? dd->size_buf1 : dd->size_buf2;
  #ifdef _TEMP_BUFSIZING
  _size_buf = size_buf*6;
  #endif
  int final_size_buf = _MAX(_size_buf, size_buf);//(_size_buf > size_buf) ? _size_buf : size_buf;
  int buf_batch_size = final_size_buf/size_buf; //this is integer division // number of buf1/2 in a single buffer
  int gemm_batch_size = final_size_buf/(norb2*norb2); // this is integer division // number of tdm2 in a single buf
  int gemv_batch_size = final_size_buf/norb2; // this is integer division // number of tdm1 in a single buf
  int num_buf_batches; 
  int num_gemm_batches; 
  int num_gemv_batches; 
  grow_array(dd->d_buf1,final_size_buf, dd->size_buf1, "buf1", FLERR); 
  grow_array(dd->d_buf2,final_size_buf, dd->size_buf2, "buf2", FLERR); 
  grow_array(dd->d_buf3,final_size_buf, dd->size_buf3, "buf3", FLERR); 
  int bits_buf = sizeof(double)*buf_batch_size*size_buf;
  ml->memset(dd->d_buf1, &zero, &bits_buf); 
  ml->memset(dd->d_buf2, &zero, &bits_buf); 
  grow_array(dd->d_tdm1, size_tdm1, dd->size_tdm1, "tdm1", FLERR);
  grow_array(dd->d_tdm2, size_tdm2, dd->size_tdm2, "tdm2", FLERR); 
  ml->memset(dd->d_tdm1, &zero, &bits_tdm1);
  ml->memset(dd->d_tdm2, &zero, &bits_tdm2);
 

  for (int stra_id = 0; stra_id<na; stra_id += buf_batch_size){
  //for (int stra_id = 0; stra_id<na; ++stra_id) { 
    num_buf_batches = _MIN(buf_batch_size, na-stra_id);
    //compute_FCIrdm2_a_t1ci( dd->d_cibra, dd->d_buf2, stra_id, nb, norb, nlinka, dd->d_clinka); 
    //compute_FCIrdm2_b_t1ci( dd->d_ciket, dd->d_buf1, stra_id, nb, norb, nlinkb, dd->d_clinkb); 
    compute_FCIrdm2_a_t1ci_v2( dd->d_cibra, dd->d_buf2, stra_id, num_buf_batches, nb, norb, nlinka, dd->d_clinka); 
    compute_FCIrdm2_b_t1ci_v2( dd->d_ciket, dd->d_buf1, stra_id, num_buf_batches, nb, norb, nlinkb, dd->d_clinkb); 

    //ml->gemm((char *) "N",(char *) "T", &norb2, &norb2, &nb, 
    //            &alpha, dd->d_buf1, &norb2, dd->d_buf2, &norb2,
    //            &beta, dd->d_tdm2, &norb2); //convert to gemm_batched, edit na loops to batches
    for (int i=0; i<num_buf_batches; i+=gemm_batch_size) {
      num_gemm_batches = _MIN(gemm_batch_size, num_buf_batches-i);
      ml->gemm_batch((char *) "N",(char *) "T", &norb2, &norb2, &nb, 
        &alpha, 
        &(dd->d_buf1[i*size_buf]), &norb2, &size_buf, 
        &(dd->d_buf2[i*size_buf]), &norb2, &size_buf, 
        &beta, dd->d_buf3, &norb2, &size_tdm2, &num_gemm_batches); 
      reduce_buf3_to_rdm(dd->d_buf3, dd->d_tdm2, size_tdm2, num_gemm_batches);
      }

    ml->memset(dd->d_buf1, &zero, &bits_buf); 
    ml->memset(dd->d_buf2, &zero, &bits_buf); 
  }     

  transpose_jikl(dd->d_tdm2, dd->d_buf1, norb);

  pm->dev_barrier();
  pm->dev_profile_stop();
  double t1 = omp_get_wtime();
  t_array[24] += t1 - t0;
  count_array[14]++;
}
/* ---------------------------------------------------------------------- */
void Device::compute_rdm12kern_sf_v2 (int na, int nb, int nlinka, int nlinkb, int norb)
{
  double t0 = omp_get_wtime();
  int id = 0;
  pm->dev_set_device(id);
  ml->set_handle(id);
  my_device_data * dd = &(device_data[id]);
  pm->dev_profile_start("tdms :: compute_tdm12kern_sf");
  int norb2 = norb*norb;
  int size_buf = norb2*nb;
  int size_tdm2 = norb2*norb2;
  int size_tdm1 = norb2;

  int zero = 0;
  int one = 1;
  const double alpha = 1.0;
  const double beta = 0.0;
  int bits_tdm1 = sizeof(double)*size_tdm1;
  int bits_tdm2 = sizeof(double)*size_tdm2;
  int _size_buf = _MAX(dd->size_buf1, dd->size_buf2);// (dd->size_buf1 > dd->size_buf2) ? dd->size_buf1 : dd->size_buf2;
  #ifdef _TEMP_BUFSIZING
  _size_buf = size_buf*6;
  #endif
  int final_size_buf = _MAX(_size_buf, size_buf);//(_size_buf > size_buf) ? _size_buf : size_buf;
  int buf_batch_size = final_size_buf/size_buf; //this is integer division // number of buf1/2 in a single buffer
  int gemm_batch_size = final_size_buf/(norb2*norb2); // this is integer division // number of tdm2 in a single buf
  int gemv_batch_size = final_size_buf/norb2; // this is integer division // number of tdm1 in a single buf
  int num_buf_batches; 
  int num_gemm_batches; 
  int num_gemv_batches; 
  grow_array(dd->d_buf1,final_size_buf, dd->size_buf1, "buf1", FLERR); 
  //grow_array(dd->d_buf2,final_size_buf, dd->size_buf2, "buf2", FLERR); 
  grow_array(dd->d_buf3,final_size_buf, dd->size_buf3, "buf3", FLERR); 
  int bits_buf = sizeof(double)*buf_batch_size*size_buf;
  ml->memset(dd->d_buf1, &zero, &bits_buf); 
  //ml->memset(dd->d_buf2, &zero, &bits_buf); 
  grow_array(dd->d_tdm1, size_tdm1, dd->size_tdm1, "tdm1", FLERR);
  grow_array(dd->d_tdm2, size_tdm2, dd->size_tdm2, "tdm2", FLERR); 
  ml->memset(dd->d_tdm1, &zero, &bits_tdm1);
  ml->memset(dd->d_tdm2, &zero, &bits_tdm2);
 

  for (int stra_id = 0; stra_id<na; stra_id += buf_batch_size){
  //for (int stra_id = 0; stra_id<na; ++stra_id) { 
    num_buf_batches = _MIN(buf_batch_size, na-stra_id);
    //compute_FCIrdm2_b_t1ci(dd->d_ciket, dd->d_buf1, stra_id, nb, norb, nlinkb, dd->d_clinkb); 
    //compute_FCIrdm2_a_t1ci(dd->d_ciket, dd->d_buf1, stra_id, nb, norb, nlinka, dd->d_clinka); 
    compute_FCIrdm2_a_t1ci_v2( dd->d_ciket, dd->d_buf1, stra_id, num_buf_batches, nb, norb, nlinka, dd->d_clinka); 
    compute_FCIrdm2_b_t1ci_v2( dd->d_ciket, dd->d_buf1, stra_id, num_buf_batches, nb, norb, nlinkb, dd->d_clinkb); 

    //ml->gemm((char *) "N",(char *) "T", &norb2, &norb2, &nb, 
    //            &alpha, dd->d_buf1, &norb2, dd->d_buf2, &norb2,
    //            &beta, dd->d_tdm2, &norb2); //convert to gemm_batched, edit na loops to batches
    for (int i=0; i<num_buf_batches; i+=gemm_batch_size) {
      num_gemm_batches = _MIN(gemm_batch_size, num_buf_batches-i);
      ml->gemm_batch((char *) "N",(char *) "T", &norb2, &norb2, &nb, 
        &alpha, 
        &(dd->d_buf1[i*size_buf]), &norb2, &size_buf, 
        &(dd->d_buf1[i*size_buf]), &norb2, &size_buf, 
        &beta, dd->d_buf3, &norb2, &size_tdm2, &num_gemm_batches); 
      reduce_buf3_to_rdm(dd->d_buf3, dd->d_tdm2, size_tdm2, num_gemm_batches);
      }

    for (int i=0; i<num_buf_batches; i+=gemv_batch_size){
      double * ketvec = &(dd->d_ciket[(stra_id+i)*nb]);
      num_gemv_batches = _MIN(gemv_batch_size, num_buf_batches-i);
      ml->gemv_batch((char *) "N", &norb2, &nb,
          &alpha, &(dd->d_buf1[i*size_buf]), &norb2, &size_buf,
          ketvec, &one, &nb, 
          &beta, dd->d_buf3, &one, &size_tdm1,
          &num_gemv_batches);
      reduce_buf3_to_rdm(dd->d_buf3, dd->d_tdm1, size_tdm1, num_gemv_batches);
      }

    ml->memset(dd->d_buf1, &zero, &bits_buf); 
    ml->memset(dd->d_buf2, &zero, &bits_buf); 
  }     

  transpose_jikl(dd->d_tdm2, dd->d_buf1, norb);

  pm->dev_barrier();
  pm->dev_profile_stop();
  double t1 = omp_get_wtime();
  t_array[25] += t1 - t0;
  count_array[15]++;
}
/* ---------------------------------------------------------------------- */
void Device::compute_tdm13h_spin_v4(int na, int nb, 
                                 int nlinka, int nlinkb, 
                                 int norb, int spin, int _reorder,
                                 int ia_bra, int ja_bra, int ib_bra, int jb_bra, int sgn_bra, 
                                 int ia_ket, int ja_ket, int ib_ket, int jb_ket, int sgn_ket )
{
  //na, nb is same for both zero-padded ci vectors, but not necessarily for non padded vectors
  double t0 = omp_get_wtime();
  int id = 0;
  pm->dev_set_device(id);
  ml->set_handle(id);
  my_device_data * dd = &(device_data[id]);

  int na_bra = ja_bra - ia_bra;
  int nb_bra = jb_bra - ib_bra;
  int na_ket = ja_ket - ia_ket;
  int nb_ket = jb_ket - ib_ket;

  int norb2 = norb*norb;
  int size_buf = norb2*nb;
  int size_tdm3h = norb2*norb2;
  int size_tdm1h = norb2;

  int zero = 0;
  int one = 1;
  const double alpha = 1.0*sgn_bra*sgn_ket;
  const double beta = 1.0;
  int bits_buf = sizeof(double)*size_buf;
  int bits_nbket = sizeof(double)*nb_ket*norb2;
  int bits_tdm1h = sizeof(double)*size_tdm1h;
  int bits_tdm3h = sizeof(double)*size_tdm3h;
  grow_array(dd->d_tdm1, size_tdm1h, dd->size_tdm1, "tdm1", FLERR);
  grow_array(dd->d_tdm2, size_tdm3h, dd->size_tdm2, "tdm2", FLERR); 
  grow_array(dd->d_tdm2_p, size_tdm3h, dd->size_tdm2_p, "tdm2_p", FLERR); 
  grow_array(dd->d_buf1,size_buf, dd->size_buf1, "buf1", FLERR); 
  grow_array(dd->d_buf2,size_buf, dd->size_buf2, "buf2", FLERR); 
  dd->d_tdm1h = dd->d_tdm1;
  ml->memset(dd->d_buf1, &zero, &bits_buf); 
  ml->memset(dd->d_buf2, &zero, &bits_buf); 
  ml->memset(dd->d_tdm1, &zero, &bits_tdm1h);
  ml->memset(dd->d_tdm2, &zero, &bits_tdm3h);
  ml->memset(dd->d_tdm2_p, &zero, &bits_tdm3h);
  dd->d_tdm3ha = dd->d_tdm2;
  dd->d_tdm3hb = dd->d_tdm2_p;

  /*
  tdm12kern_a
    a_t1ci: cibra, clinka -> buf2
    a_t1ci: ciket, clinka -> buf1
    tdm1 = gemv buf1, bravec
    tdm2 = gemm buf1, buf2
  tdm12kern_b 
    b_t1ci: cibra, clinkb -> buf2
    b_t1ci: ciket, clinkb -> buf1
    tdm1 = gemv buf1, bravec
    tdm2 = gemm buf1, buf2
  tdm12kern_ab
    a_t1ci: cibra, clinka -> buf2
    b_t1ci: ciket, clinkb -> buf1
    tdm2 = gemm buf1, buf2

  if spin ==0  
    tdm1, tdm3ha = tdm12kern_a, cibra, ciket, get 1 and 2
      a_t1ci: cibra, clinka -> buf2
      a_t1ci: ciket, clinka -> buf1
      tdm1h = gemv buf1, bravec
      tdm3ha = gemm buf1, buf2
    tdm3hb = tdm12kern_ab, cibra, ciket, get 2
      a_t1ci: cibra, clinka -> buf2  //same
      b_t1ci: ciket, clinkb -> buf1  
      tdm3hb = gemm buf1, buf2
      
  if spin ==1
    tdm1, tdm3hb = tdm12kern_b, cibra, ciket, get 1 and 2
      b_t1ci: cibra, clinkb -> buf2
      b_t1ci: ciket, clinkb -> buf1
      tdm1h = gemv buf1, bravec
      tdm3hb = gemm buf1, buf2
    tdm3ha = tdm12kern_ab, ciket, cibra, get 2
      // !caution ciket and cibra switched
      a_t1ci: ciket, clinka -> buf2
      b_t1ci: cibra, clinkb -> buf1 
      tdm3ha = gemm buf1, buf2
      //therefore
      b_t1ci: cibra, clinkb -> buf2 //doesn't matter where you store in 
      a_t1ci: ciket, clinka -> buf1
      tdm3ha = gemm buf2, buf1

  ci is zero except for [ia:ja, ib:jb] for both bra and ket. in v3, the full ci won't be passed, only non zero elements
  */
  if (spin){
    for (int stra_id = ia_bra; stra_id<ja_bra; ++stra_id){
    
      //buf2 is 0, so the whole thing is meaningless. tdm1 uses buf1 and bravec = cibra[stra_id, :]
        compute_FCIrdm3h_b_t1ci_v2(dd->d_cibra, dd->d_buf2, stra_id, nb, nb_bra, norb, nlinkb, ia_bra, ja_bra, ib_bra, jb_bra, dd->d_clinkb);
        if ((stra_id >= ia_ket) && (stra_id < ja_ket)) {
          //buf1 is 0, so tdm3hb and tdm1hb don't calculate anything
        
          compute_FCIrdm3h_b_t1ci_v2(dd->d_ciket, dd->d_buf1, stra_id, nb, nb_ket, norb, nlinkb, ia_ket, ja_ket, ib_ket, jb_ket, dd->d_clinkb);

          ml->gemm((char *) "N", (char *) "T", &norb2, &norb2, &nb, &alpha, 
                dd->d_buf1, &norb2, dd->d_buf2, &norb2, 
                &beta, dd->d_tdm3hb, &norb2);
          double * bravec = &(dd->d_cibra[(stra_id-ia_bra)*nb_bra]);
          ml->gemv((char *) "N", &norb2, &nb_bra, &alpha, 
                &(dd->d_buf1[ib_bra*norb2]), &norb2, bravec, &one, 
                &beta, dd->d_tdm1h, &one);
          ml->memset(dd->d_buf1, &zero, &bits_buf);
          }
        compute_FCIrdm3h_a_t1ci_v2(dd->d_ciket, dd->d_buf1, stra_id, nb_ket, norb, nlinka, ia_ket, ja_ket, ib_ket, jb_ket, dd->d_clinka);
        // buf1 is only populated from ib_ket:jb_ket, so don't need to run the multiplication over the whole thing 
        ml->gemm((char *) "N", (char *) "T", &norb2, &norb2, &nb_ket, &alpha, 
               &(dd->d_buf2[ib_ket*norb2]), &norb2, &(dd->d_buf1[ib_ket*norb2]), &norb2, //remember the switch?
               &beta, dd->d_tdm3ha, &norb2);
        ml->memset(dd->d_buf2, &zero, &bits_buf);
        ml->memset(&(dd->d_buf1[ib_ket*norb2]), &zero, &bits_nbket);
      }
  }
  else {
    int ib_max = (ib_bra > ib_ket) ? ib_bra : ib_ket;
    int jb_min = (jb_bra < jb_ket) ? jb_bra : jb_ket;
    int b_len  = jb_min - ib_max;

    for (int stra_id = 0; stra_id<na; ++stra_id){
        /* buf2      buf1              tdm2      bravec  
          0 0 0 0   0 0 0 0          # # # #     0  
  ib_bra  # # # #   0 0 0 0          # # # #     # ib_bra
          # # # #   # # # # ib_ket   # # # #     #
  jb_bra  # # # #   # # # #          # # # #     # jb_bra
          0 0 0 0   # # # # jb_ket               0 
          0 0 0 0   0 0 0 0                      0  
          
          given buf2, don't need to calculate from all ib_ket to jb_ket for buf1, can only do max(ib_bra, ib_ket) to min(jb_bra, jb_ket)
          gemm can also just go over the same limits.
          gemv calculation can also be reduced
        */

      compute_FCIrdm3h_a_t1ci_v2(dd->d_cibra, dd->d_buf2, stra_id, nb_bra, norb, nlinka, ia_bra, ja_bra, ib_bra, jb_bra, dd->d_clinka);
      if (b_len>0){
        compute_FCIrdm3h_a_t1ci_v2(dd->d_ciket, dd->d_buf1, stra_id, nb_ket, norb, nlinka, ia_ket, ja_ket, ib_max, jb_min, dd->d_clinka);// !limits

        ml->gemm((char *) "N", (char *) "T", &norb2, &norb2, &b_len, &alpha, 
                &(dd->d_buf1[ib_max*norb2]), &norb2, &(dd->d_buf2[ib_max*norb2]), &norb2, 
                &beta, dd->d_tdm3ha, &norb2);
        if ((stra_id >= ia_bra) && (stra_id < ja_bra)){
          double * bravec = &(dd->d_cibra[(stra_id-ia_bra)*nb_bra]);
          ml->gemv((char *) "N", &norb2, &nb_bra, &alpha, 
                &(dd->d_buf1[ib_bra*nb]), &norb2, bravec, &one, 
                &beta, dd->d_tdm1h, &one);
        }
      }

      if ((stra_id>=ia_ket) && (stra_id<ja_ket)){

      ml->memset(dd->d_buf1, &zero, &bits_buf); // can be optimized
 
      //when populated, rdm3h_b has the capability to populate the entire matrix, but buf2 is still blocked zero from a
      //can rdm3h_b take in what should be range of str0 (nb) because we are only need a specific range here (ib_bra -> jb_bra)
      //similar to the plot above of rdm3h_a * rdm3h_b, but buf1 is fully filled. 
      compute_FCIrdm3h_b_t1ci_v2(dd->d_ciket, dd->d_buf1, stra_id, nb, nb_bra, norb, nlinkb, ia_ket, ja_ket, ib_ket, jb_ket, dd->d_clinkb); 
      ml->gemm((char *) "N", (char *) "T", &norb2, &norb2, &nb_bra, &alpha, 
               &(dd->d_buf1[ib_bra*norb2]),&norb2, &(dd->d_buf2[ib_bra*norb2]), &norb2, 
               &beta, dd->d_tdm3hb, &norb2);
      }
      ml->memset(dd->d_buf2, &zero, &bits_buf); //can be optimized based
      ml->memset(dd->d_buf1, &zero, &bits_buf); 
    }
  }
  transpose_jikl(dd->d_tdm3ha, dd->d_buf1, norb);
  transpose_jikl(dd->d_tdm3hb, dd->d_buf2, norb);
  #ifdef _ENABLE_REORDER
  if (_reorder){
    if (spin) 
      {reorder(dd->d_tdm1h, dd->d_tdm3hb, dd->d_buf1, norb);}
    else
      {reorder(dd->d_tdm1h, dd->d_tdm3ha, dd->d_buf1, norb);}
  }
  #endif

  pm->dev_barrier();
  pm->dev_profile_stop();
  double t1 = omp_get_wtime();
  t_array[26] += t1-t0;//TODO: fix this
  count_array[16]++;//TODO: fix this
} 
/* ---------------------------------------------------------------------- */
void Device::compute_tdm13h_spin_v5(int na, int nb, 
                                 int nlinka, int nlinkb, 
                                 int norb, int spin, int _reorder,
                                 int ia_bra, int ja_bra, int ib_bra, int jb_bra, int sgn_bra, 
                                 int ia_ket, int ja_ket, int ib_ket, int jb_ket, int sgn_ket )
{
  #if 0
  //na, nb is same for both zero-padded ci vectors, but not necessarily for non padded vectors
  double t0 = omp_get_wtime();
  int id = 0;
  pm->dev_set_device(id);
  ml->set_handle(id);
  my_device_data * dd = &(device_data[id]);

  int na_bra = ja_bra - ia_bra;
  int nb_bra = jb_bra - ib_bra;
  int na_ket = ja_ket - ia_ket;
  int nb_ket = jb_ket - ib_ket;
  int zero = 0;
  int one = 1;
  const double alpha = 1.0*sgn_bra*sgn_ket;
  const double beta = 0.0;
  int bits_tdm1 = sizeof(double)*size_tdm1;
  int bits_tdm2 = sizeof(double)*size_tdm2;
  int _size_buf = _MAX(dd->size_buf1, dd->size_buf2);// (dd->size_buf1 > dd->size_buf2) ? dd->size_buf1 : dd->size_buf2;
  #ifdef _TEMP_BUFSIZING
  _size_buf = size_buf*6;
  #endif
  int final_size_buf = _MAX(_size_buf, size_buf);//(_size_buf > size_buf) ? _size_buf : size_buf;
  int buf_batch_size = final_size_buf/size_buf; //this is integer division // number of buf1/2 in a single buffer
  int gemm_batch_size = final_size_buf/(norb2*norb2); // this is integer division // number of tdm2 in a single buf
  int gemv_batch_size = final_size_buf/norb2; // this is integer division // number of tdm1 in a single buf
  int num_buf_batches; 
  int num_gemm_batches; 
  int num_gemv_batches; 
  grow_array(dd->d_buf1,final_size_buf, dd->size_buf1, "buf1", FLERR); 
  grow_array(dd->d_buf2,final_size_buf, dd->size_buf2, "buf2", FLERR); 
  grow_array(dd->d_buf3,final_size_buf, dd->size_buf3, "buf3", FLERR); 
  int bits_buf = sizeof(double)*buf_batch_size*size_buf;
  int bits_buf3;
  ml->memset(dd->d_buf1, &zero, &bits_buf); 
  ml->memset(dd->d_buf2, &zero, &bits_buf); 
  grow_array(dd->d_tdm1, size_tdm1, dd->size_tdm1, "tdm1", FLERR);
  grow_array(dd->d_tdm2, size_tdm2, dd->size_tdm2, "tdm2", FLERR); 
  ml->memset(dd->d_tdm1, &zero, &bits_tdm1);
  ml->memset(dd->d_tdm2, &zero, &bits_tdm2);
 
  /*
  tdm12kern_a
    a_t1ci: cibra, clinka -> buf2
    a_t1ci: ciket, clinka -> buf1
    tdm1 = gemv buf1, bravec
    tdm2 = gemm buf1, buf2
  tdm12kern_b 
    b_t1ci: cibra, clinkb -> buf2
    b_t1ci: ciket, clinkb -> buf1
    tdm1 = gemv buf1, bravec
    tdm2 = gemm buf1, buf2
  tdm12kern_ab
    a_t1ci: cibra, clinka -> buf2
    b_t1ci: ciket, clinkb -> buf1
    tdm2 = gemm buf1, buf2

  if spin ==0  
    tdm1, tdm3ha = tdm12kern_a, cibra, ciket, get 1 and 2
      a_t1ci: cibra, clinka -> buf2
      a_t1ci: ciket, clinka -> buf1
      tdm1h = gemv buf1, bravec
      tdm3ha = gemm buf1, buf2
    tdm3hb = tdm12kern_ab, cibra, ciket, get 2
      a_t1ci: cibra, clinka -> buf2  //same
      b_t1ci: ciket, clinkb -> buf1  
      tdm3hb = gemm buf1, buf2
      
  if spin ==1
    tdm1, tdm3hb = tdm12kern_b, cibra, ciket, get 1 and 2
      b_t1ci: cibra, clinkb -> buf2
      b_t1ci: ciket, clinkb -> buf1
      tdm1h = gemv buf1, bravec
      tdm3hb = gemm buf1, buf2
    tdm3ha = tdm12kern_ab, ciket, cibra, get 2
      // !caution ciket and cibra switched
      a_t1ci: ciket, clinka -> buf2
      b_t1ci: cibra, clinkb -> buf1 
      tdm3ha = gemm buf1, buf2
      //therefore
      b_t1ci: cibra, clinkb -> buf2 //doesn't matter where you store in 
      a_t1ci: ciket, clinka -> buf1
      tdm3ha = gemm buf2, buf1

  ci is zero except for [ia:ja, ib:jb] for both bra and ket. in v3, the full ci won't be passed, only non zero elements
  */
  if (spin){
    for (int stra_id = ia_bra; stra_id<ja_bra; stra_id+=buf_batch_size){
      num_buf_batches = _MIN(buf_batch_size, ja_bra-stra_id);
      //buf2 is 0, so the whole thing is meaningless. tdm1 uses buf1 and bravec = cibra[stra_id, :]
      //compute_FCIrdm3h_b_t1ci_v2(dd->d_cibra, dd->d_buf2, stra_id, nb, nb_bra, norb, nlinkb, ia_bra, ja_bra, ib_bra, jb_bra, dd->d_clinkb);
      compute_FCIrdm3h_b_t1ci_v3(dd->d_cibra, dd->d_buf2, stra_id, num_buf_batches, nb, nb_bra, norb, nlinkb, ia_bra, ja_bra, ib_bra, jb_bra, dd->d_clinkb);
      //if ((stra_id >= ia_ket) && (stra_id < ja_ket)) {  //I am going to not worry about it for now because it messes up gemm/v_batch
      //buf1 is 0, so tdm3hb and tdm1hb don't calculate anything
      //compute_FCIrdm3h_b_t1ci_v2(dd->d_ciket, dd->d_buf1, stra_id, nb, nb_ket, norb, nlinkb, ia_ket, ja_ket, ib_ket, jb_ket, dd->d_clinkb);
      compute_FCIrdm3h_b_t1ci_v3(dd->d_ciket, dd->d_buf1, stra_id, num_buf_batches, nb, nb_ket, norb, nlinkb, ia_ket, ja_ket, ib_ket, jb_ket, dd->d_clinkb);
      for (int i=0; i<num_buf_batches; i+=gemm_batch_size) {
        num_gemm_batches = _MIN(gemm_batch_size, num_buf_batches-i);
        ml->gemm_batch((char *) "N",(char *) "T", &norb2, &norb2, &nb, 
          &alpha, 
          &(dd->d_buf1[i*size_buf]), &norb2, &size_buf, 
          &(dd->d_buf2[i*size_buf]), &norb2, &size_buf, 
          &beta, dd->d_buf3, &norb2, &size_tdm2, &num_gemm_batches); 
        reduce_buf3_to_rdm(dd->d_buf3, dd->d_tdm3hb, size_tdm2, num_gemm_batches);
        }

      if ((stra_id + num_buf_batches >= ia_bra) && (stra_id < ia_bra)){
        num_buf_batches_for_gemv = stra_id+num_buf_batches - ia_bra;}
      else if ((stra_id<ja_bra)&&(stra_id+num_buf_batches >=ja_bra){
        num_buf_batches_for_gemv = ja_bra-stra_id;}
      else if ((stra_id>=ia_bra) && (stra_id+num_buf_batches <ja_bra)){
        num_buf_batches_for_gemv = num_buf_batches;}
      else num_buf_batches_for_gemv = 0;  

      for (int i=0; i<num_buf_batches_for_gemv; i+=gemv_batch_size){
        double * bravec = &(dd->d_cibra[(stra_id-ia_bra+i)*nb_bra]);//ia_bra, not ia_max
        num_gemv_batches = _MIN(gemv_batch_size, num_buf_batches_for_gemv-i);
        ml->gemv_batch((char *) "N", &norb2, &nb_bra, 
             &alpha, 
             &(dd->d_buf1[i*size_buf + ib_bra*nb]), &norb2, &size_buf,
             bravec, &one, &nb_bra, 
             &beta, dd->d_buf3, &one, &size_tdm1,
             &num_gemv_batches);
        reduce_buf3_to_rdm(dd->d_buf3, dd->d_tdm1h, size_tdm1, num_gemv_batches);
        }
       
        //double * bravec = &(dd->d_cibra[(stra_id-ia_bra)*nb_bra]);
        //ml->gemv((char *) "N", &norb2, &nb_bra, &alpha, 
        //        &(dd->d_buf1[ib_bra*norb2]), &norb2, bravec, &one, 
        //        &beta, dd->d_tdm1h, &one);
      ml->memset(dd->d_buf1, &zero, &bits_buf);
      compute_FCIrdm3h_a_t1ci_v3(dd->d_ciket, dd->d_buf1, stra_id, num_buf_batches, nb, nb_ket, norb, nlinka, ia_ket, ja_ket, ib_ket, jb_ket, dd->d_clinka);
      //compute_FCIrdm3h_a_t1ci_v2(dd->d_ciket, dd->d_buf1, stra_id, nb_ket, norb, nlinka, ia_ket, ja_ket, ib_ket, jb_ket, dd->d_clinka);
      // buf1 is only populated from ib_ket:jb_ket, so don't need to run the multiplication over the whole thing 

      for (int i=0; i<num_buf_batches; i+=gemm_batch_size) {
        num_gemm_batches = _MIN(gemm_batch_size, num_buf_batches-i);
        ml->gemm_batch((char *) "N",(char *) "T", &norb2, &norb2, &nb, 
          &alpha, 
          &(dd->d_buf1[i*size_buf]), &norb2, &size_buf, 
          &(dd->d_buf2[i*size_buf]), &norb2, &size_buf, 
          &beta, dd->d_buf3, &norb2, &size_tdm2, &num_gemm_batches); 
        reduce_buf3_to_rdm(dd->d_buf3, dd->d_tdm3ha, size_tdm2, num_gemm_batches);
        }
      //ml->gemm((char *) "N", (char *) "T", &norb2, &norb2, &nb_ket, &alpha, 
      //       &(dd->d_buf2[ib_ket*norb2]), &norb2, &(dd->d_buf1[ib_ket*norb2]), &norb2, //remember the switch?
      //       &beta, dd->d_tdm3ha, &norb2);
      ml->memset(dd->d_buf2, &zero, &bits_buf);
      ml->memset(dd->d_buf1, &zero, &bits_buf);
      }
  }
  else {
    int ib_max = (ib_bra > ib_ket) ? ib_bra : ib_ket;
    int jb_min = (jb_bra < jb_ket) ? jb_bra : jb_ket;
    int b_len  = jb_min - ib_max;

    for (int stra_id = 0; stra_id<na; ++stra_id){
        /* buf2      buf1              tdm2      bravec  
          0 0 0 0   0 0 0 0          # # # #     0  
  ib_bra  # # # #   0 0 0 0          # # # #     # ib_bra
          # # # #   # # # # ib_ket   # # # #     #
  jb_bra  # # # #   # # # #          # # # #     # jb_bra
          0 0 0 0   # # # # jb_ket               0 
          0 0 0 0   0 0 0 0                      0  
          
          given buf2, don't need to calculate from all ib_ket to jb_ket for buf1, can only do max(ib_bra, ib_ket) to min(jb_bra, jb_ket)
          gemm can also just go over the same limits.
          gemv calculation can also be reduced
        */

      compute_FCIrdm3h_a_t1ci_v3(dd->d_ciket, dd->d_buf1, stra_id, num_buf_batches, nb, nb_ket, norb, nlinka, ia_ket, ja_ket, ib_ket, jb_ket, dd->d_clinka);
      compute_FCIrdm3h_a_t1ci_v3(dd->d_cibra, dd->d_buf2, stra_id, num_buf_batches, nb, nb_bra, norb, nlinka, ia_bra, ja_bra, ib_bra, jb_bra, dd->d_clinka);
      //compute_FCIrdm3h_a_t1ci_v2(dd->d_cibra, dd->d_buf2, stra_id, nb_bra, norb, nlinka, ia_bra, ja_bra, ib_bra, jb_bra, dd->d_clinka);
      //if (b_len>0){
        //compute_FCIrdm3h_a_t1ci_v2(dd->d_ciket, dd->d_buf1, stra_id, nb_ket, norb, nlinka, ia_ket, ja_ket, ib_max, jb_min, dd->d_clinka);// !limits
      for (int i=0; i<num_buf_batches; i+=gemm_batch_size) {
        num_gemm_batches = _MIN(gemm_batch_size, num_buf_batches-i);
        ml->gemm_batch((char *) "N",(char *) "T", &norb2, &norb2, &nb, 
          &alpha, 
          &(dd->d_buf1[i*size_buf]), &norb2, &size_buf, 
          &(dd->d_buf2[i*size_buf]), &norb2, &size_buf, 
          &beta, dd->d_buf3, &norb2, &size_tdm2, &num_gemm_batches); 
        reduce_buf3_to_rdm(dd->d_buf3, dd->d_tdm3ha, size_tdm2, num_gemm_batches);
        }

      if ((stra_id + num_buf_batches >= ia_bra) && (stra_id < ia_bra)){
        num_buf_batches_for_gemv = stra_id+num_buf_batches - ia_bra;}
      else if ((stra_id<ja_bra)&&(stra_id+num_buf_batches >=ja_bra){
        num_buf_batches_for_gemv = ja_bra-stra_id;}
      else if ((stra_id>=ia_bra) && (stra_id+num_buf_batches <ja_bra)){
        num_buf_batches_for_gemv = num_buf_batches;}
      else num_buf_batches_for_gemv = 0;  

      for (int i=0; i<num_buf_batches_for_gemv; i+=gemv_batch_size){
        double * bravec = &(dd->d_cibra[(stra_id-ia_bra+i)*nb_bra]);//ia_bra, not ia_max
        num_gemv_batches = _MIN(gemv_batch_size, num_buf_batches_for_gemv-i);
        ml->gemv_batch((char *) "N", &norb2, &nb_bra, 
             &alpha, 
             &(dd->d_buf1[i*size_buf + ib_bra*nb]), &norb2, &size_buf,
             bravec, &one, &nb_bra, 
             &beta, dd->d_buf3, &one, &size_tdm1h,
             &num_gemv_batches);
        reduce_buf3_to_rdm(dd->d_buf3, dd->d_tdm1h, size_tdm1h, num_gemv_batches);
        }
  
        //ml->gemm((char *) "N", (char *) "T", &norb2, &norb2, &b_len, &alpha, 
        //        &(dd->d_buf1[ib_max*norb2]), &norb2, &(dd->d_buf2[ib_max*norb2]), &norb2, 
        //        &beta, dd->d_tdm3ha, &norb2);
        //if ((stra_id >= ia_bra) && (stra_id < ja_bra)){
        //  double * bravec = &(dd->d_cibra[(stra_id-ia_bra)*nb_bra]);
        //  ml->gemv((char *) "N", &norb2, &nb_bra, &alpha, 
        //        &(dd->d_buf1[ib_bra*nb]), &norb2, bravec, &one, 
        //        &beta, dd->d_tdm1h, &one);
        // }
      //}

      //if ((stra_id>=ia_ket) && (stra_id<ja_ket)){

      ml->memset(dd->d_buf1, &zero, &bits_buf); // can be optimized
 
      //when populated, rdm3h_b has the capability to populate the entire matrix, but buf2 is still blocked zero from a
      //can rdm3h_b take in what should be range of str0 (nb) because we are only need a specific range here (ib_bra -> jb_bra)
      //similar to the plot above of rdm3h_a * rdm3h_b, but buf1 is fully filled. 
      compute_FCIrdm3h_b_t1ci_v3(dd->d_ciket, dd->d_buf1, stra_id, num_buf_batches, nb, nb_ket, norb, nlinkb, ia_ket, ja_ket, ib_ket, jb_ket, dd->d_clinkb);
      //compute_FCIrdm3h_b_t1ci_v2(dd->d_ciket, dd->d_buf1, stra_id, nb, nb_ket, norb, nlinkb, ia_ket, ja_ket, ib_ket, jb_ket, dd->d_clinkb); 

      for (int i=0; i<num_buf_batches; i+=gemm_batch_size) {
        num_gemm_batches = _MIN(gemm_batch_size, num_buf_batches-i);
        ml->gemm_batch((char *) "N",(char *) "T", &norb2, &norb2, &nb, 
          &alpha, 
          &(dd->d_buf1[i*size_buf]), &norb2, &size_buf, 
          &(dd->d_buf2[i*size_buf]), &norb2, &size_buf, 
          &beta, dd->d_buf3, &norb2, &size_tdm2, &num_gemm_batches); 
        reduce_buf3_to_rdm(dd->d_buf3, dd->d_tdm3hb, size_tdm2, num_gemm_batches);
        }

      //ml->gemm((char *) "N", (char *) "T", &norb2, &norb2, &nb_bra, &alpha, 
      //         &(dd->d_buf1[ib_bra*norb2]),&norb2, &(dd->d_buf2[ib_bra*norb2]), &norb2, 
      //         &beta, dd->d_tdm3hb, &norb2);
      //}
      ml->memset(dd->d_buf2, &zero, &bits_buf); //can be optimized based
      ml->memset(dd->d_buf1, &zero, &bits_buf); 
    }
  }
  transpose_jikl(dd->d_tdm3ha, dd->d_buf1, norb);
  transpose_jikl(dd->d_tdm3hb, dd->d_buf2, norb);
  #ifdef _ENABLE_REORDER
  if (_reorder){
    if (spin) 
      {reorder(dd->d_tdm1h, dd->d_tdm3hb, dd->d_buf1, norb);}
    else
      {reorder(dd->d_tdm1h, dd->d_tdm3ha, dd->d_buf1, norb);}
  }
  #endif

  pm->dev_barrier();
  pm->dev_profile_stop();
  double t1 = omp_get_wtime();
  t_array[26] += t1-t0;//TODO: fix this
  count_array[16]++;//TODO: fix this
  #endif
} 


/* ---------------------------------------------------------------------- */
void Device::compute_tdmpp_spin_v2(int na, int nb, int nlinka, int nlinkb, int norb, int spin,
                                 int ia_bra, int ja_bra, int ib_bra, int jb_bra, int sgn_bra, 
                                 int ia_ket, int ja_ket, int ib_ket, int jb_ket, int sgn_ket )
{
  double t0 = omp_get_wtime();
  int id = 0;
  pm->dev_set_device(id);
  ml->set_handle(id);
  my_device_data * dd = &(device_data[id]);
  int norb2 = norb*norb;
  int size_buf = norb2*nb;
  int size_tdm2 = norb2*norb2;
  int size_tdm1 = norb2;

  int na_bra = ja_bra - ia_bra;
  int nb_bra = jb_bra - ib_bra;
  int na_ket = ja_ket - ia_ket;
  int nb_ket = jb_ket - ib_ket;
  int zero = 0;
  int one = 1;
  const double alpha = 1.0*sgn_bra*sgn_ket;
  const double beta = 1.0;
  int bits_buf = sizeof(double)*size_buf;
  int bits_tdm1 = sizeof(double)*size_tdm1;
  int bits_tdm2 = sizeof(double)*size_tdm2;
  grow_array(dd->d_tdm1, size_tdm1, dd->size_tdm1, "tdm1", FLERR);
  grow_array(dd->d_tdm2, size_tdm2, dd->size_tdm2, "tdm2", FLERR); 
  grow_array(dd->d_buf1, size_buf, dd->size_buf1, "buf1", FLERR); 
  grow_array(dd->d_buf2, size_buf, dd->size_buf2, "buf2", FLERR); 
  ml->memset(dd->d_buf1, &zero, &bits_buf); 
  ml->memset(dd->d_buf2, &zero, &bits_buf); 
  ml->memset(dd->d_tdm1, &zero, &bits_tdm1);
  ml->memset(dd->d_tdm2, &zero, &bits_tdm2);
  
 /*
 tdm12kern_a
    a_t1ci: cibra, clinka -> buf2
    a_t1ci: ciket, clinka -> buf1
    tdm1 = gemv buf1, bravec
    tdm2 = gemm buf1, buf2
  tdm12kern_b 
    b_t1ci: cibra, clinkb -> buf2
    b_t1ci: ciket, clinkb -> buf1
    tdm1 = gemv buf1, bravec
    tdm2 = gemm buf1, buf2
  tdm12kern_ab
    a_t1ci: cibra, clinka -> buf2
    b_t1ci: ciket, clinkb -> buf1
    tdm2 = gemm buf1, buf2
   
  if spin == 0
    tdm1, tdm2 = tdm12kern_a, cibra, ciket, get 1 and 2
  if spin == 2
    tdm1, tdm2 = tdm12kern_b, cibra, ciket, get 1 and 2
  if spin == 1
    tdm1, tdm2 = tdm12kern_ab, cibra, ciket, get 2
  */
  // since the difference between tdmhh and tdm13h is that zeros are added twice, only sending in the largest number should be sufficient, right?
  if (spin== 0)
      //refer to diagram in tdm3h_spin_v4
      {
      int ib_max = (ib_bra > ib_ket) ? ib_bra : ib_ket;
      int jb_min = (jb_bra < jb_ket) ? jb_bra : jb_ket;
      int b_len  = jb_min - ib_max;
      double result = 0.0;
      double buf1 = 0.0;
      double buf2 = 0.0;
      for (int stra_id = 0; stra_id<na; ++stra_id){
        compute_FCIrdm3h_a_t1ci_v2(dd->d_cibra, dd->d_buf2, stra_id, nb_bra, norb, nlinka, ia_bra, ja_bra, ib_bra, jb_bra, dd->d_clinka);
        compute_FCIrdm3h_a_t1ci_v2(dd->d_ciket, dd->d_buf1, stra_id, nb_ket, norb, nlinka, ia_ket, ja_ket, ib_ket, jb_ket, dd->d_clinka);// !limits

          ml->gemm((char *) "N",(char *) "T", &norb2, &norb2, &nb, 
                &alpha,
                dd->d_buf1, &norb2,
                dd->d_buf2, &norb2,
                &beta,
                dd->d_tdm2, &norb2); //convert to gemm_batched, edit na loops to batches
        if ((stra_id >= ia_bra) && (stra_id < ja_bra)){
          double * bravec = &(dd->d_cibra[(stra_id-ia_bra)*nb_bra]);
          ml->gemv((char *) "N", &norb2, &nb_bra, &alpha, 
                &(dd->d_buf1[ib_bra*nb]), &norb2, bravec, &one, 
                &beta, dd->d_tdm1, &one);
          }
        ml->memset(dd->d_buf1, &zero, &bits_buf);
        ml->memset(dd->d_buf2, &zero, &bits_buf);
        }
      }
    else if (spin==1) 
      { 
        for (int stra_id = 0; stra_id<na; ++stra_id){
          if ((stra_id>=ia_ket) && (stra_id<ja_ket)){//buf1 is zero otherwise
            compute_FCIrdm3h_a_t1ci_v2(dd->d_cibra, dd->d_buf2, stra_id, nb_bra, norb, nlinka, ia_bra, ja_bra, ib_bra, jb_bra, dd->d_clinka);
            compute_FCIrdm3h_b_t1ci_v2(dd->d_ciket, dd->d_buf1, stra_id, nb,nb_ket, norb, nlinkb, ia_ket, ja_ket, ib_ket, jb_ket, dd->d_clinkb);
    
            ml->gemm((char *) "N",(char *) "T", &norb2, &norb2, &nb, 
                &alpha,
                dd->d_buf1, &norb2,
                dd->d_buf2, &norb2,
                &beta,
                dd->d_tdm2, &norb2); //convert to gemm_batched, edit na loops to batches
            ml->memset(dd->d_buf1, &zero, &bits_buf);
            ml->memset(dd->d_buf2, &zero, &bits_buf);
        } } }
    else if (spin==2)
       {
       for (int stra_id = 0; stra_id<na; ++stra_id){
         if ((stra_id>=ia_bra) && (stra_id>=ia_ket) && (stra_id<ja_bra) && (stra_id<ja_ket)){
           compute_FCIrdm3h_b_t1ci_v2(dd->d_cibra, dd->d_buf2, stra_id, nb, nb_bra, norb, nlinkb, ia_bra, ja_bra, ib_bra, jb_bra, dd->d_clinkb);
           compute_FCIrdm3h_b_t1ci_v2(dd->d_ciket, dd->d_buf1, stra_id, nb, nb_ket, norb, nlinkb, ia_ket, ja_ket, ib_ket, jb_ket, dd->d_clinkb);
    
           ml->gemm((char *) "N", (char *) "T", &norb2, &norb2, &nb, &alpha, 
                dd->d_buf1, &norb2, dd->d_buf2, &norb2, 
                &beta, dd->d_tdm2, &norb2);
          double * bravec = &(dd->d_cibra[(stra_id-ia_bra)*nb_bra]);
          ml->gemv((char *) "N", &norb2, &nb_bra, &alpha, 
                &(dd->d_buf1[ib_bra*nb]), &norb2, bravec, &one, 
                &beta, dd->d_tdm1, &one);
           ml->memset(dd->d_buf1, &zero, &bits_buf);
           ml->memset(dd->d_buf2, &zero, &bits_buf);
          } } }
  //ml->memset(dd->d_buf1, &zero, &bits_buf);
  transpose_jikl(dd->d_tdm2, dd->d_buf1, norb);

  #ifdef _ENABLE_REORDER
  if (spin!=1){
    //reorder(dd->d_tdm1, dd->d_tdm2, dd->d_buf1, norb);
  }
  #endif

  pm->dev_barrier();
  pm->dev_profile_stop();
  double t1 = omp_get_wtime();
  t_array[29] += t1-t0;//TODO: fix this
  count_array[19]++;//TODO: fix this
  
}
/* ---------------------------------------------------------------------- */
void Device::compute_tdmpp_spin_v3(int na, int nb, int nlinka, int nlinkb, int norb, int spin,
                                 int ia_bra, int ja_bra, int ib_bra, int jb_bra, int sgn_bra, 
                                 int ia_ket, int ja_ket, int ib_ket, int jb_ket, int sgn_ket )
{
  double t0 = omp_get_wtime();
  int id = 0;
  pm->dev_set_device(id);
  ml->set_handle(id);
  my_device_data * dd = &(device_data[id]);
  int norb2 = norb*norb;
  int size_buf = norb2*nb;
  int size_tdm2 = norb2*norb2;
  int size_tdm1 = norb2;

  int na_bra = ja_bra - ia_bra;
  int nb_bra = jb_bra - ib_bra;
  int na_ket = ja_ket - ia_ket;
  int nb_ket = jb_ket - ib_ket;
  int zero = 0;
  int one = 1;
  const double alpha = 1.0*sgn_bra*sgn_ket;
  const double beta = 0.0;
  int bits_tdm1 = sizeof(double)*size_tdm1;
  int bits_tdm2 = sizeof(double)*size_tdm2;
  int _size_buf = _MAX(dd->size_buf1, dd->size_buf2);// (dd->size_buf1 > dd->size_buf2) ? dd->size_buf1 : dd->size_buf2;
  #ifdef _TEMP_BUFSIZING
  _size_buf = size_buf*6;
  #endif
  int final_size_buf = _MAX(_size_buf, size_buf);//(_size_buf > size_buf) ? _size_buf : size_buf;
  int buf_batch_size = final_size_buf/size_buf; //this is integer division // number of buf1/2 in a single buffer
  int gemm_batch_size = final_size_buf/(norb2*norb2); // this is integer division // number of tdm2 in a single buf
  int gemv_batch_size = final_size_buf/norb2; // this is integer division // number of tdm1 in a single buf
  int num_buf_batches; 
  int num_buf_batches_for_gemv; 
  int num_gemm_batches; 
  int num_gemv_batches; 
  grow_array(dd->d_buf1,final_size_buf, dd->size_buf1, "buf1", FLERR); 
  grow_array(dd->d_buf2,final_size_buf, dd->size_buf2, "buf2", FLERR); 
  grow_array(dd->d_buf3,final_size_buf, dd->size_buf3, "buf3", FLERR); 
  int bits_buf = sizeof(double)*buf_batch_size*size_buf;
  int bits_buf3;
  ml->memset(dd->d_buf1, &zero, &bits_buf); 
  ml->memset(dd->d_buf2, &zero, &bits_buf); 
  grow_array(dd->d_tdm1, size_tdm1, dd->size_tdm1, "tdm1", FLERR);
  grow_array(dd->d_tdm2, size_tdm2, dd->size_tdm2, "tdm2", FLERR); 
  ml->memset(dd->d_tdm1, &zero, &bits_tdm1);
  ml->memset(dd->d_tdm2, &zero, &bits_tdm2);
  
 /*
 tdm12kern_a
    a_t1ci: cibra, clinka -> buf2
    a_t1ci: ciket, clinka -> buf1
    tdm1 = gemv buf1, bravec
    tdm2 = gemm buf1, buf2
  tdm12kern_b 
    b_t1ci: cibra, clinkb -> buf2
    b_t1ci: ciket, clinkb -> buf1
    tdm1 = gemv buf1, bravec
    tdm2 = gemm buf1, buf2
  tdm12kern_ab
    a_t1ci: cibra, clinka -> buf2
    b_t1ci: ciket, clinkb -> buf1
    tdm2 = gemm buf1, buf2
   
  if spin == 0
    tdm1, tdm2 = tdm12kern_a, cibra, ciket, get 1 and 2
  if spin == 2
    tdm1, tdm2 = tdm12kern_b, cibra, ciket, get 1 and 2
  if spin == 1
    tdm1, tdm2 = tdm12kern_ab, cibra, ciket, get 2
  */
  // since the difference between tdmhh and tdm13h is that zeros are added twice, only sending in the largest number should be sufficient, right?
  int ib_max = _MAX(ib_bra, ib_ket);
  int jb_min = _MIN(jb_bra, jb_ket);
  int ia_max = _MAX(ia_bra, ia_ket);
  int ja_min = _MIN(ja_bra, ja_ket);
  int b_len  = jb_min - ib_max;
  if (spin== 0)
      //refer to diagram in tdm3h_spin_v4
      {
      double result = 0.0;
      double buf1 = 0.0;
      double buf2 = 0.0;
      //for (int stra_id = 0; stra_id<na; ++stra_id){
      for (int stra_id = 0; stra_id<na; stra_id += buf_batch_size){
        num_buf_batches = _MIN(buf_batch_size, na-stra_id);//(buf_batch_size < ja_ket - stra_id) ? buf_batch_size : ja_ket - stra_id; 
        compute_FCIrdm3h_a_t1ci_v3(dd->d_cibra, dd->d_buf2, stra_id, num_buf_batches, nb, nb_bra, norb, nlinka, ia_bra, ja_bra, ib_bra, jb_bra, dd->d_clinka);
        compute_FCIrdm3h_a_t1ci_v3(dd->d_ciket, dd->d_buf1, stra_id, num_buf_batches, nb, nb_ket, norb, nlinka, ia_ket, ja_ket, ib_ket, jb_ket, dd->d_clinka);
        for (int i=0; i<num_buf_batches; i+=gemm_batch_size) {
          num_gemm_batches = _MIN(gemm_batch_size, num_buf_batches-i);
          ml->gemm_batch((char *) "N",(char *) "T", &norb2, &norb2, &nb, 
            &alpha, 
            &(dd->d_buf1[i*size_buf]), &norb2, &size_buf, 
            &(dd->d_buf2[i*size_buf]), &norb2, &size_buf, 
            &beta, dd->d_buf3, &norb2, &size_tdm2, &num_gemm_batches); 
          reduce_buf3_to_rdm(dd->d_buf3, dd->d_tdm2, size_tdm2, num_gemm_batches);
          }

        //if ((stra_id + num_buf_batches >= ia_bra) && (stra_id < ja_bra)){
        if ((stra_id + num_buf_batches >= ia_bra) && (stra_id < ia_bra)){
          num_buf_batches_for_gemv = stra_id+num_buf_batches - ia_bra;}
        else if ((stra_id<ja_bra)&&(stra_id+num_buf_batches >=ja_bra)){
          num_buf_batches_for_gemv = ja_bra-stra_id;}
        else if ((stra_id>=ia_bra) && (stra_id+num_buf_batches <ja_bra)){
          num_buf_batches_for_gemv = num_buf_batches;}
        else num_buf_batches_for_gemv = 0;  
        if (num_buf_batches_for_gemv){
          for (int i=0; i<num_buf_batches_for_gemv; i+=gemv_batch_size){
            double * bravec = &(dd->d_cibra[(stra_id-ia_bra+i)*nb_bra]);//ia_bra, not ia_max
            num_gemv_batches = _MIN(gemv_batch_size, num_buf_batches_for_gemv-i);
            ml->gemv_batch((char *) "N", &norb2, &nb_bra, 
                &alpha, 
                &(dd->d_buf1[i*size_buf + ib_bra*nb]), &norb2, &size_buf,
                bravec, &one, &nb_bra, 
                &beta, dd->d_buf3, &one, &size_tdm1,
                &num_gemv_batches);
            reduce_buf3_to_rdm(dd->d_buf3, dd->d_tdm1, size_tdm1, num_gemv_batches);
            }
          }
          ml->memset(dd->d_buf1, &zero, &bits_buf);
          ml->memset(dd->d_buf2, &zero, &bits_buf);
        }
      }
    else if (spin==1) { 
        for (int stra_id = ia_ket; stra_id<ja_ket; stra_id += buf_batch_size){
          num_buf_batches = _MIN(buf_batch_size, ja_ket-stra_id);
          compute_FCIrdm3h_a_t1ci_v3(dd->d_cibra, dd->d_buf2, stra_id, num_buf_batches, nb, nb_bra, norb, nlinka, ia_bra, ja_bra, ib_bra, jb_bra, dd->d_clinka);
          compute_FCIrdm3h_b_t1ci_v3(dd->d_ciket, dd->d_buf1, stra_id, num_buf_batches, nb, nb_ket, norb, nlinkb, ia_ket, ja_ket, ib_ket, jb_ket, dd->d_clinkb);

          for (int i=0; i<num_buf_batches; i+=gemm_batch_size) {
            num_gemm_batches = _MIN(gemm_batch_size, num_buf_batches-i);
            ml->gemm_batch((char *) "N",(char *) "T", &norb2, &norb2, &nb, 
              &alpha, 
              &(dd->d_buf1[i*size_buf]), &norb2, &size_buf, 
              &(dd->d_buf2[i*size_buf]), &norb2, &size_buf, 
              &beta, dd->d_buf3, &norb2, &size_tdm2, &num_gemm_batches); 
            reduce_buf3_to_rdm(dd->d_buf3, dd->d_tdm2, size_tdm2, num_gemm_batches);
            }
            ml->memset(dd->d_buf1, &zero, &bits_buf);
            ml->memset(dd->d_buf2, &zero, &bits_buf);
          }
        } 
    else if (spin==2){
       for (int stra_id = ia_max; stra_id<ja_min; stra_id += buf_batch_size){
         num_buf_batches = _MIN(buf_batch_size, ja_min-stra_id);
         compute_FCIrdm3h_b_t1ci_v3(dd->d_cibra, dd->d_buf2, stra_id, num_buf_batches, nb, nb_bra, norb, nlinkb, ia_bra, ja_bra, ib_bra, jb_bra, dd->d_clinkb);
         compute_FCIrdm3h_b_t1ci_v3(dd->d_ciket, dd->d_buf1, stra_id, num_buf_batches, nb, nb_ket, norb, nlinkb, ia_ket, ja_ket, ib_ket, jb_ket, dd->d_clinkb);
         
         for (int i=0; i<num_buf_batches; i+=gemm_batch_size) {
           num_gemm_batches = _MIN(gemm_batch_size, num_buf_batches-i);
           ml->gemm_batch((char *) "N",(char *) "T", &norb2, &norb2, &nb, 
             &alpha, 
             &(dd->d_buf1[i*size_buf]), &norb2, &size_buf, 
             &(dd->d_buf2[i*size_buf]), &norb2, &size_buf, 
             &beta, dd->d_buf3, &norb2, &size_tdm2, &num_gemm_batches); 
           reduce_buf3_to_rdm(dd->d_buf3, dd->d_tdm2, size_tdm2, num_gemm_batches);
         }
         
        if ((stra_id + num_buf_batches >= ia_bra) && (stra_id < ia_bra)){
          num_buf_batches_for_gemv = stra_id+num_buf_batches - ia_bra;}
        else if ((stra_id<ja_bra)&&(stra_id+num_buf_batches >=ja_bra)){
          num_buf_batches_for_gemv = ja_bra-stra_id;}
        else if ((stra_id>=ia_bra) && (stra_id+num_buf_batches <ja_bra)){
          num_buf_batches_for_gemv = num_buf_batches;}
        else num_buf_batches_for_gemv = 0;  
        if (num_buf_batches_for_gemv){
        for (int i=0; i<num_buf_batches_for_gemv; i+=gemv_batch_size){
          double * bravec = &(dd->d_cibra[(stra_id-ia_bra+i)*nb_bra]);//ia_bra, not ia_max
          num_gemv_batches = _MIN(gemv_batch_size, num_buf_batches_for_gemv-i);
          ml->gemv_batch((char *) "N", &norb2, &nb_bra, 
               &alpha, 
               &(dd->d_buf1[i*size_buf + ib_bra*nb]), &norb2, &size_buf,
               bravec, &one, &nb_bra, 
               &beta, dd->d_buf3, &one, &size_tdm1,
               &num_gemv_batches);
          reduce_buf3_to_rdm(dd->d_buf3, dd->d_tdm1, size_tdm1, num_gemv_batches);
          }
        }
         ml->memset(dd->d_buf1, &zero, &bits_buf);
         ml->memset(dd->d_buf2, &zero, &bits_buf);
         } 
       }
  transpose_jikl(dd->d_tdm2, dd->d_buf1, norb);

  #ifdef _ENABLE_REORDER
  if (spin!=1){
    //reorder(dd->d_tdm1, dd->d_tdm2, dd->d_buf1, norb);
  }
  #endif

  pm->dev_barrier();
  pm->dev_profile_stop();
  double t1 = omp_get_wtime();
  t_array[29] += t1-t0;//TODO: fix this
  count_array[19]++;//TODO: fix this
  
}

/* ---------------------------------------------------------------------- */
void Device::compute_sfudm(int na, int nb, int nlinka, int nlinkb, int norb, 
                             int ia_bra, int ja_bra, int ib_bra, int jb_bra, int sgn_bra, 
                             int ia_ket, int ja_ket, int ib_ket, int jb_ket, int sgn_ket )
{
  double t0 = omp_get_wtime();
  int id = 0;
  pm->dev_set_device(id);
  ml->set_handle(id);
  my_device_data * dd = &(device_data[id]);
  int norb2 = norb*norb;
  int size_buf = norb2*nb;
  int size_tdm2 = norb2*norb2;
  int size_tdm1 = norb2;
  int zero = 0;
  int one = 1;
  //const double alpha = 1.0;
  const double alpha = 1.0*sgn_bra*sgn_ket;
  printf("alpha: %f\n",alpha);
  const double beta = 1.0;
  int na_bra = ja_bra - ia_bra;
  int nb_bra = jb_bra - ib_bra;
  int na_ket = ja_ket - ia_ket;
  int nb_ket = jb_ket - ib_ket;
  int bits_buf = sizeof(double)*size_buf;
  int bits_tdm2 = sizeof(double)*size_tdm2;
  grow_array(dd->d_tdm2, size_tdm2, dd->size_tdm2, "tdm2", FLERR); 
  grow_array(dd->d_buf1,size_buf, dd->size_buf1, "buf1", FLERR); 
  grow_array(dd->d_buf2,size_buf, dd->size_buf2, "buf2", FLERR); 
  ml->memset(dd->d_buf1, &zero, &bits_buf); 
  ml->memset(dd->d_buf2, &zero, &bits_buf); 
  ml->memset(dd->d_tdm2, &zero, &bits_tdm2);
  
  /*
  tdm12kern_ab
    a_t1ci: cibra, clinka -> buf2
    b_t1ci: ciket, clinkb -> buf1
    tdm2 = gemm buf1, buf2
  */
  int bra_b_len = jb_bra - ib_bra;
  for (int stra_id = 0; stra_id<na; ++stra_id){
    if ((stra_id>=ia_ket) && (stra_id<ja_ket)){//buf1 is zero otherwise

      compute_FCIrdm3h_a_t1ci_v2(dd->d_cibra, dd->d_buf2, stra_id, nb_bra, norb, nlinka, ia_bra, ja_bra, ib_bra, jb_bra, dd->d_clinka);
      compute_FCIrdm3h_b_t1ci_v2(dd->d_ciket, dd->d_buf1, stra_id, nb,nb_ket, norb, nlinkb, ia_ket, ja_ket, ib_ket, jb_ket, dd->d_clinkb);
      ml->gemm((char *) "N",(char *) "T", &norb2, &norb2, &nb, 
        &alpha,
        dd->d_buf1, &norb2,
        dd->d_buf2, &norb2,
        &beta,
        dd->d_tdm2, &norb2); //convert to gemm_batched, edit na loops to batches


      ml->memset(dd->d_buf1, &zero, &bits_buf);
      ml->memset(dd->d_buf2, &zero, &bits_buf);
    }
  }
  transpose_jikl(dd->d_tdm2, dd->d_buf1, norb);

  pm->dev_barrier();
  pm->dev_profile_stop();
  double t1 = omp_get_wtime();
  t_array[28] += t1-t0;//TODO: fix this
  count_array[18]++;//TODO: fix this
}
/* ---------------------------------------------------------------------- */
void Device::compute_sfudm_v2(int na, int nb, int nlinka, int nlinkb, int norb, 
                             int ia_bra, int ja_bra, int ib_bra, int jb_bra, int sgn_bra, 
                             int ia_ket, int ja_ket, int ib_ket, int jb_ket, int sgn_ket )
{
  double t0 = omp_get_wtime();
  int id = 0;
  pm->dev_set_device(id);
  ml->set_handle(id);
  my_device_data * dd = &(device_data[id]);
  int norb2 = norb*norb;
  int size_buf = norb2*nb;
  int size_tdm2 = norb2*norb2;
  int size_tdm1 = norb2;
  int zero = 0;
  int one = 1;
  //const double alpha = 1.0;
  const double alpha = 1.0*sgn_bra*sgn_ket;
  const double beta = 0.0;
  int na_bra = ja_bra - ia_bra;
  int nb_bra = jb_bra - ib_bra;
  int na_ket = ja_ket - ia_ket;
  int nb_ket = jb_ket - ib_ket;
  int bits_tdm2 = sizeof(double)*size_tdm2;
  int _size_buf = (dd->size_buf1 > dd->size_buf2) ? dd->size_buf1 : dd->size_buf2;
  #ifdef _TEMP_BUFSIZING
  _size_buf = size_buf*6;
  #endif
  int final_size_buf = (_size_buf > size_buf) ? _size_buf : size_buf;
  int buf_batch_size = final_size_buf/size_buf; //this is integer division // number of buf1/2 in a single buffer
  int gemm_batch_size = final_size_buf/(norb2*norb2); // this is integer division // number of tdm2 in a single buf
  int num_buf_batches; 
  int num_gemm_batches; 
  grow_array(dd->d_buf1,final_size_buf, dd->size_buf1, "buf1", FLERR); 
  grow_array(dd->d_buf2,final_size_buf, dd->size_buf2, "buf2", FLERR); 
  grow_array(dd->d_buf3,final_size_buf, dd->size_buf3, "buf3", FLERR); 
  int bits_buf = sizeof(double)*buf_batch_size*size_buf;
  int bits_buf3;
  printf("total_size: %i nb_ket: %i\n", final_size_buf, nb_ket);
  grow_array(dd->d_tdm2, size_tdm2, dd->size_tdm2, "tdm2", FLERR); 
  ml->memset(dd->d_buf1, &zero, &bits_buf); 
  ml->memset(dd->d_buf2, &zero, &bits_buf); 
  ml->memset(dd->d_tdm2, &zero, &bits_tdm2);
  
  /*
  tdm12kern_ab
    a_t1ci: cibra, clinka -> buf2
    b_t1ci: ciket, clinkb -> buf1
    tdm2 = gemm buf1, buf2
  */
  int bra_b_len = jb_bra - ib_bra;
  for (int stra_id = ia_ket; stra_id<ja_ket; stra_id += buf_batch_size){
      num_buf_batches = (buf_batch_size < ja_ket - stra_id) ? buf_batch_size : ja_ket - stra_id; 
      compute_FCIrdm3h_a_t1ci_v3(dd->d_cibra, dd->d_buf2, stra_id, num_buf_batches, nb, nb_bra, norb, nlinka, ia_bra, ja_bra, ib_bra, jb_bra, dd->d_clinka);
      compute_FCIrdm3h_b_t1ci_v3(dd->d_ciket, dd->d_buf1, stra_id, num_buf_batches, nb, nb_ket, norb, nlinkb, ia_ket, ja_ket, ib_ket, jb_ket, dd->d_clinkb);

      for (int i=0; i<num_buf_batches; i+=gemm_batch_size) {
        num_gemm_batches = (gemm_batch_size < num_buf_batches - i) ? gemm_batch_size : num_buf_batches - i;
        printf("stra_id: %i num_buf_batches: %i num_gemm_batches: %i norb2: %i nb: %i\n",stra_id, num_buf_batches, num_gemm_batches, norb2, nb);
        ml->gemm_batch((char *) "N",(char *) "T", &norb2, &norb2, &nb, 
          &alpha, 
          &(dd->d_buf1[i*size_buf]), &norb2, &size_buf, 
          &(dd->d_buf2[i*size_buf]), &norb2, &size_buf, 
          &beta, dd->d_buf3, &norb2, &size_tdm2, &num_gemm_batches); 
       reduce_buf3_to_rdm(dd->d_buf3, dd->d_tdm2, size_tdm2, num_gemm_batches);
      }
      

      ml->memset(dd->d_buf1, &zero, &bits_buf);
      ml->memset(dd->d_buf2, &zero, &bits_buf);
    }
  transpose_jikl(dd->d_tdm2, dd->d_buf1, norb);

  pm->dev_barrier();
  pm->dev_profile_stop();
  double t1 = omp_get_wtime();
  t_array[28] += t1-t0;//TODO: fix this
  count_array[18]++;//TODO: fix this
}

/* ---------------------------------------------------------------------- */
void Device::compute_tdm1h_spin( int na, int nb, int nlinka, int nlinkb, int norb, int spin, 
                             int ia_bra, int ja_bra, int ib_bra, int jb_bra, int sgn_bra, 
                             int ia_ket, int ja_ket, int ib_ket, int jb_ket, int sgn_ket )
{
  double t0 = omp_get_wtime();
  int id = 0;
  pm->dev_set_device(id);
  ml->set_handle(id);
  my_device_data * dd = &(device_data[id]);
  int norb2 = norb*norb;
  int size_tdm1 = norb2;

  grow_array(dd->d_tdm1,size_tdm1, dd->size_tdm1, "tdm1", FLERR); //actual returned
  set_to_zero(dd->d_tdm1, size_tdm1);
  /* 
     spin = 0: 
       trans_rdm1a: cibra, ciket -> tdm1
     spin = 1:
       trans_rdm1b: cibra, ciket -> tdm1
  */
  if (spin==0)
  {
    compute_FCItrans_rdm1a_v2 (dd->d_cibra, dd->d_ciket, dd->d_tdm1, 
                                norb, nlinka, 
                                ia_bra, ja_bra, ib_bra, jb_bra, 
                                ia_ket, ja_ket, ib_ket, jb_ket, sgn_bra*sgn_ket,  
                                dd->d_clinka);
  }
  else
  {
    compute_FCItrans_rdm1b_v2(dd->d_cibra, dd->d_ciket, dd->d_tdm1,  
                                norb, nlinkb, 
                                ia_bra, ja_bra, ib_bra, jb_bra, 
                                ia_ket, ja_ket, ib_ket, jb_ket, sgn_bra*sgn_ket,  
                                dd->d_clinkb);
  }

  pm->dev_barrier();
  pm->dev_profile_stop();
  double t1 = omp_get_wtime();
  t_array[27] += t1 - t0;
  count_array[17]++;
}
/* ---------------------------------------------------------------------- */
void Device::pull_tdm1(py::array_t<double> _tdm1, int norb)
{
  double t0 = omp_get_wtime();
  int id = 0;
  pm->dev_set_device(id); 
  my_device_data * dd = &(device_data[id]);
  pm->dev_profile_start("tdms :: pull tdm1");
  py::buffer_info info_tdm1 = _tdm1.request(); //2D array (norb, norb)
  double * tdm1 = static_cast<double*>(info_tdm1.ptr);
  pm->dev_pull_async(dd->d_tdm1, tdm1, norb*norb*sizeof(double));

  pm->dev_barrier();
  pm->dev_profile_stop();
  double t1 = omp_get_wtime();
  t_array[30] += t1-t0;
  count_array[20]++;

}
/* ---------------------------------------------------------------------- */
void Device::pull_tdm2(py::array_t<double> _tdm2, int norb)
{
  double t0 = omp_get_wtime();
  int id = 0;
  pm->dev_set_device(id); 
  my_device_data * dd = &(device_data[id]);
  pm->dev_profile_start("tdms :: pull tdm2");
  py::buffer_info info_tdm2 = _tdm2.request(); //4D array (norb, norb, norb, norb)
  double * tdm2 = static_cast<double*>(info_tdm2.ptr);
  pm->dev_pull_async(dd->d_tdm2, tdm2, norb*norb*norb*norb*sizeof(double));

  pm->dev_barrier();
  pm->dev_profile_stop();
  double t1 = omp_get_wtime();
  t_array[31] += t1-t0;
  count_array[21]++;
}
/* ---------------------------------------------------------------------- */
void Device::pull_tdm3hab(py::array_t<double> _tdm3ha, py::array_t<double> _tdm3hb, int norb)
{
  double t0 = omp_get_wtime();
  int id = 0;
  pm->dev_set_device(id); 
  my_device_data * dd = &(device_data[id]);
  pm->dev_profile_start("tdms :: pull tdm2");
  py::buffer_info info_tdm3ha = _tdm3ha.request(); //4D array (norb, norb, norb, norb)
  double * tdm3ha = static_cast<double*>(info_tdm3ha.ptr);
  pm->dev_pull_async(dd->d_tdm3ha, tdm3ha, norb*norb*norb*norb*sizeof(double));
  py::buffer_info info_tdm3hb = _tdm3hb.request(); //4D array (norb, norb, norb, norb)
  double * tdm3hb = static_cast<double*>(info_tdm3hb.ptr);
  pm->dev_pull_async(dd->d_tdm3hb, tdm3hb, norb*norb*norb*norb*sizeof(double));

  pm->dev_barrier();
  pm->dev_profile_stop();
  double t1 = omp_get_wtime();
  t_array[32] += t1-t0;
  count_array[22]++;
}

/* ---------------------------------------------------------------------- */
