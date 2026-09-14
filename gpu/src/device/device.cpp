/* -*- c++ -*- */

#include <stdio.h>

#include "device.h"

#include <string>
#include <vector>
#include <algorithm>
#include <unistd.h>
#include <string.h>
#include <sched.h>
#define _MIN(A,B) (A<B)?A:B
#define _MAX(A,B) (A>B)?A:B
#define _SIZE_FCI_BATCHES 6

/* ---------------------------------------------------------------------- */

Device::Device()
{  
  pm = new PM();

  ml = new MATHLIB(pm);

  verbose_level = 0;
  
  rho = nullptr;
  //vj = nullptr;
  _vktmp = nullptr;

  _pdft = nullptr;
  _jk = nullptr;
  _impham = nullptr;
  _lassi = nullptr;
  _h2eff = nullptr;
  _ao2mo = nullptr;
  _fci = nullptr;
  _comm = nullptr;
  _cache = nullptr;
  _utils = nullptr;

  // DEPRECATED: legacy integral engine state (orbital_response/fdrv)
  //buf_fdrv = nullptr;

  //ao2mo (dead members, kept until ao2mo_v3 removal)
  size_fxpp = 0;//remove when ao2mo_v3 is running
  size_bufpa = 0;//remove when ao2mo_v4 is running

  pin_fxpp = nullptr;//remove when ao2mo_v3 is running
  pin_bufpa = nullptr;//remove when ao2mo_v4 is running

  num_threads = 1;
#pragma omp parallel
  num_threads = omp_get_num_threads();

  grid_size = _SIZE_GRID;
  block_size = _SIZE_BLOCK;

  num_devices = pm->dev_num_devices();
  
  device_data = new my_device_data[num_devices];

  for(int i=0; i<num_devices; ++i) {
    pm->dev_set_device(i);
    
    device_data[i].device_id = i;
    device_data[i].active = 0;
    
    device_data[i].jk.size_rho = 0;
    device_data[i].jk.size_vj = 0;
    device_data[i].jk.size_vk = 0;
    device_data[i].jk.size_buf1 = 0;
    device_data[i].jk.size_buf2 = 0;
    device_data[i].jk.size_buf3 = 0;
    device_data[i].jk.size_dms = 0;
    device_data[i].jk.size_dmtril = 0;
    device_data[i].size_ucas = 0;
    device_data[i].size_umat = 0;
    device_data[i].size_h2eff = 0;
    device_data[i].size_mo_coeff = 0;
    device_data[i].size_mo_cas = 0;
    device_data[i].h2eff.size_eri_unpacked = 0; // this variable is not needed
    device_data[i].ao2mo.size_j_pc = 0;
    device_data[i].ao2mo.size_k_pc = 0;
    device_data[i].ao2mo.size_bufd = 0;
    device_data[i].ao2mo.size_bufpa = 0;
    device_data[i].ao2mo.size_bufaa = 0;
    device_data[i].h2eff.size_eri_h2eff=0;
    //pdft
    device_data[i].pdft.size_mo_grid=0;
    device_data[i].pdft.size_ao_grid=0;
    device_data[i].pdft.size_buf_pdft=0;
    device_data[i].pdft.size_cascm2=0;
    device_data[i].pdft.size_Pi=0;
    device_data[i].jk.size_rho=0;
    //fci
    device_data[i].fci.size_clinka=0;
    device_data[i].fci.size_clinkb=0;
    device_data[i].fci.size_cibra=0;
    device_data[i].fci.size_ciket=0;
    device_data[i].fci.size_tdm1=0;
    device_data[i].fci.size_tdm2=0;
    device_data[i].fci.size_tdm2_p=0;
    //matvecs
    
    
    device_data[i].jk.d_rho = nullptr;
    device_data[i].jk.d_vj = nullptr;
    device_data[i].jk.d_buf1 = nullptr;
    device_data[i].jk.d_buf2 = nullptr;
    device_data[i].jk.d_buf3 = nullptr;
    device_data[i].jk.d_vkk = nullptr;
    device_data[i].jk.d_dms = nullptr;
    device_data[i].d_mo_coeff=nullptr;
    device_data[i].d_mo_cas=nullptr;
    device_data[i].jk.d_dmtril = nullptr;
    device_data[i].jk.d_dms = nullptr;
    device_data[i].d_mo_coeff=nullptr;
    device_data[i].d_mo_cas=nullptr;
    device_data[i].jk.d_dmtril = nullptr;
    device_data[i].d_ucas = nullptr;
    device_data[i].d_umat = nullptr;
    device_data[i].d_h2eff = nullptr;
    device_data[i].h2eff.d_eri_h2eff = nullptr; //for h2eff_df_v2
    
    device_data[i].fci.d_pumap_ptr = nullptr;
    
    device_data[i].ao2mo.d_j_pc = nullptr;
    device_data[i].ao2mo.d_k_pc = nullptr;
    device_data[i].ao2mo.d_bufd = nullptr;
    device_data[i].ao2mo.d_bufpa = nullptr;
    device_data[i].ao2mo.d_bufaa = nullptr;
    device_data[i].ao2mo.d_papa = nullptr;//initialized, but not allocated (used dd->jk.d_buf3)

    // pdft
    device_data[i].pdft.d_ao_grid=nullptr;
    device_data[i].pdft.d_mo_grid=nullptr;
    device_data[i].pdft.d_cascm2=nullptr;
    device_data[i].pdft.d_Pi=nullptr;
    device_data[i].pdft.d_buf_pdft1=nullptr;
    device_data[i].pdft.d_buf_pdft2=nullptr;
    //fci
    device_data[i].fci.d_clinka=nullptr;
    device_data[i].fci.d_clinkb=nullptr;
    device_data[i].fci.d_cibra=nullptr;
    device_data[i].fci.d_ciket=nullptr;
    device_data[i].fci.d_tdm1=nullptr;
    device_data[i].fci.d_tdm2=nullptr;
    device_data[i].fci.d_tdm2_p=nullptr;

#if defined (_USE_GPU)
    device_data[i].handle = nullptr;
    device_data[i].stream = nullptr;
#endif

    ml->create_handle();
  }

  // subdomains borrow shared infrastructure through the DeviceContext
  dev_ctx.pm = pm;
  dev_ctx.ml = ml;
  dev_ctx.num_devices = num_devices;
  dev_ctx.verbose_level = verbose_level;
  dev_ctx.grid_size = grid_size;
  dev_ctx.block_size = block_size;
  dev_ctx.device_data = device_data;

  _comm = new DeviceComm(dev_ctx);
  dev_ctx.comm = _comm;

  _cache = new DeviceCache(dev_ctx);
  dev_ctx.cache = _cache;

  _utils = new DeviceUtils(dev_ctx);
  dev_ctx.utils = _utils;

  _pdft = new DevicePdft(dev_ctx);
  _jk = new DeviceJk(dev_ctx);
  _impham = new DeviceImpham(dev_ctx);
  _lassi = new DeviceLassi(dev_ctx);
  _h2eff = new DeviceH2eff(dev_ctx);
  _ao2mo = new DeviceAo2mo(dev_ctx);
  _fci = new DeviceFci(dev_ctx);

  // check device connectivity

  int rank = 0;
  int peer_error = pm->dev_check_peer(rank, num_devices);
  if(!peer_error) pm->dev_enable_peer(rank, num_devices);
}

/* ---------------------------------------------------------------------- */

Device::~Device()
{
  if(verbose_level) printf("LIBGPU: destroying device\n");

  delete _impham;
  _impham = nullptr;

  delete _jk;
  _jk = nullptr;

  delete _pdft;
  _pdft = nullptr;

  delete _lassi;
  _lassi = nullptr;

  delete _h2eff;
  _h2eff = nullptr;

  delete _ao2mo;
  _ao2mo = nullptr;

  delete _fci;
  _fci = nullptr;

  delete _comm;
  _comm = nullptr;

  delete _cache;
  _cache = nullptr;

  delete _utils;
  _utils = nullptr;

  pm->dev_free_host(rho);
  //pm->dev_free_host(vj);
  pm->dev_free_host(_vktmp);

  // DEPRECATED: legacy integral engine state (orbital_response/fdrv)
  //pm->dev_free_host(buf_fdrv);
  
  pm->dev_free_host(pin_fxpp);//remove 
  pm->dev_free_host(pin_bufpa);//remove when ao2mo_v3 is running
  if(verbose_level) get_dev_properties(num_devices);

  if(verbose_level) {
    double total = 0.0;
    for(size_t k=0; k<dev_ctx.profile_sites.size(); ++k)
      total += dev_ctx.profile_sites[k].time;

    // The report is driven by the accumulation sites themselves: every
    // LIBGPU_PROFILE call registers its class + function (via __PRETTY_FUNCTION__)
    // on first use, so the report and the increment sites cannot disagree on
    // which timer/counter a method owns. Every timed site is also counted.
    printf("\nLIBGPU :: SIMPLE_TIMER / SIMPLE_COUNTER\n");
    // Collect non-empty sites, sizing the name column to the longest method name
    // so it never truncates.
    std::vector<ProfileSite> rows;
    size_t wname = 22;
    for(size_t k=0; k<dev_ctx.profile_sites.size(); ++k) {
      const ProfileSite & s = dev_ctx.profile_sites[k];
      if(s.time == 0.0 && s.count == 0) continue; // site never touched
      rows.push_back(s);
      if(s.name.size() > wname) wname = s.name.size();
    }
    // Group all rows from the same class together (stable: within a class, keep
    // first-use order). A blank line separates the class groups.
    std::stable_sort(rows.begin(), rows.end(),
                     [](const ProfileSite & a, const ProfileSite & b) { return a.cls < b.cls; });
    printf("%-*s %-18s %14s %10s %14s\n", (int)wname, "name", "class", "time (s)", "count", "avg (s/call)");
    printf("\n");
    std::string prev_cls = "\x01"; // sentinel: cannot equal a real class name
    double cls_time = 0.0;
    size_t cls_count = 0;
    for(size_t k=0; k<rows.size(); ++k) {
      const ProfileSite & s = rows[k];
      if(s.cls != prev_cls) {
        if(prev_cls != "\x01") // subtotal for the completed class group
          printf("%-*s %-18s %14.6f %10zu\n", (int)wname, "", prev_cls.c_str(), cls_time, cls_count);
        if(k) printf("\n");
        cls_time = 0.0;
        cls_count = 0;
      }
      prev_cls = s.cls;
      cls_time += s.time;
      cls_count += s.count;
      if(s.count > 0)
        printf("%-*s %-18s %14.6f %10zu %14.6f\n", (int)wname, s.name.c_str(), s.cls.c_str(), s.time, s.count, s.time / s.count);
      else
        printf("%-*s %-18s %14.6f %10s %14s\n", (int)wname, s.name.c_str(), s.cls.c_str(), s.time, "-", "-");
    }
    if(prev_cls != "\x01") // subtotal for the final class group
      printf("%-*s %-18s %14.6f %10zu\n", (int)wname, "", prev_cls.c_str(), cls_time, cls_count);
    printf("%-*s %-18s %14.6f\n", (int)wname, "TOTAL", "", total);
  }

  for(int i=0; i<num_devices; ++i) {
  
    pm->dev_set_device(i);
    
    my_device_data * dd = &(device_data[i]);
    
    pm->dev_free(dd->jk.d_rho, "rho");
    pm->dev_free(dd->jk.d_vj, "vj");
    pm->dev_free(dd->jk.d_buf1, "buf1");
    pm->dev_free(dd->jk.d_buf2, "buf2");
    pm->dev_free(dd->jk.d_buf3, "buf3");
    pm->dev_free(dd->jk.d_vkk, "vkk");
    pm->dev_free(dd->jk.d_dms, "dms");
    pm->dev_free(dd->d_mo_coeff, "mo_coeff");
    pm->dev_free(dd->d_mo_cas, "mo_cas");
    pm->dev_free(dd->jk.d_dmtril, "dmtril");
    pm->dev_free(dd->d_ucas, "ucas");
    pm->dev_free(dd->d_umat, "umat");
    pm->dev_free(dd->d_h2eff, "h2eff");
    pm->dev_free(dd->h2eff.d_eri_h2eff, "eri_h2eff");
    
    pm->dev_free(dd->ao2mo.d_j_pc, "j_pc");
    pm->dev_free(dd->ao2mo.d_k_pc, "k_pc");

    pm->dev_free(dd->pdft.d_ao_grid, "ao_grid");
    pm->dev_free(dd->pdft.d_mo_grid, "ao_grid");
    pm->dev_free(dd->pdft.d_cascm2, "cascm2");
    pm->dev_free(dd->pdft.d_Pi, "Pi");
    pm->dev_free(dd->pdft.d_buf_pdft1, "buf_pdft1");
    pm->dev_free(dd->pdft.d_buf_pdft2, "buf_pdft2");

    pm->dev_free(dd->ao2mo.d_bufpa, "bufpa");
    pm->dev_free(dd->ao2mo.d_bufd, "bufd");
    pm->dev_free(dd->ao2mo.d_bufaa, "bufaa");

    pm->dev_free(dd->fci.d_clinka, "clinka");
    pm->dev_free(dd->fci.d_clinkb, "clinkb");
    pm->dev_free(dd->fci.d_cibra, "cibra");
    pm->dev_free(dd->fci.d_ciket, "ciket");
    pm->dev_free(dd->fci.d_tdm1, "tdm1");
    pm->dev_free(dd->fci.d_tdm2, "tdm2");
    pm->dev_free(dd->fci.d_tdm2_p, "tdm2_p");


    for(int i=0; i<dd->fci.size_pumap.size(); ++i) {
      pm->dev_free_host(dd->fci.pumap[i]);
      
      std::string name = "pumap-" + std::to_string(i);
      pm->dev_free(dd->fci.d_pumap[i], name);
    }
    dd->fci.type_pumap.clear();
    dd->fci.size_pumap.clear();
    dd->fci.pumap.clear();
    dd->fci.d_pumap.clear();
  }

  if(verbose_level) {
    pm->print_mem_summary();
    
    printf("LIBGPU :: Finished\n");
  }

  delete [] device_data;
  
  delete ml;
  
  delete pm;
}

/* ---------------------------------------------------------------------- */

// xthi.c from http://docs.cray.com/books/S-2496-4101/html-S-2496-4101/cnlexamples.html

#if !defined(__linux__)
#define CPU_SETSIZE 1024
#endif

// util-linux-2.13-pre7/schedutils/taskset.c
void Device::get_cores(char *str)
{
#if defined(__linux__)
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
#else
  str[0] = 0;
#endif
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
    
void Device::barrier_all()
{
  //if(verbose_level) printf("LIBGPU: barrier on all devices\n");

  for(int i=0; i<num_devices; ++i) {
    pm->dev_set_device(i);
    pm->dev_barrier();
  }
}

/* ---------------------------------------------------------------------- */
    
void Device::set_update_dfobj_(int _val)
{
  _cache->set_update_dfobj_(_val);
}

/* ---------------------------------------------------------------------- */
    
void Device::set_verbose_(int _verbose)
{
  verbose_level = _verbose; // setting nonzero prints affinity + timing info
}

/* ---------------------------------------------------------------------- */

// forwarder -> DeviceCache::get_dfobj_status

void Device::get_dfobj_status(size_t addr_dfobj, py::array_t<int> _arg)
{
  _cache->get_dfobj_status(addr_dfobj, _arg);
}

/* ---------------------------------------------------------------------- */


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
    
  _comm->mgpu_bcast(mo_vec, mo_coeff, _size_mo_coeff*sizeof(double)); // host -> gpu 0, then Bcast to all gpu

#else
  for(int id=0; id<num_devices; ++id) {
    
    pm->dev_set_device(id);
  
    my_device_data * dd = &(device_data[id]);
    
    grow_array(dd->d_mo_coeff, _size_mo_coeff, dd->size_mo_coeff, "mo_coeff", FLERR);
    
    pm->dev_push_async(dd->d_mo_coeff, mo_coeff, _size_mo_coeff*sizeof(double));
  }
#endif
  
  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(dev_ctx, t1 - t0);
}

/* ---------------------------------------------------------------------- */


/* ---------------------------------------------------------------------- */

// ============================================================================
// DEPRECATED: legacy integral engine -- Device::orbital_response().
// Commented out for the time being (removed from the build + Python export).
// Kept intact for possible revival; see refactor_plan.md. Was exported to
// Python as libgpu.orbital_response (used by mcscf/lasscf_sync_o0.py).
// If revived, also restore: device.h decl, libgpu.h/cpp binding, fdrv helper
// (device.h + pm/<backend>/jk.cpp), size_fdrv/buf_fdrv state, and the
// SIMPLE_TIMER/COUNTER printfs in the dtor.
// ============================================================================
#if 0
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
  LIBGPU_PROFILE(dev_ctx, t1  - t0);
  
#if 0
  pm->dev_free_host(ar_global);
#endif
  
  pm->dev_free_host(g_f1_prime);
  pm->dev_free_host(ecm2);
  pm->dev_free_host(_ocm2t);
  pm->dev_free_host(f1_prime);

}
#endif // end DEPRECATED legacy integral engine orbital_response

/* ---------------------------------------------------------------------- */
