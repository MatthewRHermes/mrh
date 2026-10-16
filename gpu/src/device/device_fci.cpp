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

DeviceFci::DeviceFci(DeviceContext & ctx) : ctx(ctx)
{
  h_bravecs = nullptr;
  h_ketvecs = nullptr;
  h_dm1_full = nullptr;
  h_dm2_full = nullptr;
  h_dm2_p_full = nullptr;
  size_bravecs = 0;
  size_ketvecs = 0;
  size_dm1_full = 0;
  size_dm2_full = 0;
}

/* ---------------------------------------------------------------------- */

DeviceFci::~DeviceFci()
{
  if(h_bravecs) ctx.pm->dev_free_host(h_bravecs);
  if(h_ketvecs) ctx.pm->dev_free_host(h_ketvecs);
  if(h_dm1_full) ctx.pm->dev_free_host(h_dm1_full);
  if(h_dm2_full) ctx.pm->dev_free_host(h_dm2_full);
  if(h_dm2_p_full) ctx.pm->dev_free_host(h_dm2_p_full);
  h_bravecs = nullptr;
  h_ketvecs = nullptr;
  h_dm1_full = nullptr;
  h_dm2_full = nullptr;
  h_dm2_p_full = nullptr;
  size_bravecs = 0;
  size_ketvecs = 0;
  size_dm1_full = 0;
  size_dm2_full = 0;
}

/* ---------------------------------------------------------------------- */

void DeviceFci::init_tdm1(int norb)
{
  double t0 = omp_get_wtime();
  int size_tdm1 = norb*norb; 
  //int id=0;
  for (int device_id=0; device_id<ctx.num_devices; ++device_id){
  ctx.pm->dev_set_device(device_id);
  //ctx.pm->dev_profile_start("tdms :: init tdm1");
  my_device_data * dd = &(ctx.device_data[device_id]);
  ::grow_array(ctx.pm, dd->fci.d_tdm1, size_tdm1, dd->fci.size_tdm1, "tdm1", FLERR);
  //ctx.pm->dev_profile_stop();
  }
  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1 - t0);
} 
/* ---------------------------------------------------------------------- */
void DeviceFci::init_tdm2(int norb)
{
  double t0 = omp_get_wtime();
  int size_tdm2 = norb*norb*norb*norb; 
  for (int device_id=0; device_id<ctx.num_devices; ++device_id){
  ctx.pm->dev_set_device(device_id);
  //ctx.pm->dev_profile_start("tdms :: init tdm1");
  my_device_data * dd = &(ctx.device_data[device_id]);
  ::grow_array(ctx.pm, dd->fci.d_tdm2, size_tdm2, dd->fci.size_tdm2, "tdm2", FLERR);
  //ctx.pm->dev_profile_stop();
  }
  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1 - t0);
} 
/* ---------------------------------------------------------------------- */
void DeviceFci::init_tdm3hab(int norb)
{
  double t0 = omp_get_wtime();
  int size_tdm2 = norb*norb*norb*norb; 
  for (int device_id=0; device_id<ctx.num_devices; ++device_id){
    ctx.pm->dev_set_device(device_id);
    //ctx.pm->dev_profile_start("tdms :: init tdm1");
    my_device_data * dd = &(ctx.device_data[device_id]);
    ::grow_array(ctx.pm, dd->fci.d_tdm2, size_tdm2, dd->fci.size_tdm2, "tdm2", FLERR);
    ::grow_array(ctx.pm, dd->fci.d_tdm2_p, size_tdm2, dd->fci.size_tdm2_p, "tdm2_p", FLERR);
  //ctx.pm->dev_profile_stop();
  }

  //ctx.pm->dev_profile_stop();
  double t1 = omp_get_wtime();
  //ctx.t_array[15] += t1 - t0;//TODO: Fix timing array position
} 
/* ---------------------------------------------------------------------- */
void DeviceFci::init_tdm1_host(int _size_dm1)
{
  double t0 = omp_get_wtime();
  ::grow_array_host(ctx.pm, h_dm1_full, _size_dm1, size_dm1_full, "h:dm1_full");  
  double t1 = omp_get_wtime();
}
/* ---------------------------------------------------------------------- */
void DeviceFci::init_tdm2_host(int _size_dm2)
{
  double t0 = omp_get_wtime();
  ::grow_array_host(ctx.pm, h_dm2_full, _size_dm2, size_dm2_full, "h:dm2_full");  
  double t1 = omp_get_wtime();
}
/* ---------------------------------------------------------------------- */
void DeviceFci::init_tdm3h_host(int _size_dm2)
{
  double t0 = omp_get_wtime();
  ::grow_array_host(ctx.pm, h_dm2_full, _size_dm2, size_dm2_full, "h:dm2_full");  
  ::grow_array_host(ctx.pm, h_dm2_p_full, _size_dm2, size_dm2_full, "h:dm2_p_full");  
  double t1 = omp_get_wtime();
}

/* ---------------------------------------------------------------------- */
void DeviceFci::push_cibra(py::array_t<double> _cibra, int na, int nb, int count)
{
  double t0 = omp_get_wtime();
  int id = count % ctx.num_devices;
  ctx.pm->dev_set_device(id); 
  my_device_data * dd = &(ctx.device_data[id]);
  ctx.pm->dev_profile_start("tdms :: push cibra");

  py::buffer_info info_cibra = _cibra.request(); //2D array (na, nb)
  double * cibra = static_cast<double*>(info_cibra.ptr);
  int size_cibra = na*nb;
  ::grow_array(ctx.pm, dd->fci.d_cibra, size_cibra, dd->fci.size_cibra, "cibra", FLERR);

  ctx.pm->dev_push_async(dd->fci.d_cibra, cibra, size_cibra*sizeof(double));
  ctx.pm->dev_profile_stop();
  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1 - t0);
  
} 
 /* ---------------------------------------------------------------------- */
void DeviceFci::push_ciket(py::array_t<double> _ciket, int na, int nb, int count)
{
  double t0 = omp_get_wtime();
  int id = count % ctx.num_devices;
  ctx.pm->dev_set_device(id); 
  my_device_data * dd = &(ctx.device_data[id]);
  ctx.pm->dev_profile_start("tdms :: push ciket");

  py::buffer_info info_ciket = _ciket.request(); //2D array (na, nb)
  double * ciket = static_cast<double*>(info_ciket.ptr);
  int size_ciket = na*nb;
  ::grow_array(ctx.pm, dd->fci.d_ciket, size_ciket, dd->fci.size_ciket, "ciket", FLERR);
  ctx.pm->dev_push_async(dd->fci.d_ciket, ciket, size_ciket*sizeof(double));
  ctx.pm->dev_profile_stop();
  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1 - t0);
} 
/* ---------------------------------------------------------------------- */
void DeviceFci::copy_bravecs_host(py::array_t<double> _bravecs, int nvecs, int na, int nb)
{
  double t0 = omp_get_wtime();
  py::buffer_info info_bravecs = _bravecs.request(); //3D array (nvecs, na, nb)
  double * bravecs = static_cast<double*>(info_bravecs.ptr);
  int _size_bravecs = nvecs*na*nb;
  ::grow_array_host(ctx.pm, h_bravecs, _size_bravecs, size_bravecs, "h:bravecs");
#pragma omp parallel for
  for (int i=0;i<_size_bravecs;++i){h_bravecs[i] = bravecs[i];}
  double t1 = omp_get_wtime();
}
/* ---------------------------------------------------------------------- */
void DeviceFci::copy_ketvecs_host(py::array_t<double> _ketvecs, int nvecs, int na, int nb)
{
  double t0 = omp_get_wtime();
  py::buffer_info info_ketvecs = _ketvecs.request(); //3D array (nvecs*na, nb)
  double * ketvecs = static_cast<double*>(info_ketvecs.ptr);
  int _size_ketvecs = nvecs*na*nb;
  ::grow_array_host(ctx.pm, h_ketvecs, _size_ketvecs, size_ketvecs, "h:ketvecs");
#pragma omp parallel for
  for (int i=0;i<_size_ketvecs;++i){h_ketvecs[i] = ketvecs[i];}
  double t1 = omp_get_wtime();
}
/* ---------------------------------------------------------------------- */
void DeviceFci::push_cibra_from_host(int bra_index, int na, int nb, int count)
{
  double t0 = omp_get_wtime();
  int id = count % ctx.num_devices;
  ctx.pm->dev_set_device(id);
  ctx.pm->dev_profile_start("tdms :: push_ci");
  my_device_data * dd = &(ctx.device_data[id]);
  int size_cibra = na*nb;
  ::grow_array(ctx.pm, dd->fci.d_cibra, size_cibra, dd->fci.size_cibra, "cibra", FLERR);
  double * h_bra_loc = &(h_bravecs[bra_index*size_cibra]);
  ctx.pm->dev_push_async(dd->fci.d_cibra, h_bra_loc, size_cibra*sizeof(double));
  ctx.pm->dev_profile_stop();
  double t1 = omp_get_wtime();
}
/* ---------------------------------------------------------------------- */
void DeviceFci::push_ciket_from_host(int ket_index, int na, int nb, int count)
{
  double t0 = omp_get_wtime();
  int id = count % ctx.num_devices;
  ctx.pm->dev_set_device(id);
  ctx.pm->dev_profile_start("tdms :: push_ci");
  my_device_data * dd = &(ctx.device_data[id]);
  int size_ciket = na*nb;
  ::grow_array(ctx.pm, dd->fci.d_ciket, size_ciket, dd->fci.size_ciket, "ciket", FLERR);
  double * h_ket_loc = &(h_ketvecs[ket_index*size_ciket]);
  ctx.pm->dev_push_async(dd->fci.d_ciket, h_ket_loc, size_ciket*sizeof(double));
  ctx.pm->dev_profile_stop();
  double t1 = omp_get_wtime();
}

/* ---------------------------------------------------------------------- */
void DeviceFci::push_link_indexa(int na, int nlinka, py::array_t<int> _link_indexa)
{
  double t0 = omp_get_wtime();
  if (nlinka>0){
    py::buffer_info info_link_indexa = _link_indexa.request(); //3D array (na, nlinka, 4)
    int * link_indexa = static_cast<int*>(info_link_indexa.ptr);
    int size_clinka = na*nlinka*4; //a,i,str,sign
    for (int device_id=0;device_id<ctx.num_devices;++device_id){
      ctx.pm->dev_set_device(device_id); 
      my_device_data * dd = &(ctx.device_data[device_id]);
      ::grow_array(ctx.pm, dd->fci.d_clinka, size_clinka, dd->fci.size_clinka, "clinka", FLERR);
      ctx.pm->dev_push_async(dd->fci.d_clinka, link_indexa, size_clinka*sizeof(int));
    }
  }
  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1 - t0);
}
/* ---------------------------------------------------------------------- */
void DeviceFci::push_link_indexb(int nb, int nlinkb, py::array_t<int> _link_indexb)
{
  double t0 = omp_get_wtime();
  if (nlinkb>0){
    py::buffer_info info_link_indexb = _link_indexb.request(); //3D array (nb, nlinkb, 4)
    int * link_indexb = static_cast<int*>(info_link_indexb.ptr);
    int size_clinkb = nb*nlinkb*4; //a,i,str,sign
    for (int device_id=0;device_id<ctx.num_devices;++device_id){
      ctx.pm->dev_set_device(device_id); 
      my_device_data * dd = &(ctx.device_data[device_id]);
      ::grow_array(ctx.pm, dd->fci.d_clinkb, size_clinkb, dd->fci.size_clinkb, "clinkb", FLERR);
      ctx.pm->dev_push_async(dd->fci.d_clinkb, link_indexb, size_clinkb*sizeof(int));
    } 
  }
  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1 - t0);
}
/* ---------------------------------------------------------------------- */
void DeviceFci::compute_trans_rdm1a(int na, int nb, int nlinka, int nlinkb, int norb, int count)
{
  double t0 = omp_get_wtime();
  int id = count % ctx.num_devices;
  ctx.pm->dev_set_device(id); 
  my_device_data * dd = &(ctx.device_data[id]);
  ctx.pm->dev_profile_start("tdms :: compute_trans_rdm1a");

  int norb2 = norb*norb;
  int size_tdm1 = norb2;
  ::grow_array(ctx.pm, dd->fci.d_tdm1,size_tdm1, dd->fci.size_tdm1, "tdm1", FLERR); //actual returned
  ctx.utils->set_to_zero(dd->fci.d_tdm1, size_tdm1);
  if (nlinka>0){
    compute_FCItrans_rdm1a(dd->fci.d_cibra, dd->fci.d_ciket, dd->fci.d_tdm1, norb, na, nb, nlinka, dd->fci.d_clinka);
  }
  ctx.pm->dev_profile_stop();
  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1 - t0);
}
/* ---------------------------------------------------------------------- */
void DeviceFci::compute_trans_rdm1b(int na, int nb, int nlinka, int nlinkb, int norb, int count)
{
  double t0 = omp_get_wtime();
  int id = count % ctx.num_devices;
  ctx.pm->dev_set_device(id); 
  my_device_data * dd = &(ctx.device_data[id]);
  ctx.pm->dev_profile_start("tdms :: compute_trans_rdm1b");

  int norb2 = norb*norb;
  int size_tdm1 = norb2;
  ::grow_array(ctx.pm, dd->fci.d_tdm1,size_tdm1, dd->fci.size_tdm1, "tdm1", FLERR); //actual returned
  ctx.utils->set_to_zero(dd->fci.d_tdm1, size_tdm1);
  if (nlinkb>0){
    compute_FCItrans_rdm1b(dd->fci.d_cibra, dd->fci.d_ciket, dd->fci.d_tdm1, norb, na, nb, nlinkb, dd->fci.d_clinkb);
  }
  ctx.pm->dev_profile_stop();
  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1 - t0);
}
/* ---------------------------------------------------------------------- */
void DeviceFci::compute_make_rdm1a(int na, int nb, int nlinka, int nlinkb, int norb, int count)
{
  double t0 = omp_get_wtime();
  int id = count % ctx.num_devices;
  ctx.pm->dev_set_device(id); 
  my_device_data * dd = &(ctx.device_data[id]);

  ctx.pm->dev_profile_start("tdms :: compute_make_rdm1a");
  int norb2 = norb*norb;
  int size_tdm1 = norb2;
  ::grow_array(ctx.pm, dd->fci.d_tdm1,size_tdm1, dd->fci.size_tdm1, "tdm1", FLERR); //actual returned
  ctx.utils->set_to_zero(dd->fci.d_tdm1, size_tdm1);
  if (nlinka>0){
  compute_FCImake_rdm1a(dd->fci.d_cibra, dd->fci.d_ciket, dd->fci.d_tdm1, norb, na, nb, nlinka, dd->fci.d_clinka);
  }
  ctx.pm->dev_profile_stop();
  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1 - t0);
}
/* ---------------------------------------------------------------------- */
void DeviceFci::compute_make_rdm1b(int na, int nb, int nlinka, int nlinkb, int norb, int count)
{
  double t0 = omp_get_wtime();
  int id = count % ctx.num_devices;
  ctx.pm->dev_set_device(id); 
  my_device_data * dd = &(ctx.device_data[id]);
  ctx.pm->dev_profile_start("tdms :: make_rdm1b");

  int norb2 = norb*norb;
  int size_tdm1 = norb2;
  ::grow_array(ctx.pm, dd->fci.d_tdm1,size_tdm1, dd->fci.size_tdm1, "tdm1", FLERR); //actual returned
  ctx.utils->set_to_zero(dd->fci.d_tdm1, size_tdm1);
  if (nlinkb>0){
  compute_FCImake_rdm1b(dd->fci.d_cibra, dd->fci.d_ciket, dd->fci.d_tdm1, norb, na, nb, nlinkb, dd->fci.d_clinkb);
  }
  ctx.pm->dev_profile_stop();
  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1 - t0);
}

/* ---------------------------------------------------------------------- */
void DeviceFci::compute_tdm12kern_a_v2(int na, int nb, int nlinka, int nlinkb, int norb, int count )
{
  double t0 = omp_get_wtime();
  int id = count % ctx.num_devices;
  ctx.pm->dev_set_device(id);
  ctx.ml->set_handle(id);
  my_device_data * dd = &(ctx.device_data[id]);
  ctx.pm->dev_profile_start("tdms :: compute_tdm12kern_a_v2");
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
  int _size_buf = _MAX(dd->jk.size_buf1, dd->jk.size_buf2);// (dd->jk.size_buf1 > dd->jk.size_buf2) ? dd->jk.size_buf1 : dd->jk.size_buf2;
  #ifdef _TEMP_BUFSIZING
  _size_buf = size_buf*6;
  #endif
  int final_size_buf = _MAX(_size_buf, size_buf);//(_size_buf > size_buf) ? _size_buf : size_buf;
  final_size_buf = _MAX(size_tdm2, final_size_buf);//(_size_buf > size_buf) ? _size_buf : size_buf;
  int buf_batch_size = final_size_buf/size_buf; //this is integer division // number of buf1/2 in a single buffer
  int gemm_batch_size = final_size_buf/size_tdm2; // this is integer division // number of tdm2 in a single buf
  int gemv_batch_size = final_size_buf/size_tdm1; // this is integer division // number of tdm1 in a single buf
  int num_buf_batches; 
  int num_buf_batches_for_gemv; 
  int num_gemm_batches; 
  int num_gemv_batches; 
  ::grow_array(ctx.pm, dd->jk.d_buf1,final_size_buf, dd->jk.size_buf1, "buf1", FLERR); 
  ::grow_array(ctx.pm, dd->jk.d_buf2,final_size_buf, dd->jk.size_buf2, "buf2", FLERR); 
  ::grow_array(ctx.pm, dd->jk.d_buf3,final_size_buf, dd->jk.size_buf3, "buf3", FLERR); 
  size_t bits_buf = sizeof(double)*buf_batch_size*size_buf;
  ctx.ml->memset(dd->jk.d_buf1, &zero, &bits_buf); 
  ctx.ml->memset(dd->jk.d_buf2, &zero, &bits_buf); 
  ::grow_array(ctx.pm, dd->fci.d_tdm1, size_tdm1, dd->fci.size_tdm1, "tdm1", FLERR);
  ::grow_array(ctx.pm, dd->fci.d_tdm2, size_tdm2, dd->fci.size_tdm2, "tdm2", FLERR); 
  ctx.ml->memset(dd->fci.d_tdm1, &zero, &bits_tdm1);
  ctx.ml->memset(dd->fci.d_tdm2, &zero, &bits_tdm2);
 

  for (int stra_id = 0; stra_id<na; stra_id += buf_batch_size){
    num_buf_batches = _MIN(buf_batch_size, na-stra_id);
    compute_FCIrdm2_a_t1ci_v2( dd->fci.d_cibra, dd->jk.d_buf2, stra_id, num_buf_batches, nb, norb, nlinka, dd->fci.d_clinka); 
    compute_FCIrdm2_a_t1ci_v2( dd->fci.d_ciket, dd->jk.d_buf1, stra_id, num_buf_batches, nb, norb, nlinka, dd->fci.d_clinka); 
    for (int i=0; i<num_buf_batches; i+=gemv_batch_size){
      double * bravec = &(dd->fci.d_cibra[(stra_id+i)*nb]);
      num_gemv_batches = _MIN(gemv_batch_size, num_buf_batches-i);
      ctx.ml->gemv_batch((char *) "N", &norb2, &nb,
          &alpha, &(dd->jk.d_buf1[i*size_buf]), &norb2, &size_buf,
          bravec, &one, &nb, 
          &beta, dd->jk.d_buf3, &one, &size_tdm1,
          &num_gemv_batches);
      reduce_buf3_to_rdm(dd->jk.d_buf3, dd->fci.d_tdm1, size_tdm1, num_gemv_batches);
      }

    for (int i=0; i<num_buf_batches; i+=gemm_batch_size) {
      num_gemm_batches = _MIN(gemm_batch_size, num_buf_batches-i);
      ctx.ml->gemm_batch((char *) "N",(char *) "T", &norb2, &norb2, &nb, 
        &alpha, 
        &(dd->jk.d_buf1[i*size_buf]), &norb2, &size_buf, 
        &(dd->jk.d_buf2[i*size_buf]), &norb2, &size_buf, 
        &beta, dd->jk.d_buf3, &norb2, &size_tdm2, &num_gemm_batches); 
      reduce_buf3_to_rdm(dd->jk.d_buf3, dd->fci.d_tdm2, size_tdm2, num_gemm_batches);
      }

    ctx.ml->memset(dd->jk.d_buf1, &zero, &bits_buf); 
    ctx.ml->memset(dd->jk.d_buf2, &zero, &bits_buf); 
  }     

  transpose_jikl(dd->fci.d_tdm2, dd->jk.d_buf1, norb);

  ctx.pm->dev_profile_stop();
  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1 - t0);
}

/* ---------------------------------------------------------------------- */
void DeviceFci::compute_tdm12kern_b_v2(int na, int nb, int nlinka, int nlinkb, int norb, int count)
{
  double t0 = omp_get_wtime();
  int id = count % ctx.num_devices;
  ctx.pm->dev_set_device(id);
  ctx.ml->set_handle(id);
  my_device_data * dd = &(ctx.device_data[id]);
  ctx.pm->dev_profile_start("tdms :: compute_tdm12kern_b_v2");
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
  int _size_buf = _MAX(dd->jk.size_buf1, dd->jk.size_buf2);// (dd->jk.size_buf1 > dd->jk.size_buf2) ? dd->jk.size_buf1 : dd->jk.size_buf2;
  #ifdef _TEMP_BUFSIZING
  _size_buf = size_buf*6;
  #endif
  int final_size_buf = _MAX(_size_buf, size_buf);//(_size_buf > size_buf) ? _size_buf : size_buf;
  final_size_buf = _MAX(size_tdm2, final_size_buf);//(_size_buf > size_buf) ? _size_buf : size_buf;
  int buf_batch_size = final_size_buf/size_buf; //this is integer division // number of buf1/2 in a single buffer
  int gemm_batch_size = final_size_buf/size_tdm2; // this is integer division // number of tdm2 in a single buf
  int gemv_batch_size = final_size_buf/size_tdm1; // this is integer division // number of tdm1 in a single buf
  int num_buf_batches; 
  int num_buf_batches_for_gemv; 
  int num_gemm_batches; 
  int num_gemv_batches; 
  //  printf("buf_batches: %i gemm_batches = %i\n",buf_batch_size, gemm_batch_size);
  ::grow_array(ctx.pm, dd->jk.d_buf1,final_size_buf, dd->jk.size_buf1, "buf1", FLERR); 
  ::grow_array(ctx.pm, dd->jk.d_buf2,final_size_buf, dd->jk.size_buf2, "buf2", FLERR); 
  ::grow_array(ctx.pm, dd->jk.d_buf3,final_size_buf, dd->jk.size_buf3, "buf3", FLERR); 
  size_t bits_buf = sizeof(double)*buf_batch_size*size_buf;
  ctx.ml->memset(dd->jk.d_buf1, &zero, &bits_buf); 
  ctx.ml->memset(dd->jk.d_buf2, &zero, &bits_buf); 
  ::grow_array(ctx.pm, dd->fci.d_tdm1, size_tdm1, dd->fci.size_tdm1, "tdm1", FLERR);
  ::grow_array(ctx.pm, dd->fci.d_tdm2, size_tdm2, dd->fci.size_tdm2, "tdm2", FLERR); 
  ctx.ml->memset(dd->fci.d_tdm1, &zero, &bits_tdm1);
  ctx.ml->memset(dd->fci.d_tdm2, &zero, &bits_tdm2);
 

  for (int stra_id = 0; stra_id<na; stra_id += buf_batch_size){
    num_buf_batches = _MIN(buf_batch_size, na-stra_id);
    compute_FCIrdm2_b_t1ci_v2( dd->fci.d_cibra, dd->jk.d_buf2, stra_id, num_buf_batches, nb, norb, nlinkb, dd->fci.d_clinkb); 
    compute_FCIrdm2_b_t1ci_v2( dd->fci.d_ciket, dd->jk.d_buf1, stra_id, num_buf_batches, nb, norb, nlinkb, dd->fci.d_clinkb); 
    for (int i=0; i<num_buf_batches; i+=gemv_batch_size){
      double * bravec = &(dd->fci.d_cibra[(stra_id+i)*nb]);
      num_gemv_batches = _MIN(gemv_batch_size, num_buf_batches-i);
      ctx.ml->gemv_batch((char *) "N", &norb2, &nb,
          &alpha, &(dd->jk.d_buf1[i*size_buf]), &norb2, &size_buf,
          bravec, &one, &nb, 
          &beta, dd->jk.d_buf3, &one, &size_tdm1,
          &num_gemv_batches);
      reduce_buf3_to_rdm(dd->jk.d_buf3, dd->fci.d_tdm1, size_tdm1, num_gemv_batches);
      }

    for (int i=0; i<num_buf_batches; i+=gemm_batch_size) {
      num_gemm_batches = _MIN(gemm_batch_size, num_buf_batches-i);
      ctx.ml->gemm_batch((char *) "N",(char *) "T", &norb2, &norb2, &nb, 
        &alpha, 
        &(dd->jk.d_buf1[i*size_buf]), &norb2, &size_buf, 
        &(dd->jk.d_buf2[i*size_buf]), &norb2, &size_buf, 
        &beta, dd->jk.d_buf3, &norb2, &size_tdm2, &num_gemm_batches); 
      reduce_buf3_to_rdm(dd->jk.d_buf3, dd->fci.d_tdm2, size_tdm2, num_gemm_batches);
      }

    ctx.ml->memset(dd->jk.d_buf1, &zero, &bits_buf); 
    ctx.ml->memset(dd->jk.d_buf2, &zero, &bits_buf); 
  }     

  transpose_jikl(dd->fci.d_tdm2, dd->jk.d_buf1, norb);

  ctx.pm->dev_profile_stop();
  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1 - t0);
}

/* ---------------------------------------------------------------------- */
void DeviceFci::compute_tdm12kern_ab_v2(int na, int nb, int nlinka, int nlinkb, int norb, int count)
{
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Inside Device::compute_tdm12kern_ab_v2()\n");
#endif
  
  double t0 = omp_get_wtime();
  int id = count % ctx.num_devices;
  ctx.pm->dev_set_device(id);
  ctx.ml->set_handle(id);
  my_device_data * dd = &(ctx.device_data[id]);
  ctx.pm->dev_profile_start("tdms :: compute_tdm12kern_ab_v2");
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
  int _size_buf = _MAX(dd->jk.size_buf1, dd->jk.size_buf2);// (dd->jk.size_buf1 > dd->jk.size_buf2) ? dd->jk.size_buf1 : dd->jk.size_buf2;
  #ifdef _TEMP_BUFSIZING
  _size_buf = size_buf*6;
  #endif
  int final_size_buf = _MAX(_size_buf, size_buf);//(_size_buf > size_buf) ? _size_buf : size_buf;
  final_size_buf = _MAX(size_tdm2, final_size_buf);//(_size_buf > size_buf) ? _size_buf : size_buf;
  int buf_batch_size = final_size_buf/size_buf; //this is integer division // number of buf1/2 in a single buffer
  int gemm_batch_size = final_size_buf/size_tdm2; // this is integer division // number of tdm2 in a single buf
  int gemv_batch_size = final_size_buf/size_tdm1; // this is integer division // number of tdm1 in a single buf
  int num_buf_batches; 
  int num_gemm_batches; 
  int num_gemv_batches; 
  ::grow_array(ctx.pm, dd->jk.d_buf1,final_size_buf, dd->jk.size_buf1, "buf1", FLERR); 
  ::grow_array(ctx.pm, dd->jk.d_buf2,final_size_buf, dd->jk.size_buf2, "buf2", FLERR); 
  ::grow_array(ctx.pm, dd->jk.d_buf3,final_size_buf, dd->jk.size_buf3, "buf3", FLERR); 
  size_t bits_buf = sizeof(double)*buf_batch_size*size_buf;
  ctx.ml->memset(dd->jk.d_buf1, &zero, &bits_buf); 
  ctx.ml->memset(dd->jk.d_buf2, &zero, &bits_buf); 
  ::grow_array(ctx.pm, dd->fci.d_tdm1, size_tdm1, dd->fci.size_tdm1, "tdm1", FLERR);
  ::grow_array(ctx.pm, dd->fci.d_tdm2, size_tdm2, dd->fci.size_tdm2, "tdm2", FLERR); 
  ctx.ml->memset(dd->fci.d_tdm1, &zero, &bits_tdm1);
  ctx.ml->memset(dd->fci.d_tdm2, &zero, &bits_tdm2);
 

  for (int stra_id = 0; stra_id<na; stra_id += buf_batch_size){
    num_buf_batches = _MIN(buf_batch_size, na-stra_id);
    compute_FCIrdm2_a_t1ci_v2( dd->fci.d_cibra, dd->jk.d_buf2, stra_id, num_buf_batches, nb, norb, nlinka, dd->fci.d_clinka); 
    compute_FCIrdm2_b_t1ci_v2( dd->fci.d_ciket, dd->jk.d_buf1, stra_id, num_buf_batches, nb, norb, nlinkb, dd->fci.d_clinkb); 

    for (int i=0; i<num_buf_batches; i+=gemm_batch_size) {
      num_gemm_batches = _MIN(gemm_batch_size, num_buf_batches-i);
      ctx.ml->gemm_batch((char *) "N",(char *) "T", &norb2, &norb2, &nb, 
        &alpha, 
        &(dd->jk.d_buf1[i*size_buf]), &norb2, &size_buf, 
        &(dd->jk.d_buf2[i*size_buf]), &norb2, &size_buf, 
        &beta, dd->jk.d_buf3, &norb2, &size_tdm2, &num_gemm_batches); 
      reduce_buf3_to_rdm(dd->jk.d_buf3, dd->fci.d_tdm2, size_tdm2, num_gemm_batches);
      }

    ctx.ml->memset(dd->jk.d_buf1, &zero, &bits_buf); 
    ctx.ml->memset(dd->jk.d_buf2, &zero, &bits_buf); 
  }     

  transpose_jikl(dd->fci.d_tdm2, dd->jk.d_buf1, norb);

  ctx.pm->dev_profile_stop();
  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1 - t0);
  
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: -- Leaving Device::compute_tdm12kern_ab_v2()\n");
#endif
}
/* ---------------------------------------------------------------------- */
void DeviceFci::compute_rdm12kern_sf_v2(int na, int nb, int nlinka, int nlinkb, int norb, int count)
{
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Inside Device::compute_rdm12kern_sf_v2()\n");
#endif
  
  double t0 = omp_get_wtime();
  int id = count % ctx.num_devices;
  ctx.pm->dev_set_device(id);
  ctx.ml->set_handle(id);
  my_device_data * dd = &(ctx.device_data[id]);
  ctx.pm->dev_profile_start("tdms :: compute_tdm12kern_sf_v2");
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
  int _size_buf = _MAX(dd->jk.size_buf1, dd->jk.size_buf2);// (dd->jk.size_buf1 > dd->jk.size_buf2) ? dd->jk.size_buf1 : dd->jk.size_buf2;
  _size_buf = _MAX(_size_buf, dd->jk.size_buf3);//
  #ifdef _TEMP_BUFSIZING
  _size_buf = size_buf*6;
  #endif

  int final_size_buf = _MAX(_size_buf, size_buf);//(_size_buf > size_buf) ? _size_buf : size_buf;
  final_size_buf = _MAX(size_tdm2, final_size_buf);//(_size_buf > size_buf) ? _size_buf : size_buf;
  int buf_batch_size = final_size_buf/size_buf; //this is integer division // number of buf1/2 in a single buffer
  int gemm_batch_size = final_size_buf/size_tdm2; // this is integer division // number of tdm2 in a single buf
  int gemv_batch_size = final_size_buf/size_tdm1; // this is integer division // number of tdm1 in a single buf
  int num_buf_batches; 
  int num_gemm_batches; 
  int num_gemv_batches; 
  ::grow_array(ctx.pm, dd->jk.d_buf1,final_size_buf, dd->jk.size_buf1, "buf1", FLERR); 
  ::grow_array(ctx.pm, dd->jk.d_buf2,final_size_buf, dd->jk.size_buf2, "buf1", FLERR); 
  ::grow_array(ctx.pm, dd->jk.d_buf3,final_size_buf, dd->jk.size_buf3, "buf3", FLERR); 
  size_t bits_buf = sizeof(double)*buf_batch_size*size_buf;
  ::grow_array(ctx.pm, dd->fci.d_tdm1, size_tdm1, dd->fci.size_tdm1, "tdm1", FLERR);
  ::grow_array(ctx.pm, dd->fci.d_tdm2, size_tdm2, dd->fci.size_tdm2, "tdm2", FLERR); 

  ctx.ml->memset(dd->jk.d_buf1, &zero, &bits_buf); 
  ctx.ml->memset(dd->fci.d_tdm1, &zero, &bits_tdm1);
  ctx.ml->memset(dd->fci.d_tdm2, &zero, &bits_tdm2);
  //double * h_buf3 = (double *)ctx.pm->dev_malloc_host(final_size_buf*sizeof(double));

  for (int stra_id = 0; stra_id<na; stra_id += buf_batch_size){
    num_buf_batches = _MIN(buf_batch_size, na-stra_id);
    compute_FCIrdm2_a_t1ci_v2( dd->fci.d_ciket, dd->jk.d_buf1, stra_id, num_buf_batches, nb, norb, nlinka, dd->fci.d_clinka); 
    compute_FCIrdm2_b_t1ci_v2( dd->fci.d_ciket, dd->jk.d_buf1, stra_id, num_buf_batches, nb, norb, nlinkb, dd->fci.d_clinkb); 

    for (int i=0; i<num_buf_batches; i+=gemm_batch_size) {
 
      num_gemm_batches = _MIN(gemm_batch_size, num_buf_batches-i);
      ctx.ml->gemm_batch((char *) "N",(char *) "T", &norb2, &norb2, &nb, 
        &alpha, 
        &(dd->jk.d_buf1[i*size_buf]), &norb2, &size_buf, 
        &(dd->jk.d_buf1[i*size_buf]), &norb2, &size_buf, 
        &beta, dd->jk.d_buf3, &norb2, &size_tdm2, &num_gemm_batches); 
      reduce_buf3_to_rdm(dd->jk.d_buf3, dd->fci.d_tdm2, size_tdm2, num_gemm_batches);
      }

    for (int i=0; i<num_buf_batches; i+=gemv_batch_size){
      double * ketvec = &(dd->fci.d_ciket[(stra_id+i)*nb]);
      num_gemv_batches = _MIN(gemv_batch_size, num_buf_batches-i);
      ctx.ml->gemv_batch((char *) "N", &norb2, &nb,
          &alpha, &(dd->jk.d_buf1[i*size_buf]), &norb2, &size_buf,
          ketvec, &one, &nb, 
          &beta, dd->jk.d_buf3, &one, &size_tdm1,
          &num_gemv_batches);
      reduce_buf3_to_rdm(dd->jk.d_buf3, dd->fci.d_tdm1, size_tdm1, num_gemv_batches);
      }

    ctx.ml->memset(dd->jk.d_buf1, &zero, &bits_buf); 
    ctx.ml->memset(dd->jk.d_buf2, &zero, &bits_buf); 
  }     

  transpose_jikl(dd->fci.d_tdm2, dd->jk.d_buf1, norb);

  ctx.pm->dev_profile_stop();
  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1 - t0);
  
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Leaving Device::compute_rdm12kern_sf_v2()\n");
#endif
}
/* ---------------------------------------------------------------------- */
void DeviceFci::compute_tdm13h_spin_v4(int na, int nb, 
                                 int nlinka, int nlinkb, 
                                 int norb, int spin, int _reorder,
                                 int ia_bra, int ja_bra, int ib_bra, int jb_bra, int sgn_bra, 
                                 int ia_ket, int ja_ket, int ib_ket, int jb_ket, int sgn_ket, int count )
{
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Inside Device::compute_tdm13h_spin_v4()\n");
#endif
  
  //na, nb is same for both zero-padded ci vectors, but not necessarily for non padded vectors
  double t0 = omp_get_wtime();
  int id = count % ctx.num_devices;
  ctx.pm->dev_set_device(id);
  ctx.ml->set_handle(id);
  my_device_data * dd = &(ctx.device_data[id]);
  ctx.pm->dev_profile_start("tdms :: compute_tdm13h_spin_v4");

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
  size_t bits_buf = sizeof(double)*size_buf;
  int bits_nbket = sizeof(double)*nb_ket*norb2;
  int bits_tdm1h = sizeof(double)*size_tdm1h;
  int bits_tdm3h = sizeof(double)*size_tdm3h;
  ::grow_array(ctx.pm, dd->fci.d_tdm1, size_tdm1h, dd->fci.size_tdm1, "tdm1", FLERR);
  ::grow_array(ctx.pm, dd->fci.d_tdm2, size_tdm3h, dd->fci.size_tdm2, "tdm2", FLERR); 
  ::grow_array(ctx.pm, dd->fci.d_tdm2_p, size_tdm3h, dd->fci.size_tdm2_p, "tdm2_p", FLERR); 
  ::grow_array(ctx.pm, dd->jk.d_buf1,size_buf, dd->jk.size_buf1, "buf1", FLERR); 
  ::grow_array(ctx.pm, dd->jk.d_buf2,size_buf, dd->jk.size_buf2, "buf2", FLERR); 
  //dd->fci.d_tdm1h = dd->fci.d_tdm1;
  ctx.ml->memset(dd->jk.d_buf1, &zero, &bits_buf); 
  ctx.ml->memset(dd->jk.d_buf2, &zero, &bits_buf); 
  ctx.ml->memset(dd->fci.d_tdm1, &zero, &bits_tdm1h);
  ctx.ml->memset(dd->fci.d_tdm2, &zero, &bits_tdm3h);
  ctx.ml->memset(dd->fci.d_tdm2_p, &zero, &bits_tdm3h);
  //dd->fci.d_tdm3ha = dd->fci.d_tdm2;
  //dd->fci.d_tdm3hb = dd->fci.d_tdm2_p;

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
 

    for (int stra_id = ia_ket; stra_id<ja_ket; ++stra_id){
    
      //buf2 is 0, so the whole thing is meaningless. tdm1 uses buf1 and bravec = cibra[stra_id, :]
        compute_FCIrdm3h_b_t1ci_v2(dd->fci.d_cibra, dd->jk.d_buf2, stra_id, nb, nb_bra, norb, nlinkb, ia_bra, ja_bra, ib_bra, jb_bra, dd->fci.d_clinkb);
        if ((stra_id >= ia_ket) && (stra_id < ja_ket)) {
          //buf1 is 0, so tdm3hb and tdm1hb don't calculate anything
        
          compute_FCIrdm3h_b_t1ci_v2(dd->fci.d_ciket, dd->jk.d_buf1, stra_id, nb, nb_ket, norb, nlinkb, ia_ket, ja_ket, ib_ket, jb_ket, dd->fci.d_clinkb);

          ctx.ml->gemm((char *) "N", (char *) "T", &norb2, &norb2, &nb, &alpha, 
                dd->jk.d_buf1, &norb2, dd->jk.d_buf2, &norb2, 
                &beta, dd->fci.d_tdm2, &norb2);
          double * bravec = &(dd->fci.d_cibra[(stra_id-ia_bra)*nb_bra]);
          ctx.ml->gemv((char *) "N", &norb2, &nb_bra, &alpha, 
                &(dd->jk.d_buf1[ib_bra*norb2]), &norb2, bravec, &one, 
                &beta, dd->fci.d_tdm1, &one);
          ctx.ml->memset(dd->jk.d_buf1, &zero, &bits_buf);
          }
        compute_FCIrdm3h_a_t1ci_v2(dd->fci.d_ciket, dd->jk.d_buf1, stra_id, nb_ket, norb, nlinka, ia_ket, ja_ket, ib_ket, jb_ket, dd->fci.d_clinka);
        // buf1 is only populated from ib_ket:jb_ket, so don't need to run the multiplication over the whole thing 
        ctx.ml->gemm((char *) "N", (char *) "T", &norb2, &norb2, &nb_ket, &alpha, 
               &(dd->jk.d_buf2[ib_ket*norb2]), &norb2, &(dd->jk.d_buf1[ib_ket*norb2]), &norb2, //remember the switch?
               &beta, dd->fci.d_tdm2_p, &norb2);
        ctx.ml->memset(dd->jk.d_buf2, &zero, &bits_buf);
        ctx.ml->memset(&(dd->jk.d_buf1[ib_ket*norb2]), &zero, &bits_nbket);
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

      compute_FCIrdm3h_a_t1ci_v2(dd->fci.d_cibra, dd->jk.d_buf2, stra_id, nb_bra, norb, nlinka, ia_bra, ja_bra, ib_bra, jb_bra, dd->fci.d_clinka);
      if (b_len>0){
        compute_FCIrdm3h_a_t1ci_v2(dd->fci.d_ciket, dd->jk.d_buf1, stra_id, nb_ket, norb, nlinka, ia_ket, ja_ket, ib_max, jb_min, dd->fci.d_clinka);// !limits

        ctx.ml->gemm((char *) "N", (char *) "T", &norb2, &norb2, &b_len, &alpha, 
                &(dd->jk.d_buf1[ib_max*norb2]), &norb2, &(dd->jk.d_buf2[ib_max*norb2]), &norb2, 
                &beta, dd->fci.d_tdm2, &norb2);
        if ((stra_id >= ia_bra) && (stra_id < ja_bra)){
          double * bravec = &(dd->fci.d_cibra[(stra_id-ia_bra)*nb_bra]);
          ctx.ml->gemv((char *) "N", &norb2, &nb_bra, &alpha, 
                &(dd->jk.d_buf1[ib_bra*nb]), &norb2, bravec, &one, 
                &beta, dd->fci.d_tdm1, &one);
        }
      }

      if ((stra_id>=ia_ket) && (stra_id<ja_ket)){

      ctx.ml->memset(dd->jk.d_buf1, &zero, &bits_buf); // can be optimized
 
      //when populated, rdm3h_b has the capability to populate the entire matrix, but buf2 is still blocked zero from a
      //can rdm3h_b take in what should be range of str0 (nb) because we are only need a specific range here (ib_bra -> jb_bra)
      //similar to the plot above of rdm3h_a * rdm3h_b, but buf1 is fully filled. 
      compute_FCIrdm3h_b_t1ci_v2(dd->fci.d_ciket, dd->jk.d_buf1, stra_id, nb, nb_bra, norb, nlinkb, ia_ket, ja_ket, ib_ket, jb_ket, dd->fci.d_clinkb); 
      ctx.ml->gemm((char *) "N", (char *) "T", &norb2, &norb2, &nb_bra, &alpha, 
               &(dd->jk.d_buf1[ib_bra*norb2]),&norb2, &(dd->jk.d_buf2[ib_bra*norb2]), &norb2, 
               &beta, dd->fci.d_tdm2_p, &norb2);
      }
      ctx.ml->memset(dd->jk.d_buf2, &zero, &bits_buf); //can be optimized based
      ctx.ml->memset(dd->jk.d_buf1, &zero, &bits_buf); 
    }
  }
  transpose_jikl(dd->fci.d_tdm2, dd->jk.d_buf1, norb);
  transpose_jikl(dd->fci.d_tdm2_p, dd->jk.d_buf2, norb);

  ctx.pm->dev_profile_stop();
  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1-t0); //TODO: fix this
  
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Leaving Device::compute_tdm13h_spin_v4()\n");
#endif
} 
/* ---------------------------------------------------------------------- */
void DeviceFci::compute_tdm13h_spin_v5(int na, int nb, 
                                 int nlinka, int nlinkb, 
                                 int norb, int spin, int _reorder,
                                 int ia_bra, int ja_bra, int ib_bra, int jb_bra, int sgn_bra, 
                                 int ia_ket, int ja_ket, int ib_ket, int jb_ket, int sgn_ket, int count )
{
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Inside Device::compute_tdm13h_spin_v5()\n");
#endif
  
  //na, nb is same for both zero-padded ci vectors, but not necessarily for non padded vectors
  double t0 = omp_get_wtime();
  int id = count % ctx.num_devices;
  ctx.pm->dev_set_device(id);
  ctx.ml->set_handle(id);
  my_device_data * dd = &(ctx.device_data[id]);
  ctx.pm->dev_profile_start("tdms :: compute_tdm13h_spin_v4");

  int na_bra = ja_bra - ia_bra;
  int nb_bra = jb_bra - ib_bra;
  int na_ket = ja_ket - ia_ket;
  int nb_ket = jb_ket - ib_ket;
  int ia_max = _MAX(ia_bra, ia_ket);
  int ja_min = _MIN(ja_bra, ja_ket);
  int ib_max = _MAX(ib_ket, ib_ket);
  int jb_min = _MIN(jb_bra, jb_ket);
  int zero = 0;
  int one = 1;

  const double alpha = 1.0*sgn_bra*sgn_ket;
  const double beta = 0.0;
  const double beta_one = 1.0;

  int norb2 = norb*norb;
  int size_tdm1 = norb2;
  int size_tdm2 = norb2*norb2;
  int size_buf = norb2*nb;
  int bits_tdm1 = sizeof(double)*size_tdm1;
  int bits_tdm2 = sizeof(double)*size_tdm2;

  int _size_buf = _MAX(dd->jk.size_buf1, dd->jk.size_buf2);// (dd->jk.size_buf1 > dd->jk.size_buf2) ? dd->jk.size_buf1 : dd->jk.size_buf2;
  int final_size_buf = _MAX(_size_buf, size_buf);//(_size_buf > size_buf) ? _size_buf : size_buf;
  final_size_buf = _MAX(final_size_buf, size_tdm2);//(_size_buf > size_buf) ? _size_buf : size_buf;

  int buf_batch_size = final_size_buf/size_buf; //this is integer division // number of buf1/2 in a single buffer
  int gemm_batch_size = final_size_buf/size_tdm2; // this is integer division // number of tdm2 in a single buf
  int num_buf_batches; 
  int num_gemm_batches; 

  ::grow_array(ctx.pm, dd->jk.d_buf1,final_size_buf, dd->jk.size_buf1, "buf1", FLERR); 
  ::grow_array(ctx.pm, dd->jk.d_buf2,final_size_buf, dd->jk.size_buf2, "buf2", FLERR); 
  ::grow_array(ctx.pm, dd->jk.d_buf3,final_size_buf, dd->jk.size_buf3, "buf3", FLERR); 
  size_t bits_buf = sizeof(double)*buf_batch_size*size_buf;

  ctx.ml->memset(dd->jk.d_buf1, &zero, &bits_buf); 
  ctx.ml->memset(dd->jk.d_buf2, &zero, &bits_buf); 

  ::grow_array(ctx.pm, dd->fci.d_tdm1, size_tdm1, dd->fci.size_tdm1, "tdm1", FLERR);
  ::grow_array(ctx.pm, dd->fci.d_tdm2, size_tdm2, dd->fci.size_tdm2, "tdm2", FLERR); 
  ::grow_array(ctx.pm, dd->fci.d_tdm2_p, size_tdm2, dd->fci.size_tdm2_p, "tdm2_p", FLERR); 

  ctx.ml->memset(dd->fci.d_tdm1, &zero, &bits_tdm1);
  ctx.ml->memset(dd->fci.d_tdm2, &zero, &bits_tdm2);
  ctx.ml->memset(dd->fci.d_tdm2_p, &zero, &bits_tdm2);
 


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

  */

  if (spin){
    int start_id;
    int end_id; 
    int buf_starting_index;
    int bravec_starting_index;
    int num_gemv_batches;
    for (int stra_id = ia_ket; stra_id<ja_ket; stra_id+=buf_batch_size){
        num_buf_batches = _MIN(buf_batch_size, ja_ket-stra_id);
        compute_FCIrdm3h_b_t1ci_v3(dd->fci.d_cibra, dd->jk.d_buf2, stra_id, num_buf_batches, nb, nb_bra, norb, nlinkb, ia_bra, ja_bra, ib_bra, jb_bra, dd->fci.d_clinkb);
        //compute_FCIrdm3h_b_t1ci_v2(dd->fci.d_cibra, dd->jk.d_buf2, stra_id, nb, nb_bra, norb, nlinkb, ia_bra, ja_bra, ib_bra, jb_bra, dd->fci.d_clinkb);
        //if ((stra_id >= ia_ket) && (stra_id < ja_ket)) {
        //buf1 is 0, so tdm3hb and tdm1hb don't calculate anything
        
        compute_FCIrdm3h_b_t1ci_v3(dd->fci.d_ciket, dd->jk.d_buf1, stra_id, num_buf_batches, nb, nb_ket, norb, nlinkb, ia_ket, ja_ket, ib_ket, jb_ket, dd->fci.d_clinkb);
        //compute_FCIrdm3h_b_t1ci_v2(dd->fci.d_ciket, dd->jk.d_buf1, stra_id, nb, nb_ket, norb, nlinkb, ia_ket, ja_ket, ib_ket, jb_ket, dd->fci.d_clinkb);
        //ctx.ml->gemm((char *) "N", (char *) "T", &norb2, &norb2, &nb, &alpha, 
        //        dd->jk.d_buf1, &norb2, dd->jk.d_buf2, &norb2, 
        //        &beta, dd->fci.d_tdm2, &norb2);
        for (int i=0; i<num_buf_batches; i+=gemm_batch_size) {
          num_gemm_batches = _MIN(gemm_batch_size, num_buf_batches-i);
          ctx.ml->gemm_batch((char *) "N",(char *) "T", &norb2, &norb2, &nb, 
            &alpha, 
            &(dd->jk.d_buf1[i*size_buf]), &norb2, &size_buf, 
            &(dd->jk.d_buf2[i*size_buf]), &norb2, &size_buf, 
            &beta, dd->jk.d_buf3, &norb2, &size_tdm2, &num_gemm_batches); 
          reduce_buf3_to_rdm(dd->jk.d_buf3, dd->fci.d_tdm2, size_tdm2, num_gemm_batches);
          }

        /*  10 possibilities
            a                  |---------na_bra---------------------|
         |--batch---|     |--batch---|         |--batch---|      |--batch---|   |--batch---|          
            a1                 a2                  a3                a4             a5
                               |--batch---|              |--batch---|
                                  ae1                        ae2
                    |--batch---|                                    |--batch---|
                         ae3                                           ae4

            b                  |---------batch---------------------|
         |--na_bra--|     |--na_bra--|         |--na_bra--|      |--na_bra--|   |--na_bra--|          
            b5                 b4                  b3                b2             b1
                               |--batch---|              |--batch---|
                                 be1                        be2
                    |--na_bra--|                                    |--na_bra---|
                         be3                                           be4
          
                               |---------batch---------------------|
                               |---------na_bra--------------------|
                                         abe1
         for both a and b set of cases
         if stra_id<ja_bra, stra_id+batch>=ia_bra //when in a1, a5, b1, b5
           start is always start_id = max(stra_id, ia_bra)
           end is always end_id = min(stra_id+batch, ja_bra)

         buf_starting_index = start_id-stra_id;
         bravec_starting_index = start_id - ia_bra;
         num_gemv_batches = end_id - start_id
         
         calculations only happen when num_gemv_batches is positive

          | cases | ia_b| ja_b| batch| stra_id | stra_id+batch | start_id | end_id | buf_startidx| bravec_startidx| gemv_batch |
          |-------|-----|-----|------|---------|---------------|----------|--------|-------------|----------------|------------|
          | a1    | 20  | 40  | 8    | 5       | 13            | 20       | 13     | 15          | 0              | -7         |
          | a2    | 20  | 40  | 8    | 17      | 25            | 20       | 25     | 3           | 0              | 5          |
          | a3    | 20  | 40  | 8    | 28      | 36            | 28       | 36     | 0           | 8              | 8          |
          | a4    | 20  | 40  | 8    | 38      | 46            | 38       | 40     | 0           | 18             | 2          |
          | a5    | 20  | 40  | 8    | 50      | 58            | 50       | 40     | 0           | 30             | -10        |
          | ae1   | 20  | 40  | 8    | 20      | 28            | 20       | 28     | 0           | 0              | 8          |
          | ae2   | 20  | 40  | 8    | 32      | 40            | 32       | 40     | 0           | 12             | 8          |
          | ae3   | 20  | 40  | 8    | 12      | 20            | 20       | 20     | 8           | 0              | 0          |
          | ae4   | 20  | 40  | 8    | 40      | 48            | 40       | 40     | 0           | 20             | 0          |
          | b1    | 20  | 30  | 15   | 0       | 15            | 20       | 15     | 20          | 0              | -5         |
          | b2    | 20  | 30  | 15   | 10      | 25            | 20       | 25     | 10          | 0              | 5          |
          | b3    | 20  | 30  | 15   | 17      | 32            | 20       | 30     | 3           | 0              | 10         |
          | b4    | 20  | 30  | 15   | 25      | 40            | 25       | 30     | 0           | 5              | 5          |
          | b5    | 20  | 30  | 15   | 35      | 50            | 35       | 30     | 0           | 15             | -5         |
          | be1   | 20  | 30  | 15   | 20      | 35            | 20       | 30     | 0           | 0              | 10         |
          | be2   | 20  | 30  | 15   | 15      | 30            | 20       | 30     | 5           | 0              | 10         |
          | be3   | 20  | 30  | 15   | 5       | 20            | 20       | 20     | 15          | 0              | 0          |
          | be4   | 20  | 30  | 15   | 30      | 45            | 30       | 30     | 0           | 10             | 0          |
          | abe1  | 20  | 40  | 20   | 20      | 40            | 20       | 40     | 0           | 0              | 20         |

         */
         
         
        start_id = _MAX(stra_id, ia_bra);
        end_id = _MIN(stra_id + num_buf_batches, ja_bra);
        num_gemv_batches = start_id-end_id;
        if (num_gemv_batches > 0){
          buf_starting_index = start_id - stra_id;
          bravec_starting_index = start_id - ia_bra;
          double * bravec = &(dd->fci.d_cibra[bravec_starting_index*nb_bra]);
          double * buf_mat = &(dd->jk.d_buf1[buf_starting_index*size_buf]);
          ctx.ml->gemv_batch((char *) "N", &norb2, &nb_bra, &alpha,
                         &(buf_mat[ib_bra*norb2]), &norb2, &size_buf,
                         bravec, &one, &nb_bra,
                         &beta, dd->jk.d_buf3, &one, &norb2, &num_gemv_batches);
           
          //double * bravec = &(dd->fci.d_cibra[(stra_id-ia_bra)*nb_bra]);
          //ctx.ml->gemv((char *) "N", &norb2, &nb_bra, &alpha, 
          //      &(dd->jk.d_buf1[ib_bra*norb2]), &norb2, bravec, &one, 
          //      &beta, dd->fci.d_tdm1, &one);
          reduce_buf3_to_rdm(dd->jk.d_buf3, dd->fci.d_tdm1, size_tdm1, num_gemv_batches);
          }
        ctx.ml->memset(dd->jk.d_buf1, &zero, &bits_buf);

        //compute_FCIrdm3h_a_t1ci_v2(dd->fci.d_ciket, dd->jk.d_buf1, stra_id, nb_ket, norb, nlinka, ia_ket, ja_ket, ib_ket, jb_ket, dd->fci.d_clinka);
        compute_FCIrdm3h_a_t1ci_v3(dd->fci.d_ciket, dd->jk.d_buf1, stra_id, num_buf_batches, nb, nb_ket, norb, nlinka, ia_ket, ja_ket, ib_ket, jb_ket, dd->fci.d_clinka);
        // buf1 is only populated from ib_ket:jb_ket, so don't need to run the multiplication over the whole thing 
        for (int i=0; i<num_buf_batches; i+=gemm_batch_size) {
          num_gemm_batches = _MIN(gemm_batch_size, num_buf_batches-i);
          ctx.ml->gemm_batch((char *) "N",(char *) "T", &norb2, &norb2, &nb_ket, 
            &alpha, 
            &(dd->jk.d_buf2[ib_ket*norb2]), &norb2, &size_buf,
            &(dd->jk.d_buf1[ib_ket*norb2]), &norb2, &size_buf,//remember the switch?
            &beta, 
            dd->jk.d_buf3, &norb2, &size_tdm2, &num_gemm_batches); 
          reduce_buf3_to_rdm(dd->jk.d_buf3, dd->fci.d_tdm2_p, size_tdm2, num_gemm_batches);
          }

        //ctx.ml->gemm((char *) "N", (char *) "T", &norb2, &norb2, &nb_ket, &alpha, 
        //       &(dd->jk.d_buf2[ib_ket*norb2]), &norb2, &(dd->jk.d_buf1[ib_ket*norb2]), &norb2, //remember the switch?
        //       &beta, dd->fci.d_tdm2_p, &norb2);
        ctx.ml->memset(dd->jk.d_buf2, &zero, &bits_buf);
        ctx.ml->memset(dd->jk.d_buf1, &zero, &bits_buf);
      } // for main loop
    } //for full if
  else {

    int ib_max = (ib_bra > ib_ket) ? ib_bra : ib_ket;
    int jb_min = (jb_bra < jb_ket) ? jb_bra : jb_ket;
    int b_len  = jb_min - ib_max;
    int start_id;
    int end_id;
    int num_gemv_batches;
    int num_total_gemm_batches;
    int buf_starting_index;
    int bravec_starting_index;
    //for (int stra_id = 0; stra_id<na; ++stra_id){
    for (int stra_id = 0; stra_id<na; stra_id += buf_batch_size){
      num_buf_batches = _MIN(buf_batch_size, na-stra_id);
        /* buf2      buf1              tdm2      bravec  
          0 0 0 0   0 0 0 0          # # # #     0  
  ib_bra  # # # #   0 0 0 0          # # # #     # ib_bra
          # # # #   # # # # ib_ket   # # # #     #
  jb_bra  # # # #   # # # #          # # # #     # jb_bra
          0 0 0 0   # # # # jb_ket               0 
          0 0 0 0   0 0 0 0                      0  
          
          given buf2, don't need to calculate from all ib_ket to jb_ket for buf1, can only do max(ib_bra, ib_ket) to min(jb_bra, jb_ket)
        */

      //compute_FCIrdm3h_a_t1ci_v2(dd->fci.d_cibra, dd->jk.d_buf2, stra_id, nb_bra, norb, nlinka, ia_bra, ja_bra, ib_bra, jb_bra, dd->fci.d_clinka);
      compute_FCIrdm3h_a_t1ci_v3(dd->fci.d_cibra, dd->jk.d_buf2, stra_id, num_buf_batches, nb, nb_bra, norb, nlinka, ia_bra, ja_bra, ib_bra, jb_bra, dd->fci.d_clinka);
      if (b_len>0){
        //compute_FCIrdm3h_a_t1ci_v2(dd->fci.d_ciket, dd->jk.d_buf1, stra_id, nb_ket, norb, nlinka, ia_ket, ja_ket, ib_max, jb_min, dd->fci.d_clinka);// !limits
        compute_FCIrdm3h_a_t1ci_v3(dd->fci.d_ciket, dd->jk.d_buf1, stra_id, num_buf_batches, nb, nb_ket, norb, nlinka, ia_ket, ja_ket, ib_max, jb_min, dd->fci.d_clinka);// !limits


        //ctx.ml->gemm((char *) "N", (char *) "T", &norb2, &norb2, &b_len, &alpha, 
        //        &(dd->jk.d_buf1[ib_max*norb2]), &norb2, &(dd->jk.d_buf2[ib_max*norb2]), &norb2, 
        //        &beta, dd->fci.d_tdm2, &norb2);
        for (int i=0; i<num_buf_batches; i+=gemm_batch_size) {
          num_gemm_batches = _MIN(gemm_batch_size, num_buf_batches-i);
          ctx.ml->gemm_batch((char *) "N",(char *) "T", &norb2, &norb2, &b_len, 
            &alpha, 
            &(dd->jk.d_buf1[i*size_buf+ib_max*norb2]), &norb2, &size_buf, 
            &(dd->jk.d_buf2[i*size_buf+ib_max*norb2]), &norb2, &size_buf, 
            &beta, dd->jk.d_buf3, &norb2, &size_tdm2, &num_gemm_batches); 
          reduce_buf3_to_rdm(dd->jk.d_buf3, dd->fci.d_tdm2, size_tdm2, num_gemm_batches);
          }

        //if ((stra_id >= ia_bra) && (stra_id < ja_bra)){
        //  double * bravec = &(dd->fci.d_cibra[(stra_id-ia_bra)*nb_bra]);
        //  ctx.ml->gemv((char *) "N", &norb2, &nb_bra, &alpha, 
        //        &(dd->jk.d_buf1[ib_bra*nb]), &norb2, bravec, &one, 
        //        &beta, dd->fci.d_tdm1, &one);
        //using similar logic from before
        start_id = _MAX(stra_id, ia_bra);
        end_id = _MIN(stra_id + num_buf_batches, ja_bra);
        num_gemv_batches = start_id-end_id;
        if (num_gemv_batches > 0){
          buf_starting_index = start_id - stra_id;
          bravec_starting_index = start_id - ia_bra;
          printf("stra_id:%i num_buf_batches:%i buf_starting_index:%i bravec_starting_index:%i num_gemv_batches:%i\n",stra_id, num_buf_batches, buf_starting_index, bravec_starting_index, num_gemv_batches);
          double * bravec = &(dd->fci.d_cibra[bravec_starting_index*nb_bra]);
          double * buf_mat = &(dd->jk.d_buf1[buf_starting_index*size_buf]);
          ctx.ml->gemv_batch((char *) "N", &norb2, &nb_bra, &alpha,
                         &(buf_mat[ib_bra*norb2]), &norb2, &size_buf,
                         bravec, &one, &nb_bra,
                         &beta, dd->jk.d_buf3, &one, &norb2, &num_gemv_batches); 
          reduce_buf3_to_rdm(dd->jk.d_buf3, dd->fci.d_tdm1, size_tdm1, num_gemv_batches);
          }

        }

      //if ((stra_id>=ia_ket) && (stra_id<ja_ket)){
      //similar logic as before but from ia_ket to ja_ket
      start_id = _MAX(stra_id, ia_ket);
      end_id = _MIN(stra_id + num_buf_batches, ja_ket);
      num_total_gemm_batches = start_id-end_id;//will be less than or equal to num_buf_batches, denotes the total gemms
      if (num_total_gemm_batches > 0){
        int buf1_starting_index = start_id - ia_bra;//goes as stra_id in b_t1ci,
        int buf2_starting_index = start_id - stra_id;//goes into buf2 for gemm

        ctx.ml->memset(dd->jk.d_buf1, &zero, &bits_buf); // can be optimized
        //when populated, rdm3h_b has the capability to populate the entire matrix, but buf2 is still blocked zero from a
        //can rdm3h_b take in what should be range of str0 (nb) because we are only need a specific range here (ib_bra -> jb_bra)

        //compute_FCIrdm3h_b_t1ci_v2(dd->fci.d_ciket, dd->jk.d_buf1, stra_id, nb, nb_bra, norb, nlinkb, ia_ket, ja_ket, ib_ket, jb_ket, dd->fci.d_clinkb); 
       
        compute_FCIrdm3h_b_t1ci_v3(dd->fci.d_ciket, dd->jk.d_buf1, buf1_starting_index, num_total_gemm_batches, nb, nb_bra, norb, nlinkb, ia_ket, ja_ket, ib_ket, jb_ket, dd->fci.d_clinkb); //remember that this fills up buf1 starting from 0 to upto num_total_gemm_batches
         
        //similar to the plot above of rdm3h_a * rdm3h_b, but buf1 is fully filled. 
        //ctx.ml->gemm((char *) "N", (char *) "T", &norb2, &norb2, &nb_bra, &alpha, 
        //       &(dd->jk.d_buf1[ib_bra*norb2]),&norb2, &(dd->jk.d_buf2[ib_bra*norb2]), &norb2, 
        //       &beta, dd->fci.d_tdm2_p, &norb2);
        for (int i=0; i<num_total_gemm_batches; i+=gemm_batch_size) {
          num_gemm_batches = _MIN(gemm_batch_size, num_total_gemm_batches-i);
          ctx.ml->gemm_batch((char *) "N",(char *) "T", &norb2, &norb2, &nb, 
            &alpha, 
            &(dd->jk.d_buf1[i*size_buf+ib_bra*norb2]), &norb2, &size_buf, 
            &(dd->jk.d_buf2[i*size_buf+ib_bra*norb2]), &norb2, &size_buf, 
            &beta, dd->jk.d_buf3, &norb2, &size_tdm2, &num_gemm_batches); 
          reduce_buf3_to_rdm(dd->jk.d_buf3, dd->fci.d_tdm2_p, size_tdm2, num_gemm_batches);
          }

      }//tdm2_p
      ctx.ml->memset(dd->jk.d_buf2, &zero, &bits_buf); //can be optimized based
      ctx.ml->memset(dd->jk.d_buf1, &zero, &bits_buf); 
    }
    }//for full else
  transpose_jikl(dd->fci.d_tdm2, dd->jk.d_buf1, norb);
  transpose_jikl(dd->fci.d_tdm2_p, dd->jk.d_buf2, norb);

  ctx.pm->dev_profile_stop();
  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1-t0); //TODO: fix this
  
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Leaving Device::compute_tdm13h_spin_v5()\n");
#endif
} 

/* ---------------------------------------------------------------------- */
void DeviceFci::compute_tdmpp_spin_v4(int na, int nb, int nlinka, int nlinkb, int norb, int spin,
                                 int ia_bra, int ja_bra, int ib_bra, int jb_bra, int sgn_bra, 
                                 int ia_ket, int ja_ket, int ib_ket, int jb_ket, int sgn_ket, int count )
{
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Inside Device::compute_tdmpp_spin_v4() w/ spin= %i\n", spin);
#endif
  
  //even though the cpu version of tdmpp does tdm1 and tdm2, tdm1 gets "absorbed" into tdm2 by reorder function.
  //for this function specifcally, reorder does not do anything, therefore, any calculation of tdm1 is meaningless.
  //we just need to filder tdm2 to tdm1 (see sfudm)
  double t0 = omp_get_wtime();
  int id = count % ctx.num_devices;
  ctx.pm->dev_set_device(id);
  ctx.ml->set_handle(id);
  my_device_data * dd = &(ctx.device_data[id]);
  ctx.pm->dev_profile_start("tdms :: compute_tdmpp_spin_v4");
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
  int _size_buf = _MAX(dd->jk.size_buf1, dd->jk.size_buf2);// (dd->jk.size_buf1 > dd->jk.size_buf2) ? dd->jk.size_buf1 : dd->jk.size_buf2;
  _size_buf = _MAX(_size_buf, dd->jk.size_buf3);
  #ifdef _TEMP_BUFSIZING
  _size_buf = size_buf*6;
  #endif
  int final_size_buf = _MAX(_size_buf, size_buf);//(_size_buf > size_buf) ? _size_buf : size_buf;
  final_size_buf = _MAX(final_size_buf, size_tdm2);//(_size_buf > size_buf) ? _size_buf : size_buf;
  int buf_batch_size = final_size_buf/size_buf; //this is integer division // number of buf1/2 in a single buffer
  int gemm_batch_size = final_size_buf/(norb2*norb2); // this is integer division // number of tdm2 in a single buf
  int num_buf_batches; 
  int num_gemm_batches; 
  ::grow_array(ctx.pm, dd->jk.d_buf1, final_size_buf, dd->jk.size_buf1, "buf1", FLERR); 
  ::grow_array(ctx.pm, dd->jk.d_buf2, final_size_buf, dd->jk.size_buf2, "buf2", FLERR); 
  ::grow_array(ctx.pm, dd->jk.d_buf3, final_size_buf, dd->jk.size_buf3, "buf3", FLERR); 
  size_t bits_buf = sizeof(double)*buf_batch_size*size_buf;
  size_t bits_buf1;
  size_t bits_buf2;
  size_t bits_buf3;
  ctx.ml->memset(dd->jk.d_buf1, &zero, &bits_buf); 
  ctx.ml->memset(dd->jk.d_buf2, &zero, &bits_buf); 
  ::grow_array(ctx.pm, dd->fci.d_tdm1, size_tdm1, dd->fci.size_tdm1, "tdm1", FLERR);
  ::grow_array(ctx.pm, dd->fci.d_tdm2, size_tdm2, dd->fci.size_tdm2, "tdm2", FLERR); 
  ctx.ml->memset(dd->fci.d_tdm2, &zero, &bits_tdm2);
  
 /*
 tdm12kern_a
    a_t1ci: cibra, clinka -> buf2
    a_t1ci: ciket, clinka -> buf1
    tdm2 = gemm buf1, buf2
  tdm12kern_b 
    b_t1ci: cibra, clinkb -> buf2
    b_t1ci: ciket, clinkb -> buf1
    tdm2 = gemm buf1, buf2
  tdm12kern_ab
    a_t1ci: cibra, clinka -> buf2
    b_t1ci: ciket, clinkb -> buf1
    tdm2 = gemm buf1, buf2
   
  if spin == 0
    tdm1, tdm2 = tdm12kern_a, cibra, ciket, get 2 
  if spin == 2
    tdm1, tdm2 = tdm12kern_b, cibra, ciket, get 2
  if spin == 1
    tdm1, tdm2 = tdm12kern_ab, cibra, ciket, get 2
  */
  int ib_max = _MAX(ib_bra, ib_ket);
  int jb_min = _MIN(jb_bra, jb_ket);
  int ia_max = _MAX(ia_bra, ia_ket);
  int ja_min = _MIN(ja_bra, ja_ket);
  int b_len  = jb_min - ib_max;
  
  if (spin== 0)
      //refer to diagram in tdm3h_spin_v4
      {
      bits_buf1 = sizeof(double)*nb_ket*norb2;//rdm3h_a fills only ib:jb
      bits_buf2 = sizeof(double)*nb_bra*norb2;//rdm3h_a fills only ib:jb
      for (int stra_id = 0; stra_id<na; stra_id += buf_batch_size){
        num_buf_batches = _MIN(buf_batch_size, na-stra_id);
        compute_FCIrdm3h_a_t1ci_v3(dd->fci.d_cibra, dd->jk.d_buf2, stra_id, num_buf_batches, nb, nb_bra, norb, nlinka, ia_bra, ja_bra, ib_bra, jb_bra, dd->fci.d_clinka);
        compute_FCIrdm3h_a_t1ci_v3(dd->fci.d_ciket, dd->jk.d_buf1, stra_id, num_buf_batches, nb, nb_ket, norb, nlinka, ia_ket, ja_ket, ib_ket, jb_ket, dd->fci.d_clinka);
        for (int i=0; i<num_buf_batches; i+=gemm_batch_size) {
          num_gemm_batches = _MIN(gemm_batch_size, num_buf_batches-i);
          ctx.ml->gemm_batch((char *) "N",(char *) "T", &norb2, &norb2, &nb, 
            &alpha, 
            &(dd->jk.d_buf1[i*size_buf]), &norb2, &size_buf, 
            &(dd->jk.d_buf2[i*size_buf]), &norb2, &size_buf, 
            &beta, dd->jk.d_buf3, &norb2, &size_tdm2, &num_gemm_batches); 
          reduce_buf3_to_rdm(dd->jk.d_buf3, dd->fci.d_tdm2, size_tdm2, num_gemm_batches);
          }
	ctx.utils->memset_zero_batch_stride(dd->jk.d_buf1, size_buf, ib_ket*norb2, nb_ket*norb2, num_buf_batches);
	ctx.utils->memset_zero_batch_stride(dd->jk.d_buf2, size_buf, ib_bra*norb2, nb_bra*norb2, num_buf_batches);
        }
      }
    else if (spin==1) { 
        bits_buf2 = sizeof(double)*nb_bra*norb2;//rdm3h_a fills only ib:jb
        for (int stra_id = ia_ket; stra_id<ja_ket; stra_id += buf_batch_size){
          num_buf_batches = _MIN(buf_batch_size, ja_ket-stra_id);
          compute_FCIrdm3h_a_t1ci_v3(dd->fci.d_cibra, dd->jk.d_buf2, stra_id, num_buf_batches, nb, nb_bra, norb, nlinka, ia_bra, ja_bra, ib_bra, jb_bra, dd->fci.d_clinka);
          compute_FCIrdm3h_b_t1ci_v3(dd->fci.d_ciket, dd->jk.d_buf1, stra_id, num_buf_batches, nb, nb_ket, norb, nlinkb, ia_ket, ja_ket, ib_ket, jb_ket, dd->fci.d_clinkb);

          for (int i=0; i<num_buf_batches; i+=gemm_batch_size) {
            num_gemm_batches = _MIN(gemm_batch_size, num_buf_batches-i);
            ctx.ml->gemm_batch((char *) "N",(char *) "T", &norb2, &norb2, &nb, 
              &alpha, 
              &(dd->jk.d_buf1[i*size_buf]), &norb2, &size_buf, 
              &(dd->jk.d_buf2[i*size_buf]), &norb2, &size_buf, 
              &beta, dd->jk.d_buf3, &norb2, &size_tdm2, &num_gemm_batches); 
            reduce_buf3_to_rdm(dd->jk.d_buf3, dd->fci.d_tdm2, size_tdm2, num_gemm_batches);
            }
          ctx.ml->memset(dd->jk.d_buf1, &zero, &bits_buf);
	  ctx.utils->memset_zero_batch_stride(dd->jk.d_buf2, size_buf, ib_bra*norb2, nb_bra*norb2, num_buf_batches);
          }
        } 
    else if (spin==2){
       for (int stra_id = ia_max; stra_id<ja_min; stra_id += buf_batch_size){
         num_buf_batches = _MIN(buf_batch_size, ja_min-stra_id);
         compute_FCIrdm3h_b_t1ci_v3(dd->fci.d_cibra, dd->jk.d_buf2, stra_id, num_buf_batches, nb, nb_bra, norb, nlinkb, ia_bra, ja_bra, ib_bra, jb_bra, dd->fci.d_clinkb);
         compute_FCIrdm3h_b_t1ci_v3(dd->fci.d_ciket, dd->jk.d_buf1, stra_id, num_buf_batches, nb, nb_ket, norb, nlinkb, ia_ket, ja_ket, ib_ket, jb_ket, dd->fci.d_clinkb);
         
         for (int i=0; i<num_buf_batches; i+=gemm_batch_size) {
           num_gemm_batches = _MIN(gemm_batch_size, num_buf_batches-i);
           ctx.ml->gemm_batch((char *) "N",(char *) "T", &norb2, &norb2, &nb, 
             &alpha, 
             &(dd->jk.d_buf1[i*size_buf]), &norb2, &size_buf, 
             &(dd->jk.d_buf2[i*size_buf]), &norb2, &size_buf, 
             &beta, dd->jk.d_buf3, &norb2, &size_tdm2, &num_gemm_batches); 
           reduce_buf3_to_rdm(dd->jk.d_buf3, dd->fci.d_tdm2, size_tdm2, num_gemm_batches);
         }
         
         ctx.ml->memset(dd->jk.d_buf1, &zero, &bits_buf);
         ctx.ml->memset(dd->jk.d_buf2, &zero, &bits_buf);
         } 
       }
  transpose_jikl(dd->fci.d_tdm2, dd->jk.d_buf1, norb);

  filter_tdmpp (dd->fci.d_tdm2, dd->fci.d_tdm1, norb, spin);
  ctx.pm->dev_profile_stop();
  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1-t0); //TODO: fix this
  
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Leaving Device::compute_tdmpp_spin_v4()\n");
#endif
  
}


/* ---------------------------------------------------------------------- */
void DeviceFci::compute_sfudm_v2(int na, int nb, int nlinka, int nlinkb, int norb, 
                             int ia_bra, int ja_bra, int ib_bra, int jb_bra, int sgn_bra, 
                             int ia_ket, int ja_ket, int ib_ket, int jb_ket, int sgn_ket, int count )
{
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Inside Device::compute_sfudm_v2()\n");
#endif
  
  double t0 = omp_get_wtime();
  int id = count % ctx.num_devices;
  ctx.pm->dev_set_device(id);
  ctx.ml->set_handle(id);
  my_device_data * dd = &(ctx.device_data[id]);
  ctx.pm->dev_profile_start("tdms :: compute_sfudm_v2");
  int norb2 = norb*norb;
  int size_buf = norb2*nb;
  int size_tdm2 = norb2*norb2;
  int size_tdm1 = norb2;
  int zero = 0;
  int one = 1;
  //const double alpha = 1.0;
  const double alpha = -1.0*sgn_bra*sgn_ket;//negative is returned
  const double beta = 0.0;
  int na_bra = ja_bra - ia_bra;
  int nb_bra = jb_bra - ib_bra;
  int na_ket = ja_ket - ia_ket;
  int nb_ket = jb_ket - ib_ket;
  int bits_tdm2 = sizeof(double)*size_tdm2;
  int _size_buf = (dd->jk.size_buf1 > dd->jk.size_buf2) ? dd->jk.size_buf1 : dd->jk.size_buf2;
  #ifdef _TEMP_BUFSIZING
  _size_buf = size_buf*6;
  #endif
  int final_size_buf = _MAX(_size_buf, size_buf);//(_size_buf > size_buf) ? _size_buf : size_buf;
  final_size_buf = _MAX(final_size_buf, size_tdm2);//(_size_buf > size_buf) ? _size_buf : size_buf;
  int buf_batch_size = final_size_buf/size_buf; //this is integer division // number of buf1/2 in a single buffer
  int gemm_batch_size = final_size_buf/(norb2*norb2); // this is integer division // number of tdm2 in a single buf
  
  int num_buf_batches; 
  int num_gemm_batches; 
  ::grow_array(ctx.pm, dd->jk.d_buf1,final_size_buf, dd->jk.size_buf1, "buf1", FLERR); 
  ::grow_array(ctx.pm, dd->jk.d_buf2,final_size_buf, dd->jk.size_buf2, "buf2", FLERR); 
  ::grow_array(ctx.pm, dd->jk.d_buf3,final_size_buf, dd->jk.size_buf3, "buf3", FLERR); 
  size_t bits_buf = sizeof(double)*buf_batch_size*size_buf;
  size_t bits_buf3;
  //printf("total_size: %i nb_ket: %i\n", final_size_buf, nb_ket);
  ::grow_array(ctx.pm, dd->fci.d_tdm1, size_tdm1, dd->fci.size_tdm1, "tdm1", FLERR); 
  ::grow_array(ctx.pm, dd->fci.d_tdm2, size_tdm2, dd->fci.size_tdm2, "tdm2", FLERR); 
  ctx.ml->memset(dd->jk.d_buf1, &zero, &bits_buf); 
  ctx.ml->memset(dd->jk.d_buf2, &zero, &bits_buf); 
  ctx.ml->memset(dd->fci.d_tdm2, &zero, &bits_tdm2);
  
  /*
  tdm12kern_ab
    a_t1ci: cibra, clinka -> buf2
    b_t1ci: ciket, clinkb -> buf1
    tdm2 = gemm buf1, buf2
  */
  int bra_b_len = jb_bra - ib_bra;
  size_t bits_buf2 = sizeof(double)*nb_bra*norb2;
  for (int stra_id = ia_ket; stra_id<ja_ket; stra_id += buf_batch_size){
      num_buf_batches = (buf_batch_size < ja_ket - stra_id) ? buf_batch_size : ja_ket - stra_id; 
      compute_FCIrdm3h_a_t1ci_v3(dd->fci.d_cibra, dd->jk.d_buf2, stra_id, num_buf_batches, nb, nb_bra, norb, nlinka, ia_bra, ja_bra, ib_bra, jb_bra, dd->fci.d_clinka);
      compute_FCIrdm3h_b_t1ci_v3(dd->fci.d_ciket, dd->jk.d_buf1, stra_id, num_buf_batches, nb, nb_ket, norb, nlinkb, ia_ket, ja_ket, ib_ket, jb_ket, dd->fci.d_clinkb);

      for (int i=0; i<num_buf_batches; i+=gemm_batch_size) {
        num_gemm_batches = (gemm_batch_size < num_buf_batches - i) ? gemm_batch_size : num_buf_batches - i;
        ctx.ml->gemm_batch((char *) "N",(char *) "T", &norb2, &norb2, &nb, 
          &alpha, 
          &(dd->jk.d_buf1[i*size_buf]), &norb2, &size_buf, 
          &(dd->jk.d_buf2[i*size_buf]), &norb2, &size_buf, 
          &beta, dd->jk.d_buf3, &norb2, &size_tdm2, &num_gemm_batches); 
       reduce_buf3_to_rdm(dd->jk.d_buf3, dd->fci.d_tdm2, size_tdm2, num_gemm_batches);
      }
      ctx.ml->memset(dd->jk.d_buf1, &zero, &bits_buf);
      ctx.utils->memset_zero_batch_stride(dd->jk.d_buf2, size_buf, ib_bra*norb2, nb_bra*norb2, num_buf_batches);
    }
  transpose_jikl(dd->fci.d_tdm2, dd->jk.d_buf1, norb);

  filter_sfudm(&(dd->fci.d_tdm2[norb2*norb*(norb-1)]), dd->fci.d_tdm1, norb);

  ctx.pm->dev_profile_stop();
  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1-t0); //TODO: fix this
  
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Leaving Device::compute_sfudm_v2()\n");
#endif
}

/* ---------------------------------------------------------------------- */
void DeviceFci::compute_tdm1h_spin( int na, int nb, int nlinka, int nlinkb, int norb, int spin, 
                             int ia_bra, int ja_bra, int ib_bra, int jb_bra, int sgn_bra, 
                             int ia_ket, int ja_ket, int ib_ket, int jb_ket, int sgn_ket, int count )
{
  double t0 = omp_get_wtime();
  int id = count % ctx.num_devices;
  ctx.pm->dev_set_device(id);
  ctx.ml->set_handle(id);
  my_device_data * dd = &(ctx.device_data[id]);
  int norb2 = norb*norb;
  int size_tdm1 = norb2;

  ::grow_array(ctx.pm, dd->fci.d_tdm1,size_tdm1, dd->fci.size_tdm1, "tdm1", FLERR); //actual returned
  ctx.utils->set_to_zero(dd->fci.d_tdm1, size_tdm1);
  /* 
     spin = 0: 
       trans_rdm1a: cibra, ciket -> tdm1
     spin = 1:
       trans_rdm1b: cibra, ciket -> tdm1
  */
  if (spin==0)
  {
    compute_FCItrans_rdm1a_v2 (dd->fci.d_cibra, dd->fci.d_ciket, dd->fci.d_tdm1, 
                                norb, nlinka, 
                                ia_bra, ja_bra, ib_bra, jb_bra, 
                                ia_ket, ja_ket, ib_ket, jb_ket, sgn_bra*sgn_ket,  
                                dd->fci.d_clinka);
  }
  else
  {
    compute_FCItrans_rdm1b_v2(dd->fci.d_cibra, dd->fci.d_ciket, dd->fci.d_tdm1,  
                                norb, nlinkb, 
                                ia_bra, ja_bra, ib_bra, jb_bra, 
                                ia_ket, ja_ket, ib_ket, jb_ket, sgn_bra*sgn_ket,  
                                dd->fci.d_clinkb);
  }

  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1 - t0);
}
/* ---------------------------------------------------------------------- */
void DeviceFci::reorder_rdm(int norb, int count)
{
  double t0 = omp_get_wtime();
  int id = count % ctx.num_devices;
  ctx.pm->dev_set_device(id); 
  my_device_data * dd = &(ctx.device_data[id]);
  //get buffer array
  int _size_buf = norb*norb*norb*norb;
  ::grow_array(ctx.pm, dd->jk.d_buf1, _size_buf, dd->jk.size_buf1, "buf1", FLERR);
  reorder(dd->fci.d_tdm1, dd->fci.d_tdm2, dd->jk.d_buf1, norb);
  double t1 = omp_get_wtime();
  //ctx.t_array[30] += t1-t0;
  //ctx.count_array[20]++;

}
/* ---------------------------------------------------------------------- */
void DeviceFci::transpose_tdm2(int norb, int count)
{
  double t0 = omp_get_wtime();
  int id = count % ctx.num_devices;
  ctx.pm->dev_set_device(id); 
  my_device_data * dd = &(ctx.device_data[id]);
  //get buffer array
  int size_tdm2 = norb*norb*norb*norb;
  ctx.utils->transpose_3210(dd->fci.d_tdm2, dd->jk.d_buf2, norb, norb);
  ctx.utils->veccopy(dd->jk.d_buf2, dd->fci.d_tdm2, size_tdm2);
  
  double t1 = omp_get_wtime();
  //ctx.t_array[30] += t1-t0;
  //ctx.count_array[20]++;

}

/* ---------------------------------------------------------------------- */
void DeviceFci::pull_tdm1(py::array_t<double> _tdm1, int norb, int count)
{
  double t0 = omp_get_wtime();
  int id = count % ctx.num_devices;
  ctx.pm->dev_set_device(id); 
  my_device_data * dd = &(ctx.device_data[id]);
  ctx.pm->dev_profile_start("tdms :: pull tdm1");
  py::buffer_info info_tdm1 = _tdm1.request(); //2D array (norb, norb)
  double * tdm1 = static_cast<double*>(info_tdm1.ptr);
  ctx.pm->dev_pull_async(dd->fci.d_tdm1, tdm1, norb*norb*sizeof(double));

  ctx.pm->dev_profile_stop();
  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1-t0);

}
/* ---------------------------------------------------------------------- */
void DeviceFci::pull_tdm2(py::array_t<double> _tdm2, int norb, int count)
{
  double t0 = omp_get_wtime();
  int id = count % ctx.num_devices;
  ctx.pm->dev_set_device(id); 
  my_device_data * dd = &(ctx.device_data[id]);
  ctx.pm->dev_profile_start("tdms :: pull tdm2");
  py::buffer_info info_tdm2 = _tdm2.request(); //4D array (norb, norb, norb, norb)
  double * tdm2 = static_cast<double*>(info_tdm2.ptr);
  ctx.pm->dev_pull_async(dd->fci.d_tdm2, tdm2, norb*norb*norb*norb*sizeof(double));

  ctx.pm->dev_profile_stop();
  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1-t0);
}
/* ---------------------------------------------------------------------- */
void DeviceFci::pull_tdm1_host(int i, int j, int n_bra, int n_ket, int size_tdm1, int factor, int count)
{
  double t0 = omp_get_wtime();
  int id = count % ctx.num_devices;
  ctx.pm->dev_set_device(id); 
  my_device_data * dd = &(ctx.device_data[id]);
  ctx.pm->dev_profile_start("tdms :: pull tdm1");
  int loc_tdm1 = (i*n_ket+j)*size_tdm1;
  double * h_dm1_loc = &(h_dm1_full[loc_tdm1]);
  ctx.pm->dev_pull_async(dd->fci.d_tdm1, h_dm1_loc, size_tdm1*sizeof(double));
  ctx.pm->dev_profile_stop();
   
  if ((factor*(count+1) == n_bra*n_ket)&&(n_ket == j+1) &&(n_bra = i+1)){
    for (int device_id =0; device_id<ctx.num_devices; ++device_id){
      ctx.pm->dev_set_device(device_id); 
      ctx.pm->dev_barrier();
      }
    }
  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1-t0);
}
/* ---------------------------------------------------------------------- */
void DeviceFci::pull_tdm2_host(int i, int j, int n_bra, int n_ket, int size_tdm2, int factor, int count)
{
  double t0 = omp_get_wtime();
  int id = count % ctx.num_devices;
  ctx.pm->dev_set_device(id); 
  my_device_data * dd = &(ctx.device_data[id]);
  ctx.pm->dev_profile_start("tdms :: pull tdm2");
  int loc_tdm2 = (i*n_ket+j)*size_tdm2;
  double * h_dm2_loc = &(h_dm2_full[loc_tdm2]);
  ctx.pm->dev_pull_async(dd->fci.d_tdm2, h_dm2_loc, size_tdm2*sizeof(double));
  ctx.pm->dev_profile_stop();
  
  if (factor*(count+1) == n_bra*n_ket){
    for (int device_id =0; device_id<ctx.num_devices; ++device_id){
      ctx.pm->dev_set_device(device_id); 
      ctx.pm->dev_barrier();
      }
    }

  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1-t0);
}
/* ---------------------------------------------------------------------- */
void DeviceFci::pull_tdm3h_host(int loc, int size_tdm2, int count)
{
  double t0 = omp_get_wtime();
  int id = count % ctx.num_devices;
  ctx.pm->dev_set_device(id); 
  my_device_data * dd = &(ctx.device_data[id]);
  ctx.pm->dev_profile_start("tdms :: pull tdm2");
  ctx.pm->dev_pull_async(dd->fci.d_tdm2, &h_dm2_full[loc], size_tdm2*sizeof(double));
  ctx.pm->dev_pull_async(dd->fci.d_tdm2_p, &h_dm2_p_full[loc], size_tdm2*sizeof(double));
  ctx.pm->dev_profile_stop();
  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1-t0);
}

/* ---------------------------------------------------------------------- */
void DeviceFci::pull_tdm3hab(py::array_t<double> _tdm3ha, py::array_t<double> _tdm3hb, int norb, int count)
{
  double t0 = omp_get_wtime();
  int id = count % ctx.num_devices;
  ctx.pm->dev_set_device(id); 
  my_device_data * dd = &(ctx.device_data[id]);
  ctx.pm->dev_profile_start("tdms :: pull tdm2");
  py::buffer_info info_tdm3ha = _tdm3ha.request(); //4D array (norb, norb, norb, norb)
  double * tdm3ha = static_cast<double*>(info_tdm3ha.ptr);
  ctx.pm->dev_pull_async(dd->fci.d_tdm2, tdm3ha, norb*norb*norb*norb*sizeof(double));
  py::buffer_info info_tdm3hb = _tdm3hb.request(); //4D array (norb, norb, norb, norb)
  double * tdm3hb = static_cast<double*>(info_tdm3hb.ptr);
  ctx.pm->dev_pull_async(dd->fci.d_tdm2_p, tdm3hb, norb*norb*norb*norb*sizeof(double));

  ctx.pm->dev_profile_stop();
  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1-t0);
}
/* ---------------------------------------------------------------------- */
void DeviceFci::pull_tdm3hab_v2(py::array_t<double> _tdm1h, py::array_t<double> _tdm3ha, py::array_t<double> _tdm3hb, int norb, int cre, int spin, int count)
{
  double t0 = omp_get_wtime();
  int id = count % ctx.num_devices;
  ctx.pm->dev_set_device(id); 
  my_device_data * dd = &(ctx.device_data[id]);
  ctx.pm->dev_profile_start("tdms :: pull tdm2_v2");
  py::buffer_info info_tdm1h = _tdm1h.request(); //1D array (norb)
  double * tdm1h = static_cast<double*>(info_tdm1h.ptr);
  py::buffer_info info_tdm3ha = _tdm3ha.request(); //3D array (norb, norb, norb)
  py::buffer_info info_tdm3hb = _tdm3hb.request(); //3D array (norb, norb, norb)
  double * tdm3ha;
  double * tdm3hb;
  
  filter_tdm1h(dd->fci.d_tdm1, dd->jk.d_buf3, norb);
  ctx.pm->dev_pull_async(dd->jk.d_buf3, tdm1h, norb*sizeof(double));
  if (spin){ //SWITCH is important
    tdm3hb = static_cast<double*>(info_tdm3ha.ptr);
    tdm3ha = static_cast<double*>(info_tdm3hb.ptr);
    }
  else{
    tdm3ha = static_cast<double*>(info_tdm3ha.ptr);
    tdm3hb = static_cast<double*>(info_tdm3hb.ptr);
    }
  int norb1 = norb+1;
  int norb2 = norb*norb;
  if (spin)
    { 
      ctx.utils->transpose_3210(dd->fci.d_tdm2_p, dd->jk.d_buf2, norb+1, norb+1);//using a function from before
      filter_tdm3h(dd->jk.d_buf2, &(dd->jk.d_buf3[norb+norb*norb2]), norb);
    }
  else
    {
      filter_tdm3h(dd->fci.d_tdm2_p, &(dd->jk.d_buf3[norb+norb*norb2]), norb);
    }
  filter_tdm3h(dd->fci.d_tdm2, &(dd->jk.d_buf3[norb]), norb);
  
  if (cre==0){
    ctx.utils->transpose_021(&(dd->jk.d_buf3[norb]),dd->fci.d_tdm2, norb, norb, norb);
    ctx.utils->transpose_021(&(dd->jk.d_buf3[norb+norb*norb2]),dd->fci.d_tdm2_p, norb, norb, norb);
    ctx.pm->dev_pull_async(dd->fci.d_tdm2, tdm3ha, norb*norb2*sizeof(double));
    ctx.pm->dev_pull_async(dd->fci.d_tdm2_p, tdm3hb, norb*norb2*sizeof(double));
    }
  else{
    ctx.pm->dev_pull_async(&(dd->jk.d_buf3[norb]), tdm3ha, norb*norb2*sizeof(double));
    ctx.pm->dev_pull_async(&(dd->jk.d_buf3[norb+norb*norb2]), tdm3hb, norb*norb2*sizeof(double));
    }

  //printf("3ha sgpu\n");
  //for (int i=0; i<norb; ++i){for (int j=0;j<norb2;++j){printf("%f\t",tdm3hb[i*norb2+j]);}printf("\n");}
  //printf("3hb sgpu\n");
  //for (int i=0; i<norb; ++i){for (int j=0;j<norb2;++j){printf("%f\t",tdm3hb[i*norb2+j]);}printf("\n");}
  ctx.pm->dev_profile_stop();
  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1-t0);
}
/* ---------------------------------------------------------------------- */
void DeviceFci::pull_tdm3hab_v2_host(int i, int j, int n_bra, int n_ket, int norb, int cre, int spin, int count)
{
  double t0 = omp_get_wtime();
  int id = count % ctx.num_devices;
  ctx.pm->dev_set_device(id); 
  my_device_data * dd = &(ctx.device_data[id]);
  int norb1 = norb+1;
  int norb2 = norb*norb;
  int size_tdm1h = norb;
  int size_tdm3h = norb*norb2;
  int loc_tdm1h = (i*n_bra+j)*size_tdm1h;
  int loc_tdm3h = (i*n_bra+j)*2*size_tdm3h;
  double * h_dm1_loc = &(h_dm1_full[loc_tdm1h]);
  double * h_dm3ha_loc;
  double * h_dm3hb_loc;

  ::grow_array(ctx.pm, dd->jk.d_buf3, norb, dd->jk.size_buf3, "buf3", FLERR);
  
  filter_tdm1h(dd->fci.d_tdm1, dd->jk.d_buf3, norb);
  ctx.pm->dev_pull_async(dd->jk.d_buf3, h_dm1_loc, norb*sizeof(double));
  h_dm3hb_loc = &(h_dm2_full[loc_tdm3h+(1-spin)*size_tdm3h]);
  h_dm3ha_loc = &(h_dm2_full[loc_tdm3h+spin*size_tdm3h]);
  filter_tdm3h(dd->fci.d_tdm2, &(dd->jk.d_buf3[norb]), norb);
  if (spin)
    { 
      ctx.utils->transpose_3210(dd->fci.d_tdm2_p, dd->jk.d_buf2, norb+1, norb+1);//using a function from before, it was for transpose of ncas,ncas,ncas,nmo shaped
      filter_tdm3h(dd->jk.d_buf2, &(dd->jk.d_buf3[norb+norb*norb2]), norb);
    }
  else
    {
      filter_tdm3h(dd->fci.d_tdm2_p, &(dd->jk.d_buf3[norb+norb*norb2]), norb);
    }
  
  if (cre==0){
    ctx.utils->transpose_021(&(dd->jk.d_buf3[norb]),dd->fci.d_tdm2, norb, norb, norb);
    ctx.utils->transpose_021(&(dd->jk.d_buf3[norb+norb*norb2]),dd->fci.d_tdm2_p, norb, norb, norb);
    ctx.pm->dev_pull_async(dd->fci.d_tdm2, h_dm3ha_loc, norb*norb2*sizeof(double));
    ctx.pm->dev_pull_async(dd->fci.d_tdm2_p, h_dm3hb_loc, norb*norb2*sizeof(double));
    }
  else{
    ctx.pm->dev_pull_async(&(dd->jk.d_buf3[norb]), h_dm3ha_loc, norb*norb2*sizeof(double));
    ctx.pm->dev_pull_async(&(dd->jk.d_buf3[norb+norb*norb2]), h_dm3hb_loc, norb*norb2*sizeof(double));
    }
  ctx.pm->dev_profile_stop();
  if (count+1 == n_bra*n_ket){
    for (int device_id =0; device_id<ctx.num_devices; ++device_id){
      ctx.pm->dev_set_device(device_id); 
      ctx.pm->dev_barrier();
      }
    }
  //printf("i:%i j:%i\n",i,j);
  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1-t0);
}

/* ---------------------------------------------------------------------- */
void DeviceFci::copy_tdm1_host_to_page(py::array_t<double> _dm1_full, int size_dm1_full)
{
  double t0 = omp_get_wtime();
  py::buffer_info info_dm1_full = _dm1_full.request(); // (size_dm1_full)
  double * dm1_full = static_cast<double*>(info_dm1_full.ptr);
#pragma omp parallel for
  for (int i=0; i<size_dm1_full; ++i){
    dm1_full[i] = h_dm1_full[i];
  }
  double t1 = omp_get_wtime();
}
/* ---------------------------------------------------------------------- */
void DeviceFci::copy_tdm2_host_to_page(py::array_t<double> _dm2_full, int size_dm2_full)
{
  double t0 = omp_get_wtime();
  py::buffer_info info_dm2_full = _dm2_full.request(); // (size_dm2_full)
  double * dm2_full = static_cast<double*>(info_dm2_full.ptr);
#pragma omp parallel for
  for (int i=0; i<size_dm2_full; ++i){
    dm2_full[i] = h_dm2_full[i];
  }
  double t1 = omp_get_wtime();
}

/* ---------------------------------------------------------------------- */

void Device::init_tdm1(int norb)
{ _fci->init_tdm1(norb); }

void Device::init_tdm2(int norb)
{ _fci->init_tdm2(norb); }

void Device::init_tdm3hab(int norb)
{ _fci->init_tdm3hab(norb); }

void Device::init_tdm1_host(int _size_dm1)
{ _fci->init_tdm1_host(_size_dm1); }

void Device::init_tdm2_host(int _size_dm2)
{ _fci->init_tdm2_host(_size_dm2); }

void Device::init_tdm3h_host(int _size_dm2)
{ _fci->init_tdm3h_host(_size_dm2); }

void Device::push_cibra(py::array_t<double> _cibra, int na, int nb, int count)
{ _fci->push_cibra(_cibra, na, nb, count); }

void Device::push_ciket(py::array_t<double> _ciket, int na, int nb, int count)
{ _fci->push_ciket(_ciket, na, nb, count); }

void Device::copy_bravecs_host(py::array_t<double> _bravecs, int nvecs, int na, int nb)
{ _fci->copy_bravecs_host(_bravecs, nvecs, na, nb); }

void Device::copy_ketvecs_host(py::array_t<double> _ketvecs, int nvecs, int na, int nb)
{ _fci->copy_ketvecs_host(_ketvecs, nvecs, na, nb); }

void Device::push_cibra_from_host(int bra_index, int na, int nb, int count)
{ _fci->push_cibra_from_host(bra_index, na, nb, count); }

void Device::push_ciket_from_host(int ket_index, int na, int nb, int count)
{ _fci->push_ciket_from_host(ket_index, na, nb, count); }

void Device::push_link_indexa(int na, int nlinka, py::array_t<int> _link_indexa)
{ _fci->push_link_indexa(na, nlinka, _link_indexa); }

void Device::push_link_indexb(int nb, int nlinkb, py::array_t<int> _link_indexb)
{ _fci->push_link_indexb(nb, nlinkb, _link_indexb); }

void Device::compute_trans_rdm1a(int na, int nb, int nlinka, int nlinkb, int norb, int count)
{ _fci->compute_trans_rdm1a(na, nb, nlinka, nlinkb, norb, count); }

void Device::compute_trans_rdm1b(int na, int nb, int nlinka, int nlinkb, int norb, int count)
{ _fci->compute_trans_rdm1b(na, nb, nlinka, nlinkb, norb, count); }

void Device::compute_make_rdm1a(int na, int nb, int nlinka, int nlinkb, int norb, int count)
{ _fci->compute_make_rdm1a(na, nb, nlinka, nlinkb, norb, count); }

void Device::compute_make_rdm1b(int na, int nb, int nlinka, int nlinkb, int norb, int count)
{ _fci->compute_make_rdm1b(na, nb, nlinka, nlinkb, norb, count); }

void Device::compute_tdm12kern_a_v2(int na, int nb, int nlinka, int nlinkb, int norb, int count)
{ _fci->compute_tdm12kern_a_v2(na, nb, nlinka, nlinkb, norb, count); }

void Device::compute_tdm12kern_b_v2(int na, int nb, int nlinka, int nlinkb, int norb, int count)
{ _fci->compute_tdm12kern_b_v2(na, nb, nlinka, nlinkb, norb, count); }

void Device::compute_tdm12kern_ab_v2(int na, int nb, int nlinka, int nlinkb, int norb, int count)
{ _fci->compute_tdm12kern_ab_v2(na, nb, nlinka, nlinkb, norb, count); }

void Device::compute_rdm12kern_sf_v2(int na, int nb, int nlinka, int nlinkb, int norb, int count)
{ _fci->compute_rdm12kern_sf_v2(na, nb, nlinka, nlinkb, norb, count); }

void Device::compute_tdm13h_spin_v4(int na, int nb, int nlinka, int nlinkb, int norb, int spin, int _reorder, int ia_bra, int ja_bra, int ib_bra, int jb_bra, int sgn_bra, int ia_ket, int ja_ket, int ib_ket, int jb_ket, int sgn_ket, int count)
{ _fci->compute_tdm13h_spin_v4(na, nb, nlinka, nlinkb, norb, spin, _reorder, ia_bra, ja_bra, ib_bra, jb_bra, sgn_bra, ia_ket, ja_ket, ib_ket, jb_ket, sgn_ket, count); }

void Device::compute_tdm13h_spin_v5(int na, int nb, int nlinka, int nlinkb, int norb, int spin, int _reorder, int ia_bra, int ja_bra, int ib_bra, int jb_bra, int sgn_bra, int ia_ket, int ja_ket, int ib_ket, int jb_ket, int sgn_ket, int count)
{ _fci->compute_tdm13h_spin_v5(na, nb, nlinka, nlinkb, norb, spin, _reorder, ia_bra, ja_bra, ib_bra, jb_bra, sgn_bra, ia_ket, ja_ket, ib_ket, jb_ket, sgn_ket, count); }

void Device::compute_tdmpp_spin_v4(int na, int nb, int nlinka, int nlinkb, int norb, int spin, int ia_bra, int ja_bra, int ib_bra, int jb_bra, int sgn_bra, int ia_ket, int ja_ket, int ib_ket, int jb_ket, int sgn_ket, int count)
{ _fci->compute_tdmpp_spin_v4(na, nb, nlinka, nlinkb, norb, spin, ia_bra, ja_bra, ib_bra, jb_bra, sgn_bra, ia_ket, ja_ket, ib_ket, jb_ket, sgn_ket, count); }

void Device::compute_sfudm_v2(int na, int nb, int nlinka, int nlinkb, int norb, int ia_bra, int ja_bra, int ib_bra, int jb_bra, int sgn_bra, int ia_ket, int ja_ket, int ib_ket, int jb_ket, int sgn_ket, int count)
{ _fci->compute_sfudm_v2(na, nb, nlinka, nlinkb, norb, ia_bra, ja_bra, ib_bra, jb_bra, sgn_bra, ia_ket, ja_ket, ib_ket, jb_ket, sgn_ket, count); }

void Device::compute_tdm1h_spin(int na, int nb, int nlinka, int nlinkb, int norb, int spin, int ia_bra, int ja_bra, int ib_bra, int jb_bra, int sgn_bra, int ia_ket, int ja_ket, int ib_ket, int jb_ket, int sgn_ket, int count)
{ _fci->compute_tdm1h_spin(na, nb, nlinka, nlinkb, norb, spin, ia_bra, ja_bra, ib_bra, jb_bra, sgn_bra, ia_ket, ja_ket, ib_ket, jb_ket, sgn_ket, count); }

void Device::reorder_rdm(int norb, int count)
{ _fci->reorder_rdm(norb, count); }

void Device::transpose_tdm2(int norb, int count)
{ _fci->transpose_tdm2(norb, count); }

void Device::pull_tdm1(py::array_t<double> _tdm1, int norb, int count)
{ _fci->pull_tdm1(_tdm1, norb, count); }

void Device::pull_tdm2(py::array_t<double> _tdm2, int norb, int count)
{ _fci->pull_tdm2(_tdm2, norb, count); }

void Device::pull_tdm1_host(int i, int j, int n_bra, int n_ket, int size_tdm1, int factor, int count)
{ _fci->pull_tdm1_host(i, j, n_bra, n_ket, size_tdm1, factor, count); }

void Device::pull_tdm2_host(int i, int j, int n_bra, int n_ket, int size_tdm2, int factor, int count)
{ _fci->pull_tdm2_host(i, j, n_bra, n_ket, size_tdm2, factor, count); }

void Device::pull_tdm3h_host(int loc, int size_tdm2, int count)
{ _fci->pull_tdm3h_host(loc, size_tdm2, count); }

void Device::pull_tdm3hab(py::array_t<double> _tdm3ha, py::array_t<double> _tdm3hb, int norb, int count)
{ _fci->pull_tdm3hab(_tdm3ha, _tdm3hb, norb, count); }

void Device::pull_tdm3hab_v2(py::array_t<double> _tdm1h, py::array_t<double> _tdm3ha, py::array_t<double> _tdm3hb, int norb, int cre, int spin, int count)
{ _fci->pull_tdm3hab_v2(_tdm1h, _tdm3ha, _tdm3hb, norb, cre, spin, count); }

void Device::pull_tdm3hab_v2_host(int i, int j, int n_bra, int n_ket, int norb, int cre, int spin, int count)
{ _fci->pull_tdm3hab_v2_host(i, j, n_bra, n_ket, norb, cre, spin, count); }

void Device::copy_tdm1_host_to_page(py::array_t<double> _dm1_full, int size_dm1_full)
{ _fci->copy_tdm1_host_to_page(_dm1_full, size_dm1_full); }

void Device::copy_tdm2_host_to_page(py::array_t<double> _dm2_full, int size_dm2_full)
{ _fci->copy_tdm2_host_to_page(_dm2_full, size_dm2_full); }
