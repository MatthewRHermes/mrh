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

DeviceLassi::DeviceLassi(DeviceContext & ctx) : ctx(ctx)
{
  size_new_sivecs = 0;
  size_old_sivecs = 0;
  size_ox1 = 0;
  size_instruction_list = 0;
  size_op = 0;
  ox1_on_gpu = 0;
  h_new_sivecs = nullptr;
  h_old_sivecs = nullptr;
  h_ox1 = nullptr;
  h_instruction_list = nullptr;
}

DeviceLassi::~DeviceLassi()
{
  if(h_new_sivecs) ctx.pm->dev_free_host(h_new_sivecs);
  if(h_old_sivecs) ctx.pm->dev_free_host(h_old_sivecs);
  if(h_ox1) ctx.pm->dev_free_host(h_ox1);
  if(h_instruction_list) ctx.pm->dev_free_host(h_instruction_list);
  h_new_sivecs = nullptr;
  h_old_sivecs = nullptr;
  h_ox1 = nullptr;
  h_instruction_list = nullptr;
}

/* ---------------------------------------------------------------------- */

void DeviceLassi::push_op(py::array_t<double> _op, int m, int k, int counts)
{
  double t0 = omp_get_wtime();
  py::buffer_info info_op = _op.request(); // (2D array of m * k)
  double * op = static_cast<double*>(info_op.ptr);
  int _size_op = m*k;
  #if defined(_ENABLE_P2P)
  counts = _MIN(counts, ctx.num_devices);
  std::vector<double *> op_vec(counts); // array of device addresses 
  for(int id=0; id<counts; ++id) {
    ctx.pm->dev_set_device(id);
    my_device_data * dd = &(ctx.device_data[id]);
    ::grow_array(ctx.pm, dd->jk.d_buf1, _size_op, dd->jk.size_buf1, "buf1", FLERR);
    op_vec[id] = dd->jk.d_buf1;}
  ctx.comm->mgpu_bcast(op_vec, op, _size_op*sizeof(double));
  #else
  for (int i=0; i<ctx.num_devices;++i){
    ctx.pm->dev_set_device(i);
    my_device_data * dd = &(ctx.device_data[i]);
    ::grow_array(ctx.pm, dd->jk.d_buf1, _size_op, dd->jk.size_buf1, "buf1", FLERR);
    ctx.pm->dev_push_async(dd->jk.d_buf1, op, _size_op*sizeof(double));
  }
  #endif
  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1-t0);
}
/* ---------------------------------------------------------------------- */
void DeviceLassi::push_op_4frag(py::array_t<double> _op, int size_op, int size_req, int counts)
{
  double t0 = omp_get_wtime();
  py::buffer_info info_op = _op.request();
  double * op = static_cast<double*>(info_op.ptr);
  int _size_op = size_op;

  counts = _MIN(counts, ctx.num_devices);
  std::vector<double *> op_vec(counts); // array of device addresses 
  //printf("size_req: %i, size_op: %i\t",size_req, size_op);

  for(int id=0; id<counts; ++id) {
    ctx.pm->dev_set_device(id);
    my_device_data * dd = &(ctx.device_data[id]);
    ::grow_array(ctx.pm, dd->jk.d_buf1, size_req, dd->jk.size_buf1, "buf1", FLERR);
    //printf("size_buf: %i\n", dd->jk.size_buf1);
    op_vec[id] = dd->jk.d_buf1;}

  ctx.comm->mgpu_bcast(op_vec, op, _size_op*sizeof(double));

  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1-t0);
}
/* ---------------------------------------------------------------------- */
void DeviceLassi::push_d2(py::array_t<double> _d2, int size_d2, int loc_d2, int counts)
{
  double t0 = omp_get_wtime();
  py::buffer_info info_d2 = _d2.request();
  double * d2 = static_cast<double*>(info_d2.ptr);

  counts = _MIN(counts, ctx.num_devices);
  std::vector<double *> d2_vec(counts); // array of device addresses 

  for(int id=0; id<counts; ++id) {
    ctx.pm->dev_set_device(id);
    my_device_data * dd = &(ctx.device_data[id]);
    d2_vec[id] = &(dd->jk.d_buf1[loc_d2]);}

  ctx.comm->mgpu_bcast(d2_vec, d2, size_d2*sizeof(double));

  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1-t0);
}
/* ---------------------------------------------------------------------- */
void DeviceLassi::push_d3(py::array_t<double> _d3, int size_d3, int loc_d3, int counts)
{
  double t0 = omp_get_wtime();
  py::buffer_info info_d3 = _d3.request();
  double * d3 = static_cast<double*>(info_d3.ptr);

  counts = _MIN(counts, ctx.num_devices);
  std::vector<double *> d3_vec(counts); // array of device addresses 

  for(int id=0; id<counts; ++id) {
    ctx.pm->dev_set_device(id);
    my_device_data * dd = &(ctx.device_data[id]);
    d3_vec[id] = &(dd->jk.d_buf1[loc_d3]);}

  ctx.comm->mgpu_bcast(d3_vec, d3, size_d3*sizeof(double));

  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1-t0);
}



/* ---------------------------------------------------------------------- */
void DeviceLassi::init_new_sivecs_host(int m, int n)
{
  double t0 = omp_get_wtime();
  int _size_sivecs = m*n;
  ::grow_array_host(ctx.pm, h_new_sivecs, _size_sivecs, size_new_sivecs, "h:new_sivecs");
  double t1 = omp_get_wtime();
}
/* ---------------------------------------------------------------------- */
void DeviceLassi::init_ox1_pinned(int size)
{
  double t0 = omp_get_wtime();
  ::grow_array_host(ctx.pm, h_ox1, size, size_ox1, "h:ox1");
  int overall_max_size_buf=0;
  int _max_size_buf, device_id;
  for (int i=0; i<ctx.num_devices; ++i){
    device_id = i%ctx.num_devices;
    ctx.pm->dev_set_device(device_id);
    my_device_data * dd = &(ctx.device_data[device_id]);
    _max_size_buf = _MAX(dd->jk.size_buf1, dd->jk.size_buf2);
    _max_size_buf = _MAX(_max_size_buf, dd->jk.size_buf3);
    overall_max_size_buf = _MAX(overall_max_size_buf,_max_size_buf); 
    dd->active=0; //setting it to zero in case there is not enough work.
    }
  if (overall_max_size_buf>size) 
    {ox1_on_gpu = 1;}
  else{ox1_on_gpu=0;}
  for (int i=0; i<ctx.num_devices;++i){
    ctx.pm->dev_set_device(i);
    my_device_data * dd = &(ctx.device_data[i]);
    ::grow_array(ctx.pm, dd->jk.d_buf1, overall_max_size_buf, dd->jk.size_buf1, "buf1", FLERR);
    ::grow_array(ctx.pm, dd->jk.d_buf2, overall_max_size_buf, dd->jk.size_buf2, "buf2", FLERR);
    ::grow_array(ctx.pm, dd->jk.d_buf3, overall_max_size_buf, dd->jk.size_buf3, "buf3", FLERR);
    if (ox1_on_gpu){ ctx.utils->set_to_zero(dd->jk.d_buf3, size); } 
    } 
  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1-t0);
}

/* ---------------------------------------------------------------------- */
void DeviceLassi::init_old_sivecs_host(int k, int n)
{
  double t0 = omp_get_wtime();
   
  int _size_sivecs = k*n;

  ctx.pm->dev_set_device(0);
  my_device_data * dd = &(ctx.device_data[0]);
  if (dd->jk.size_buf2<_size_sivecs){printf("there is an issue with the calculation\n");}

  ::grow_array_host(ctx.pm, h_old_sivecs, _size_sivecs, size_old_sivecs, "h:old_sivecs");
  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1-t0);
}
/* ---------------------------------------------------------------------- */
void DeviceLassi::push_sivecs_to_host(py::array_t<double> _vec, int loc, int size)
{
  double t0 = omp_get_wtime();
  py::buffer_info info_vec = _vec.request(); // (2D array of n * k)
  double * vec = static_cast<double*>(info_vec.ptr);
  double * h_old_sivecs_loc = &(h_old_sivecs[loc]);
#pragma omp parallel for
  for (int i=0;i<size;++i){h_old_sivecs_loc[i] = vec[i];}
  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1-t0);
}
/* ---------------------------------------------------------------------- */
void DeviceLassi::push_sivecs_to_device(py::array_t<double> _vec, int loc, int size, int count)
{
  double t0 = omp_get_wtime();
  py::buffer_info info_vec = _vec.request(); // (2D array of n * k)
  double * vec = static_cast<double*>(info_vec.ptr);
  //int device_id = count%num_devices;
  int device_id=0;
  ctx.pm->dev_set_device(device_id);
  my_device_data * dd = &(ctx.device_data[device_id]);
  ctx.pm->dev_push_async(&(dd->jk.d_buf2[loc]), vec, size*sizeof(double)); 
  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1-t0);
}
/* ---------------------------------------------------------------------- */
void DeviceLassi::bcast_vec(int size, int counts)
{
  double t0 = omp_get_wtime();
  counts = _MIN(counts, ctx.num_devices);
  std::vector<double *> vec_vec(counts); // array of device addresses 
  for(int id=0; id<counts; ++id) {
    ctx.pm->dev_set_device(id);
    my_device_data * dd = &(ctx.device_data[id]);
    vec_vec[id] = dd->jk.d_buf2;}
  //ctx.comm->mgpu_bcast(vec_vec, op, _size_op*sizeof(double));
  //not doing bcast directly because it copies from host to devices
  for(int i=1; i<vec_vec.size(); ++i){
    ctx.pm->dev_memcpy_peer(vec_vec[i], i, vec_vec[0], 0, size*sizeof(double));
    }

  for(int id=0; id<counts; ++id) {
    ctx.pm->dev_set_device(id);
    ctx.pm->dev_barrier();}
  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1-t0);

}
/* ---------------------------------------------------------------------- */
void DeviceLassi::push_instruction_list(py::array_t<int> _instruction_list, int len)
{
  double t0 = omp_get_wtime();
  py::buffer_info info_instruction_list = _instruction_list.request();
  int * instruction_list = static_cast<int*>(info_instruction_list.ptr);
  int _size_list = 4*len;
  ::grow_array_host(ctx.pm, h_instruction_list, _size_list, size_instruction_list, "h:instruction_list");
#pragma omp parallel for
  for (int i=0;i<_size_list; ++i){h_instruction_list[i] = instruction_list[i];}
  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1-t0);
}
/* ---------------------------------------------------------------------- */
void DeviceLassi::compute_sivecs (int m, int n, int k)
{
  double t0 = omp_get_wtime();
  int _max_size_buf;
  int max_batch_n;
  int batch_n;
  int device_id;
  double alpha = 1.0;
  double beta = 0.0;
  for (int i=0; i<ctx.num_devices; ++i){
    device_id = i%ctx.num_devices;
    ctx.pm->dev_set_device(device_id);
    my_device_data * dd = &(ctx.device_data[device_id]);
    _max_size_buf = _MAX(dd->jk.size_buf1, dd->jk.size_buf2);
    _max_size_buf = _MAX(_max_size_buf, dd->jk.size_buf3);
    ::grow_array(ctx.pm, dd->jk.d_buf2, _max_size_buf, dd->jk.size_buf2, "buf2", FLERR);//not doing buf1 because it is used for op
    ::grow_array(ctx.pm, dd->jk.d_buf3, _max_size_buf, dd->jk.size_buf3, "buf3", FLERR);
    max_batch_n = _MIN(_max_size_buf/k, _max_size_buf/m);
    max_batch_n = 6;//_MIN(_max_size_buf/k, _max_size_buf/m);
    }
  int device_id_counter = 0;
  for (int i=0; i<n; i+=max_batch_n){
    device_id = device_id_counter%ctx.num_devices;
    ++device_id_counter;
    ctx.pm->dev_set_device(device_id);
    my_device_data * dd = &(ctx.device_data[device_id]);
    ctx.ml->set_handle(device_id);
    batch_n = _MIN(max_batch_n, n-i);
    double * old_sivecs = &(h_old_sivecs[i*k]);
    double * new_sivecs = &(h_new_sivecs[i*m]);

    ctx.pm->dev_push_async(dd->jk.d_buf2, old_sivecs, k*batch_n*sizeof(double));
    ctx.ml->gemm((char *) "T", (char *) "N", 
             &m, &batch_n, &k, 
             &alpha, 
             dd->jk.d_buf1, &k,
             dd->jk.d_buf2, &k,
             &beta, 
             dd->jk.d_buf3, &m);
    ctx.pm->dev_pull_async( dd->jk.d_buf3, new_sivecs, batch_n*m*sizeof(double));
    }
  for (int i=0; i<ctx.num_devices; ++i){
    ctx.pm->dev_set_device(device_id);
    ctx.pm->dev_barrier();}
  double t1 = omp_get_wtime();
}
/* ---------------------------------------------------------------------- */
void DeviceLassi::compute_sivecs_full (int _m, int _k, int counts, int op_t)
{
  double t0 = omp_get_wtime();
  int device_id = 0, device_id_counter = 0;
  double alpha, beta;
  
  int m, k, n, vec_loc, vec_size, ox1_loc, ox1_size, fac;
  double * result;
  if (op_t){
    m = _k;
    k = _m;}
  else{
    m = _m;
    k = _k;}
  for (int count=0; count<counts; ++count){
    vec_loc = h_instruction_list[count*4];
    vec_size = h_instruction_list[count*4+1];
    ox1_loc = h_instruction_list[count*4+2];
    fac = h_instruction_list[count*4+3];
    n = vec_size/k;
    ox1_size=n*m;

    device_id = count%ctx.num_devices;
    ctx.pm->dev_set_device(device_id);
    ctx.pm->dev_profile_start("op_vec :: compute_opvec");
    my_device_data * dd = &(ctx.device_data[device_id]);
    ctx.ml->set_handle(device_id);
    dd->active = 1;

    alpha = fac*1.0;
    if (ox1_on_gpu){ 
      beta = 1.0;
      result = &(dd->jk.d_buf3[ox1_loc]); }
    else {
      beta = 0.0;
      result = dd->jk.d_buf3; }

    //for most calculations m*n_i and k*n_i should fit on a single gpu
      //if that is not the case, k*n_i can be split, but we need to be careful about how we update the ox1 on pinned
    
    double * old_sivecs = &(h_old_sivecs[vec_loc]);
    ctx.pm->dev_push_async(dd->jk.d_buf2, old_sivecs, vec_size*sizeof(double));
    if (op_t){
      ctx.ml->gemm((char *) "T", (char *) "T", 
           &n,&m,&k, 
           &alpha, dd->jk.d_buf2, &k, dd->jk.d_buf1, &m,
           &beta, result, &n); }
      
    else {
      ctx.ml->gemm((char *) "T", (char *) "N", 
           &n,&m,&k, 
           &alpha, dd->jk.d_buf2, &k, dd->jk.d_buf1, &k,
           &beta, result, &n); }
 
    if (!ox1_on_gpu){
      printf("shouldn't be here!\n");
      double * new_sivecs = &(h_ox1[ox1_loc]);
      ctx.pm->dev_pull_async( dd->jk.d_buf3, new_sivecs, ox1_size*sizeof(double));
      }
    ctx.pm->dev_profile_stop();
    }
  if (!ox1_on_gpu){
      printf("shouldn't be here!\n");
    for (int i=0; i<ctx.num_devices; ++i){
      ctx.pm->dev_set_device(device_id);
      ctx.pm->dev_barrier();}
    }
  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1-t0);
}
/* ---------------------------------------------------------------------- */
void DeviceLassi::compute_sivecs_full_v2 (int _m, int _k, int counts, int op_t)
{
  double t0 = omp_get_wtime();
  int device_id = 0, device_id_counter = 0;
  double alpha, beta;
  
  int m, k, n, vec_loc, vec_size, ox1_loc, ox1_size, fac;
  double * result;
  if (op_t){
    m = _k;
    k = _m;}
  else{
    m = _m;
    k = _k;}
  for (int count=0; count<counts; ++count){
    vec_loc = h_instruction_list[count*4];
    vec_size = h_instruction_list[count*4+1];
    ox1_loc = h_instruction_list[count*4+2];
    fac = h_instruction_list[count*4+3];
    n = vec_size/k;
    ox1_size=n*m;

    //device_id = count%num_devices;
    device_id = 0;
    ctx.pm->dev_set_device(device_id);
    ctx.pm->dev_profile_start("op_vec :: compute_opvec");
    my_device_data * dd = &(ctx.device_data[device_id]);
    ctx.ml->set_handle(device_id);
    dd->active = 1;

    alpha = fac*1.0;
    //if (ox1_on_gpu){ 
    //  beta = 1.0;
    //  result = &(dd->jk.d_buf3[ox1_loc]); }
    //else {
    //  beta = 0.0;
    //  result = dd->jk.d_buf3; }
    beta = 1.0;
    result = &(dd->jk.d_buf3[ox1_loc]);
    
    double * vec = &(dd->jk.d_buf2[vec_loc]);
    if (op_t){
      ctx.ml->gemm((char *) "T", (char *) "T", 
           &n,&m,&k, 
           &alpha, vec, &k, dd->jk.d_buf1, &m,
           &beta, result, &n); }
      
    else {
      ctx.ml->gemm((char *) "T", (char *) "N", 
           &n,&m,&k, 
           &alpha, vec, &k, dd->jk.d_buf1, &k,
           &beta, result, &n); }
 
    if (!ox1_on_gpu){
      printf("shouldn't be here!\n");
      double * new_sivecs = &(h_ox1[ox1_loc]);
      ctx.pm->dev_pull_async( dd->jk.d_buf3, new_sivecs, ox1_size*sizeof(double));
      }
    ctx.pm->dev_profile_stop();
    }
  if (!ox1_on_gpu){
      printf("shouldn't be here!\n");
    for (int i=0; i<ctx.num_devices; ++i){
      ctx.pm->dev_set_device(device_id);
      ctx.pm->dev_barrier();}
    }
  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1-t0);
}
/* ---------------------------------------------------------------------- */
void DeviceLassi::compute_sivecs_full_v3 (int _m, int _k, int n, int vec_loc, int ox1_loc, int fac, int op_t, int count)
{
  double t0 = omp_get_wtime();
  int device_id = count%ctx.num_devices;
  double alpha, beta;
  
  int m, k, vec_size, ox1_size; 
  double * result;
  if (op_t){
    m = _k;
    k = _m;}
  else{
    m = _m;
    k = _k;}
  ox1_size=n*m;
  ctx.pm->dev_set_device(device_id);
  ctx.pm->dev_profile_start("op_vec :: compute_opvec");
  my_device_data * dd = &(ctx.device_data[device_id]);
  ctx.ml->set_handle(device_id);
  dd->active = 1;

  alpha = fac*1.0;
  beta = 1.0;
  result = &(dd->jk.d_buf3[ox1_loc]);
  
  double * vec = &(dd->jk.d_buf2[vec_loc]);

  double * h_vec;
  double * h_res;
  #if 0
  if (ox1_loc == 209){
    printf("m: %i k: %i n: %i vec_loc: %i ox1_loc: %i device_id: %i\n",m,k,n,vec_loc, ox1_loc, device_id);
    h_vec = (double*) ctx.pm->dev_malloc_host(k*n*sizeof(double));
    h_res = (double*) ctx.pm->dev_malloc_host(m*n*sizeof(double));
    ctx.pm->dev_pull_async(vec, h_vec, k*n*sizeof(double));
    ctx.pm->dev_pull_async(result, h_res, m*n*sizeof(double));
    ctx.pm->dev_barrier();
    printf("vec\n");
    for (int i=0;i<k*n;++i){printf("%f\t",h_vec[i]);}printf("\n");
    printf("res\n");
    for (int i=0;i<m*n;++i){printf("%f\t",h_res[i]);}printf("\n");
    }
  #endif
  if (op_t){
    ctx.ml->gemm((char *) "T", (char *) "T", 
         &n,&m,&k, 
         &alpha, vec, &k, dd->jk.d_buf1, &m,
         &beta, result, &n); }
  else {
    ctx.ml->gemm((char *) "T", (char *) "N", 
         &n,&m,&k, 
         &alpha, vec, &k, dd->jk.d_buf1, &k,
         &beta, result, &n); }
  #if 0
  if (ox1_loc == 209){
    ctx.pm->dev_pull_async(result, h_res, m*n*sizeof(double));
    ctx.pm->dev_barrier();
    printf("res\n");
    for (int i=0;i<m*n;++i){printf("%f\t",h_res[i]*100000);}printf("\n");
    }
  #endif
  ctx.pm->dev_profile_stop();
  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1-t0);
}

/* ---------------------------------------------------------------------- */
void DeviceLassi::compute_4frag_matvec( int i, int j, int k, int l, 
                             int a, int b, int c, int d,
                             int z, 
                             int r, int s, 
                             int vec_loc, int ox1_loc, int fac, int op_t, int count)
{
  //c === einsum(mk,nk->mn, a,b)  === dot(a,b.T) === cublasDgemm(T,N,n,m,k,alpha,b,k,a,k,beta,c,n)
  double t0 = omp_get_wtime();
  int device_id = count;
  int _m, _n, _k; //actual multiplication factors
  double * matA, * matB, * matC; //matrices
  //buf1: op, d[2], d[3], [op*vec], transpose_23140(op*vec)
  //buf2: vec_loc = vec
  //final_result = &buf3[ox1_loc]
  //tmp1 = &buf1[size_op + size_d2 + size_d3]
  //tmp1 = op*vec
  //tmp2 = &tmp[size_tmp]
  //tmp2 = transpose(tmp1)
  //tmp1 = d2*tmp2
  //final_result += d3*tmp1
  //buf1[size_op + size_d2 + size_d3 + size_tmp] = transpose_23140(
  //printf("r: %i s: %i b: %i a: %i i: %i j: %i\n",r,s,b,a,i,j);
  int size_op = r*s*b*a*i*j;
  int size_d2 = c*k*r;
  int size_d3 = d*l*s;
  int size_other = z*l*k*j*i;
  int size_tmp_result1 = r*s*b*a*z*l*k;
  int size_tmp_result2 = c*b*a*z*l*s;
  int size_tmp_result3 = d*c*b*a*z;
  int loc_C;
  double alpha = 1.0;
  double beta = 0.0; 
  ctx.pm->dev_set_device(device_id);
  ctx.pm->dev_profile_start("op_vec :: compute_4frag");
  my_device_data * dd = &(ctx.device_data[device_id]);
  ctx.ml->set_handle(device_id);
  dd->active = 1;
  double * d_op = dd->jk.d_buf1;
  double * d_d2 = &(dd->jk.d_buf1[size_op]);
  double * d_d3 = &(dd->jk.d_buf1[size_op+size_d2]);
  loc_C = size_op + size_d2+size_d3;
  double * h_matC;
  double * h_op;
  double * h_di;
  #if 0
    //h_op = (double*)ctx.pm->dev_malloc_host(size_op*sizeof(double));
    //ctx.pm->dev_pull_async(d_op, h_op, size_op*sizeof(double));
    double * h_vec = (double*)ctx.pm->dev_malloc_host(size_other*sizeof(double));
    double * d_vec = &(dd->jk.d_buf2[vec_loc]);
    ctx.pm->dev_pull_async(d_vec, h_vec, size_other*sizeof(double));
    ctx.pm->dev_barrier();
    //printf("%i op\n",size_op);
    //for (int _i=0; _i<size_op; ++_i){printf("%f\t",h_op[_i]*1e7);}printf("\n");
    printf("vec\n");
    for (int _i=0; _i<size_other; ++_i){printf("%f\t",h_vec[_i]);}printf("\n");
  #endif
  if (op_t){
  double * buf_d2 = &(dd->jk.d_buf1[loc_C]); //buffer to store transposed d2, will copy it back to it's original position after done.
  //transpose d[2] kcr->ckr
  ctx.utils->transpose_102(d_d2, buf_d2, c,k,r);
  ctx.utils->veccopy(buf_d2, d_d2, size_d2);
  //transpose d[3] lds->dls
  ctx.utils->transpose_102(d_d3, buf_d2, d,l,s);
  ctx.utils->veccopy(buf_d2, d_d3, size_d3);
  //transpose op rsjiba -> rsbaji
  //transpose_021(in,out,ax1,ax2,ax3) reads in as (ax1,ax3,ax2) and writes out as (ax1,ax2,ax3),
  //so recovering rsbaji (ax2=b*a,ax3=j*i) from a pushed rsjiba buffer needs args in that order,
  //not (r*s, j*i, b*a) -- the latter instead converts a canonical rsbaji buffer to rsjiba, i.e.
  //backwards. Only invisible when j*i == b*a (masked by every square test config).
  ctx.utils->transpose_021(d_op, buf_d2, r*s, b*a, j*i);
  ctx.utils->veccopy(buf_d2, d_op, size_op);
  }

  #if 0
    printf("ckr\n");
    h_di = (double*)ctx.pm->dev_malloc_host(size_d2*sizeof(double));
    ctx.pm->dev_pull_async(d_d2, h_di, size_d2*sizeof(double));
    ctx.pm->dev_barrier();
    for (int _i=0; _i<size_d2; ++_i){printf("%f\t",h_di[_i]);}printf("\n");
  #endif

  //ox = lib.einsum ('rsbaji,zlkji->rsbazlk', self.op, other)
  _m = r*s*b*a;
  _n = z*l*k;
  _k =  i*j;
  matB = &(dd->jk.d_buf2[vec_loc]);
  matA = dd->jk.d_buf1;
  //loc_C = size_op + size_d2+size_d3;
  matC = &(dd->jk.d_buf1[loc_C]);
  ctx.ml->gemm((char *) "T", (char *) "N", 
       &_n,&_m,&_k, 
       &alpha, matB, &_k, matA, &_k,
       &beta, matC, &_n); 

  #if 0
    if (ox1_loc==1378){
    printf("rsjiba\n");
    h_op = (double*)ctx.pm->dev_malloc_host(_m*_k*sizeof(double));
    ctx.pm->dev_pull_async(matA, h_op, _m*_k*sizeof(double));
    ctx.pm->dev_barrier();
    for (int _i=0; _i<_m*_k; ++_i){printf("%f\t",h_op[_i]);}printf("\n");
   }
  #endif



  //rsbazlk->bazlskr

  double * matC_T = &(matC[size_tmp_result1]); //buf1 after matC
  ctx.utils->transpose_2130(matC, matC_T, r, s, b*a*z*l, k); 
  
  //original: ox = lib.einsum ('ckr,rsbazlk->scbazl', self.d[2], ox)
  //post transpose: ox = lib.einsum('ckr, bazlskr->cbazlk',self.d[2],ox)
  _m = c;
  _n = b*a*z*l*s;
  _k =  k*r;
  matB = matC_T;
  matA = &(dd->jk.d_buf1[size_op]); 
  loc_C = size_op+size_d2+size_d3;
  matC = &(dd->jk.d_buf1[loc_C]);
  ctx.ml->gemm((char *) "T", (char *) "N", 
       &_n,&_m,&_k, 
       &alpha, matB, &_k, matA, &_k,
       &beta, matC, &_n); 

  #if 0
    printf("cbazlk\n");
    h_matC = (double*)ctx.pm->dev_malloc_host(_m*_n*sizeof(double));
    ctx.pm->dev_pull_async(matC, h_matC, _m*_n*sizeof(double));
    ctx.pm->dev_barrier();
    for (int _i=0; _i<_m*_n; ++_i){printf("%f\t",h_matC[_i]);}printf("\n");
  #endif
  
  //original: ox = lib.einsum ('dls,scbazl->dcbaz', self.d[3], ox)
  //my_vers : ox = lib.einsum ('dls,cbazlk->dcbaz', self.d[3], ox) 
  
  _m = d;
  _n = c*b*a*z;
  _k =  l*s;
  matB = matC; //from previous calculation
  matA = &(dd->jk.d_buf1[size_op+size_d2]); //d3  
  matC = &(dd->jk.d_buf3[ox1_loc]);
  #if 0
    printf("dcbaz\n");
    h_matC = (double*)ctx.pm->dev_malloc_host(_m*_n*sizeof(double));
    ctx.pm->dev_pull_async(matC, h_matC, _m*_n*sizeof(double));
    ctx.pm->dev_barrier();
    for (int _i=0; _i<_m*_n; ++_i){printf("%f\t",h_matC[_i]);}printf("\n");
  #endif


  #if 0
    if (ox1_loc == 1378){
    printf("before start\n"); 
    h_matC = (double*)ctx.pm->dev_malloc_host(_m*_n*sizeof(double));
    ctx.pm->dev_pull_async(matC, h_matC, _m*_n*sizeof(double));
    ctx.pm->dev_barrier();
    for (int _i=0; _i<_m*_n; ++_i){printf("%f\t",h_matC[_i]*1e8);}printf("\n");
    }
  #endif

  alpha = 1.0*fac;
  beta = 1.0;
  ctx.ml->gemm((char *) "T", (char *) "N", 
       &_n,&_m,&_k, 
       &alpha, matB, &_k, matA, &_k,
       &beta, matC, &_n); 

  #if 0
    if (ox1_loc == 1378){
    printf("after calc start\n"); 
    h_matC = (double*)ctx.pm->dev_malloc_host(_m*_n*sizeof(double));
    ctx.pm->dev_pull_async(matC, h_matC, _m*_n*sizeof(double));
    ctx.pm->dev_barrier();
    for (int _i=0; _i<_m*_n; ++_i){printf("%f\t",h_matC[_i]*1e8);}printf("\n");
    }
  #endif


  ctx.pm->dev_profile_stop();
  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1-t0);
}


/* ---------------------------------------------------------------------- */
void DeviceLassi::print_sivecs(int start, int size)
{
  //for (int i=start;i<start+size;++i){printf("%f\t",h_old_sivecs[i]);}printf("\n");
}
/* ---------------------------------------------------------------------- */
void DeviceLassi::pull_sivecs_from_pinned(py::array_t<double> _vec, int n_loc, int m, int n)
{
  double t0 = omp_get_wtime();
  py::buffer_info info_vec = _vec.request(); // (empty 2D array of n * m)
  double * vec = static_cast<double*>(info_vec.ptr);
  int _size_vec = n*m;
  double * h_new_sivecs_loc = &(h_new_sivecs[n_loc*m]);
#pragma omp parallel for
  for (int i=0; i<_size_vec; ++i){vec[i] = h_new_sivecs_loc[i];}
  double t1 = omp_get_wtime();
}
/* ---------------------------------------------------------------------- */
void DeviceLassi::add_ox1_pinned(py::array_t<double> _ox1, int size)
{
  double t0 = omp_get_wtime();
  if (!ox1_on_gpu){
  printf("Pulling onto pinned\n");
  py::buffer_info info_ox1 = _ox1.request(); // (empty 1D array of size)
  double * ox1 = static_cast<double*>(info_ox1.ptr);
#pragma omp parallel for
  for (int i=0; i<size; ++i){ox1[i] += h_ox1[i];}
#pragma omp parallel for
  for (int i=0; i<size; ++i){h_ox1[i] = 0.0;}
  }
  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1-t0);
}
/* ---------------------------------------------------------------------- */
void DeviceLassi::finalize_ox1_pinned(py::array_t<double> _ox1, int size)
{
  double t0 = omp_get_wtime();
  if (ox1_on_gpu){
  py::buffer_info info_ox1 = _ox1.request(); // (empty 1D array of size)
  double * ox1 = static_cast<double*>(info_ox1.ptr);

  #if 0
    for (int _i=0; _i<size; ++_i){printf("%f\t",ox1[_i]);}printf("\n");
  #endif

  std::vector<double *> ox1_vec(ctx.num_devices);
  std::vector<double *> buf_vec(ctx.num_devices);
  std::vector<int> active(ctx.num_devices);
  int count_active = 0;
  for(int i=0; i<ctx.num_devices; ++i) {
    my_device_data * dd = &(ctx.device_data[i]);
    ox1_vec[i] = dd->jk.d_buf3;//results
    buf_vec[i] = dd->jk.d_buf2;//buffer
    active[i] = dd->active;
    if (dd->active) ++count_active;
    }
  if (count_active){
    ctx.comm->mgpu_reduce(ox1_vec, h_ox1, size, true, buf_vec, active);
#pragma omp parallel for
    for (int i=0;i<size;++i){ox1[i]+=h_ox1[i];}
    }
  
  }
  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1-t0);
}
/* ---------------------------------------------------------------------- */

/* ---------------------------------------------------------------------- */
/* Device facade forwarders (Phase 5 Option A: keep the flat Python API)    */
/* ---------------------------------------------------------------------- */

void Device::push_op(py::array_t<double> _op, int m, int k, int counts)
{ _lassi->push_op(_op, m, k, counts); }

void Device::push_op_4frag(py::array_t<double> _op, int size_op, int size_req, int counts)
{ _lassi->push_op_4frag(_op, size_op, size_req, counts); }

void Device::push_d2(py::array_t<double> _d2, int size_d2, int loc_d2, int counts)
{ _lassi->push_d2(_d2, size_d2, loc_d2, counts); }

void Device::push_d3(py::array_t<double> _d3, int size_d3, int loc_d3, int counts)
{ _lassi->push_d3(_d3, size_d3, loc_d3, counts); }

void Device::init_ox1_pinned(int size)
{ _lassi->init_ox1_pinned(size); }

void Device::init_new_sivecs_host(int m, int n)
{ _lassi->init_new_sivecs_host(m, n); }

void Device::init_old_sivecs_host(int k, int n)
{ _lassi->init_old_sivecs_host(k, n); }

void Device::push_sivecs_to_host(py::array_t<double> _vec, int loc, int size)
{ _lassi->push_sivecs_to_host(_vec, loc, size); }

void Device::push_sivecs_to_device(py::array_t<double> _vec, int loc, int size, int count)
{ _lassi->push_sivecs_to_device(_vec, loc, size, count); }

void Device::bcast_vec(int size, int counts)
{ _lassi->bcast_vec(size, counts); }

void Device::push_instruction_list(py::array_t<int> _instruction_list, int len)
{ _lassi->push_instruction_list(_instruction_list, len); }

void Device::compute_sivecs(int m, int n, int k)
{ _lassi->compute_sivecs(m, n, k); }

void Device::compute_sivecs_full(int m, int k, int counts, int op_t)
{ _lassi->compute_sivecs_full(m, k, counts, op_t); }

void Device::compute_sivecs_full_v2(int m, int k, int counts, int op_t)
{ _lassi->compute_sivecs_full_v2(m, k, counts, op_t); }

void Device::compute_sivecs_full_v3(int m, int k, int n, int vec_loc, int ox1_loc, int fac, int op_t, int count)
{ _lassi->compute_sivecs_full_v3(m, k, n, vec_loc, ox1_loc, fac, op_t, count); }

void Device::compute_4frag_matvec(int i, int j, int k, int l,
                                  int a, int b, int c, int d,
                                  int z, int r, int s,
                                  int vec_loc, int ox1_loc, int fac, int op_t, int count)
{ _lassi->compute_4frag_matvec(i, j, k, l, a, b, c, d, z, r, s, vec_loc, ox1_loc, fac, op_t, count); }

void Device::print_sivecs(int start, int size)
{ _lassi->print_sivecs(start, size); }

void Device::pull_sivecs_from_pinned(py::array_t<double> _vec, int n_loc, int m, int n)
{ _lassi->pull_sivecs_from_pinned(_vec, n_loc, m, n); }

void Device::add_ox1_pinned(py::array_t<double> _ox1, int size)
{ _lassi->add_ox1_pinned(_ox1, size); }

void Device::finalize_ox1_pinned(py::array_t<double> _ox1, int size)
{ _lassi->finalize_ox1_pinned(_ox1, size); }
