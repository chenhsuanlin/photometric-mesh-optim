#ifdef __cplusplus
extern "C" {
#endif

int forward_kernel_launcher(int num_faces,int H,int W,float *CI,float *FVT,int *BFI,int *IM,float *BM,float *IDM,int *LM);
__global__ void forward_kernel(int num_faces,int H,int W,float *CI,float *FVT,int *BFI,int *IM,float *BM,float *IDM,int *LM);
__device__ void vector_sub_cuda(float* res,float* a,float* b);
__device__ float dot_product_cuda(float* a,float* b);
__device__ void cross_product_cuda(float* res,float* a,float* b);
__device__ void calib_vertex_cuda(float* v_persp,float* v_trans,float* CI);

#ifdef __cplusplus
}
#endif
