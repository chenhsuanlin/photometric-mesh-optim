#include <THC/THC.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "gpu.h"
#include "gpu_kernel.h"

// this symbol will be resolved automatically from PyTorch libs
extern THCState *state;

int forward_cuda(THCudaTensor *cam_intr, // [3,3]
				 THCudaTensor *face_vertices_trans, // [F,3,3]
				 THCudaIntTensor *batch_face_index, // [F]
				 THCudaIntTensor *index_map, // [B,H,W]
				 THCudaTensor *baryc_map, // [B,H,W,3]
				 THCudaTensor *inv_depth_map, // [B,H,W]
				 THCudaIntTensor *lock_map) // [B,H,W]
{
	int num_faces = face_vertices_trans->size[0],
		H = index_map->size[1],
		W = index_map->size[2];

	float *CI = THCudaTensor_data(state,cam_intr);
	float *FVT = THCudaTensor_data(state,face_vertices_trans);
	int *BFI = THCudaIntTensor_data(state,batch_face_index);
	int *IM = THCudaIntTensor_data(state,index_map);
	float *BM = THCudaTensor_data(state,baryc_map);
	float *IDM = THCudaTensor_data(state,inv_depth_map);
	int *LM = THCudaIntTensor_data(state,lock_map);

	int success = forward_kernel_launcher(num_faces,H,W,CI,FVT,BFI,IM,BM,IDM,LM);

	if (!success) THError("abort...");

	return 1;
}


