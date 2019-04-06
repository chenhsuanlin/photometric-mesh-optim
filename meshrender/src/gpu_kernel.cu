#include <stdio.h>
#include "gpu_kernel.h"

#define T 256

int forward_kernel_launcher(int num_faces,int H,int W,float *CI,float *FVT,int *BFI,int *IM,float *BM,float *IDM,int *LM)
{
	forward_kernel<<<(num_faces+T-1)/T,T>>>(num_faces,H,W,CI,FVT,BFI,IM,BM,IDM,LM);
	cudaError_t err = cudaGetLastError();
	if (err!=cudaSuccess) {
		printf("error: %s\n",cudaGetErrorString(err));
		return 0;
	}
	return 1;
}

__device__ enum Coord3D {X=0,Y=1,Z=2};
__device__ float ray_origin[3] = {0.0,0.0,0.0};

__global__
void forward_kernel(int num_faces,int H,int W,float *CI,float *FVT,int *BFI,int *IM,float *BM,float *IDM,int *LM)
{
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	if (i>=num_faces) return;

	float scale_x = CI[0], center_x = CI[2],
		   scale_y = CI[4], center_y = CI[5];

	int b = BFI[i*2+0],
		f = BFI[i*2+1];
	float *v0_trans = FVT+i*9+0,
		   *v1_trans = FVT+i*9+3,
		   *v2_trans = FVT+i*9+6;
	float v0_persp[2], v1_persp[2], v2_persp[2];
	calib_vertex_cuda(v0_persp,v0_trans,CI);
	calib_vertex_cuda(v1_persp,v1_trans,CI);
	calib_vertex_cuda(v2_persp,v2_trans,CI);

	// precompute edges
	float edge1[3], edge2[3];
	vector_sub_cuda(edge1,v1_trans,v0_trans);
	vector_sub_cuda(edge2,v2_trans,v0_trans);
	float s[3], q[3];
	vector_sub_cuda(s,ray_origin,v0_trans);
	cross_product_cuda(q,s,edge1);

	int vert0 = fabsf(v2_persp[X]-v1_persp[X])<1e-8,
		vert1 = fabsf(v0_persp[X]-v2_persp[X])<1e-8,
		vert2 = fabsf(v1_persp[X]-v0_persp[X])<1e-8;
	float y0b = v1_persp[Y]+(v0_persp[X]-v1_persp[X])/(v2_persp[X]-v1_persp[X])*(v2_persp[Y]-v1_persp[Y]),
		   y1b = v2_persp[Y]+(v1_persp[X]-v2_persp[X])/(v0_persp[X]-v2_persp[X])*(v0_persp[Y]-v2_persp[Y]),
		   y2b = v0_persp[Y]+(v2_persp[X]-v0_persp[X])/(v1_persp[X]-v0_persp[X])*(v1_persp[Y]-v0_persp[Y]);
	int lower0 = v0_persp[Y]>y0b,
		lower1 = v1_persp[Y]>y1b,
		lower2 = v2_persp[Y]>y2b;
	int right0 = v0_persp[X]>v1_persp[X],
		right1 = v1_persp[X]>v2_persp[X],
		right2 = v2_persp[X]>v0_persp[X];

	int x_min = fmaxf(0,fminf(v0_persp[X],fminf(v1_persp[X],v2_persp[X]))),
		x_max = fminf(W-1,fmaxf(v0_persp[X],fmaxf(v1_persp[X],v2_persp[X])));

	if (vert0) { if (right0) x_min = (int)ceilf(fmaxf(v1_persp[X],(float)x_min)); else x_max = (int)floorf(fminf(v1_persp[X],(float)x_max)); }
	if (vert1) { if (right1) x_min = (int)ceilf(fmaxf(v2_persp[X],(float)x_min)); else x_max = (int)floorf(fminf(v2_persp[X],(float)x_max)); }
	if (vert2) { if (right2) x_min = (int)ceilf(fmaxf(v0_persp[X],(float)x_min)); else x_max = (int)floorf(fminf(v0_persp[X],(float)x_max)); }

	for (int x = x_min; x<=x_max; x++) {

		int y_min = 0,
			y_max = H-1;

		float y0 = v1_persp[Y]+(x-v1_persp[X])/(v2_persp[X]-v1_persp[X])*(v2_persp[Y]-v1_persp[Y]),
			   y1 = v2_persp[Y]+(x-v2_persp[X])/(v0_persp[X]-v2_persp[X])*(v0_persp[Y]-v2_persp[Y]),
			   y2 = v0_persp[Y]+(x-v0_persp[X])/(v1_persp[X]-v0_persp[X])*(v1_persp[Y]-v0_persp[Y]);
		if (!vert0) { if (lower0) y_min = (int)ceilf(fmaxf(y0,(float)y_min)); else y_max = (int)floorf(fminf(y0,(float)y_max)); }
		if (!vert1) { if (lower1) y_min = (int)ceilf(fmaxf(y1,(float)y_min)); else y_max = (int)floorf(fminf(y1,(float)y_max)); }
		if (!vert2) { if (lower2) y_min = (int)ceilf(fmaxf(y2,(float)y_min)); else y_max = (int)floorf(fminf(y2,(float)y_max)); }

		for (int y = y_min; y<=y_max; y++) {

			int idx = (b*H+y)*W+x;

			// Möller–Trumbore ray-triangle intersection
			float ray_vector[3] = {((float)x-center_x)/scale_x,
									((float)y-center_y)/scale_y,
									1.0};

			float h[3];
			cross_product_cuda(h,ray_vector,edge2);
			float det = dot_product_cuda(edge1,h);
			// if ray is parallel to triangle
			if (fabsf(det)<1e-8) continue;
			float inv_det = 1.0/det;

			float u = inv_det*dot_product_cuda(s,h);
			float v = inv_det*dot_product_cuda(ray_vector,q);
			float z = inv_det*dot_product_cuda(edge2,q);
			if (z<0) continue;
			float iz = 1.0/z;
			
	 		// update and lock
			int locked = 0;
			do {
				if ((locked = atomicCAS(LM+idx,0,1))==0) {
					if (iz>atomicAdd(IDM+idx,0)) {
						atomicExch(IDM+idx,iz);
						atomicExch(BM+idx*3+0,1.-u-v);
						atomicExch(BM+idx*3+1,u);
						atomicExch(BM+idx*3+2,v);
						atomicExch(IM+idx,f);
					}
					atomicExch(LM+idx,0);
				}
			}
			while (locked);

		}
		
	}

}

__device__
void vector_sub_cuda(float* res,float* a,float* b)
{
	for (int i = 0; i<3; i++) res[i] = a[i]-b[i];
}
__device__
float dot_product_cuda(float* a,float* b)
{
	return a[X]*b[X]+a[Y]*b[Y]+a[Z]*b[Z];
}
__device__
void cross_product_cuda(float* res,float* a,float* b)
{
	res[0] = a[Y]*b[Z]-a[Z]*b[Y];
	res[1] = a[Z]*b[X]-a[X]*b[Z];
	res[2] = a[X]*b[Y]-a[Y]*b[X];
}
__device__
void calib_vertex_cuda(float* v_persp,float* v_trans,float* CI)
{
	float v_calib_x = dot_product_cuda(v_trans,CI+0);
	float v_calib_y = dot_product_cuda(v_trans,CI+3);
	float v_calib_z = dot_product_cuda(v_trans,CI+6);
	v_persp[X] = v_calib_x/(v_calib_z+1e-16);
	v_persp[Y] = v_calib_y/(v_calib_z+1e-16);
}
