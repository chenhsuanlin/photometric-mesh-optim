#include <TH/TH.h>
#include <math.h>
#include <assert.h>
#include "cpu.h"

enum Coord3D {X=0,Y=1,Z=2};
float ray_origin[3] = {0.0,0.0,0.0};

int forward(THFloatTensor *cam_intr, // [3,3]
			THFloatTensor *face_vertices_trans, // [F,3,3]
			THIntTensor *batch_face_index, // [F,2]
			THIntTensor *index_map, // [B,H,W]
			THFloatTensor *baryc_map, // [B,H,W,3]
			THFloatTensor *inv_depth_map) // [B,H,W]
{
	int num_faces = face_vertices_trans->size[0],
		H = index_map->size[1],
		W = index_map->size[2];

	float *CI = THFloatTensor_data(cam_intr);
	float *FVT = THFloatTensor_data(face_vertices_trans);
	int *BFI = THIntTensor_data(batch_face_index);
	int *IM = THIntTensor_data(index_map);
	float *BM = THFloatTensor_data(baryc_map);
	float *IDM = THFloatTensor_data(inv_depth_map);

	float scale_x = CI[0], center_x = CI[2],
		   scale_y = CI[4], center_y = CI[5];

	for (int i = 0; i<num_faces; i++) {

		int b = BFI[i*2+0],
			f = BFI[i*2+1];
		float *v0_trans = FVT+i*9+0,
			   *v1_trans = FVT+i*9+3,
			   *v2_trans = FVT+i*9+6;
		float v0_persp[2], v1_persp[2], v2_persp[2];
		calib_vertex(v0_persp,v0_trans,CI);
		calib_vertex(v1_persp,v1_trans,CI);
		calib_vertex(v2_persp,v2_trans,CI);

		// precompute edges
		float edge1[3], edge2[3];
		vector_sub(edge1,v1_trans,v0_trans);
		vector_sub(edge2,v2_trans,v0_trans);
		float s[3], q[3];
		vector_sub(s,ray_origin,v0_trans);
		cross_product(q,s,edge1);

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
				cross_product(h,ray_vector,edge2);
				float det = dot_product(edge1,h);
				// if ray is parallel to triangle
				if (fabsf(det)<1e-8) continue;
				float inv_det = 1.0/det;

				float u = inv_det*dot_product(s,h);
				float v = inv_det*dot_product(ray_vector,q);
				float z = inv_det*dot_product(edge2,q);
				if (z<0) continue;
				float iz = 1.0/z;

				// update (no need to lock for single-thread CPU)
				if (iz>IDM[idx]) {
					IDM[idx] = iz;
					BM[idx*3+0] = 1.0-u-v;
					BM[idx*3+1] = u;
					BM[idx*3+2] = v;
					IM[idx] = f;
				}

			}
			
		}

	}

	return 1;
}

void vector_sub(float* res,float* a,float* b)
{
	for (int i = 0; i<3; i++) res[i] = a[i]-b[i];
}
float dot_product(float* a,float* b)
{
	return a[X]*b[X]+a[Y]*b[Y]+a[Z]*b[Z];
}
void cross_product(float* res,float* a,float* b)
{
	res[0] = a[Y]*b[Z]-a[Z]*b[Y];
	res[1] = a[Z]*b[X]-a[X]*b[Z];
	res[2] = a[X]*b[Y]-a[Y]*b[X];
}
void calib_vertex(float* v_persp,float* v_trans,float* CI)
{
	float v_calib_x = dot_product(v_trans,CI+0);
	float v_calib_y = dot_product(v_trans,CI+3);
	float v_calib_z = dot_product(v_trans,CI+6);
	v_persp[X] = v_calib_x/(v_calib_z+1e-8);
	v_persp[Y] = v_calib_y/(v_calib_z+1e-8);
}

