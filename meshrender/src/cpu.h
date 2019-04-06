int forward(THFloatTensor *cam_intr, // [3,3]
			THFloatTensor *face_vertices_trans, // [F,3,3]
			THIntTensor *batch_face_index, // [F]
			THIntTensor *index_map, // [B,H,W]
			THFloatTensor *baryc_map, // [B,H,W,3]
			THFloatTensor *inv_depth_map); // [B,H,W]

void vector_sub(float* res,float* a,float* b);
float dot_product(float* a,float* b);
void cross_product(float* res,float* a,float* b);
void calib_vertex(float* v_persp,float* v_trans,float* CI);
