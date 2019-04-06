import numpy as np
import torch
import meshrender

def rasterize_3D_mesh(opt,vertices,faces,cam_extr,cam_intr):
	"""
	input: vertices [V,3], faces [F,3], cam_extr [B,3,4], cam_intr [3,3]
	output: index_map [B,H,W], baryc_map [B,H,W,3], mask_map [B,H,W], inv_depth_map [B,H,W], vertices [B,V,3]
	"""
	B = len(cam_extr)
	vertices = vertices.repeat(B,1,1) # [B,V,3]
	faces = faces.repeat(B,1,1) # [B,F,3]
	vertices = calib_extrinsic(opt,vertices,cam_extr) # [B,V,3]
	with torch.no_grad():
		batch_face_index = get_batch_face_index(opt,faces,B) # [BF,2]
		face_vertices = get_face_vertices(opt,vertices,faces,B) # [BF,3,3]
		index_map,baryc_map,mask_map,inv_depth_map = Rasterize().apply(opt,B,cam_intr,face_vertices,batch_face_index)
	return index_map,baryc_map,mask_map,inv_depth_map,vertices

def calib_extrinsic(opt,vertices,cam_extr):
	"""
	input: vertices [B,V,3], cam_extr [B,3,4]
	output: vertices_trans [B,V,3]
	"""
	ones = torch.ones_like(vertices)[...,:1]
	vertices_homo = torch.cat([vertices,ones],dim=-1)
	vertices_trans = vertices_homo@cam_extr.transpose(-1,-2)
	return vertices_trans

def calib_intrinsic(opt,vertices_trans,cam_intr):
	"""
	input: vertices_trans [B,V,3], cam_intr [3,3]
	output: vertices_trans [B,V,3]
	"""
	vertices_persp = vertices_trans@cam_intr.t()
	vertices_persp[...,0] /= (vertices_persp[...,2]+1e-8)
	vertices_persp[...,1] /= (vertices_persp[...,2]+1e-8)
	return vertices_persp

def get_batch_face_index(opt,faces,B):
	"""
	input: faces [B,F,3]
	output: batch_face_index [BF,2] (batch index, face index)
	"""
	num_faces = faces.shape[1]
	face_index,batch_index = np.meshgrid(range(num_faces),range(B))
	face_index = torch.tensor(face_index,dtype=torch.int32,device=opt.device).reshape(-1)
	batch_index = torch.tensor(batch_index,dtype=torch.int32,device=opt.device).reshape(-1)
	batch_face_index = torch.stack([batch_index,face_index],dim=-1)
	return batch_face_index

def get_face_vertices(opt,vertices,faces,B):
	"""
	input: vertices [B,V,3], faces [B,F,3]
	output: face_vertices [BF,3,3]
	"""
	face_vertices_list = []
	for b in range(B):
		face_vertices_list.append(vertices[b][faces[b].long()]) # [F,3,3]
	face_vertices = torch.cat(face_vertices_list,dim=0)
	return face_vertices

# ---------- rasterize function below ----------

class Rasterize(torch.autograd.Function):

	@staticmethod
	def forward(ctx,opt,B,cam_intr,face_vertices_trans,batch_face_index):
		index_map = torch.ones(B,opt.H,opt.W,device=opt.device,dtype=torch.int32).mul_(-1)
		baryc_map = torch.zeros(B,opt.H,opt.W,3,device=opt.device,dtype=torch.float32)
		inv_depth_map = torch.zeros(B,opt.H,opt.W,device=opt.device,dtype=torch.float32)
		if "cuda" in opt.device:
			lock_map = torch.zeros(B,opt.H,opt.W,device=opt.device,dtype=torch.int32)
			meshrender.forward_cuda(cam_intr,face_vertices_trans,batch_face_index,index_map,baryc_map,inv_depth_map,lock_map)
		else:
			meshrender.forward(cam_intr,face_vertices_trans,batch_face_index,index_map,baryc_map,inv_depth_map)
		mask_map = (index_map!=-1).float()
		return index_map,baryc_map,mask_map,inv_depth_map

	@staticmethod
	def backward(ctx):
		# we don't need to backpropagate rasterization!
		raise NotImplementedError
