import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils
import os,sys,time
import termcolor
import skvideo.io
import render

# convert to colored strings
def red(content): return termcolor.colored(str(content),"red",attrs=["bold"])
def green(content): return termcolor.colored(str(content),"green",attrs=["bold"])
def blue(content): return termcolor.colored(str(content),"blue",attrs=["bold"])
def cyan(content): return termcolor.colored(str(content),"cyan",attrs=["bold"])
def yellow(content): return termcolor.colored(str(content),"yellow",attrs=["bold"])
def magenta(content): return termcolor.colored(str(content),"magenta",attrs=["bold"])

def get_time(sec):
	h = int(sec//3600)
	m = int((sec//60)%60)
	s = sec%60
	return h,m,s

def restore_checkpoint_from_epoch(opt,model,keys):
	print(magenta("resuming from epoch {}...".format(opt.from_epoch)))
	with torch.cuda.device(opt.gpu):
		checkpoint = torch.load("checkpoint/{0}/{1}/ep{2}.npz".format(opt.group,opt.name,opt.from_epoch),map_location=opt.device)
		for k in keys:
			getattr(model,k).load_state_dict(checkpoint[k])

def restore_checkpoint(opt,model,load_name,keys):
	print(magenta("loading checkpoint {}...".format(load_name)))
	with torch.cuda.device(opt.gpu):
		checkpoint = torch.load(load_name,map_location=opt.device)
		for k in keys:
			getattr(model,k).load_state_dict(checkpoint[k])

def save_checkpoint(opt,model,keys,ep):
	os.makedirs("checkpoint/{0}/{1}".format(opt.group,opt.name),exist_ok=True)
	checkpoint = {}
	with torch.cuda.device(opt.gpu):
		for k in keys:
			checkpoint[k] = getattr(model,k).state_dict()
		torch.save(checkpoint,"checkpoint/{0}/{1}/ep{2}.npz".format(opt.group,opt.name,ep))
	print(green("checkpoint saved: ({0}) {1}, epoch {2}".format(opt.group,opt.name,ep)))

def make_tb_image(opt,image):
	return torchvision.utils.make_grid(image[:15],nrow=5,pad_value=0.5)

def save_mesh(opt,model,vertices):
	os.makedirs("optimized_mesh/{0}/{1}".format(opt.group,opt.name),exist_ok=True)
	with torch.cuda.device(opt.gpu):
		torch.save({
			"code": model.code,
			"sim3": model.sim3,
			"faces": model.faces,
			"vertices": vertices,
		},"optimized_mesh/{0}/{1}/{2}_{3}.npz".format(opt.group,opt.name,model.c,model.m))
	print(green("optimized mesh saved: ({0}) {1}".format(opt.group,opt.name)))

def get_edge_map(opt,mask,inv_depth):
	edge = torch.zeros_like(mask)
	depth = 1/inv_depth
	depth[~mask] = 0
	lapl_vert = depth[:-2,:]-2*depth[1:-1,:]+depth[2:,:]
	lapl_horiz = depth[:,:-2]-2*depth[:,1:-1]+depth[:,2:]
	thres = 0.1 # heuristically set for now
	edge[1:-1,:] |= lapl_vert>thres
	edge[:,1:-1] |= lapl_horiz>thres
	return edge

def get_normal_map(opt,index,vertices,faces):
	face_vertices = vertices[faces.long()]
	v1,v2,v3 = torch.unbind(face_vertices,dim=1)
	normal = F.normalize(torch.cross(v2-v1,v3-v2),dim=1)
	# face normals towards camera
	normal[normal[:,2]<0] *= -1
	normal_color = (normal+1)/2
	normal_color = torch.cat([torch.zeros(1,3,device=opt.device),normal_color],dim=0)
	normal_color[0] = 0
	normal_map = normal_color[index.long()+1].permute(2,0,1)
	return normal_map

def visualize_frames(opt,sequence,vertices,faces,vis_idx):
	vis_idx = torch.tensor(vis_idx,dtype=torch.int64,device=opt.device)
	RGB = sequence.RGB[vis_idx]
	cam_extr = sequence.cam_extr[vis_idx]
	cam_intr = sequence.cam_intr
	with torch.no_grad():
		index,_,mask,inv_depth,vertices_t = render.rasterize_3D_mesh(opt,vertices,faces,cam_extr,cam_intr)
	frames = []
	for RGB_i,index_i,mask_i,inv_depth_i,vertices_i in zip(RGB,index,mask,inv_depth,vertices_t):
		frame = visualize_frame_with_mesh(opt,RGB_i,index_i,mask_i,inv_depth_i,vertices_i,faces)
		frames.append(frame)
	return frames

def visualize_frame_with_mesh(opt,RGB,index,mask,inv_depth,vertices,faces):
	normal_color = get_normal_map(opt,index,vertices,faces)
	mask = mask.byte()
	edge = get_edge_map(opt,mask,inv_depth)
	frame = RGB*0.9+0.1
	frame[mask.repeat(3,1,1)] *= 0.4
	frame += normal_color*0.6
	frame[edge.repeat(3,1,1)] = 0
	return frame

def write_video(opt,sequence,vertices,faces,c,m):
	os.makedirs("video/{0}/{1}".format(opt.group,opt.name),exist_ok=True)
	fname = "video/{0}/{1}/{2}_{3}.mp4".format(opt.group,opt.name,c,m)
	video_writer = skvideo.io.FFmpegWriter(fname,outputdict={"-b":"30M","-r":"15"})
	frames = []
	batch_size = sequence.len() if opt.batch_size_pmo==-1 else opt.batch_size_pmo
	for b in range(0,sequence.len(),batch_size):
		idx = torch.arange(b,min(b+batch_size,sequence.len()),dtype=torch.int64,device=opt.device)
		frames_batch = visualize_frames(opt,sequence,vertices,faces,idx)
	frames += frames_batch
	for f in frames:
		f_uint8 = (f*255).byte().cpu().permute(1,2,0).numpy()
		video_writer.writeFrame(f_uint8)
	video_writer.close()

