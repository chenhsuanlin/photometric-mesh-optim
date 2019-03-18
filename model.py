import numpy as np
import os,sys,time
import torch
import torch.nn.functional as F
import easydict
import atlasnet,data,pose,render,util
import tensorboardX
import visdom

class Model():

	def __init__(self,opt):
		super(Model,self).__init__()
		self.opt = opt
		torch.backends.cudnn.benchmark = True
		# torch.backends.cudnn.deterministic = True

	def load_sequence(self,opt,c,m):
		print(util.magenta("loading sequence..."))
		self.sequence = data.load_sequence(opt,c,m)
		self.c,self.m = c,m

	def build_network(self,opt):
		print(util.magenta("building AtlasNet..."))
		self.network = atlasnet.AtlasNet(opt,eval_enc=True,eval_dec=True)
		self.faces = self.network.faces_regular

	def setup_optimizer(self,opt):
		optim_list = [{ "params": [v for k,v in self.sim3.items()], "lr": opt.lr_pmo },
					  { "params": [self.code], "lr": opt.lr_pmo }]
		self.optim = torch.optim.Adam(optim_list)

	def restore_checkpoint(self,opt):
		util.restore_checkpoint(opt,self,opt.load,["network"])

	def setup_visualizer(self,opt):
		if opt.log_tb: self.tb = tensorboardX.SummaryWriter(log_dir="summary/{0}/{1}/{2}/{3}".format(opt.group,self.c,self.m,opt.name))
		if opt.log_visdom: self.vis = visdom.Visdom(server=opt.vis_server,port=opt.vis_port,env=opt.group)

	def time_start(self,opt): self.time_start = time.time()

	def setup_variables(self,opt):
		input_image = self.sequence.RGB[opt.init_idx]
		self.code_init = self.network.encoder.forward(input_image[None]).detach()
		self.code = self.code_init.clone()
		self.code.requires_grad_(True)
		self.sim3 = easydict.EasyDict({
			"scale": torch.tensor(0.,device=opt.device,requires_grad=True),
			"rot": torch.tensor([0.,0.,0.],device=opt.device,requires_grad=True),
			"trans": torch.tensor([0.,0.,0.],device=opt.device,requires_grad=True),
		})
		self.cs_map_mtrx = pose.add_noise(opt,self.sequence.cs_map_mtrx) if opt.noise else \
						   self.sequence.cs_map_mtrx

	def graph_forward(self,opt):
		var = easydict.EasyDict()
		# forward through AtlasNet decoder
		var.vertices_canon = self.network.decoder_forward(opt,self.code,regular=True)[0]
		# apply 3D similarity transformation
		var.vertices_world = pose.apply_3Dsim(opt,var.vertices_canon,self.cs_map_mtrx)
		var.sim3_mtrx = pose.params_to_mtrx(opt,self.sim3)
		var.vertices = pose.apply_3Dsim(opt,var.vertices_world,var.sim3_mtrx)
		return var

	def compute_loss(self,opt,var):
		loss = easydict.EasyDict()
		loss.photom = 0
		# accumulate photometric gradients to avoid multiple network forward calls
		var.vertices_clone = var.vertices.detach()
		var.vertices_clone.requires_grad_(True)
		batch_size = self.sequence.len()-1 if opt.batch_size_pmo==-1 else opt.batch_size_pmo
		for b in range(0,self.sequence.len()-1,batch_size):
			idx_a = torch.arange(b,min(b+batch_size,self.sequence.len()-1),dtype=torch.int64,device=opt.device)
			idx_b = idx_a+1
			loss.photom += self.compute_photometric_loss_batch(opt,var,idx_a,idx_b)
		var.vertices.backward(gradient=var.vertices_clone.grad)
		# regularization
		loss.code = ((self.code-self.code_init)**2).sum()
		loss.scale = -self.sim3.scale
		# weight all losses
		loss.all = loss.photom \
				  +opt.code*loss.code \
				  +opt.scale*loss.scale
		return loss

	def compute_photometric_loss_batch(self,opt,var,idx_a,idx_b):
		# get RGB images and cameras
		RGB_a = self.sequence.RGB[idx_a]
		RGB_b = self.sequence.RGB[idx_b]
		cam_extr_a = self.sequence.cam_extr[idx_a]
		cam_extr_b = self.sequence.cam_extr[idx_b]
		cam_intr = self.sequence.cam_intr
		# compute virtual viewpoint
		cam_extr_V = pose.interpolate_camera(cam_extr_a,cam_extr_b,alpha=0.5)
		# rasterization
		index_a,_,_,_,vertices_a = render.rasterize_3D_mesh(opt,var.vertices_clone,self.faces,cam_extr_a,cam_intr)
		index_b,_,_,_,vertices_b = render.rasterize_3D_mesh(opt,var.vertices_clone,self.faces,cam_extr_b,cam_intr)
		index_V,baryc_V,mask_V,_,_ = render.rasterize_3D_mesh(opt,var.vertices_clone,self.faces,cam_extr_V,cam_intr)
		# project 3D points to 2D sampling coordinates
		points_a = self.project_coordinates(opt,vertices_a,self.faces,index_V,baryc_V,cam_intr)
		points_b = self.project_coordinates(opt,vertices_b,self.faces,index_V,baryc_V,cam_intr)
		# synthesize images
		image_synth_a = self.sample_RGB(opt,RGB_a,points_a)
		image_synth_b = self.sample_RGB(opt,RGB_b,points_b)
		# mask out invalid points
		valid_range_a = self.in_range(opt,points_a)
		valid_range_b = self.in_range(opt,points_b)
		valid_index_a = self.compare_valid_index(opt,index_a,index_V,points_a)
		valid_index_b = self.compare_valid_index(opt,index_b,index_V,points_b)
		valid_mask = mask_V.byte()
		valid = (valid_range_a&valid_range_b)& \
				(valid_index_a&valid_index_b)& \
				valid_mask
		# compute photometric loss
		loss = self.photometric_loss2(opt,image_synth_a,image_synth_b,valid=valid) if opt.avg_frame else \
			   self.photometric_loss(opt,image_synth_a,image_synth_b,valid=valid)
		loss *= len(idx_a)/(self.sequence.len()-1)
		# backpropagate photometric gradients
		loss.backward()
		return loss.detach()

	def in_range(self,opt,points):
		return (points[...,0]>=0)&(points[...,0]<=opt.W-1)& \
			   (points[...,1]>=0)&(points[...,1]<=opt.H-1)

	def project_coordinates(self,opt,vertices,faces,index_V,baryc_V,cam_intr):
		pixel_vertices_list = []
		batch_size = len(index_V)
		for b in range(batch_size):
			pixel_vertices_list.append(vertices[b][faces.long()][index_V[b].long()])
		pixel_vertices = torch.stack(pixel_vertices_list,dim=0)
		points_3D = (pixel_vertices*baryc_V[...,None]).sum(dim=3)
		points_2D = render.calib_intrinsic(opt,points_3D,cam_intr)[...,:2]
		return points_2D

	def sample_RGB(self,opt,RGB,points):
		X = points[...,0]/(opt.W-1)*2-1
		Y = points[...,1]/(opt.H-1)*2-1
		grid = torch.stack([X,Y],dim=-1)
		image_synth = F.grid_sample(RGB,grid,mode="bilinear")
		return image_synth

	def compare_valid_index(self,opt,index_s,index_V,points):
		batch_size = len(index_V)
		index_synth_list = []
		index_vec = index_s.reshape(batch_size,opt.H*opt.W)
		# get index map from 4 integer corners
		for Y in [points[...,1].floor(),points[...,1].ceil()]:
			for X in [points[...,0].floor(),points[...,0].ceil()]:
				grid_sample = Y.long().clamp(min=0,max=opt.H-1)*opt.W \
							 +X.long().clamp(min=0,max=opt.W-1)
				grid_sample_vec = grid_sample.reshape(batch_size,opt.H*opt.W)
				index_synth_vec = torch.gather(index_vec,1,grid_sample_vec)
				index_synth = index_synth_vec.reshape(batch_size,opt.H,opt.W)
				index_synth_list.append(index_synth)
		# consider only points where projected coordinates have consistent triangle indices
		valid_index = (index_synth_list[0]==index_V) \
					 &(index_synth_list[1]==index_V) \
					 &(index_synth_list[2]==index_V) \
					 &(index_synth_list[3]==index_V)
		return valid_index

	def photometric_loss(self,opt,image_a,image_b,valid):
		valid = valid[:,None].repeat(1,3,1,1)
		image_a = image_a[valid]
		image_b = image_b[valid]
		loss = F.l1_loss(image_a,image_b)
		return loss

	def photometric_loss2(self,opt,image_a,image_b,valid):
		valid = valid[:,None,:,:].float()
		valid_count = valid.sum(dim=-1).sum(dim=-1)
		image_a = image_a*valid
		image_b = image_b*valid
		loss_l1 = (image_a-image_b).abs()
		loss_sample = loss_l1.sum(dim=-1).sum(dim=-1)/(valid_count+1e-8)
		loss = loss_sample.mean()
		return loss

	def optimize(self,opt):
		vis_idx = np.linspace(0,self.sequence.len()-1,15).astype(int)
		for it in range(opt.to_it):
			self.optim.zero_grad()
			var = self.graph_forward(opt)
			loss = self.compute_loss(opt,var)
			loss.all.backward() # photometric gradients has been backpropagated already
			self.optim.step()
			if (it+1)%1==0: self.show_progress(opt,it,loss)
			if (it+1)%1==0: self.log_losses(opt,it,loss)
			if (it+1)%5==0: self.visualize(opt,it,var,vis_idx=vis_idx)

	def show_progress(self,opt,it,loss):
		time_elapsed = util.get_time(time.time()-self.time_start)
		print("it {0}/{1}, lr:{3}, loss:{4}, time:{2}"
			.format(util.cyan("{}".format(it+1)),
					opt.to_it,
					util.green("{0}:{1:02d}:{2:05.2f}".format(*time_elapsed)),
					util.yellow("{:.2e}".format(opt.lr_pmo)),
					util.red("{:.4e}".format(loss.all)),
		))

	def log_losses(self,opt,it,loss):
		if opt.log_tb:
			self.tb.add_scalar("optim/loss",loss.all,it+1)
			self.tb.add_scalar("optim/loss_photom",loss.photom,it+1)
			self.tb.add_scalar("optim/loss_code",loss.code,it+1)
			self.tb.add_scalar("optim/loss_scale",loss.scale,it+1)

	def visualize(self,opt,it,var,vis_idx):
		if opt.log_tb:
			frames = util.visualize_frames(opt,self.sequence,var.vertices,self.faces,vis_idx)
			self.tb.add_image("optim/mesh",util.make_tb_image(opt,frames),it+1)
		if opt.log_visdom:
			self.vis.mesh(var.vertices,self.faces,
				env="{0}/{1}/{2}".format(opt.group,opt.name,self.c),win=self.m,
				opts={"opacity":1.0,
					  "title":"{0}, it {1}".format(self.m,it+1),})

	def write_video(self,opt):
		if opt.video:
			with torch.no_grad():
				var = self.graph_forward(opt)
			util.write_video(opt,self.sequence,var.vertices,self.faces,self.c,self.m)

	def save_mesh(self,opt):
		with torch.no_grad():
			var = self.graph_forward(opt)
		util.save_mesh(opt,self,var.vertices)
