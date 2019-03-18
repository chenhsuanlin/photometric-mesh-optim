import numpy as np
import torch
import torch.utils.data
import os
import easydict
import imageio

def load_sequence_list(opt,subset=None):
	seq_list = []
	list_file = "data/lists/all_test.list"
	with open(list_file) as file:
		for line in file:
			c,m = line.strip().split()
			if opt.category is not None and c!=opt.category: continue
			seq_list.append((c,m))
	if subset is not None:
		seq_list = seq_list[:subset]
	print("number of sequences: {0}".format(len(seq_list)))
	return seq_list

def load_sequence(opt,c,m):
	sequence = easydict.EasyDict()
	# load RGB images
	print("reading RGB .npy file...")
	seq_fname = "{0}/{1}/{2}.npy".format(opt.seq_path,c,m)
	RGBA = np.load(seq_fname)/255.0
	sequence.RGB = RGBA[...,:3] # discard mask channel
	# load coordinate system transform (from canonical mesh to world cameras)
	sequence.cs_map_mtrx = np.load("{0}/estim_3Dsim.npy".format(path)) if opt.sfm else \
						   np.array([[1,0,0,0],
									 [0,0,-1,0],
									 [0,1,0,0]])
	# load camera matrices
	if opt.sfm:
		print("reading camera from SfM output...")
		raise NotImplementedError
		# dataset["extr"],dataset["intr"] = pose.get_SfM_cameras(opt,"{0}/sparse/0".format(path))
	else:
		print("reading ground-truth camera .npz file...")
		cam_fname = "data/camera.npz"
		cam = np.load(cam_fname)
		sequence.cam_extr = cam["extr"]
		sequence.cam_intr = cam["intr"]
	# convert to Pytorch
	for k in sequence:
		sequence[k] = torch.tensor(sequence[k],dtype=torch.float32,device=opt.device)
	sequence.RGB = sequence.RGB.permute(0,3,1,2)
	# set length
	sequence.len = lambda: len(sequence.RGB)
	return sequence

def setup_loader(opt,dataset,shuffle=False):
	loader = torch.utils.data.DataLoader(dataset,
		batch_size=opt.batch_size,
		num_workers=opt.num_workers,
		shuffle=shuffle,
	)
	print("number of samples: {}".format(len(dataset)))
	return loader

class DatasetPretrain(torch.utils.data.Dataset):

	def __init__(self,opt,load_test=False):
		super(DatasetPretrain,self).__init__()
		self.opt = opt
		self.load_test = load_test
		# read object data list
		categories = []
		with open("data/categories.txt") as file:
			for line in file:
				c = line.strip().split()[0]
				categories.append(c)
		self.models = []
		for c in categories:
			if opt.category is not None and c!=opt.category: continue
			with open("data/lists/{0}_{1}.list".format(c,"test" if load_test else "train")) as file:
				for line in file:
					m = line.strip()
					self.models.append((c,m))
		# read background data list
		self.backgrounds = []
		with open("data/lists/indoor_{0}.list".format("test" if load_test else "train")) as file:
			for line in file:
				self.backgrounds.append(line.strip().split())

	def __getitem__(self,idx):
		opt = self.opt
		c,m = self.models[idx]
		# get 3D points
		ply_fname = "{0}/{1}/ply/{2}.points.ply".format(opt.pointcloud_path,c,m)
		points = self.sample_points_from_ply(opt,ply_fname)
		points = torch.tensor(points,dtype=torch.float32)
		# ShapeNet rendering
		view_idx = 0 if self.load_test else np.random.randint(72)
		image_obj_fname = "{0}/{1}/{2}/{3}.png".format(opt.rendering_path,c,m,view_idx)
		image_obj = imageio.imread(image_obj_fname)/255.0
		if not self.load_test and opt.aug_transl is not None:
			image_obj = self.random_translate(opt,image_obj)
		alpha = image_obj[...,-1:]
		# SUN360 background
		ei = view_idx//24
		ai = 0 if self.load_test else np.random.randint(24)
		view_idx2 = ei*24+ai
		rand_idx = 0 if self.load_test else np.random.randint(len(self.backgrounds))
		bg = self.backgrounds[rand_idx]
		image_bg_fname = "{0}/{1}/{2}.png".format(opt.sun360_path,bg[1],view_idx2)
		image_bg = imageio.imread(image_bg_fname)/255.0
		# composite images
		image = image_obj[...,:3]*alpha+image_bg*(1-alpha)
		image = torch.tensor(image,dtype=torch.float32).permute(2,0,1)
		return image,points

	def sample_points_from_ply(self,opt,ply_fname):
		points_str = []
		# fast but dirty data reading...
		# assume points in line are shuffled, and each line is about of length 60
		file_size = os.stat(ply_fname).st_size
		chunk_size = 60*(opt.num_points_all+2)
		with open(ply_fname) as file:
			file.seek(np.random.randint(400,file_size-chunk_size))
			chunk = file.read(chunk_size)
			points_str = chunk.split(os.linesep)[1:opt.num_points_all+1]
		points = [[float(n) for n in s.split()] for s in points_str]
		assert(len(points)==opt.num_points_all)
		points = np.array(points,dtype=np.float32)[:,:3] # ignore normals
		return points

	def random_translate(self,opt,image):
		y_shift,x_shift = np.random.randint(opt.aug_transl*2+1,size=[2])-opt.aug_transl
		image_new = np.zeros_like(image)
		image_new[max(0,-y_shift):min(opt.H,-y_shift+opt.H),max(0,-x_shift):min(opt.W,-x_shift+opt.W)] \
			= image[max(0,y_shift):min(opt.H,y_shift+opt.H),max(0,x_shift):min(opt.W,x_shift+opt.W)]
		return image_new

	def __len__(self):
		return len(self.models)

# helper functions for icosahedron (spherical mesh)
def get_icosahedron(opt):
	npz = np.load("data/icosahedron.npz")
	V = npz["V"][:,[1,0,2]]
	F = npz["F"]
	for _ in range(opt.sphere_densify):
		V,F = densify_icosahedron(opt,V,F)
	V = torch.tensor(V,dtype=torch.float32,device=opt.device)
	F = torch.tensor(F,dtype=torch.int64,device=opt.device)
	return V,F

def densify_icosahedron(opt,V,F,from_pytorch=False):
	if from_pytorch:
		V = V.detach().cpu().numpy()
		F = F.detach().cpu().numpy()
	V_list = list(V)
	D_insert = {}
	D_new_v = {}
	cur_idx = len(V)
	# add new vertices
	for f in F:
		for i in range(3):
			fi1,fi2 = f[i],f[(i+1)%3]
			hash_idx = min(fi1,fi2)*len(V)+max(fi1,fi2)
			if hash_idx not in D_new_v:
				new_v = (V[fi1]+V[fi2])/2.0
				if not from_pytorch:
					new_v /= np.linalg.norm(new_v)
				D_new_v[hash_idx] = [new_v,cur_idx,fi1,fi2]
				V_list.append(new_v)
				if fi1 not in D_insert:
					D_insert[fi1] = {}
				D_insert[fi1][fi2] = cur_idx
				if fi2 not in D_insert:
					D_insert[fi2] = {}
				D_insert[fi2][fi1] = cur_idx
				cur_idx += 1
	F_new_list = []
	for f in F:
		for i in range(3):
			fi1,fi2,fi3 = f[i],f[(i+1)%3],f[(i+2)%3]
			fi2_new = D_insert[fi1][fi2]
			fi3_new = D_insert[fi1][fi3]
			F_new_list.append([fi1,fi2_new,fi3_new])
		fi1_new = D_insert[fi1][fi2]
		fi2_new = D_insert[fi2][fi3]
		fi3_new = D_insert[fi3][fi1]
		F_new_list.append([fi1_new,fi2_new,fi3_new])
	V = np.array(V_list)
	F = np.array(F_new_list)
	if from_pytorch:
		V = torch.tensor(V,dtype=torch.float32,device=opt.device,requires_grad=True)
		F = torch.tensor(F,dtype=torch.int64,device=opt.device)
	return V,F
