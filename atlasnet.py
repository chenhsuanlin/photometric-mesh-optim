import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import data
import util
import chamfer

# network
class AtlasNet(torch.nn.Module):

	def __init__(self,opt,eval_enc=False,eval_dec=False):
		super(AtlasNet,self).__init__()
		# define UV
		self.UV_sphere,self.faces_sphere = data.get_icosahedron(opt)
		self.UV_regular,self.faces_regular = self.get_regular_patch_grid(opt)
		self.faces_regular = self.duplicate_faces_original(opt)
		# define and load pretrained weights
		self.define_weights(opt)
		if opt.pretrained_dec is not None:
			self.load_pretrained_decoder(opt)
		for p in self.encoder.parameters(): p.requires_grad_(not eval_enc)
		for p in self.decoder.parameters(): p.requires_grad_(not eval_dec)
		(self.encoder.eval if eval_enc else self.encoder.train)()
		(self.decoder.eval if eval_dec else self.decoder.train)()

	def define_weights(self,opt):
		embed_size = 1024
		self.encoder = resnet18(pretrained=opt.imagenet_enc,num_classes=embed_size)
		code_size = embed_size+3 if opt.sphere else embed_size+2
		self.decoder = torch.nn.ModuleList([PointGenCon(code_size=code_size) for _ in range(opt.num_prim)])
		self.encoder = self.encoder.to(opt.device)
		self.decoder = self.decoder.to(opt.device)

	def load_pretrained_decoder(self,opt):
		print(util.magenta("loading pretrained decoder ({})...".format(opt.pretrained_dec)))
		weight_dict = torch.load(opt.pretrained_dec,map_location=opt.device)
		# remove "decoder/" prefix in dictionary
		decoder_weight_dict = {k[8:]: weight_dict[k] for k in weight_dict if "decoder" in k}
		self.decoder.load_state_dict(decoder_weight_dict)

	def decoder_forward(self,opt,code,regular=False):
		batch_size = code.shape[0]
		points_list = []
		for p in range(opt.num_prim):
			if opt.sphere:
				UV = self.UV_sphere.repeat(batch_size,1,1).permute(0,2,1)
			else:
				if regular:
					UV = self.UV_regular.repeat(batch_size,1,1).permute(0,2,1)
				else:
					UV = torch.rand(batch_size,2,opt.num_points,device=opt.device)
			concat = torch.cat([UV,code[...,None].repeat(1,1,UV.shape[2])],dim=1)
			points_prim = self.decoder[p](concat)
			points_list.append(points_prim)
		points = torch.cat(points_list,dim=-1).permute(0,2,1)
		return points

	def forward(self,opt,image,regular=False):
		code = self.encoder.forward(image)
		points = self.decoder_forward(opt,code,regular=regular)
		return points

	def get_regular_patch_grid(self,opt):
		N = opt.num_meshgrid
		# vertices (UV space)
		U,V = np.meshgrid(range(N+1),range(N+1))
		U = (U.astype(np.float32)/N).reshape([-1])
		V = (V.astype(np.float32)/N).reshape([-1])
		UV = np.stack([U,V],axis=-1)
		UV = torch.tensor(UV,dtype=torch.float32,device=opt.device)
		# facess
		J,I = np.meshgrid(range(N),range(N))
		face_upper = np.stack([I*(N+1)+J,I*(N+1)+J+1,(I+1)*(N+1)+J],axis=-1).reshape([-1,3])
		face_lower = np.stack([I*(N+1)+J+1,(I+1)*(N+1)+J+1,(I+1)*(N+1)+J],axis=-1).reshape([-1,3])
		faces = np.concatenate([face_upper,face_lower],axis=0)
		faces = torch.tensor(faces,dtype=torch.int32,device=opt.device)
		return UV,faces

	def duplicate_faces_original(self,opt):
		faces_list = [self.faces_regular+(opt.num_meshgrid+1)**2*p for p in range(opt.num_prim)]
		self.faces_regular = torch.cat(faces_list,dim=0)
		return self.faces_regular

# ---------- AtlasNet decoder blackbox below ----------

class PointGenCon(torch.nn.Module):
	def __init__(self,code_size):
		self.bottleneck_size = code_size
		super(PointGenCon,self).__init__()
		self.conv1 = torch.nn.Conv1d(self.bottleneck_size,self.bottleneck_size,1)
		self.conv2 = torch.nn.Conv1d(self.bottleneck_size,self.bottleneck_size//2,1)
		self.conv3 = torch.nn.Conv1d(self.bottleneck_size//2,self.bottleneck_size//4,1)
		self.conv4 = torch.nn.Conv1d(self.bottleneck_size//4,3,1)

		self.th = torch.nn.Tanh()
		self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
		self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size//2)
		self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size//4)

	def forward(self,x):
		batchsize = x.size()[0]
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)))
		x = self.th(self.conv4(x))
		return x

# ---------- ResNet blackbox below ----------

def resnet18(pretrained=False,**kwargs):
	model = ResNet(BasicBlock,[2,2,2,2],**kwargs)
	if pretrained:
		print(util.magenta("loading pretrained encoder..."))
		weight_dict = model_zoo.load_url("https://download.pytorch.org/models/resnet18-5c106cde.pth")
		block_names = list(set([k.split(".")[0] for k in weight_dict.keys()]))
		for b in block_names:
			if b=="fc": continue
			block_weight_dict = {".".join(k.split(".")[1:]): weight_dict[k] for k in weight_dict if k[:len(b)]==b}
			getattr(model,b).load_state_dict(block_weight_dict)
	return model

class BasicBlock(torch.nn.Module):

	expansion = 1

	def __init__(self,inplanes,planes,stride=1,downsample=None):
		super(BasicBlock,self).__init__()
		self.conv1 = torch.nn.Conv2d(inplanes,planes,kernel_size=3,stride=stride,padding=1,bias=False)
		self.bn1 = torch.nn.BatchNorm2d(planes)
		self.relu = torch.nn.ReLU(inplace=True)
		self.conv2 = torch.nn.Conv2d(planes,planes,kernel_size=3,stride=1,padding=1,bias=False)
		self.bn2 = torch.nn.BatchNorm2d(planes)
		self.downsample = downsample
		self.stride = stride

	def forward(self,x):
		residual = x
		out = self.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		if self.downsample is not None:
			residual = self.downsample(x)
		out += residual
		out = self.relu(out)
		return out

class ResNet(torch.nn.Module):

	def __init__(self,block,layers,num_classes=1000):
		self.inplanes = 64
		super(ResNet,self).__init__()
		self.conv1 = torch.nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
		self.bn1 = torch.nn.BatchNorm2d(64)
		self.relu = torch.nn.ReLU(inplace=True)
		self.maxpool = torch.nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
		self.layer1 = self._make_layer(block,64,layers[0])
		self.layer2 = self._make_layer(block,128,layers[1],stride=2)
		self.layer3 = self._make_layer(block,256,layers[2],stride=2)
		self.layer4 = self._make_layer(block,512,layers[3],stride=2)
		self.avgpool = torch.nn.AvgPool2d(7)
		self.fc = torch.nn.Linear(512*block.expansion,num_classes)
		for m in self.modules():
			if isinstance(m,torch.nn.Conv2d):
				n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
				m.weight.data.normal_(0,np.sqrt(2./n))
			elif isinstance(m,torch.nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def _make_layer(self,block,planes,blocks,stride=1):
		downsample = None
		if stride!=1 or self.inplanes!=planes*block.expansion:
			downsample = torch.nn.Sequential(
				torch.nn.Conv2d(self.inplanes,planes*block.expansion,kernel_size=1,stride=stride,bias=False),
				torch.nn.BatchNorm2d(planes*block.expansion),
			)
		layers = []
		layers.append(block(self.inplanes,planes,stride,downsample))
		self.inplanes = planes*block.expansion
		for i in range(1,blocks):
			layers.append(block(self.inplanes,planes))
		return torch.nn.Sequential(*layers)

	def forward(self,x):
		x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.avgpool(x)
		x = x.reshape(x.shape[0],-1)
		x = self.fc(x)
		return x

# ---------- chamfer distance blackbox below ----------

class ChamferDistance(torch.autograd.Function):

	@staticmethod
	def forward(ctx,opt,p1,p2):
		batch_size = p1.shape[0]
		num_p1_points = p1.shape[1]
		num_p2_points = p2.shape[1]
		dist1 = torch.zeros(batch_size,num_p1_points,device=opt.device)
		dist2 = torch.zeros(batch_size,num_p2_points,device=opt.device)
		idx1 = torch.zeros(batch_size,num_p1_points,dtype=torch.int32,device=opt.device)
		idx2 = torch.zeros(batch_size,num_p2_points,dtype=torch.int32,device=opt.device)
		p1 = p1.contiguous()
		p2 = p2.contiguous()
		if "cuda" in opt.device:
			chamfer.nnd_forward_cuda(p1,p2,dist1,dist2,idx1,idx2)
		else:
			chamfer.nnd_forward(p1,p2,dist1,dist2,idx1,idx2)
		ctx.opt = opt
		ctx.save_for_backward(p1,p2,dist1,dist2,idx1,idx2)
		return dist1,dist2

	@staticmethod
	def backward(ctx,grad_dist1,grad_dist2):
		opt = ctx.opt
		p1,p2,dist1,dist2,idx1,idx2 = ctx.saved_tensors
		grad_p1 = torch.zeros_like(p1)
		grad_p2 = torch.zeros_like(p2)
		if "cuda" in opt.device:
			chamfer.nnd_backward_cuda(p1,p2,grad_p1,grad_p2,grad_dist1,grad_dist2,idx1,idx2)
		else:
			chamfer.nnd_backward(p1,p2,grad_p1,grad_p2,grad_dist1,grad_dist2,idx1,idx2)
		return None,grad_p1,grad_p2
