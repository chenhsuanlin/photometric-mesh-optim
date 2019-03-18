import numpy as np
import os,sys,time
import torch
import torch.nn.functional as F
import easydict
import atlasnet,data
import util
import tensorboardX
import visdom

class Model():

	def __init__(self,opt):
		super(Model,self).__init__()
		self.opt = opt
		torch.backends.cudnn.benchmark = True
		# torch.backends.cudnn.deterministic = True

	def load_dataset(self,opt):
		print(util.magenta("loading training data..."))
		train_data = data.DatasetPretrain(opt,load_test=False)
		self.train_loader = data.setup_loader(opt,train_data,shuffle=True)
		print(util.magenta("loading test data..."))
		test_data = data.DatasetPretrain(opt,load_test=True)
		self.test_loader = data.setup_loader(opt,test_data,shuffle=False)

	def build_network(self,opt):
		print(util.magenta("building AtlasNet..."))
		self.network = atlasnet.AtlasNet(opt)

	def setup_optimizer(self,opt):
		optim_list = [{ "params": [p for p in self.network.parameters() if p.requires_grad], "lr": opt.lr_pretrain }]
		self.optim = torch.optim.Adam(optim_list)
		self.sched = torch.optim.lr_scheduler.StepLR(self.optim,step_size=opt.lr_step,gamma=opt.lr_decay)

	def restore_checkpoint(self,opt):
		if opt.from_epoch!=0:
			util.restore_checkpoint_from_epoch(opt,self,["network","optim","sched"])
		elif opt.load is not None:
			util.restore_checkpoint(opt,self,opt.load,["network","optim","sched"])
		elif opt.imagenet_enc or opt.pretrained_dec is not None: pass
		else:
			print(util.magenta("training from scratch..."))

	def setup_visualizer(self,opt):
		self.tb = tensorboardX.SummaryWriter(log_dir="summary/{0}/{1}".format(opt.group,opt.name))
		self.vis = visdom.Visdom(server=opt.vis_server,port=opt.vis_port,env=opt.group)

	def time_start(self,opt): self.time_start = time.time()

	def graph_forward(self,opt,batch,training=False):
		var = easydict.EasyDict()
		[var.image,var.points_GT] = [v.to(opt.device) for v in batch]
		var.points_pred = self.network.forward(opt,var.image)
		return var

	def compute_loss(self,opt,var,ep=None):
		loss = easydict.EasyDict()
		dist1,dist2 = atlasnet.ChamferDistance().apply(opt,var.points_GT,var.points_pred)
		loss.chamfer = dist1.mean()+dist2.mean()
		loss.all = loss.chamfer
		return loss

	def train_epoch(self,opt,ep):
		self.network.train()
		self.sched.step()
		for it,batch in enumerate(self.train_loader):
			self.optim.zero_grad()
			var = self.graph_forward(opt,batch,training=True)
			loss = self.compute_loss(opt,var,ep=ep)
			loss.all.backward()
			self.optim.step()
		if (ep+1)%1==0: self.show_progress(opt,ep,loss)
		if (ep+1)%1==0: self.log_losses(opt,ep,loss,training=True)
		if (ep+1)%5==0: self.visualize(opt,ep,var,training=True)

	def evaluate(self,opt,ep=None):
		self.network.eval()
		loss_eval = easydict.EasyDict()
		count = 0
		with torch.no_grad():
			for it,batch in enumerate(self.test_loader):
				var = self.graph_forward(opt,batch,training=False)
				loss = self.compute_loss(opt,var,ep=ep)
				batch_size = len(batch[0])
				for k in loss:
					if k not in loss_eval: loss_eval[k] = 0
					loss_eval[k] += loss[k]*batch_size
				count += batch_size
		for k in loss_eval: loss_eval[k] /= count
		print("[EVAL] loss:{0}".format(util.red("{:.4e}".format(loss_eval.all))))
		if ep is not None:
			self.log_losses(opt,ep,loss_eval,training=False)
			self.visualize(opt,ep,var,training=False)

	def show_progress(self,opt,ep,loss):
		[lr] = self.sched.get_lr()
		time_elapsed = util.get_time(time.time()-self.time_start)
		print("ep {0}/{1}, lr:{3}, loss:{4}, time:{2}"
			.format(util.cyan("{}".format(ep+1)),
					opt.to_epoch,
					util.green("{0}:{1:02d}:{2:05.2f}".format(*time_elapsed)),
					util.yellow("{:.2e}".format(lr)),
					util.red("{:.4e}".format(loss.all)),
		))

	def log_losses(self,opt,ep,loss,training=True):
		group = "train" if training else "eval"
		self.tb.add_scalar("{}/loss".format(group),loss.all,ep+1)

	def visualize(self,opt,ep,var,training=True):
		group = "train" if training else "eval"
		self.tb.add_image("{}/input".format(group),util.make_tb_image(opt,var.image),ep+1)
		self.vis.scatter(var.points_GT[0],env="{0}/{1}".format(opt.group,opt.name),win="{0}/GT".format(group),
										  opts={"markersize":2,"title":"{0} (GT), ep {1}".format(group,ep+1)})
		self.vis.scatter(var.points_pred[0],env="{0}/{1}".format(opt.group,opt.name),win="{0}/pred".format(group),
											opts={"markersize":2,"title":"{0} (pred), ep {1}".format(group,ep+1)})

	def save_checkpoint(self,opt,ep):
		util.save_checkpoint(opt,self,["network","optim","sched"],ep+1)
