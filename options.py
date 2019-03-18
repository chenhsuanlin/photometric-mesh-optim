import numpy as np
import argparse
import util
import torch

def set():

	# parse input arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--eval",			action="store_true",			help="evaluation phase")
	parser.add_argument("--group",						default="0",		help="name for group")
	parser.add_argument("--name",						default="debug",	help="name for model instance")
	parser.add_argument("--seed",			type=int,	default=0,			help="fix random seed")
	parser.add_argument("--gpu",			type=int,	default=0,			help="GPU device")
	parser.add_argument("--cpu",			action="store_true",			help="use CPU only")
	parser.add_argument("--load",						default=None,		help="load (pre)trained model")
	# dataset
	parser.add_argument("--rendering-path",	default="data/rendering",		help="path to ShapeNet rendering")
	parser.add_argument("--pointcloud-path",default="data/customShapeNet",	help="path to ShapeNet 3D point cloud")
	parser.add_argument("--sun360-path",	default="data/background",		help="path to SUN360 background")
	parser.add_argument("--seq-path",		default="data/sequences",		help="path to RGB sequences for evaluation")
	parser.add_argument("--category",					default=None,		help="train on specific category")
	parser.add_argument("--num-workers",	type=int,	default=8,			help="number of data loading threads")
	parser.add_argument("--size",						default="224x224",	help="rendered image size")
	parser.add_argument("--sfm",			action="store_true",			help="use coordinate system mapping from SfM output")
	parser.add_argument("--init-idx",		type=int,	default=27,			help="initial frame index")
	parser.add_argument("--noise",			type=float,	default=None,		help="gaussian noise in coordinate system mapping")
	# visualization
	parser.add_argument("--log-tb",			action="store_true",			help="output loss in TensorBoard")
	parser.add_argument("--log-visdom",		action="store_true",			help="visualize mesh in Visdom")
	parser.add_argument("--vis-server",		default="http://localhost",		help="visdom port server")
	parser.add_argument("--vis-port",		type=int,	default=8097,		help="visdom port number")
	parser.add_argument("--video",			action="store_true",			help="write video sequence with optimized mesh")
	# AtlasNet
	parser.add_argument("--num-prim",		type=int,	default=25,			help="number of primitives")
	parser.add_argument("--num-points",		type=int,	default=100,		help="number of points (per primitive)")
	parser.add_argument("--num-meshgrid",	type=int,	default=5,			help="number of regular grids for mesh")
	parser.add_argument("--sphere",			action="store_true",			help="use closed sphere for AtlasNet")
	parser.add_argument("--sphere-densify",	type=int,	default=3,			help="densify levels")	
	parser.add_argument("--imagenet-enc",	action="store_true",			help="initialize with pretrained ResNet encoder")
	parser.add_argument("--pretrained-dec",				default=None,		help="initialize with pretrained AtlasNet decoder")
	# photometric optimization
	parser.add_argument("--batch-size-pmo",	type=int,	default=-1,			help="batch size for photometric optimization (-1 for all)")
	parser.add_argument("--lr-pmo",			type=float,	default=1e-3,		help="base learning rate for photometric optimization")
	parser.add_argument("--code",			type=float,	default=None,		help="penalty on code difference")
	parser.add_argument("--scale",			type=float,	default=None,		help="penalty on scale")
	parser.add_argument("--to-it",			type=int,	default=100,		help="run optimization to iteration number")
	parser.add_argument("--avg-frame",		action="store_true",			help="average photo. loss across frames instead of sampled pixels")
	# AtlasNet training
	parser.add_argument("--batch-size",		type=int,	default=32,			help="input batch size")
	parser.add_argument("--aug-transl",		type=int,	default=None,		help="augment with random translation (for new dataset)")
	parser.add_argument("--lr-pretrain",	type=float,	default=1e-4,		help="base learning rate")
	parser.add_argument("--lr-decay",		type=float,	default=1.0,		help="learning rate decay")
	parser.add_argument("--lr-step",		type=int,	default=100,		help="learning rate decay step size")
	parser.add_argument("--from-epoch",		type=int,	default=0,			help="train from epoch number")
	parser.add_argument("--to-epoch",		type=int,	default=500,		help="train to epoch number")
	opt = parser.parse_args()

	# --- below are automatically set ---
	if opt.seed is not None:
		np.random.seed(opt.seed)
		torch.manual_seed(opt.seed)
		torch.cuda.manual_seed_all(opt.seed)
		opt.name += "_seed{}".format(opt.seed)
	opt.device = "cpu" if opt.cpu or not torch.cuda.is_available() else "cuda:{}".format(opt.gpu)
	opt.H,opt.W = [int(s) for s in opt.size.split("x")]

	if opt.sphere:
		opt.num_prim = 1
	opt.num_points_all = opt.num_points*opt.num_prim
	
	# print configurations
	for o in sorted(vars(opt)):
		print(util.green(o),":",util.yellow(getattr(opt,o)))
	print()

	return opt
