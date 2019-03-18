import numpy as np
import os,sys,time
import torch
import options,data,util
import model

print(util.yellow("======================================================="))
print(util.yellow("main.py (photometric mesh optimization)"))
print(util.yellow("======================================================="))

print(util.magenta("setting configurations..."))
opt = options.set()

print(util.magenta("reading list of sequences..."))
seq_list = data.load_sequence_list(opt,subset=1)
seq_list = [("02958343","eebbce8b77bdb53c82382fde2cafeb9")]

with torch.cuda.device(opt.gpu):

	pmo = model.Model(opt)
	pmo.build_network(opt)
	pmo.restore_checkpoint(opt)

	print(util.yellow("======= OPTIMIZATION START ======="))
	for c,m in seq_list:
		pmo.load_sequence(opt,c,m)
		pmo.setup_visualizer(opt)
		pmo.setup_variables(opt)
		pmo.setup_optimizer(opt)
		pmo.time_start(opt)
		pmo.optimize(opt)
	print(util.yellow("======= OPTIMIZATION DONE ======="))
	pmo.write_video(opt)
	pmo.save_mesh(opt)
