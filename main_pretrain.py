import numpy as np
import os,sys,time
import torch
import options,util
import model_pretrain

print(util.yellow("======================================================="))
print(util.yellow("main_pretrain.py (pretraining with AtlasNet reimplementation)"))
print(util.yellow("======================================================="))

print(util.magenta("setting configurations..."))
opt = options.set()

with torch.cuda.device(opt.gpu):

	trainer = model_pretrain.Model(opt)
	trainer.load_dataset(opt)
	trainer.build_network(opt)
	trainer.setup_optimizer(opt)
	trainer.restore_checkpoint(opt)
	trainer.setup_visualizer(opt)

	print(util.yellow("======= TRAINING START ======="))
	trainer.time_start(opt)
	for ep in range(opt.from_epoch,opt.to_epoch):
		trainer.train_epoch(opt,ep)
		if (ep+1)%10==0: trainer.evaluate(opt,ep)
		if (ep+1)%50==0: trainer.save_checkpoint(opt,ep)
	print(util.yellow("======= TRAINING DONE ======="))
