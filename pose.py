import numpy as np
import torch
import easydict

def params_to_mtrx(opt,sim3):
	R = get_lie_rotation_matrix(opt,sim3.rot)
	mtrx = torch.cat([sim3.scale.exp()*R,sim3.trans[:,None]],dim=1)
	return mtrx

def apply_3Dsim(opt,points,sim3_mtrx,inv=False):
	sR,t = sim3_mtrx[:,:3],sim3_mtrx[:,3]
	points = (points-t)@sR.inverse().t() if inv else \
			 points@sR.t()+t
	return points

def get_lie_rotation_matrix(opt,r):
	O = torch.tensor(0.0,device=opt.device,dtype=torch.float32)
	rx = torch.stack([torch.stack([O,-r[2],r[1]]),
					  torch.stack([r[2],O,-r[0]]),
					  torch.stack([-r[1],r[0],O])],dim=0)
	# matrix exponential
	R = torch.eye(3,device=opt.device,dtype=torch.float32)
	numer = torch.eye(3,device=opt.device,dtype=torch.float32)
	denom = 1.0
	for i in range(1,20):
		numer = numer.matmul(rx)
		denom *= i
		R += numer/denom
	return R

def add_noise(opt,cs_map_mtrx):
	noise = easydict.EasyDict({
		"scale": opt.noise*torch.randn(1,device=opt.device)[0],
		"rot": opt.noise*torch.randn(3,device=opt.device),
		"trans": opt.noise*torch.randn(3,device=opt.device),
	})
	noise_mtrx = params_to_mtrx(opt,noise)
	hom = torch.cat([torch.zeros(1,3,device=opt.device),
					 torch.ones(1,1,device=opt.device)],dim=1)
	cs_map_mtrx_hom = torch.cat([cs_map_mtrx,hom],dim=0)
	noisy_cs_map_mtrx = noise_mtrx@cs_map_mtrx_hom
	print("noise -- scale: {0:.4f}, rot: [{1:.4f},{2:.4f},{3:.4f}], trans: [{4:.4f},{5:.4f},{6:.4f}]"
		.format(noise.scale,noise.rot[0],noise.rot[1],noise.rot[2],noise.trans[0],noise.trans[1],noise.trans[2]))
	return noisy_cs_map_mtrx

def interpolate_camera(extr1,extr2,alpha=0.5): # [B,3,4]
	R1,t1 = extr1[...,:3],extr1[...,3:]
	R2,t2 = extr2[...,:3],extr2[...,3:]
	q1 = rotation_matrix_to_quaternion(R1)
	q2 = rotation_matrix_to_quaternion(R2)
	q_intp = interpolate_quaternion(q1,q2,alpha)
	R_intp = quaternion_to_rotation_matrix(q_intp)
	t_intp = (1-alpha)*t1+alpha*t2
	extr_intp = torch.cat([R_intp,t_intp],dim=-1)
	return extr_intp

def interpolate_quaternion(q1,q2,alpha=0.5):
	cos_angle = (q1*q2).sum(dim=-1)
	flip = cos_angle<0
	q1[flip] *= -1
	cos_angle[flip] *= -1
	theta = cos_angle.acos()[...,None]
	slerp = (((1-alpha)*theta).sin()*q1+(alpha*theta).sin()*q2)/theta.sin()
	return slerp

def rotation_matrix_to_quaternion(R): # [B,3,3]
	row0,row1,row2 = torch.unbind(R,dim=-2)
	R00,R01,R02 = torch.unbind(row0,dim=-1)
	R10,R11,R12 = torch.unbind(row1,dim=-1)
	R20,R21,R22 = torch.unbind(row2,dim=-1)
	t = R[...,0,0]+R[...,1,1]+R[...,2,2]
	r = (1+t).sqrt()
	qa = 0.5*r
	qb = (R21-R12).sign()*0.5*(1+R00-R11-R22).sqrt()
	qc = (R02-R20).sign()*0.5*(1-R00+R11-R22).sqrt()
	qd = (R10-R01).sign()*0.5*(1-R00-R11+R22).sqrt()
	q = torch.stack([qa,qb,qc,qd],dim=-1)
	for i,qi in enumerate(q):
		if torch.isnan(qi).any():
			print(i)
			K = torch.stack([torch.stack([R00-R11-R22,R10+R01,R20+R02,R12-R21],dim=-1),
							 torch.stack([R10+R01,R11-R00-R22,R21+R12,R20-R20],dim=-1),
							 torch.stack([R20+R02,R21+R12,R22-R00-R11,R01-R10],dim=-1),
							 torch.stack([R12-R21,R20-R02,R01-R10,R00+R11+R22],dim=-1)],dim=-2)/3.0
			K = K[i]
			eigval,eigvec = K.eig(eigenvectors=True)
			idx = eigval[:,0].argmax()
			V = eigvec[:,idx]
			q[i] = torch.stack([V[3],V[0],V[1],V[2]])
	return q

def quaternion_to_rotation_matrix(q): # [B,4]
	qa,qb,qc,qd = torch.unbind(q,dim=-1)
	R = torch.stack([torch.stack([1-2*(qc**2+qd**2),2*(qb*qc-qa*qd),2*(qa*qc+qb*qd)],dim=-1),
					 torch.stack([2*(qb*qc+qa*qd),1-2*(qb**2+qd**2),2*(qc*qd-qa*qb)],dim=-1),
					 torch.stack([2*(qb*qd-qa*qc),2*(qa*qb+qc*qd),1-2*(qb**2+qc**2)],dim=-1)],dim=-2)
	return R
