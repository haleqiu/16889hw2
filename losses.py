import torch
import torch.nn as nn
from pytorch3d.loss import chamfer_distance, mesh_laplacian_smoothing

# define losses
# (B, C)
BCEloss = nn.BCELoss()
sigmoid = nn.Sigmoid()

def voxel_loss(voxel_src,voxel_tgt):
	voxel_sigmoid = sigmoid(voxel_src.reshape(-1,1))
	prob_loss = BCEloss(voxel_sigmoid, voxel_tgt.reshape(-1,1))
	# implement some loss for binary voxel grids
	return prob_loss

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	loss_chamfer, _ = chamfer_distance(point_cloud_src,point_cloud_tgt)
	# implement chamfer loss from scratch
	return loss_chamfer

def smoothness_loss(mesh_src):
	loss = mesh_laplacian_smoothing(mesh_src)
	# implement laplacian smoothening loss
	return loss