import argparse
import os
import time
import pickle

import torch
import losses
from pytorch3d.utils import ico_sphere
from r2n2_custom import R2N2
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
import dataset_location



def get_args_parser_fit(parents = []):
    parser = argparse.ArgumentParser('Model Fit', add_help=False, parents = parents)
    parser.add_argument('--lr', default=4e-4, type=float)
    parser.add_argument('--max_iter', default=100000, type=int)
    parser.add_argument('--log_freq', default=1000, type=int)
    parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh'], type=str)
    parser.add_argument('--n_points', default=5000, type=int)
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=0.1, type=float)
    parser.add_argument('--save', default='image', type=str)
    return parser

def fit_mesh(mesh_src, mesh_tgt, args):
    start_iter = 0
    start_time = time.time()

    deform_vertices_src = torch.zeros(mesh_src.verts_packed().shape,requires_grad=True,device='cuda')
    optimizer = torch.optim.Adam([deform_vertices_src], lr = args.lr)
    print("Starting training !")
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        new_mesh_src = mesh_src.offset_verts(deform_vertices_src)

        sample_trg = sample_points_from_meshes(mesh_tgt, args.n_points)
        sample_src = sample_points_from_meshes(new_mesh_src, args.n_points)

        loss_reg = losses.chamfer_loss(sample_src, sample_trg)
        loss_smooth = losses.smoothness_loss(new_mesh_src)

        loss = args.w_chamfer * loss_reg + args.w_smooth * loss_smooth

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time,  iter_time, loss_vis))        
    
    mesh_src.offset_verts_(deform_vertices_src)

    print('Done!')


def fit_pointcloud(pointclouds_src, pointclouds_tgt, args):
    start_iter = 0
    start_time = time.time()    
    optimizer = torch.optim.Adam([pointclouds_src], lr = args.lr)
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        loss = losses.chamfer_loss(pointclouds_src, pointclouds_tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time,  iter_time, loss_vis))
    
    print('Done!')


def fit_voxel(voxels_src, voxels_tgt, args):
    start_iter = 0
    start_time = time.time()    
    optimizer = torch.optim.Adam([voxels_src], lr = args.lr)
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        loss = losses.voxel_loss(voxels_src,voxels_tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time,  iter_time, loss_vis))
    
    print('Done!')


def fit_model(args):
    r2n2_dataset = R2N2("train", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=True)

    
    feed = r2n2_dataset[0]
    

    feed_cuda = {}
    for k in feed:
        if torch.is_tensor(feed[k]):
            feed_cuda[k] = feed[k].to(args.device).float()


    if args.type == "vox":
        # initialization
        voxels_src = torch.rand(feed_cuda['voxels'].shape,requires_grad=True,device='cuda')
        voxel_coords = feed_cuda['voxel_coords'].unsqueeze(0)
        voxels_tgt = feed_cuda['voxels']

        # fitting
        fit_voxel(voxels_src, voxels_tgt, args)
        out_model = voxels_src.cpu().detach().numpy()


    elif args.type == "point":
        # initialization
        pointclouds_src = torch.randn([1,args.n_points,3],requires_grad=True,device='cuda')
        mesh_tgt = Meshes(verts=[feed_cuda['verts']], faces=[feed_cuda['faces']])
        pointclouds_tgt = sample_points_from_meshes(mesh_tgt, args.n_points)

        # fitting
        fit_pointcloud(pointclouds_src, pointclouds_tgt, args)
        out_model = pointclouds_src.cpu().detach().numpy()
    
    elif args.type == "mesh":
        # initialization
        # try different ways of initializing the source mesh        
        mesh_src = ico_sphere(4,'cuda')
        mesh_tgt = Meshes(verts=[feed_cuda['verts']], faces=[feed_cuda['faces']])

        # fitting
        fit_mesh(mesh_src, mesh_tgt, args)
        out_model = mesh_src.cpu().detach()

    if args.save:
        with open(os.path.join(args.save, args.type) + ".pkl", 'wb') as file:
            pickle.dump(out_model, file)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Model Fit', parents=[get_args_parser_fit()])
    args = parser.parse_args()
    fit_model(args)



# ## Visualization 
# from visualization import visualize_mesh_360

# with open("image/mesh.pkl", 'rb') as file:
#     predict_mesh = pickle.load(file)

# color=[0.7, 0.7, 1]
# textures = torch.ones_like(predict_mesh.verts_list()[0]).unsqueeze(0)  # (1, N_v, 3)
# textures = textures * torch.tensor(color)  # (1, N_v, 3)

# predict_mesh.textures = pytorch3d.renderer.TexturesVertex(textures)
# visualize_mesh_360(predict_mesh.cuda(), save = './test.gif')

# textures = torch.ones_like(feed_cuda['verts']).unsqueeze(0).cpu() # (1, N_v, 3)
# textures = textures * torch.tensor(color)  # (1, N_v, 3)
# textures = pytorch3d.renderer.TexturesVertex(textures.to(device))

# mesh_tgt = pytorch3d.structures.Meshes(verts=[feed_cuda['verts']], faces=[feed_cuda['faces']], textures=textures).to(device)
# visualize_mesh_360(mesh_tgt, save = './mesh_gt.gif')




# # VISUALIZE VOX
# from visualization import visualize_voxel_360
# with open("image/vox.pkl", 'rb') as file:
#     predict_voxel = pickle.load(file)

# predict_voxels = torch.sigmoid(torch.tensor(predict_voxel[0]))
# visualize_voxel_360(predict_voxels.numpy(), save = './test.gif')



# from visualization import visualize_point
# with open("image/point.pkl", 'rb') as file:
#     predict_point = pickle.load(file)

# verts = torch.Tensor(predict_point)
# rgb = (verts - verts.min()) / (verts.max() - verts.min())
# point_cloud = pytorch3d.structures.Pointclouds(points=verts, features=rgb)

# visualize_point(point_cloud, save = "test.gif")