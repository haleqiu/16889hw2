import argparse
import time
import torch
from model import SingleViewto3D
from r2n2_custom import R2N2
from  pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
import dataset_location
import pytorch3d
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.ops import knn_points
import mcubes
import utils_vox
import torch.nn as nn
import pickle, os
import numpy as np
import matplotlib.pyplot as plt

from train_model import calculate_loss
sigmoid = nn.Sigmoid()


from visualization import visualize_voxel, visualize_voxel_360, visualize_mesh_360, visualize_point, reder_mesh

def get_args_parser():
    parser = argparse.ArgumentParser('Singleto3D', add_help=False)
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--max_iter', default=10000, type=str)
    parser.add_argument('--vis_freq', default=100, type=int)
    parser.add_argument('--batch_size', default=1, type=str)
    parser.add_argument('--num_workers', default=0, type=str)
    parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh'], type=str)
    parser.add_argument('--n_points', default=4096, type=int)
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=0.1, type=float) 
    parser.add_argument('--surfix', default='', type = str) 
    parser.add_argument('--save_image', action='store_true')  
    parser.add_argument('--save', default='vis')
    parser.add_argument('--device', default='cuda:0', type=str)

    return parser

def preprocess(feed_dict, device = 'cuda:0'):
    for k in ['images']:
        feed_dict[k] = feed_dict[k].to(device)

    images = feed_dict['images'].squeeze(1)
    mesh = feed_dict['mesh']

    return images, mesh

def compute_sampling_metrics(pred_points, gt_points, thresholds= [0.01, 0.02, 0.03, 0.04, 0.05], eps=1e-8):
    metrics = {}
    lengths_pred = torch.full(
        (pred_points.shape[0],), pred_points.shape[1], dtype=torch.int64, device=pred_points.device
    )
    lengths_gt = torch.full(
        (gt_points.shape[0],), gt_points.shape[1], dtype=torch.int64, device=gt_points.device
    )

    # For each predicted point, find its neareast-neighbor GT point
    knn_pred = knn_points(pred_points, gt_points, lengths1=lengths_pred, lengths2=lengths_gt, K=1)
    # Compute L1 and L2 distances between each pred point and its nearest GT
    pred_to_gt_dists2 = knn_pred.dists[..., 0]  # (N, S)
    pred_to_gt_dists = pred_to_gt_dists2.sqrt()  # (N, S)

    # For each GT point, find its nearest-neighbor predicted point
    knn_gt = knn_points(gt_points, pred_points, lengths1=lengths_gt, lengths2=lengths_pred, K=1)
    # Compute L1 and L2 dists between each GT point and its nearest pred point
    gt_to_pred_dists2 = knn_gt.dists[..., 0]  # (N, S)
    gt_to_pred_dists = gt_to_pred_dists2.sqrt()  # (N, S)

    # Compute precision, recall, and F1 based on L2 distances
    for t in thresholds:
        precision = 100.0 * (pred_to_gt_dists < t).float().mean(dim=1)
        recall = 100.0 * (gt_to_pred_dists < t).float().mean(dim=1)
        f1 = (2.0 * precision * recall) / (precision + recall + eps)
        metrics["Precision@%f" % t] = precision
        metrics["Recall@%f" % t] = recall
        metrics["F1@%f" % t] = f1

    # Move all metrics to CPU
    metrics = {k: v.cpu() for k, v in metrics.items()}
    return metrics

def evaluate(predictions, mesh_gt, args):
    if args.type == "vox":
        voxels_src = predictions
        H,W,D = voxels_src.shape[2:]
        vertices_src, faces_src = mcubes.marching_cubes(voxels_src.detach().cpu().squeeze().numpy(), isovalue=0.5)
        vertices_src = torch.tensor(vertices_src).float()
        faces_src = torch.tensor(faces_src.astype(int))
        mesh_src = pytorch3d.structures.Meshes([vertices_src], [faces_src])
        pred_points = sample_points_from_meshes(mesh_src, args.n_points)
        pred_points = utils_vox.Mem2Ref(pred_points, H, W, D)

    elif args.type == "point":
        pred_points = predictions.cpu()
    elif args.type == "mesh":
        pred_points = sample_points_from_meshes(predictions, args.n_points).cpu()

    gt_points = sample_points_from_meshes(mesh_gt, args.n_points)
        
    metrics = compute_sampling_metrics(pred_points, gt_points)
    return metrics



def evaluate_model(args):
    r2n2_dataset = R2N2("test", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=True)
    # r2n2_dataset = R2N2("train", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=True)

    loader = torch.utils.data.DataLoader(
        r2n2_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_batched_R2N2,
        pin_memory=True,
        drop_last=True)
    eval_loader = iter(loader)

    model =  SingleViewto3D(args)
    model.to(args.device)
    model.eval()

    start_iter = 0
    start_time = time.time()

    avg_f1_score = []

    checkpoint = torch.load(f'checkpoint_{args.surfix}{args.type}.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Succesfully loaded iter {start_iter}")
    
    print("Starting evaluating !")
    max_iter = len(eval_loader)
    for step in range(start_iter, max_iter):
        iter_start_time = time.time()

        read_start_time = time.time()

        feed_dict = next(eval_loader)

        images_gt, mesh_gt = preprocess(feed_dict, args.device)

        read_time = time.time() - read_start_time

        predictions = model(images_gt, args)

        if args.type == "vox":
            loss = calculate_loss(predictions, feed_dict['voxels'].float().to(args.device), args)
            print("loss", loss)
            predictions = sigmoid(predictions)
            predictions = predictions.permute(0,1,4,3,2)

        metrics = evaluate(predictions, mesh_gt, args)

        # TODO:
        if (step % args.vis_freq) == 0:
            # visualization block
            if args.type == "vox":
                vox = predictions[0][0].detach().cpu().numpy()
                visualize_voxel_360(vox, save = '%s/%i_%s.gif'%(args.save, step, args.type))
                visualize_voxel_360(feed_dict['voxels'][0][0].detach().cpu().numpy(), save = '%s/gt%i_%s.gif'%(args.save, step, args.type))
            if args.type == "point":
                verts = predictions.detach()
                rgb = (verts - verts.min()) / (verts.max() - verts.min())
                point_cloud = pytorch3d.structures.Pointclouds(points=verts, features=rgb)
                visualize_point(point_cloud, save = '%s/%i_%s.gif'%(args.save, step, args.type))

                mesh_tgt = pytorch3d.structures.Meshes(verts=feed_dict['verts'], faces=feed_dict['faces'])
                pointclouds_tgt = sample_points_from_meshes(mesh_tgt, args.n_points)
                print("pointclouds_tgt.max()", pointclouds_tgt.max())
                print("pointclouds_tgt.min()", pointclouds_tgt.min())
                rgb = (pointclouds_tgt - pointclouds_tgt.min()) / (pointclouds_tgt.max() - pointclouds_tgt.min())
                point_cloud = pytorch3d.structures.Pointclouds(points=pointclouds_tgt, features=rgb)
                visualize_point(point_cloud, save = '%s/gt%i_%s.gif'%(args.save, step, args.type))
            
            if args.type == "mesh":
                mesh = predictions
                textures = torch.ones_like(mesh.verts_list()[0]).unsqueeze(0)# (1, N_v, 3)

                mesh.textures = pytorch3d.renderer.TexturesVertex(textures)
                mesh = mesh.to(args.device)
                reder_mesh(mesh, save = '%s/%i_%s.gif'%(args.save, step, args.type))

                textures = torch.ones_like(feed_dict['verts'][0]).unsqueeze(0)# (1, N_v, 3)
                mesh_tgt = pytorch3d.structures.Meshes(verts=feed_dict['verts'], faces=feed_dict['faces'], textures=pytorch3d.renderer.TexturesVertex(textures)).to(args.device)
                reder_mesh(mesh_tgt, save = '%s/gt%i_%s.gif'%(args.save, step, args.type))
            
            if args.save_image:
                plt.imshow(images_gt[0].detach().cpu().numpy())
                plt.savefig('%s/image%i_%s.png'%(args.save, step, args.type), bbox_inches='tight',transparent=True, pad_inches=0)

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        f1_05 = metrics['F1@0.050000']

        avg_f1_score.append(f1_05.detach().cpu().numpy())

        print("[%4d/%4d]; ttime: %.0f (%.2f, %.2f); F1@0.05: %.3f; Avg F1@0.05: %.3f" % (step, max_iter, total_time, read_time, iter_time, f1_05, torch.tensor(avg_f1_score).mean()))

    avg_f1_score = np.array(avg_f1_score)
    print("average_f1_score: ", avg_f1_score.mean())
    with open(os.path.join(args.save, args.surfix + args.type + "avg_f1_score.pkl"), 'wb') as file:
        pickle.dump(avg_f1_score, file)
    print('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Singleto3D', parents=[get_args_parser()])
    args = parser.parse_args()
    evaluate_model(args)
