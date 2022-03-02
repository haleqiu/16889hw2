import argparse
import time, os, tqdm
import torch
from model import SingleViewto3D
from r2n2_custom import R2N2
import pytorch3d
from  pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
import dataset_location
from pytorch3d.ops import sample_points_from_meshes
import losses
from torch.optim.lr_scheduler import ReduceLROnPlateau
from visualization import visualize_voxel_360, visualize_point

def get_args_parser(parents = []):
    parser = argparse.ArgumentParser('Singleto3D', add_help=False, parents=parents)
    # Model parameters
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--max_iter', default=10000, type=str)
    parser.add_argument('--log_freq', default=1000, type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_workers', default=0, type=str)
    parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh'], type=str)
    parser.add_argument('--n_points', default=4096, type=int)
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=0.1, type=float)
    parser.add_argument('--save_freq', default=200, type=int)  
    parser.add_argument('--device', default='cuda:0', type = str)
    parser.add_argument('--surfix', default='', type = str)
    parser.add_argument('--load_checkpoint', action='store_true')         
    parser.add_argument('--scheduler', action='store_true')
    parser.add_argument('--vis_training', action='store_true')
    parser.add_argument('--save', default='', type = str)
    return parser

def preprocess(feed_dict,args):
    for k in ['images','voxels','mesh']:
        feed_dict[k] = feed_dict[k].to(args.device)
    images = feed_dict['images'].squeeze(1)
    if args.type == "vox":
        voxels = feed_dict['voxels'].float()
        ground_truth_3d = voxels
    elif args.type == "point":
        # mesh = feed_dict['mesh']
        # faces = [k.to(args.device) for k in feed_dict['faces']]
        mesh = pytorch3d.structures.Meshes(verts=feed_dict['verts'], faces=feed_dict['faces'])
        pointclouds_tgt = sample_points_from_meshes(mesh, args.n_points)
        ground_truth_3d = pointclouds_tgt.to(args.device)
    elif args.type == "mesh":
        ground_truth_3d = feed_dict["mesh"]
    return images, ground_truth_3d


def calculate_loss(predictions, ground_truth, args):
    if args.type == 'vox':
        loss = losses.voxel_loss(predictions,ground_truth)
    elif args.type == 'point':
        loss = losses.chamfer_loss(predictions, ground_truth)
    elif args.type == 'mesh':
        sample_trg = sample_points_from_meshes(ground_truth, args.n_points)
        sample_pred = sample_points_from_meshes(predictions, args.n_points)

        loss_reg = losses.chamfer_loss(sample_pred, sample_trg)
        loss_smooth = losses.smoothness_loss(predictions)

        loss = args.w_chamfer * loss_reg + args.w_smooth * loss_smooth        
    return loss


def train_model(args):
    r2n2_dataset = R2N2("train", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=True)

    loader = torch.utils.data.DataLoader(
        r2n2_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_batched_R2N2,
        pin_memory=True,
        drop_last=True)
    train_loader = iter(loader)

    model =  SingleViewto3D(args)
    model.to(args.device)
    model.train()

    # ============ preparing optimizer ... ============
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)  # to use with ViTs
    scheduler = ReduceLROnPlateau(optimizer, factor=0.2, mode = 'min', patience=5, min_lr= 1e-4)
    start_iter = 0
    start_time = time.time()

    if args.load_checkpoint:
        if os.path.exists(f'checkpoint_{args.surfix}{args.type}.pth'):
            checkpoint = torch.load(f'checkpoint_{args.surfix}{args.type}.pth')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_iter = checkpoint['step']
            print(f"Succesfully loaded iter {start_iter}")
    
    if args.vis_training:
        sample_dict = next(train_loader)

    print("Starting training !", len(train_loader))
    running_loss = 0
    for step in tqdm.tqdm(range(start_iter, args.max_iter)):
        iter_start_time = time.time()

        if step % len(train_loader) == 0: #restart after one epoch
            epho_loss = running_loss/len(train_loader)
            if args.scheduler:
                scheduler.step(epho_loss)
            print("one epoch, loss", epho_loss)
            train_loader = iter(loader)
            running_loss = 0

        read_start_time = time.time()

        feed_dict = next(train_loader)

        images_gt, ground_truth_3d = preprocess(feed_dict,args)
        read_time = time.time() - read_start_time

        prediction_3d = model(images_gt, args)

        loss = calculate_loss(prediction_3d, ground_truth_3d, args)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()
        running_loss += loss_vis

        if (step % args.save_freq) == 0:
            print("save " + f'checkpoint_{args.surfix}{args.type}.pth')
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, f'checkpoint_{args.surfix}{args.type}.pth')

            if args.vis_training:
                images_gt, ground_truth_3d = preprocess(sample_dict,args)
                predictions = model(images_gt, args)
                if args.type == "point":
                    verts = predictions.detach()[0].unsqueeze(0)
                    rgb = (verts - verts.min()) / (verts.max() - verts.min())
                    point_cloud = pytorch3d.structures.Pointclouds(points=verts, features=rgb)
                    visualize_point(point_cloud, save = '%s/%i_%s.gif'%(args.save, step, args.type))
        # print("[%4d/%4d]; ttime: %.0f (%.2f, %.2f); loss: %.3f" % (step, args.max_iter, total_time, read_time, iter_time, loss_vis))

    print('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Singleto3D', parents=[get_args_parser()])
    args = parser.parse_args()
    train_model(args)
