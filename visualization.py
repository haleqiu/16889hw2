import pickle
import matplotlib.pyplot as plt
import mcubes
import numpy as np
import pytorch3d
import torch
import imageio

from utils import get_device, get_mesh_renderer, get_volums_renderer, get_points_renderer

from pytorch3d.structures import Volumes, Meshes
from pytorch3d.ops import sample_points_from_meshes


# input should be numpy
def _render_voxel(voxels, device, voxel_size=64):
    min_value = -1.1
    max_value = 1.1

    vertices, faces = mcubes.marching_cubes(voxels, isovalue=0.5)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))

    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
    device
)
    return mesh


# input should be numpy
def visualize_voxel_360(voxels,image_size = 256, save = 'image/voxel_predict.gif', device = None, steps = 36):
    if not device:
        device = get_device()

    mesh = _render_voxel(voxels, device)
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3] for i in range(steps)], device=device)
    renderer = get_mesh_renderer(image_size=image_size, device=device)

    Rs = []; Ts = []
    for angle in np.linspace(0,360,steps):
        R, T = pytorch3d.renderer.look_at_view_transform(3, 0, angle)
        Rs.append(R); Ts.append(T)

    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=torch.cat(Rs), T=torch.cat(Ts), fov=60, device=device
    )

    meshes = mesh.extend(steps)

    rend = renderer(meshes, cameras=cameras, lights=lights)
    images = rend.cpu().numpy()[:, ..., :3]
    imageio.mimsave(save, images, fps=15)


def visualize_voxel(voxels,image_size = 256, save = 'image/voxel_predict.gif', device = None, steps = 36):
    if not device:
        device = get_device()

    mesh = _render_voxel(voxels, device)
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
    renderer = get_mesh_renderer(image_size=image_size, device=device)

    Rs = []; Ts = []
    R, T = pytorch3d.renderer.look_at_view_transform(3, 0, 0)
    Rs.append(R); Ts.append(T)

    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=torch.cat(Rs), T=torch.cat(Ts), fov=60, device=device
    )


    rend = renderer(mesh, cameras=cameras, lights=lights)
    image = rend.cpu().numpy()[:, ..., :3].clip(0, 1)[0]
    plt.imshow(image)
    plt.savefig(save)


def visualize_mesh_360(mesh,image_size = 256, save = 'image/mesh_predict.gif', device = None, steps = 36):
    if not device:
        device = get_device()

    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3] for i in range(steps)], device=device)
    renderer = get_mesh_renderer(image_size=image_size, device=device)

    Rs = []; Ts = []
    for angle in np.linspace(0,360,steps):
        R, T = pytorch3d.renderer.look_at_view_transform(3, 0, angle)
        Rs.append(R); Ts.append(T)

    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=torch.cat(Rs), T=torch.cat(Ts), fov=60, device=device
    )

    meshes = mesh.extend(steps)

    rend = renderer(meshes, cameras=cameras, lights=lights)
    images = rend.cpu().numpy()[:, ..., :3]
    imageio.mimsave(save, images, fps=15)


def visualize_point(
    point, image_size=256, device=None, steps = 36, save = 'image/point_predict.gif'):

    if device is None:
        device = get_device()
    renderer = get_points_renderer(
        image_size=image_size, background_color=(1, 1, 1)
    )

    point = point.to(device)

    Rs = []; Ts = []
    for angle in np.linspace(0,360,steps):
        R, T = pytorch3d.renderer.look_at_view_transform(3, 0, angle)
        Rs.append(R); Ts.append(T)

    # Prepare the camera:
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=torch.cat(Rs), T=torch.cat(Ts), fov=60, device=device
    )
    # Place a point light in front of the cow.
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3] for i in range(steps)], device=device)
    points = point.extend(steps)

    rend = renderer(points, cameras=cameras, lights=lights)
    # The .cpu moves the tensor to GPU (if needed).

    images = rend.cpu().numpy()[:, ..., :3]  # (B, H, W, 4) -> (H, W, 3)

    imageio.mimsave(save, images, fps=15)


def reder_mesh(
    mesh, image_size=256, device=None, steps = 36, save = 'image/mesh_predict.gif'):

    if device is None:
        device = get_device()

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    Rs = []; Ts = []
    for angle in np.linspace(0,360,steps):
        R, T = pytorch3d.renderer.look_at_view_transform(3, 0, angle)
        Rs.append(R); Ts.append(T)

    # Prepare the camera:
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=torch.cat(Rs), T=torch.cat(Ts), fov=60, device=device
    )
    # Place a point light in front of the cow.
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3] for i in range(steps)], device=device)
    meshes = mesh.extend(steps)

    rend = renderer(meshes, cameras=cameras, lights=lights)
    # The .cpu moves the tensor to GPU (if needed).

    images = rend.detach().cpu().numpy()[:, ..., :3]  # (B, H, W, 4) -> (H, W, 3)

    imageio.mimsave(save, images, fps=15)