import os
import numpy as np
import torch
import pytorch3d
from pytorch3d.renderer import (
    AlphaCompositor,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
)
import imageio
from scipy.spatial.transform import Rotation as R

def save_checkpoint(epoch, model, args, best=False):
    if best:
        path = os.path.join(args.checkpoint_dir, 'best_model.pt')
    else:
        path = os.path.join(args.checkpoint_dir, 'model_epoch_{}.pt'.format(epoch))
    torch.save(model.state_dict(), path)

def create_dir(directory):
    """
    Creates a directory if it does not already exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_points_renderer(
    image_size=256, device=None, radius=0.01, background_color=(1, 1, 1)
):
    """
    Returns a Pytorch3D renderer for point clouds.

    Args:
        image_size (int): The rendered image size.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
        radius (float): The radius of the rendered point in NDC.
        background_color (tuple): The background color of the rendered image.
    
    Returns:
        PointsRenderer.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius,)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer


def viz_seg (verts, labels, path, device):
    """
    visualize segmentation result
    output: a 360-degree gif
    """
    image_size=256
    background_color=(1, 1, 1)
    colors = [[1.0,1.0,1.0], [1.0,0.0,1.0], [0.0,1.0,1.0],[1.0,1.0,0.0],[0.0,0.0,1.0], [1.0,0.0,0.0]]

    # Construct various camera viewpoints
    dist = 3
    elev = 0
    azim = [180 - 12*i for i in range(30)]
    R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=dist, elev=elev, azim=azim, device=device)
    c = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)

    sample_verts = verts.unsqueeze(0).repeat(30,1,1).to(torch.float)
    sample_labels = labels.unsqueeze(0)
    num_points = verts.shape[0]  # Get actual number of points
    sample_colors = torch.zeros((1, num_points, 3))

    # Colorize points based on segmentation labels
    for i in range(6):
        sample_colors[sample_labels==i] = torch.tensor(colors[i])

    sample_colors = sample_colors.repeat(30,1,1).to(torch.float)

    point_cloud = pytorch3d.structures.Pointclouds(points=sample_verts, features=sample_colors).to(device)

    renderer = get_points_renderer(image_size=image_size, background_color=background_color, device=device)
    rend = renderer(point_cloud, cameras=c).cpu().numpy() # (30, 256, 256, 3)
    rend = (rend * 255).astype(np.uint8)

    imageio.mimsave(path, rend, fps=15)


def rotate_point_cloud(points, angle_degrees, seed=None):
    """
    Apply random rotation around all three axes (X, Y, Z) to point cloud.
    
    Args:
        points: torch.Tensor of shape (N, 3) or (B, N, 3)
        angle_degrees: float, total rotation angle in degrees
        seed: int, random seed for reproducibility (optional)
    
    Returns:
        rotated_points: torch.Tensor of same shape as input
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Convert to numpy for rotation
    is_batch = len(points.shape) == 3
    if is_batch:
        B, N, _ = points.shape
        points_np = points.cpu().numpy()
    else:
        N, _ = points.shape
        points_np = points.cpu().numpy()
        points_np = points_np.reshape(1, N, 3)
    
    # Generate random rotation angles for each axis
    # Distribute the total angle across X, Y, Z axes
    angle_rad = np.radians(angle_degrees)
    # Random direction for rotation vector
    direction = np.random.randn(3)
    direction = direction / np.linalg.norm(direction)
    rotation_vector = direction * angle_rad
    
    # Create rotation object
    rotation = R.from_rotvec(rotation_vector)
    rotation_matrix = rotation.as_matrix()
    
    # Apply rotation to all point clouds
    rotated_points_np = np.zeros_like(points_np)
    for i in range(points_np.shape[0]):
        rotated_points_np[i] = (rotation_matrix @ points_np[i].T).T
    
    # Convert back to torch tensor
    if not is_batch:
        rotated_points_np = rotated_points_np[0]
    
    rotated_points = torch.from_numpy(rotated_points_np).to(points.dtype).to(points.device)
    
    return rotated_points

