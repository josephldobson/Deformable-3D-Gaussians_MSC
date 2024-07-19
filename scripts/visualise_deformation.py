import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mayavi import mlab
import torch
from scene import Scene, DeformModel

from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import imageio
import numpy as np
import time
import sys

# python scripts/visualise_deformation.py -s '/home/joe/data/colmap/tree_garden' -m '/home/joe/repos/Deformable-3D-Gaussians_MSC/output/95456d7f-6' 


def get_deformation(deform: DeformModel, xyz, frame: int):
    N = xyz.shape[0]
    t = torch.ones((N, 1)).to('cuda')*frame
    d_xyz, d_rotation, d_scaling = deform.step(xyz, t)
    return d_xyz, d_rotation

def filter_gaussians_sphere(gaussians: GaussianModel, center: tuple, radius: float):
    
    N = 60_000 # max gaussians my memory can handle
    xyz = gaussians.get_xyz.detach()
    rot = gaussians.get_rotation.detach()
    scal = gaussians.get_scaling.detach()
    opac = gaussians.get_opacity.detach()
    
    x_c, y_c, z_c = center
    
    distances_squared = (xyz[:, 0] - x_c) ** 2 + (xyz[:, 1] - y_c) ** 2 + (xyz[:, 2] - z_c) ** 2
    
    mask = distances_squared <= radius**2
    
    filtered_xyz = xyz[mask]
    filtered_rot = rot[mask]
    filtered_scal = scal[mask]
    filtered_opac = opac[mask]

    mask = filtered_opac.view(-1) > 0.05

    filtered_xyz = filtered_xyz[mask]
    filtered_rot = filtered_rot[mask]
    filtered_scal = filtered_scal[mask]
    filtered_opac = filtered_opac[mask]

    if filtered_xyz.shape[0] > N:
        filtered_xyz = filtered_xyz[:N]
        filtered_rot = filtered_rot[:N]
        filtered_scal = filtered_rot[:N]
        filtered_opac = filtered_opac[:N]

    print(filtered_xyz.shape[0])

    return filtered_xyz, filtered_rot, filtered_scal, filtered_opac


def generate_deformation(dataset: ModelParams, iteration: int):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        deform = DeformModel(dataset.is_blender, dataset.is_6dof)
        deform.load_weights(dataset.model_path)

        # tree_garden - bad part
        # xyz, rot, scal, opac = filter_gaussians_sphere(gaussians, (-0.8, 1.5, 8.85), 2)
        
        # moving_leaf - leaf (583)
        # xyz, rot, scal, opac = filter_gaussians_sphere(gaussians, (0.8, 2, 6), 3)

        # two_leaves - leaf
        xyz, rot, scal, opac = filter_gaussians_sphere(gaussians, (-3, 2.4, 4), 3)

        
        del gaussians
        del scene

        xs, ys, zs = xyz[:, 0].cpu().numpy(), xyz[:, 1].cpu().numpy(), xyz[:, 2].cpu().numpy()
        s = opac.cpu().numpy().reshape(-1)
        points = mlab.points3d(xs, ys, zs, s, scale_factor=0.01, color=(1, 0.1, 0))

        u, v, w = rot[:, 0].cpu().numpy(), rot[:, 1].cpu().numpy(), rot[:, 2].cpu().numpy()
        vectors = mlab.quiver3d(xs, ys, zs, u, v, w, line_width=0.5, scale_factor=0.05, color=(0, 1, 0))

        mlab.xlabel('X-axis')
        mlab.ylabel('Y-axis')
        mlab.zlabel('Z-axis')
        
        @mlab.animate(delay=200)
        def animate():
            num_frames = 474
            frame = 0
            while True:
                d_xyz, d_rot = get_deformation(deform, xyz, frame / 474)
                dx, dy, dz = d_xyz[:, 0].cpu().numpy(), d_xyz[:, 1].cpu().numpy(), d_xyz[:, 2].cpu().numpy()
                du, dv, dw = d_rot[:, 0].cpu().numpy(), d_rot[:, 1].cpu().numpy(), d_rot[:, 2].cpu().numpy()


                points.mlab_source.set(x=xs + dx, y=ys + dy, z=zs + dz)
                vectors.mlab_source.set(x=xs+dx, y=ys+dy, z=zs+dz, u=u+du, v=v+dv, w=w+dw)
                
                frame += 1
                if frame == num_frames:
                    xs[:] = xs
                    ys[:] = ys
                    zs[:] = zs
                    frame = 0
                
                yield

        # Start animation
        animate()
        mlab.show()

def generate_deformation_save(dataset: ModelParams, iteration: int):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        deform = DeformModel(dataset.is_blender, dataset.is_6dof)
        deform.load_weights(dataset.model_path)

        xyz, rot, scal, opac = filter_gaussians_sphere(gaussians, (-3, 2.4, 4), 3)
        del gaussians
        del scene

        num_frames = 474

        xs, ys, zs = xyz[:, 0].cpu().numpy(), xyz[:, 1].cpu().numpy(), xyz[:, 2].cpu().numpy()
        s = opac.cpu().numpy().reshape(-1)
        u, v, w = rot[:, 0].cpu().numpy(), rot[:, 1].cpu().numpy(), rot[:, 2].cpu().numpy()
        
        x_full = np.tile(xs, (num_frames, 1))
        y_full = np.tile(ys, (num_frames, 1))
        z_full = np.tile(zs, (num_frames, 1))
        u_full = np.tile(u, (num_frames, 1))
        v_full = np.tile(v, (num_frames, 1))
        w_full = np.tile(w, (num_frames, 1))
        
        frame = 0
        for frame in range(474):
            d_xyz, d_rot = get_deformation(deform, xyz, frame / 474)
            dx, dy, dz = d_xyz[:, 0].cpu().numpy(), d_xyz[:, 1].cpu().numpy(), d_xyz[:, 2].cpu().numpy()
            du, dv, dw = d_rot[:, 0].cpu().numpy(), d_rot[:, 1].cpu().numpy(), d_rot[:, 2].cpu().numpy()

            x_full[frame] += dx
            y_full[frame] += dy
            z_full[frame] += dz
            u_full[frame] += du
            v_full[frame] += dv
            w_full[frame] += dw

        np.save('x_full.npy', x_full)
        np.save('y_full.npy', y_full)
        np.save('z_full.npy', z_full)
        np.save('u_full.npy', u_full)
        np.save('v_full.npy', v_full)
        np.save('w_full.npy', w_full)
        np.save('s.npy', s)



if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mode", default='render', choices=['render', 'time', 'view', 'all', 'pose', 'original'])
    args = get_combined_args(parser)
    print("Saving " + args.model_path)
    safe_state(args.quiet)

    generate_deformation_save(model.extract(args), args.iteration)
