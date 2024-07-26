import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from scene import Scene, DeformModel

from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import sys


def get_deformation(deform: DeformModel, xyz, frame: int):
    N = xyz.shape[0]
    t = torch.ones((N, 1)).to('cuda')*frame
    d_xyz, d_rotation, d_scaling = deform.step(xyz, t)
    return d_xyz.cpu().numpy()


def generate_deformation_save(dataset: ModelParams, iteration: int):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        deform = DeformModel(dataset.is_blender, dataset.is_6dof)
        deform.load_weights(dataset.model_path)

        num_frames = len(scene.getTrainCameras())
        xyz = gaussians.get_xyz
        np.save('xyz.npy', xyz.cpu().numpy())

        # del scene
        # del gaussians

        # xyzt = np.tile(xyz.cpu().numpy().astype(np.float16), (num_frames, 1, 1))

        # frame = 0
        # for frame in range(num_frames):
        #     d_xyz = get_deformation(deform, xyz, frame / 474)
        #     xyzt[frame] += d_xyz
        # np.save('xyzt.npy', xyzt)


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
