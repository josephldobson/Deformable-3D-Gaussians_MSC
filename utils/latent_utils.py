import torch
from scene.deform_model import DeformModel
from scene.gaussian_model import GaussianModel
import numpy as np


def get_trajectories(deform: DeformModel, xyz, num__frames) -> np.ndarray:
    trajectories = xyz.unsqueeze(0).repeat(num__frames, 1, 1)
    times = (
        torch.linspace(0, 1, num__frames)
        .unsqueeze(1)
        .unsqueeze(2)
        .repeat(1, xyz.shape[0], 1)
    )
    with torch.no_grad():
        shift, _, _, _ = deform.step(trajectories, times)
        trajectories += shift
    return trajectories


def filter_static(trajectories: torch.Tensor):
    tolerance = 0.00083

    displacements = torch.norm(trajectories[1:] - trajectories[:-1], dim=2)

    mean_displacement = torch.mean(displacements, dim=0)
    mask = mean_displacement <= tolerance
    trajectories = trajectories[:, ~mask]
    return trajectories, mask


def pairwise_distance_matrix(latent_points: torch.Tensor):
    latent_points = latent_points.unsqueeze(1)
    y = latent_points.transpose(0, 1)
    distances = torch.sqrt(torch.sum((latent_points - y) ** 2, dim=-1) + 1e-8)
    return distances


def latent_space_distance(trajectories: torch.Tensor) -> torch.Tensor:
    trajectories -= torch.mean(trajectories, dim=0)

    n = trajectories.size(1)
    sum_squares = torch.sum(trajectories**2, dim=-1)

    # Compute the pairwise distances using dot product formula (I had ram problems):
    # D_ij = sqrt((x_i - x_j) . (x_i - x_j))
    #       = sqrt(x_i . x_i + x_j . x_j - 2 * (x_i . x_j))
    pairwise_distances = torch.sqrt(
        sum_squares[:, :, None]
        + sum_squares[:, None, :]
        - 2 * torch.bmm(trajectories, trajectories.transpose(1, 2))
    )
    average_pairwise_distances = torch.sum(pairwise_distances) / (n * (n - 1))
    return average_pairwise_distances


def distance_loss(
    original_distances: torch.Tensor, mask: torch.Tensor, latent_points: torch.Tensor
):
    latent_distances = pairwise_distance_matrix(latent_points)
    return torch.mean((original_distances - latent_distances) ** 2)


def latent_space_distance(trajectories: torch.Tensor):
    trajectories -= torch.mean(trajectories, dim=0)
    expanded_trajectory = trajectories[:, :, None, :]
    pairwise_diff = expanded_trajectory - expanded_trajectory.transpose(1, 2)
    pairwise_sq_diff = pairwise_diff**2
    pairwise_distances = torch.sqrt(torch.sum(pairwise_sq_diff, dim=-1))
    average_pairwise_distances = torch.mean(pairwise_distances, dim=0)
    return average_pairwise_distances
