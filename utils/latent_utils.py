import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Encoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, latent_dim)

    def forward(self, x: torch.Tensor):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def latent_space_distance(trajectories: torch.Tensor):
    trajectories -= torch.mean(trajectories, dim=0)
    expanded_trajectory = trajectories[:, :, None, :]
    pairwise_diff = expanded_trajectory - expanded_trajectory.transpose(1, 2)
    pairwise_sq_diff = pairwise_diff ** 2
    pairwise_distances = torch.sqrt(torch.sum(pairwise_sq_diff, dim=-1))
    average_pairwise_distances = torch.mean(pairwise_distances, dim=0)
    return average_pairwise_distances


def filter_static(points: torch.Tensor, trajectories: torch.Tensor):
    tolerance = 0.00083

    displacements = torch.norm(trajectories[1:] - trajectories[:-1], dim=2)
    print(points.shape[0])

    mean_displacement = torch.mean(displacements, dim=0)
    mask = mean_displacement <= tolerance
    static_trajectories = trajectories[:, mask]
    trajectories = trajectories[:, ~mask]
    static_points = points[mask]
    points = points[~mask]
    print(points.shape[0])
    return static_points, points, static_trajectories, trajectories


def pairwise_distance_matrix(x: torch.Tensor):
    x = x.unsqueeze(1)
    y = x.transpose(0, 1)
    distances = torch.sqrt(torch.sum((x - y) ** 2, dim=-1) + 1e-8)
    return distances


def distance_loss(original_distances: torch.Tensor, latent_points: torch.Tensor):
    latent_distances = pairwise_distance_matrix(latent_points)
    return torch.mean((original_distances - latent_distances) ** 2)


def train_encoder_loop(encoder, optimizer, x, max_epochs=25):

    idx = torch.randperm(x.shape[0])
    x = x.index_select(0, idx)

    for batch_idx in range(num_batches):
        points = data_points[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        trajectories = trajectories_data[
            :, batch_idx * batch_size : (batch_idx + 1) * batch_size
        ]

        input = trajectories.cpu().numpy()
        input -= np.mean(input, axis=0)
        original_distances = latent_space_distance(input)
        original_distances = torch.from_numpy(original_distances).cuda()
        optimizer.zero_grad()
        latent_points = encoder(points)
        loss = distance_loss(original_distances, latent_points)
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    train_encoder_loop()