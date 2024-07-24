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


def latent_space_distance(trajectories: torch.Tensor):
    trajectories -= torch.mean(trajectories, dim=0)
    expanded_trajectory = trajectories[:, :, None, :]
    pairwise_diff = expanded_trajectory - expanded_trajectory.transpose(1, 2)
    pairwise_sq_diff = pairwise_diff ** 2
    pairwise_distances = torch.sqrt(torch.sum(pairwise_sq_diff, dim=-1))
    average_pairwise_distances = torch.mean(pairwise_distances, dim=0)
    return average_pairwise_distances


def train_encoder(epochs=25, lr=0.001, batch_size=1000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_points = torch.from_numpy(np.load('data/xyz.npy')).float().to(device)
    trajectories_data = torch.from_numpy(np.load('data/xyzt.npy')).float().to(device)

    static_points, data_points, static_trajectories, trajectories_data = filter_static(data_points, trajectories_data)
    np.save('data/static_points.npy', static_points.cpu().numpy())
    np.save('data/non_static_points.npy', data_points.cpu().numpy())
    num_batches = data_points.shape[0] // batch_size

    del static_points
    del static_trajectories

    encoder = Encoder(input_dim=3, latent_dim=3).to(device)
    optimizer = optim.Adam(encoder.parameters(), lr=lr)

    for epoch in range(epochs):
        epoch_loss = 0
        idx = torch.randperm(data_points.shape[0]).to(device)
        data_points = data_points.index_select(0, idx)
        trajectories_data = trajectories_data.index_select(1, idx)

        for batch_idx in range(num_batches):
            points = data_points[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            trajectories = trajectories_data[:, batch_idx * batch_size:(batch_idx + 1) * batch_size]

            original_distances = latent_space_distance(trajectories).to(device)
            optimizer.zero_grad()
            latent_points = encoder(points)
            loss = distance_loss(original_distances, latent_points)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")
    torch.save(encoder.state_dict(), 'data/encoder_model_3.pth')


if __name__ == '__main__':
    train_encoder()
