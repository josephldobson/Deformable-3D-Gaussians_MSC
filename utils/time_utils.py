import torch
import torch.nn as nn
import torch.nn.functional as F


class SinCosTransform(nn.Module):
    def __init__(self, N_freqs):
        super(SinCosTransform, self).__init__()
        self.freqs = 2 ** torch.arange(0, N_freqs)

    def forward(self, x):
        sinusoids = torch.cat([torch.sin(x * freq) for freq in self.freqs] +
                              [torch.cos(x * freq) for freq in self.freqs], dim=-1)
        return sinusoids


class DeformationNetwork(nn.Module):
    def __init__(self, D=8, W=256, latent_dim=3, N_freqs=10):
        super(DeformationNetwork, self).__init__()
        self.skips = [D // 2]
        self.D = D
        self.W = W
        self.input_d = 8*N_freqs

        self.encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.sin_cos_transform = SinCosTransform(N_freqs=10)

        self.deformation_network = nn.ModuleList(
            [nn.Linear(self.input_d, W)] + [
                nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_d, W)
                for i in range(D-1)
            ]
        )
        self.xyz_shift = nn.Linear(self.W, 3)
        self.rot_shift = nn.Linear(self.W, 4)
        self.sca_shift = nn.Linear(self.W, 3)

    def forward(self, xyz, t):
        latent = self.encoder(xyz)
        sin_cos = self.sin_cos_transform(torch.concat((latent, t), dim=1))
        h = torch.clone(sin_cos)
        for i, layer in enumerate(self.deformation_network):
            h = layer(h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat((h, sin_cos), -1)

        xyz = self.xyz_shift(h)
        rot = self.rot_shift(h)
        sca = self.sca_shift(h)

        return xyz, rot, sca, latent
