import numpy as np
import mayavi.mlab as mlab
from create_latent_space import Encoder
import torch

static_points = np.load('data/static_points.npy')
non_static_points = np.load('data/non_static_points.npy')

mlab.points3d(static_points[:, 0], static_points[:, 1], static_points[:, 2], color=(1, 1, 1), scale_factor=0.01)

encoder = Encoder(input_dim=3, latent_dim=3)
encoder.load_state_dict(torch.load('data/encoder_model_3.pth'))
encoder.eval()

non_static_tensor = torch.from_numpy(non_static_points).float()
col = encoder(non_static_tensor)
col = col.detach()

min_vals = col.min(dim=0)[0]
max_vals = col.max(dim=0)[0]
col = (col - min_vals) / (max_vals - min_vals)
col = col % 0.5
col = col * 2
col = col.numpy()

n = non_static_points.shape[0]
lut = np.zeros((n, 4), dtype=np.uint8)
lut[:, :3] = (col * 255).astype(np.uint8)
lut[:, 3] = 255

indices = np.arange(n)
p3d = mlab.points3d(non_static_points[:, 0], non_static_points[:, 1], non_static_points[:, 2], indices, scale_factor=0.01, scale_mode='none')
lut_manager = p3d.module_manager.scalar_lut_manager
lut_manager.lut.number_of_colors = n
lut_manager.lut.table = lut

mlab.draw()
mlab.show()
