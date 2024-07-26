import numpy as np
import mayavi.mlab as mlab
from create_latent_space import Encoder
import torch

def main():
    non_static_points = np.load('data/non_static_points.npy')

    encoder = Encoder(input_dim=3, latent_dim=3)
    encoder.load_state_dict(torch.load('data/encoder_model_3.pth'))
    encoder.eval()

    non_static_tensor = torch.from_numpy(non_static_points).float()
    latent = encoder(non_static_tensor)
    latent = latent.detach()

    min_vals = latent.min(dim=0)[0]
    max_vals = latent.max(dim=0)[0]
    col = (latent - min_vals) / (max_vals - min_vals)
    col = 2.0 * (0.5 - abs(col - 0.5))
    col = col.numpy()


    n = non_static_points.shape[0]
    lut = np.zeros((n, 4), dtype=np.uint8)
    lut[:, :3] = (col * 255).astype(np.uint8)
    lut[:, 3] = 255

    indices = np.arange(n)
    p3d = mlab.points3d(latent[:, 0], latent[:, 1], latent[:, 2], indices, scale_factor=0.0005, scale_mode='none')
    lut_manager = p3d.module_manager.scalar_lut_manager
    lut_manager.lut.number_of_colors = n
    lut_manager.lut.table = lut

    mlab.draw()
    mlab.show()

if __name__ == "__main__":
    main()  
