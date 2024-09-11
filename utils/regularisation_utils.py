import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

def rigidity_loss(xyz_t0, xyz_t1, k=4) -> float:
    """
    Compute the rigidity loss that penalizes changes in relative distances between neighboring points.
    """
    displacement = torch.norm(xyz_t1 - xyz_t0, dim=-1)
    moving_mask = displacement > 0.001
    xyz_t0_m = xyz_t0[moving_mask]
    xyz_t1_m = xyz_t1[moving_mask]

    if xyz_t1_m.shape[0] > 10:
        with torch.no_grad():
            knn = NearestNeighbors(n_neighbors=k).fit(xyz_t0_m.cpu().numpy())
            distances, indices = knn.kneighbors(xyz_t0_m.cpu().numpy())
            indices = torch.tensor(indices, device=xyz_t0_m.device)

        t0_neighbours = xyz_t0_m[indices]
        diffs = t0_neighbours - xyz_t0_m.unsqueeze(1)
        initial_distances = torch.norm(diffs, dim=-1)
        
        t1_neighbours = xyz_t1_m[indices]
        deformed_diffs = t1_neighbours - xyz_t1_m.unsqueeze(1)
        new_distances = torch.norm(deformed_diffs, dim=-1)

        rigidity_loss = F.mse_loss(new_distances, initial_distances, reduction='mean')
        return rigidity_loss
    else: return 0.

