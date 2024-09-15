import torch
from pytorch3d.transforms import matrix_to_quaternion
from sklearn.neighbors import NearestNeighbors


def compute_d_quat(xyz, d_xyz, k):
    """
    Compute the change in orientation (d_quat) for each point.

    Args:
        xyz (torch.Tensor): Original positions of the points (n, 3).
        d_xyz (torch.Tensor): Displacements of the points (n, 3).
        quat (torch.Tensor): Original orientations as quaternions (n, 4).
        k (int): Number of nearest neighbors to consider.

    Returns:
        torch.Tensor: Updated orientations as quaternions (n, 4).
    """
    with torch.no_grad():
        nn = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(xyz.cpu().numpy())
        _, indices = nn.kneighbors(xyz.cpu().numpy())
        indices = torch.tensor(indices[:, 1:], device='cuda')
    
    P_bef = xyz[indices]
    P_aft = P_bef + d_xyz[indices]
    
    # Compute centroids of neighbors before and after displacement
    cP_bef = P_bef.mean(dim=1, keepdim=True)  # (n, 1, 3)
    cQ_aft = P_aft.mean(dim=1, keepdim=True)
    
    # Center the positions
    P_bef_centered = P_bef - cP_bef               # (n, k, 3)
    P_aft_centered = P_aft - cQ_aft
    
    # Compute covariance matrices (n, 3, 3)
    H = torch.matmul(P_bef_centered.transpose(1, 2), P_aft_centered)
    
    # Perform SVD on the covariance matrices
    U, S, Vh = torch.linalg.svd(H)
    V = Vh.transpose(-2, -1)
    
    # Compute rotation matrices (n, 3, 3)
    R = torch.matmul(V, U.transpose(-2, -1))
        
    # Correct improper rotations (reflections)
    det = torch.det(R)
    det_sign = torch.sign(det).unsqueeze(-1)  # Shape: (n, 1)
    V[:, :, 2] *= det_sign  # Broadcast to (n, 3)
    R = torch.matmul(V, U.transpose(-2, -1))
    
    # Convert rotation matrices to quaternions
    d_quat = matrix_to_quaternion(R)  # (n, 4) 
    return d_quat
