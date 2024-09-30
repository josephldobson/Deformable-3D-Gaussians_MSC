import torch
from jaxtyping import Float, Shaped


# Methods for working with 3D transformation matrices.
# All methods here return matrices with batch dimensions. Inputs without batch dimensions are supported, but batch
# dimensions are always added to the output.
# Forked from https://github.com/facebookresearch/pytorch3d/blob/4ae25bfce7eb42042a34585acc3df81cf4be7d85/pytorch3d/transforms/rotation_conversions.py

import torch
import torch.nn.functional as F


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """Returns torch.sqrt(torch.max(0, x)) but with a zero subgradient where x is 0."""

    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def torch3d_matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)),
        dim=-1,
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        ),
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5,
        :,
    ].reshape(batch_dim + (4,))
    return standardize_quaternion(out)


def quaternion_to_torch3d_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def translation(transmat: Float[Shaped, "... 4 4"]) -> Float[Shaped, "batch ... 3"]:
    if transmat.ndim == 2:
        transmat = transmat[None]
    return transmat[..., :3, 3]


def rotation(transmat: Float[Shaped, "... 4 4"]) -> Float[Shaped, "batch ... 3 3"]:
    if transmat.ndim == 2:
        transmat = transmat[None]
    return transmat[..., :3, :3]


def rotation_from_axis_angles(axis_angles: Float[torch.Tensor, "... 3"]) -> Float[torch.Tensor, "... 3 3"]:
    if not isinstance(axis_angles, torch.Tensor):
        axis_angles = torch.tensor(axis_angles).float()
    if axis_angles.ndim == 1:
        axis_angles = axis_angles[None]
    x, y, z = axis_angles.unbind(-1)
    angle = torch.linalg.norm(axis_angles, dim=-1, keepdim=True)
    s = torch.sin(angle)
    c = torch.cos(angle)
    t = 1 - c
    x, y, z = x / angle, y / angle, z / angle
    return torch.stack(
        [
            t * x * x + c,
            t * x * y - s * z,
            t * x * z + s * y,
            t * x * y + s * z,
            t * y * y + c,
            t * y * z - s * x,
            t * x * z - s * y,
            t * y * z + s * x,
            t * z * z + c,
        ],
        dim=-1,
    ).reshape(*axis_angles.shape[:-1], 3, 3)


def from_rotation_translation(
    rotation: Float[torch.Tensor, "... 3 3"] = None,
    translation: Float[torch.Tensor, "... 3"] = None,
) -> Float[torch.Tensor, "batch ... 4 4"]:
    dtype = (
        rotation.dtype
        if rotation is not None and isinstance(rotation, torch.Tensor)
        else translation.dtype
        if translation is not None and isinstance(translation, torch.Tensor)
        else torch.float32
    )
    if rotation is None and translation is None:
        return torch.eye(4, dtype=dtype)[None]
    if rotation is not None and not isinstance(rotation, torch.Tensor):
        rotation = torch.tensor(rotation, dtype=dtype)
    if translation is not None and not isinstance(translation, torch.Tensor):
        translation = torch.tensor(translation, dtype=dtype)
    if rotation is None:
        rotation = torch.eye(3, device=translation.device, dtype=dtype).expand(*translation.shape[:-1], -1, -1)
    if translation is None:
        translation = torch.zeros(3, device=rotation.device, dtype=dtype).expand(*rotation.shape[:-2], -1)
    if rotation.ndim == 2:
        rotation = rotation[None]
    if translation.ndim == 1:
        translation = translation[None]

    transform = torch.eye(4, device=rotation.device, dtype=dtype).expand(*rotation.shape[:-2], -1, -1).contiguous()
    transform[..., :3, :3] = rotation
    transform[..., :3, 3] = translation
    return transform


def identity(dtype=torch.float) -> Float[torch.Tensor, "1 4 4"]:
    return torch.eye(4, dtype=dtype)[None]


def inverse(transmat: torch.Tensor) -> torch.Tensor:
    rot_inv = rotation(transmat).permute(0, 2, 1)
    trans = translation(transmat)
    return from_rotation_translation(rot_inv, (rot_inv @ -trans.unsqueeze(-1)).view_as(trans))


def convert_3x4_to_4x4(transmat: Float[torch.Tensor, "... 3 4"]) -> Float[torch.Tensor, "... 4 4"]:
    assert transmat.shape[-2:] == (3, 4), f"Invalid shape {transmat.shape}"
    if transmat.ndim == 2:
        transmat = transmat[None]
    transform = (
        torch.eye(4, device=transmat.device, dtype=transmat.dtype).expand(*transmat.shape[:-2], -1, -1).contiguous()
    )
    transform[..., :3, :4] = transmat
    return transform


def convert_4x4_to_3x4(transmat: Float[torch.Tensor, "... 4 4"]) -> Float[torch.Tensor, "... 3 4"]:
    assert transmat.shape[-2:] == (4, 4), f"Invalid shape {transmat.shape}"
    if transmat.ndim == 2:
        transmat = transmat[None]
    return transmat[..., :3, :4]


def matrix_from_quarternion(quat: Float[torch.Tensor, "... 4"]) -> Float[torch.Tensor, "... 3 3"]:
    if quat.ndim == 1:
        quat = quat[None]
    # Torch3d rotation matrices are row major, so we need to permute the axes.
    return quaternion_to_torch3d_matrix(quat).permute(0, 2, 1)


def quaternion_from_matrix(rotmat: Float[torch.Tensor, "... 3 3"]) -> Float[torch.Tensor, "... 4"]:
    if rotmat.ndim == 2:
        rotmat = rotmat[None]
    # Torch3d rotation matrices are row major, so we need to permute the axes.
    return torch3d_matrix_to_quaternion(rotmat.permute(0, 2, 1))


def apply(
    transmat: Float[torch.Tensor, "... 4 4"],
    points: Float[torch.Tensor, "... pts 3"],
) -> Float[torch.Tensor, "... pts 3"]:
    while points.ndim > transmat.ndim - 1:
        transmat = transmat.unsqueeze(-3)
    return (transmat[..., :3, :3] @ points.unsqueeze(-1)).squeeze(-1) + transmat[..., :3, 3]
