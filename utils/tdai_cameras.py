import math
from typing import Literal, Optional, Tuple

import numpy as np
import torch
from jaxtyping import Float

from utils import tdai_transforms as transforms
from typing import Literal, Optional, Protocol, Tuple
from typing_extensions import Self
from jaxtyping import Bool, Float, Int, Int64, Shaped, UInt8


# In Python 3.12 these shall be generic functions.
class CameraProtocol(Protocol):
    world_from_cam: Float[torch.Tensor, "#num_imgs 4 4"]

    focal_length: Float[torch.Tensor, "#num_imgs 2"]
    principal_point: Float[torch.Tensor, "#num_imgs 2"]
    image_size_wh: Int[torch.Tensor, "#num_imgs 2"]

    distortion_model: Shaped[np.ndarray, "#num_imgs"]
    distortion: Float[torch.Tensor, "#num_imgs 6"]

    camera_space: str

    def replace(self: Self, **kwargs) -> Self: ...

    def build(self: Self, **kwargs) -> Self: ...

    def float(self) -> Self: ...

    def __len__(self) -> int: ...

def resize(
    cameras: CameraProtocol,
    *,
    scale_factor = None,
    size_wh: Optional[Tuple[int, int]] = None,
    align: bool = True,
):
    """Scale the cameras.

    If align is True, this ensures that width and height are integers."""
    if size_wh is not None:
        assert scale_factor is None, "Cannot specify both scale_factor and size_wh"
        image_size_wh = torch.tensor(size_wh, dtype=torch.int32)[None].repeat(len(cameras.image_size_wh), 1)
        scale_factor = image_size_wh.float() / cameras.image_size_wh
    else:
        if isinstance(scale_factor, float):
            scale_factor = (scale_factor, scale_factor)
        if isinstance(scale_factor, tuple):
            scale_factor = torch.tensor(scale_factor)[None]
        image_size_wh = cameras.image_size_wh * scale_factor
    if align:
        image_size_wh = image_size_wh.round().to(torch.int32)
        scale_factor = image_size_wh.float() / cameras.image_size_wh
    return cameras.replace(
        focal_length=cameras.focal_length * scale_factor,
        principal_point=cameras.principal_point * scale_factor,
        image_size_wh=image_size_wh,
    )


def get_rub_from_rdf():
    rub_from_rdf = transforms.identity()
    rub_from_rdf[:, 0:3, 1:3] *= -1
    return rub_from_rdf


def get_rub_from_flu():
    return torch.tensor(
        [
            [0.0, 0.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )


def to_right_up_bottom(cameras: CameraProtocol):
    """Convert the cameras to right up bottom coordinate frame."""
    if cameras.camera_space == "RUB":
        return cameras
    if cameras.camera_space == "RDF":
        world_from_cam = cameras.world_from_cam @ get_rub_from_rdf().to(cameras.world_from_cam)
        return cameras.replace(
            world_from_cam=world_from_cam,
            camera_space="RUB",
        )
    if cameras.camera_space == "FLU":
        world_from_cam = cameras.world_from_cam @ get_rub_from_flu().to(cameras.world_from_cam)
        return cameras.replace(
            world_from_cam=world_from_cam,
            camera_space="RUB",
        )
    raise ValueError(f"Unknown camera space: {cameras.camera_space}")


def project_points_pinhole_rdf(
    points_camera_rdf: Float[torch.Tensor, "pts 3"],
    focal_length: Float[torch.Tensor, "2"],
    principal_point: Float[torch.Tensor, "2"],
) -> Float[torch.Tensor, "pts 2"]:
    """Project points from camera space to image space using pinhole model.

    Assumes that the coordinate frame is RDF."""
    assert points_camera_rdf.ndim == 2
    u, v, w = points_camera_rdf.unbind(-1)
    return torch.stack(
        [
            focal_length[0] * u / w + principal_point[0],
            focal_length[1] * v / w + principal_point[1],
        ],
        dim=-1,
    )


def project_points_pinhole_rub(
    points_camera_rub: Float[torch.Tensor, "pts 3"],
    focal_length: Float[torch.Tensor, "2"],
    principal_point: Float[torch.Tensor, "2"],
) -> Float[torch.Tensor, "pts 2"]:
    """Project points from camera space to image space using pinhole model.

    Assumes that the coordinate frame is RUB."""
    assert points_camera_rub.ndim == 2
    points_camera_rdf = transforms.apply(
        transforms.inverse(get_rub_from_rdf().to(points_camera_rub.dtype)),
        points_camera_rub,
    )
    return project_points_pinhole_rdf(points_camera_rdf, focal_length, principal_point)


def project_points_fisheye_rdf(
    points_camera_rdf: Float[torch.Tensor, "pts 3"],
    focal_length: Float[torch.Tensor, "2"],
    principal_point: Float[torch.Tensor, "2"],
    distortion: Float[torch.Tensor, "6"],
) -> Float[torch.Tensor, "pts 2"]:
    """Project points from camera space (RDF) to image space using the OPENCV_FISHEYE model.

    https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html"""

    x, y, z = points_camera_rdf.unbind(-1)
    cx, cy = principal_point.unbind(-1)
    fx, fy = focal_length.unbind(-1)
    k1, k2, k3, k4, *_ = distortion.unbind(-1)

    # 1. Normalize the coordinates
    x_prime = x / z
    y_prime = y / z

    # 2. Compute the radius
    r = torch.sqrt(x_prime**2 + y_prime**2)

    # 3. Compute the theta
    theta = torch.atan(r)

    # 4. Apply the fisheye distortion
    theta_d = theta * (1 + k1 * theta**2 + k2 * theta**4 + k3 * theta**6 + k4 * theta**8)
    theta_d_div_r = torch.where(r != 0, theta_d / r, torch.tensor(1.0, dtype=theta_d.dtype, device=theta_d.device))

    x_d = x_prime * theta_d_div_r
    y_d = y_prime * theta_d_div_r

    # 5. Transform to pixel coordinates
    u = fx * x_d + cx
    v = fy * y_d + cy

    # Stack the results into a single tensor
    return torch.stack((u, v), dim=-1)


def unproject_points_pinhole_rdf(
    points_image: Float[torch.Tensor, "pts 2"],
    depth: Float[torch.Tensor, "pts"],
    focal_length: Float[torch.Tensor, "2"],
    principal_point: Float[torch.Tensor, "2"],
) -> Float[torch.Tensor, "pts 3"]:
    """Unproject points from image space to camera space using pinhole model.

    Returns points in RDF camera frame."""
    assert points_image.ndim == 2
    u, v = points_image.unbind(-1)
    return torch.stack(
        [
            (u - principal_point[0]) * depth / focal_length[0],
            (v - principal_point[1]) * depth / focal_length[1],
            depth,
        ],
        dim=-1,
    )


def unproject_depthmap_pinhole_rdf(
    depthmap: Float[torch.Tensor, "height width"],
    focal_length: Float[torch.Tensor, "2"],
    principal_point: Float[torch.Tensor, "2"],
) -> Float[torch.Tensor, "height width 3"]:
    """Unproject a depthmap from image space to camera space using pinhole model.

    Returns points in RDF camera frame."""
    height, width = depthmap.shape
    v, u = torch.meshgrid(torch.arange(height, device=depthmap.device), torch.arange(width, device=depthmap.device))
    return unproject_points_pinhole_rdf(
        torch.stack([u, v], dim=-1).reshape(-1, 2),
        depthmap.reshape(-1),
        focal_length,
        principal_point,
    ).reshape(height, width, 3)


def interpolate_cameras(
    keyframes: CameraProtocol,
    num_steps_between_frames: int,
    easing: Literal["linear", "sine", "cubic"] = "linear",
) -> CameraProtocol:
    """Interpolates camera parameters between the keyframes."""

    if easing == "linear":
        ticks = torch.linspace(0, 1, num_steps_between_frames + 2)
    elif easing == "sine":
        ticks = torch.sin(torch.linspace(0, math.pi / 2, num_steps_between_frames + 2))
    elif easing == "cubic":
        ticks = torch.pow(torch.linspace(0, 1, num_steps_between_frames + 2), 3)
    else:
        raise ValueError(f"Unknown easing: {easing}")

    world_from_cam = _interpolate_between_poses(keyframes.world_from_cam, ticks)
    focal_length = _interpolate_between_values(keyframes.focal_length, ticks)
    principal_point = _interpolate_between_values(keyframes.principal_point, ticks)
    distortion = _interpolate_between_values(keyframes.distortion, ticks)

    return keyframes.build(
        world_from_cam=world_from_cam,
        focal_length=focal_length,
        principal_point=principal_point,
        distortion=distortion,
        distortion_model=(
            np.array([keyframes.distortion_model[0]]).repeat(len(world_from_cam), 0)
            if keyframes.distortion_model is not None
            else None
        ),
        image_size_wh=(
            keyframes.image_size_wh[[0]].repeat(len(world_from_cam), 1) if keyframes.image_size_wh is not None else None
        ),
        camera_space=keyframes.camera_space,
    )


def _interpolate_between_values(vals: torch.Tensor, fractions: torch.Tensor):
    if vals is None:
        return None
    fractions = fractions.to(vals)
    frames = torch.stack(
        [
            torch.stack(
                [torch.lerp(vals[i], vals[i + 1], x) for x in fractions],
            )
            for i in range(len(vals) - 1)
        ],
    )
    frames = frames[:, :-1, ...].reshape(-1, *vals.shape[1:])
    # Add the last frame
    frames = torch.cat([frames, vals[-1][None]], dim=0)
    return frames


def _interpolate_between_poses(poses: torch.Tensor, fractions: torch.Tensor):
    num_poses = poses.shape[0]
    fractions = fractions.to(poses)
    ts = transforms.translation(poses)
    # Interpolate between translations
    ts = _interpolate_between_values(ts, fractions)
    # Operations are done in double precision to avoid numerical issues.
    quats = transforms.quaternion_from_matrix(transforms.rotation(poses).to(torch.float64))
    # Interpolate between quaternions. Note that the interpolation is done in the log space.
    quats = torch.stack(
        [
            torch.stack(
                [_quat_slerp(quats[i], quats[i + 1], x) for x in fractions],
            )
            for i in range(num_poses - 1)
        ],
    )
    quats = quats[:, :-1, ...].reshape(-1, *quats.shape[2:])
    # Add the last frame
    quats = torch.cat([quats, quats[-1][None]], dim=0)

    # Convert back to matrices
    rotmats = transforms.matrix_from_quarternion(quats)
    return transforms.from_rotation_translation(rotmats.to(poses.dtype), ts)


def _quat_slerp(
    quat0: torch.Tensor,
    quat1: torch.Tensor,
    fraction: float,
    shortestpath: bool = True,
) -> torch.Tensor:
    """Return spherical linear interpolation between two quaternions."""
    if fraction == 0.0:
        return quat0
    if fraction == 1.0:
        return quat1
    # Normalize the input quaternions
    quat0 = _normalize_vector(quat0)
    quat1 = _normalize_vector(quat1)

    # Compute the dot product between the quaternions
    dot = torch.dot(quat0, quat1)

    # If shortestpath is true, adjust signs to ensure the shortest path
    if shortestpath and dot < 0.0:
        quat1 = -quat1
        dot = -dot

    # Clamp dot product to be within the range of acos
    dot = torch.clamp(dot, -1.0, 1.0)

    # Calculate the angle between the quaternions
    theta_0 = torch.acos(dot)  # theta_0 is the angle between input vectors

    # Compute the sin of the angles
    sin_theta_0 = torch.sin(theta_0)

    # Handle the case when the angle is very small to avoid division by zero
    if sin_theta_0.abs() < 1e-6:
        return (1.0 - fraction) * quat0 + fraction * quat1

    # Compute interpolation
    s0 = torch.sin((1.0 - fraction) * theta_0) / sin_theta_0
    s1 = torch.sin(fraction * theta_0) / sin_theta_0

    return s0 * quat0 + s1 * quat1


def add_sine_noise(
    world_from_camera: torch.Tensor,
    magnitude: float,
    n_cycles: int,
    direction: torch.Tensor = torch.tensor([1.0, 0.0, 0.0]),
):
    """Adds sine noise to camera center."""
    direction = _normalize_vector(direction)
    x = torch.linspace(0, 2 * math.pi * n_cycles, world_from_camera.shape[0], device=world_from_camera.device)
    translation = magnitude * torch.sin(x).unsqueeze(-1) * direction

    perturbation = transforms.from_rotation_translation(None, translation)
    return world_from_camera @ perturbation


def add_spiral_noise(
    world_from_camera: torch.Tensor,
    magnitude: float,
    n_cycles: int,
    zrate: float = 0.5,
):
    """Adds spiral noise to camera center."""
    theta = torch.linspace(0.0, 2.0 * torch.pi * n_cycles, len(world_from_camera))
    translation = [torch.cos(theta), -torch.sin(theta), -torch.sin(theta * zrate)]
    translation = torch.stack(translation, dim=-1) * magnitude

    perturbation = transforms.from_rotation_translation(None, translation)
    return world_from_camera @ perturbation


def _normalize_vector(vec: torch.Tensor) -> torch.Tensor:
    return vec / torch.norm(vec, dim=-1, keepdim=True)
