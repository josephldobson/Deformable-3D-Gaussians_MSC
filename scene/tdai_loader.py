import sys
import os
from tqdm import tqdm
from PIL import Image
from utils.graphics_utils import BasicPointCloud

import dataclasses
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal, Optional, Protocol, Tuple
from plyfile import PlyData, PlyElement
from typing import Any, Callable, ClassVar
import inspect

import h5py
import numpy as np
import pandas as pd
import torch
from jaxtyping import Bool, Float, Int, Int64, Shaped, UInt8
from overrides import override
from typing_extensions import Self

from utils import ns_camera_utils as camera_utils
from utils.nest import *
from typing import NamedTuple
from utils import tdai_transforms as transforms
from utils import tdai_cameras as cameras
from utils.graphics_utils import getWorld2View2
import json
from utils.graphics_utils import focal2fov, fov2focal


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    fid: float
    depth: Optional[np.array] = None
    pointcloud_camera: np.array = None

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

class TensorDataclass:
    # Ensure conformance with the DataclassInstance protocol.
    __dataclass_fields__: ClassVar[dict[str, dataclasses.Field[Any]]]

    @classmethod
    def build(cls, **kwargs):
        self = cls.make_empty()
        return self.replace(**kwargs)

    @classmethod
    def make_empty(cls) -> Self:
        return cls(**{f.name: None for f in dataclasses.fields(cls)})

    @classmethod
    def collate(cls, *items):
        def _collate(*items):
            if items[0] is None:
                return None
            if isinstance(items[0], (str, bool, int, float)):
                assert all(item is None or item == items[0] for item in items), f"Cannot collate {items}"
                return items[0]
            if isinstance(items[0], np.ndarray):
                return np.concatenate(items, axis=0)
            if isinstance(items[0], pd.DataFrame):
                return pd.concat(items, axis=0)
            if isinstance(items[0], TensorDataclass):
                raise NotImplementedError(f"Cannot collate {type(items[0])}")
            if isinstance(items[0], torch.Tensor):
                return torch.cat(items, dim=0)

            raise ValueError(f"Cannot collate {type(items[0])}")

        return cls.build(**map_structure(_collate, *[item.as_dict(shallow=True) for item in items]))

    def replace(self: Self, **kwargs) -> Self:
        return dataclasses.replace(self, **kwargs)

    def __getitem__(self: Self, index) -> Self:
        return self.apply_notnull(lambda item: self._index_field(item, index))

    def _index_field(self, item, index):
        if isinstance(item, (str, bool, int, float, Path)):
            return item

        if isinstance(item, TensorDataclass):
            return item[index]

        if isinstance(item, pd.DataFrame):
            return item.iloc[index]

        # Handle [::-1]
        if isinstance(item, torch.Tensor) and isinstance(index, slice) and index == slice(None, None, -1):
            return item.flip(0)

        return item[index]

    def apply(self: Self, fn: Callable, _unwrapped_fn=None, recursive=False) -> Self:
        source_d = self.as_dict(shallow=True)
        has_name = "name" in inspect.getfullargspec(fn).args
        for name, item in list(source_d.items()):
            if recursive and isinstance(item, TensorDataclass):
                source_d[name] = item.apply(fn, _unwrapped_fn=_unwrapped_fn, recursive=recursive)
                continue
            try:
                source_d[name] = fn(item, name=name) if has_name else fn(item)
            except Exception as e:
                raise RuntimeError(f"Error applying {_unwrapped_fn or fn} to {name}") from e

        return self.__class__(**source_d)

    def apply_notnull(self: Self, fn) -> Self:
        has_name = "name" in inspect.getfullargspec(fn).args
        if has_name:
            return self.apply(
                lambda item, name: fn(item, name=name) if item is not None else None,
                _unwrapped_fn=fn,
            )

        return self.apply(lambda item: fn(item) if item is not None else None, _unwrapped_fn=fn)

    def as_dict(self, shallow=False):
        if shallow:
            return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        return dataclasses.asdict(self)

    def shape_repr(self) -> str:
        out = [f"{self.__class__.__name__}:"]
        shapes_d = map_structure(
            lambda t: (*t.shape, t.dtype if hasattr(t, "dtype") else ()) if hasattr(t, "shape") else t,
            self.as_dict(),
        )
        out += [f" {key}: {shape}" for key, shape in shapes_d.items()]
        return "\n".join(out)

    def float(self):
        return self.apply(
            lambda x: x.float() if isinstance(x, torch.Tensor) and x.dtype == torch.float64 else x,
            recursive=True,
        )

    def copy(self):
        def _copy_field(x):
            if isinstance(x, (np.ndarray, pd.DataFrame, dict, list)):
                return x.copy()
            if isinstance(x, torch.Tensor):
                return x.clone()
            return x

        return self.apply(_copy_field, recursive=True)


def get_focal_lengths(intrmat: Float[torch.Tensor, "... 3 3"]) -> Float[torch.Tensor, "... 2"]:
    if intrmat.ndim == 2:
        intrmat = intrmat[None]
    return intrmat[..., [0, 1], [0, 1]]


def get_principal_point(intrmat: Float[torch.Tensor, "... 3 3"]) -> Float[torch.Tensor, "... 2"]:
    if intrmat.ndim == 2:
        intrmat = intrmat[None]
    return intrmat[..., :2, 2]


def get_flat(intrmat: Float[torch.Tensor, "... 3 3"]) -> Float[torch.Tensor, "... 4"]:
    return torch.cat([get_focal_lengths(intrmat), get_principal_point(intrmat)], dim=-1)


def from_f_c(
    focal_lengths: Float[torch.Tensor, "... 2"],
    principal_points: Float[torch.Tensor, "... 2"],
) -> Float[torch.Tensor, "... 3 3"]:
    if focal_lengths.ndim == 1:
        focal_lengths = focal_lengths[None]
    if principal_points.ndim == 1:
        principal_points = principal_points[None]
    # Create diagonal elements with focal lengths
    fx = focal_lengths[..., 0]
    fy = focal_lengths[..., 1]
    zeros = torch.zeros_like(fx)  # Create a tensor of zeros for efficiency

    # Create intrinsic matrices
    intrinsic_matrices = torch.stack(
        [
            fx,
            zeros,
            principal_points[..., 0],
            zeros,
            fy,
            principal_points[..., 1],
            zeros,
            zeros,
            torch.ones_like(fx),  # Last row: [0, 0, 1]
        ],
        dim=-1,
    )  # Stack along the last dimension

    # Reshape for the correct batch dimension
    return intrinsic_matrices.reshape(*fx.shape, 3, 3)


def from_flat(
    intrinsics: Float[torch.Tensor, "... 4"],
) -> Float[torch.Tensor, "... 3 3"]:
    return from_f_c(
        intrinsics[..., :2],
        intrinsics[..., 2:],
    )



class CameraProtocol(Protocol):
    world_from_cam: Float[torch.Tensor, "#batch 4 4"]

    focal_length: Float[torch.Tensor, "#batch 2"]
    principal_point: Float[torch.Tensor, "#batch 2"]
    image_size_wh: Int[torch.Tensor, "#batch 2"]

    distortion_model: Shaped[np.ndarray, "#batch"]
    distortion: Float[torch.Tensor, "#batch 6"]

    camera_space: str

    def replace(self: Self, **kwargs) -> Self: ...

    def build(self: Self, **kwargs) -> Self: ...

    def float(self) -> Self: ...


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

@dataclass(frozen=True)
class Points(TensorDataclass):
    xyz: Float[torch.Tensor, "#num_points 3"]
    rgb: UInt8[torch.Tensor, "#num_points 3"]
    error: Float[torch.Tensor, "#num_points"]
    visibility: Bool[torch.Tensor, "#num_points #batch"]
    """Whether the point is visible in each camera."""

    @property
    def visible(self) -> Self:
        return self[self.visibility.sum(-1) > 0]

    def __len__(self):
        return len(self.xyz)

    def serialize(self, grp: h5py.Group):
        for key, value in self.as_dict(shallow=True).items():
            _serialize_field(grp, key, value)

    @staticmethod
    def deserialize(grp: h5py.Group):
        fields = {key: torch.tensor(grp[key][()]) for key in grp.keys()}
        return Points.build(**fields)


@dataclass(frozen=True)
class Scene(TensorDataclass):
    world_from_cam: Float[torch.Tensor, "#batch 4 4"]
    camera_name: Shaped[np.ndarray, "#batch"]  # string or int
    timestamp_usec: Int64[np.ndarray, "#batch"]

    focal_length: Float[torch.Tensor, "#batch 2"]
    principal_point: Float[torch.Tensor, "#batch 2"]
    image_size_wh: Int[torch.Tensor, "#batch 2"]

    distortion_model: Shaped[np.ndarray, "#batch"]
    distortion: Float[torch.Tensor, "#batch 6"]
    """Contents of this tensor are dependent on the distortion_model.

    for OPENCV: [k1, k2, p1, p2, 0, 0]
    """

    # Image data. All images are stored as arrays of paths to the image files.
    rgb: Shaped[np.ndarray, "#batch"]
    ignore_mask: Shaped[np.ndarray, "#batch"]
    """Mask of pixels to ignore in the image. 0 is ignore, 1 is keep."""
    semseg: Shaped[np.ndarray, "#batch"]
    depth: Shaped[np.ndarray, "#batch"]

    depth_alignment: Float[torch.Tensor, "#batch 3"]
    """Depth alignment coefficients, [shift, scale, error]."""

    points: Points

    df: pd.DataFrame

    world_space: str
    """Defines which way the gravity points."""
    camera_space: str
    """The coordinate space of the camera. The default for tdai datasets is RDF (RIGHT DOWN FORWARD)."""

    @cached_property
    def unique_timestamps(self) -> np.ndarray:
        return np.unique(self.timestamp_usec)

    @cached_property
    def unique_camera_names(self) -> np.ndarray:
        return np.unique(self.camera_name)

    @cached_property
    def time_index(self) -> np.ndarray:
        """Reindexes the timestamps to be 0, 1, 2, ..."""
        return np.searchsorted(self.unique_timestamps, self.timestamp_usec)

    @cached_property
    def front_camera(self) -> Self:
        return self[self.camera_name == 1]

    @property
    def depth_shift(self) -> Float[torch.Tensor, "#batch"]:
        return self.depth_alignment[..., 0]

    @property
    def depth_scale(self) -> Float[torch.Tensor, "#batch"]:
        return self.depth_alignment[..., 1]

    @property
    def intrinsics(self) -> Float[torch.Tensor, "#batch 3 3"]:
        return from_f_c(self.focal_length, self.principal_point)

    def is_undistorted(self) -> bool:
        return self.distortion_model is None or self.distortion is None or torch.all(self.distortion == 0.0).item()

    def argsort(self, by_camera_name: bool = False) -> np.ndarray:
        """Return the indices that would sort the scene by timestamp."""
        if by_camera_name:
            return np.lexsort([self.timestamp_usec, self.camera_name])
        return np.lexsort([self.camera_name, self.timestamp_usec])

    def sorted(self, by_camera_name: bool = False) -> Self:
        """Sort the scene by timestamp, camera_name or camera_name, timestamp."""
        idx = self.argsort(by_camera_name)
        return self[idx]

    @override
    def _index_field(self, item, index):
        if isinstance(item, Points):
            # When slicing points, slice only the images dimension.
            return item.replace(visibility=item.visibility[:, index] if item.visibility is not None else None)
        return super()._index_field(item, index)

    def __len__(self):
        if self.world_from_cam is not None:
            return len(self.world_from_cam)
        if self.timestamp_usec is not None:
            return len(self.timestamp_usec)
        if self.rgb is not None:
            return len(self.rgb)
        raise ValueError("Scene has no length")

    def __iter__(self):
        for i in range(len(self)):
            yield self[[i]]

    def cuda(self):
        return self.apply(lambda x: x.cuda() if isinstance(x, torch.Tensor) else x, recursive=True)

    def scale_world(self, scale: float):
        world_from_cam = self.world_from_cam.clone()
        world_from_cam[:, :3, 3] *= scale
        return self.replace(
            world_from_cam=world_from_cam,
            points=self.points.replace(xyz=self.points.xyz * scale) if self.points is not None else None,
        )

    def transform(self, transform_matrix: Float[torch.Tensor, "1 4 4"]) -> Self:
        if transform_matrix.ndim == 2:
            transform_matrix = transform_matrix[None]
        world_from_cam = transform_matrix @ self.world_from_cam
        return self.replace(
            world_from_cam=world_from_cam,
            points=self.points.replace(
                xyz=transforms.apply(transform_matrix.to(self.points.xyz), self.points.xyz),
            )
            if self.points is not None
            else None,
        )

    def serialize(self, filename: Path):
        with h5py.File(str(filename), "w") as f:
            for key, value in self.as_dict(shallow=True).items():
                if value is None:
                    continue
                if key in ["rgb", "depth", "semseg", "ignore_mask"]:
                    _serialize_field(
                        f,
                        key,
                        np.array([str(path.resolve().relative_to(filename.parent.resolve())) for path in value]),
                    )
                elif key == "points":
                    value.serialize(f.create_group("points"))
                elif key == "df":
                    pass
                else:
                    _serialize_field(f, key, value)
        if self.df is not None:
            self.df.to_hdf(str(filename), key="df")

    @staticmethod
    def deserialize(filename: Path) -> "Scene":
        assert filename.exists() and filename.is_file(), f"File {filename} does not exist"
        with h5py.File(str(filename), "r") as f:
            torch_fields = [
                "world_from_cam",
                "focal_length",
                "principal_point",
                "distortion",
                "image_size_wh",
                "depth_alignment",
                "g_world_from_cam",  # legacy
            ]
            numpy_fields = ["camera_name", "timestamp_usec", "distortion_model"]
            path_fields = ["rgb", "depth", "semseg", "ignore_mask"]
            scalar_fields = ["world_space", "camera_space"]
            compound_fields = ["points", "df"]
            renames = {"g_world_from_cam": "world_from_cam"}

            # Check that the above definition accounts for all fields.
            all_fields = torch_fields + numpy_fields + path_fields + scalar_fields + compound_fields
            class_fields = [field.name for field in dataclasses.fields(Scene)]
            assert len(class_fields) == len(
                set(all_fields) - set(renames.keys()),
            ), f"Fields mismatch: missing: {set(class_fields) - set(all_fields) - set(renames.keys())}, "
            f"extra: {set(all_fields) - set(renames.keys()) - set(class_fields)}"

            fields = {key: torch.tensor(f[key][()]) for key in f.keys() if key in torch_fields}
            fields.update({key: f.attrs[key] for key in f.attrs.keys() if key in scalar_fields})
            fields.update({key: f[key][()] for key in f.keys() if key in (numpy_fields + path_fields)})
            fields.update(
                {
                    key: np.char.decode(c)
                    for key, c in fields.items()
                    if isinstance(c, np.ndarray) and c.dtype.kind == "S"
                },
            )
            fields.update(
                {
                    key: np.array([(filename.parent / path).resolve() for path in fields[key]])
                    for key in path_fields
                    if key in fields
                },
            )
            fields = {renames.get(key, key): value for key, value in fields.items()}
            points = Points.deserialize(f["points"]) if "points" in f else None
        df = pd.read_hdf(str(filename), key="df") if "df" in f else None
        return Scene.build(points=points, df=df, **fields)

    def as_row(self, row_idx: Optional[int] = None) -> "_SceneRow":
        """Returns a single row representation of the Scene."""

        def _squeeze_item(array):
            if array is None:
                return None
            if isinstance(array, (np.ndarray, torch.Tensor)):
                item = array.squeeze(0)
                return item if item.ndim > 0 else item.item()
            return self._index_field(array, 0)

        slc = self[[row_idx]] if row_idx is not None else self
        return _SceneRow(**map_structure(_squeeze_item, slc.as_dict(shallow=True)))

    def iterrows(self):
        for i in range(len(self)):
            yield self.as_row(i)


# A copy of the Scene dataclass with the batch dimension and any methods removed.
_SceneRow = dataclasses.make_dataclass(
    "SceneRow",
    [(field.name, field.type) for field in dataclasses.fields(Scene)],
)


def get_base_dir(scene: Scene) -> Path:
    assert "cam_" in scene.rgb[0].parent.name
    assert "images" == scene.rgb[0].parent.parent.name
    return scene.rgb[0].parent.parent.parent


def subsample_spatially(dist_m: np.ndarray, step_m: float = 0.5):
    """Subsample the distance array to have a consistent step size.

    Returns the indices of the subsampled array."""
    assert dist_m[0] == 0

    dists = np.arange(0, dist_m[-1], step_m)
    idxs = _find_nearest(dist_m, dists)
    idxs = np.unique(idxs)  # remove duplicates.
    return idxs


def _find_nearest(a: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Like searchsorted, but returns the position of the closest element in a."""
    idx = np.searchsorted(a, v)
    idx[idx == len(a)] = len(a) - 1
    idx[v - a[idx] > a[idx + 1] - v] += 1
    return idx


def distance_between_cameras(scene: Scene) -> Float[torch.Tensor, "#batch"]:
    """Compute the distance between the cameras."""
    return torch.norm(
        transforms.translation(scene.world_from_cam[1:] - scene.world_from_cam[:-1]),
        dim=-1,
    )


def auto_orient_and_center_poses(
    world_from_cam: Float[torch.Tensor, "*batch 4 4"],
    orientation_method: Literal["pca", "up", "vertical", "none"] = "up",
    center_method: Literal["poses", "focus", "none"] = "poses",
) -> Tuple[Float[torch.Tensor, "*batch 4 4"], Float[torch.Tensor, "1 4 4"]]:
    poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
        world_from_cam,
        method=orientation_method,
        center_method=center_method,
    )
    poses = transforms.convert_3x4_to_4x4(poses)
    transform_matrix = transforms.convert_3x4_to_4x4(transform_matrix)
    return poses, transform_matrix


def auto_scale_poses(
    world_from_cam: Float[torch.Tensor, "*batch 4 4"],
) -> Tuple[Float[torch.Tensor, "*batch 4 4"], float]:
    world_from_cam = world_from_cam.clone()
    scale = 1.0 / torch.max(torch.abs(transforms.translation(world_from_cam))).item()
    world_from_cam[..., :3, 3] *= scale
    return world_from_cam, scale


def _serialize_field(f, key: str, value):
    if value is None:
        return
    if isinstance(value, (torch.Tensor, np.ndarray)):
        if str(value.dtype).startswith("<U"):
            f.create_dataset(key, data=value.astype("S"), compression="gzip")
        else:
            f.create_dataset(
                key,
                data=value.cpu().numpy() if isinstance(value, torch.Tensor) else value,
                compression="gzip",
            )
    else:
        f.attrs[key] = value

def pad_poses(p):
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p):
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]


def transform_poses_pca(poses, fix_radius=0):
    """Transforms poses so principal components lie on XYZ axes.

  Args:
    poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

  Returns:
    A tuple (poses, transform), with the transformed poses and the applied
    camera_to_world transforms.
    
    From https://github.com/SuLvXiangXin/zipnerf-pytorch/blob/af86ea6340b9be6b90ea40f66c0c02484dfc7302/internal/camera_utils.py#L161
  """
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    # Sort eigenvectors in order of largest to smallest eigenvalue.
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = unpad_poses(transform @ pad_poses(poses))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    # Flip coordinate system if z component of y-axis is negative
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    # Just make sure it's it in the [-1, 1]^3 cube
    if fix_radius>0:
        scale_factor = 1./fix_radius
    else:
        scale_factor = 1. / (np.max(np.abs(poses_recentered[:, :3, 3])) + 1e-5)
        scale_factor = min(1 / 10, scale_factor)

    poses_recentered[:, :3, 3] *= scale_factor
    transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform

    return poses_recentered, transform, scale_factor

def cam_traj(scene_path):
    ss = Scene.deserialize(scene_path).float()
    ss = ss[ss.camera_name == 1]

    idxs = torch.tensor([transforms.translation(ss.world_from_cam)[:,2].argmin(), transforms.translation(ss.world_from_cam)[:,2].argmax()])
    start_end = ss[idxs]
    start_end.world_from_cam[0, :3, :3] = start_end.world_from_cam[1:, :3, :3]
    traj = cameras.interpolate_cameras(start_end, 250, easing="sine")
    traj = traj.replace(world_from_cam=cameras.add_spiral_noise(traj.world_from_cam, 2, 3))
    return traj

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def readTDAISceneInfo(args):
    folder_path = args.source_path
    scene_path = os.path.join(folder_path, "scene.h5")
    scene = Scene.deserialize(Path(scene_path))

    cam_infos = []
    points = []
    point_colors = []
    points_time = []
    cam_num = len(scene.unique_camera_names)
    frame_num = len(scene.timestamp_usec)

    cam_indices = np.zeros((len(scene.front_camera.camera_name), cam_num), dtype=int)
    for i in range(0, cam_num):
        cam_indices[:,i,None]=np.argwhere(scene.camera_name == scene.unique_camera_names[i])

    for j in range(0,int(frame_num/cam_num)):

        c2w=np.array([]).reshape((0,scene.world_from_cam[0].detach().cpu().numpy().shape[0], scene.world_from_cam[0].detach().cpu().numpy().shape[1]))
        for i in range(0, cam_num):
            this_c2w=scene.world_from_cam[cam_indices[j,i]].detach().cpu().numpy()
            c2w=np.concatenate((c2w, this_c2w[None]), axis=0)
        w2c = np.linalg.inv(c2w)
        images = []
        image_paths = []
        HWs = []

        for i in range(0, cam_num):
            subdir='images/cam_'+str(i+1).zfill(2)+'/'
            image_path = os.path.join(folder_path, subdir+ str(scene.timestamp_usec[cam_indices[j,i]]).zfill(9)+'.jpeg')
            im_data = Image.open(image_path)
            W, H = im_data.size
            image = np.array(im_data) / 255.
            HWs.append((H, W))
            images.append(image)
            image_paths.append(image_path)
            
        timestamp = scene.timestamp_usec[cam_indices[j,0]]
        # point_xyz_world = scene.points.xyz.detach().cpu().numpy()
        # points.append(point_xyz_world)
        # point_colors.append(scene.points.rgb.detach().cpu().numpy())
        # point_time = np.full_like(point_xyz_world[:, :1], timestamp)
        # points_time.append(point_time)

        for i in range(0, cam_num):
            # point_camera = (np.pad(point_xyz_world, ((0, 0), (0, 1)), constant_values=1) @ w2c[i])[:, :3]
            R = np.transpose(w2c[i, :3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[i, :3, 3]
            K = scene.intrinsics[cam_indices[j,i]].detach().cpu().numpy()
            fx = float(K[0, 0])
            fy = float(K[1, 1])
            cx = float(K[0, 2])
            cy = float(K[1, 2])
            FovY = 2 * np.arctan(scene.image_size_wh[cam_indices[j,i]][0] / (2 * fy))
            FovX = 2 * np.arctan(scene.image_size_wh[cam_indices[j,i]][1] / (2 * fx))

            cam_infos.append(CameraInfo(uid=j * 5 + i, R=R, T=T, FovY=FovY, FovX=FovX,
                                        image=images[i]*255, 
                                        image_path=image_paths[i], image_name=str(i)+"_"+str(j),
                                        width=HWs[i][1], height=HWs[i][0], fid=timestamp,
                                        pointcloud_camera = None))
            
    pointcloud = scene.points.xyz.detach().cpu().numpy()
    pointcloud_colors = scene.points.rgb.detach().cpu().numpy()

    w2cs = np.zeros((len(cam_infos), 4, 4))
    Rs = np.stack([c.R for c in cam_infos], axis=0)
    Ts = np.stack([c.T for c in cam_infos], axis=0)
    w2cs[:, :3, :3] = Rs.transpose((0, 2, 1))
    w2cs[:, :3, 3] = Ts
    w2cs[:, 3, 3] = 1
    c2ws = unpad_poses(np.linalg.inv(w2cs))
    c2ws, transform, scale_factor = transform_poses_pca(c2ws, fix_radius=1)

    c2ws = pad_poses(c2ws)
    for idx, cam_info in enumerate(tqdm(cam_infos, desc="Transform data")):
        c2w = c2ws[idx]
        w2c = np.linalg.inv(c2w)
        cam_info.R[:] = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        cam_info.T[:] = w2c[:3, 3]
        # cam_info.pointcloud_camera[:] *= scale_factor
    pointcloud = (np.pad(pointcloud, ((0, 0), (0, 1)), constant_values=1) @ transform.T)[:, :3]
    print(pointcloud[100])
    print(pointcloud_colors[100])

    if args.eval:
        # ## for snerf scene
        # train_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // cam_num) % testhold != 0]
        # test_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // cam_num) % testhold == 0]

        # for dynamic scene
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // args.cam_num + 1) % args.testhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // args.cam_num + 1) % args.testhold == 0]
        
        # for emernerf comparison [testhold::testhold]
        if args.testhold == 10:
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // args.cam_num) % args.testhold != 0 or (idx // args.cam_num) == 0]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // args.cam_num) % args.testhold == 0 and (idx // args.cam_num)>0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    nerf_normalization['radius'] = 1/nerf_normalization['radius']

    ply_path = os.path.join(args.source_path, "points3d.ply")
    # if not os.path.exists(ply_path):
    #     rgbs = np.random.random((pointcloud.shape[0], 3))
    #     storePly(ply_path, pointcloud, rgbs, pointcloud_timestamp)
    # try:
    #     pcd = fetchPly(ply_path)
    # except:
    #     pcd = None
    storePly(ply_path, pointcloud, pointcloud_colors)


    pcd = BasicPointCloud(pointcloud, colors=pointcloud_colors, normals=None)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)

    return scene_info


# def readTDAISceneInfo(args):
#     folder_path = args.source_path
#     scene_path = os.path.join(folder_path, "scene.h5")
#     scene = Scene.deserialize(Path(scene_path))

#     cam_infos = []
#     points = []
#     point_colors = []
#     points_time = []

#     frame_num = len(scene.timestamp_usec)
#     max_t, min_t = scene.timestamp_usec.max(), scene.timestamp_usec.min()

#     ['camera_name', 'distortion', 'distortion_model', 'focal_length', 'ignore_mask', 'image_size_wh', 'points', 'principal_point', 'rgb', 'timestamp_usec', 'world_from_cam']

#     for id in range(0, frame_num):
#         uid = id
#         timestamp = (scene.timestamp_usec[id] - min_t) / (max_t - min_t)
        
#         # Extract camera parameters
#         world_from_cam = scene.world_from_cam[id]
#         R = world_from_cam[:3, :3].T  # Transpose for CUDA compatibility
#         T = world_from_cam[:3, 3]
        
#         intrinsics = scene.
#         fx, fy = intrinsics[0, 0], intrinsics[1, 1]
#         cx, cy = intrinsics[0, 2], intrinsics[1, 2]
#         width, height = scene.image_size[id]
        
#         # Calculate FovX and FovY
#         FovY = focal2fov(fy, height)
#         FovX = focal2fov(fx, width)
        
#         image = scene.images[id]
#         image_path = os.path.join(folder_path, f"images/{id:06d}_{j:02d}.jpg")
        
#         point_xyz = point_xyz_world = scene.points.xyz.detach().cpu().numpy()
#         point_colors = scene.points.rgb[id].detach().cpu().numpy()
        
#         # Append to global lists
#         points.append(valid_points)
#         point_colors.append(valid_colors)
#         points_time.append(np.full(len(valid_points), timestamp))
        
#         cam_infos.append(CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX,
#                                     image=image, 
#                                     image_path=image_path, image_name=f"{id:06d}_{j:02d}",
#                                     width=width, height=height, fid=timestamp,
#                                     pointcloud_camera=valid_points))

            
            
#     pointcloud = np.concatenate(points, axis=0)
#     pointcloud_colors = np.concatenate(point_colors, axis=0)

#     w2cs = np.zeros((len(cam_infos), 4, 4))
#     Rs = np.stack([c.R for c in cam_infos], axis=0)
#     Ts = np.stack([c.T for c in cam_infos], axis=0)
#     w2cs[:, :3, :3] = Rs.transpose((0, 2, 1))
#     w2cs[:, :3, 3] = Ts
#     w2cs[:, 3, 3] = 1
#     c2ws = unpad_poses(np.linalg.inv(w2cs))
#     c2ws, transform, scale_factor = transform_poses_pca(c2ws, fix_radius=args.fix_radius)

#     c2ws = pad_poses(c2ws)
#     for idx, cam_info in enumerate(tqdm(cam_infos, desc="Transform data")):
#         c2w = c2ws[idx]
#         w2c = np.linalg.inv(c2w)
#         cam_info.R[:] = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
#         cam_info.T[:] = w2c[:3, 3]
#         cam_info.pointcloud_camera[:] *= scale_factor
#     pointcloud = (np.pad(pointcloud, ((0, 0), (0, 1)), constant_values=1) @ transform.T)[:, :3]


#     if args.eval:
#         train_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // args.cam_num + 1) % args.testhold != 0]
#         test_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // args.cam_num + 1) % args.testhold == 0]
        
#         if args.testhold == 10:
#             train_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // args.cam_num) % args.testhold != 0 or (idx // args.cam_num) == 0]
#             test_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // args.cam_num) % args.testhold == 0 and (idx // args.cam_num)>0]
#     else:
#         train_cam_infos = cam_infos
#         test_cam_infos = []

#     nerf_normalization = getNerfppNorm(train_cam_infos)
#     nerf_normalization['radius'] = 1/nerf_normalization['radius']

#     ply_path = os.path.join(args.source_path, "points3d.ply")
#     storePly(ply_path, pointcloud, pointcloud_colors)

#     pcd = BasicPointCloud(pointcloud, colors=pointcloud_colors, normals=None)

#     scene_info = SceneInfo(point_cloud=pcd,
#                            train_cameras=train_cam_infos,
#                            test_cameras=test_cam_infos,
#                            nerf_normalization=nerf_normalization,
#                            ply_path=ply_path)

#     return scene_info