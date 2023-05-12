# Pointersect API

A pointersect model is designed to render clean point clouds, i.e., points that are sampled on the surfaces.

Pointersect model takes the following inputs:

- a point cloud containing `xyz: (n, 3)` in the world coordinate and optionally `rgb: (n, 3)`;
- query rays `origin: (m, 3)` and `direction: (m, 3)` in the world coordinate;

and it outputs

- intersection points (between query rays and the underlying surfaces): `(m, 3)`
- surface normal: `(m, 3)` in the world coordinate; and
- interpolation weights: `(m, k)` where `k` is a preset number of neighboring points used for each ray.

## Rendering functions

We provide a few convenient functions for utilizing pointersect:

[render_point_cloud_camera_using_pointersect](/pointersect/inference/infer.py):
renders a point cloud with given pinhole cameras.

```python
def render_point_cloud_camera_using_pointersect(
        model_filename: str,  # checkpoint of the pointersect model
        k: int,  # number of neighboring points to use per ray
        point_cloud: PointCloud,
        output_cameras: Camera,  # camera to produce the query rays 
        max_ray_chunk_size: int = int(4e4),  # used to chunk the query rays to avoid out of memory
        model: SimplePointersect = None,  # preloaded pointersect model
) -> PointersectRecord:
```

[render_point_cloud_ray_using_pointersect](/pointersect/inference/infer.py):
renders a point cloud given query rays.

```python
def render_point_cloud_ray_using_pointersect(
        model: SimplePointersect,  # preloaded pointersect model
        k: int,
        point_cloud: PointCloud,
        rays: Ray,  # query rays
        max_ray_chunk_size: int = int(4e4),
) -> T.Dict[str, T.Any]:
```

[intersect_pcd_and_ray](/pointersect/inference/infer.py#35):
is the main rendering function called by all other convenience functions.
It does not chunk the camera rays, so if the number of rays is too large,
the memory would run out.

```python
def intersect_pcd_and_ray(
        point_cloud: PointCloud,
        camera_rays: Ray,
        model: SimplePointersect,  # pointersect model
        k: int,  # number of neighbor points to use
) -> T.Dict[str, T.Union[PointersectRecord, T.Dict[str, torch.Tensor], torch.Tensor, None]]:
    """
    Given point cloud and camera rays, compute intersection points, surface normal, rgb, etc.
```

## Data structure

We store information (e.g., point cloud, ray, camera, mesh)
in [structures.py](/pointersect/inference/structures.py).
The classes are well documented and contain many useful functions
for processing and rendering point clouds and meshes.