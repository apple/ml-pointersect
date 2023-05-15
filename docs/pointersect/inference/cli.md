# Pointersect Command Line Interface (CLI)

We provide two command line interfaces to render with pointersect.

### Installation

To use the CLI, we need to install the pip library.

```bash
pip install pointersect
```

or

```bash
# At the repo root (where [pyproject.toml](/pyproject.toml) is)
pip install .
```

Note that the first time the CLI are called or the package is imported,
it will try to compile a CUDA package (that we implement
to find neighbor points along a ray in the point cloud). Please
make sure a CUDA environment is set up.

### Simple point cloud rendering

The command [pointersect](/pointersect/inference/main.py:render_pcd_with_pointersect)
provide an easy-to-use interface to quickly render a point cloud.

```bash
# In terminal
# Suppose the point cloud is in your_pcd.ply. The command renders 
# the point cloud with a pre-determined spiral camera trajectory and 
# parameters.
pointersect --input_point_cloud your_pcd.ply --output_dir out 
```

##### Details

We use [fire](https://github.com/google/python-fire) to expose the
[functions](/pointersect/inference/main.py) to the command line.

Here is a brief description of the options:

###### input_point_cloud

The filename of a ply file containing the point cloud.

###### output_dir

Folder to store all rendering outputs.

###### model_filename

A pytorch checkpoint file containing the pretrained pointersect model.  
If `None`, the function uses the default pointersect model.

###### k

Number of neighboring points used in the rendering per ray

###### output_camera_trajectory

Camera trajectory to render the point cloud.
It can be a string to use a preset trajectory
(see :py:`pointersect.structures.CameraTrajectory`),
or it can be a json file containing the camera pose
in the world coordinate to be interpolated.
If `None`, it uses `spiral` with 36 images.

    The json file should contain:
        H_c2w:
            (b, q, 4, 4), a nested list containing the camera pose in the world coord.
            `b` is the batch dimension, `q` is number of camera poses in a batch.
            For example, `H_c2w[i,j]` is the 4x4 camera pose matrix that converts
            a point in the camera coordinate to the world coordinate.

###### fov

The horizontal field of view in degree of the output images.

###### width_px

The number of horizontal pixels in the rendered image.

###### height_px

The number of vertial pixels in the rendered image.

###### n_output_imgs

Total number of output images. It will interpolate the camera trajectory uniformly.
If `None` and `output_camera_trajectory` is provided as a json file, it will use the
camera poses in the json file.

###### ray_chunk_size_ratio

The setting controls the number of rays to render per batch. We use some simple
logic to determine the number to try to fill the memory as much as possible.
If run out of memory, please decrease the number. For example, if set to `0.5`,
it will render half of the number of rays in a batch at once.

###### render_surfel

Whether to render the point cloud using visibility splatting.

###### render_poisson

Whether to render the point cloud by first reconstructing a mesh using
screened poisson reconstruction.

#### Note

Since the function calls the [render_point_cloud](/pointersect/inference/main.py:render_point_cloud)
function, you can also provide additional arguments for `render_point_cloud` in
the command line to further control the rendering.

#### Output

The function outputs the following to the assigned folder.

```python
├── compare  # The folder contains stitched results for easy comparison
│     ├── batch_0  # individual frames of depth, surface normal, and rgb estimation
│     ├── depth  # estimated depth video
│     ├── hit_map  # estimation of whether a ray hit a surface 
│     ├── normal_w  # estimated surface normal map video
│     └── rgb  # estimated rgb video
├── input_pcd  # input point cloud seens by the model
│     └── pcd_0.ply
├── output_camera  # output camera 
│     ├── batch_0
│     ├── batch_0.obj
│     └── state_dict.pt
├── pointersect  # Rendering results of pointersect. Same format as above.
│     ├── batch_0
│     ├── depth
│     ├── hit_map
│     ├── model_info.json
│     ├── normal_w
│     ├── rgb
│     └── total_time.json
├── poisson  # Rendering results of poisson reconstruction. Same format as above.
│     ├── batch_0
│     ├── depth
│     ├── hit_map
│     ├── normal_w
│     ├── reconstructed_mesh.ply
│     ├── rgb
│     └── total_time.json
├── surfel  # Rendering results of visibility splatting. Same format as above.
│     ├── batch_0
│     ├── depth
│     ├── hit_map
│     ├── normal_w
│     ├── rgb
│     └── total_time.json
└── timing.json
```

### Full functionality

The command [pointersect_full](/pointersect/inference/main.py)
provide detail controls that we used to render results in the paper.
It calls functions in [the file](/pointersect/inference/main.py)
to render a mesh, a batch of meshes, a point cloud, a batch of point clouds, etc.

```bash
# In terminal
pointersect_full --config_filename config.yaml 
```

In the config file, there is a field `procedure` whose value
is used to call the corresponding function in [the file](/pointersect/inference/main.py).

The rest of the config file details the arguments that will be given to
the corresponding function called by the `procedure`.


### Examples 

We provide several [examples](/assets/examples) to use the command line interface.
