# Pointersect: Neural Rendering with Cloud-Ray Intersection

[[Website](https://machinelearning.apple.com/research/pointersect)] 
[[Paper](https://arxiv.org/abs/2304.12390)] 
[[Docs](/assets/docs/README.md)] 
[[Examples](/assets/examples)] 
[[Videos](/assets/videos)] 

Pointersect is a **plug-and-play** method for rendering point clouds.
It is differentiable and does not require per-scene optimization.


## Usage

Given a point cloud and a ray, pointersect returns:

- the intersection point between the ray and the underlying surface represented by the point cloud;
- surface normal at the intersection point; and
- color/material interpolation weights of neighboring points.

You can use point clouds containing only xyz---neither color nor vertex normal is needed.

[<img src="/assets/img/pointersect_output.png" width="66%">](https://mlr.cdn-apple.com/video/zebra_video_78ac0aabc5.mp4)



## Examples

1. We use the surface normal estimated by pointersect to relight a point cloud.

<img src="/assets/img/pointersect_relight.png" width="66%">

2. Even though pointersect is designed to render clean point clouds, here is an example where we use a pretrained pointersect model 
to render a lidar-scanned point cloud without any optimization.

[<img src="/assets/img/pointersect_preview.png" width="66%">](https://mlr.cdn-apple.com/video/lidar_results_9a7bf55e95.mp4)


3. Edit and render without re-optimization.

[<img src="/assets/img/pointersect_3dedit.png" width="66%">](https://mlr.cdn-apple.com/video/lidar_results_9a7bf55e95.mp4)


4. We use pointersect with path tracing to render the global illumination of a scene. 

<img src="/assets/img/pointersect_globallight.png" width="66%">



## How to use 

You can use pointersect by installing the pypi package:
```bash
pip install pointersect
```
or 
```bash
# in the repo root 
pip install .
```

We provide API, command line tool, and training script for using pointersect. 
See [the documentation](/docs/README.md) for instructions.

We also provide a few [examples](/assets/examples) if you want to jump in directly :) 


## Citation

If you use this software package, please cite our paper:

```
@inproceedings{chang2023pointersect,
  author={Jen-Hao Rick Chang and Wei-Yu Chen and Anurag Ranjan and Kwang Moo Yi and Oncel Tuzel},
  title={Pointersect: Neural Rendering with Cloud-Ray Intersection},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year = {2023}
}
```
