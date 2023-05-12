# Example scene

We provide a down-sampled point cloud from [ARKitScenes dataset](https://github.com/apple/ARKitScenes). 
The original point cloud is scanned by a high-resolution lidar. 
For the file `pcd_0.ply`, we perform volume downsampling with `cell_width` equal to 0.03. 
For the file `pcd_0_highres.ply`, we perform volume downsampling with `cell_width` equal to 0.01. 

To render a quick preview:
```bash
# in the folder
pointersect --input_point_cloud pcd_0.ply --output_dir out  --render_poisson 0 --output_camera_trajectory cam_pose.pt --th_hit_prob 0 --output_camera_setting '{"fov": 60, "width_px": 256, "height_px": 192}' --k 100  --n_output_imgs 20 
```

