# Example point cloud from multiview stereo 

We captured an RGB video of our desk and used multiview stereo to create a point cloud. 

To render a quick preview:
```bash
pointersect --input_point_cloud pcd.ply --output_dir out --th_hit_prob 0 --output_camera_setting '{"fov": 30, "width_px": 300, "height_px": 300}' --output_camera_trajectory state_dict.pt --n_output_imgs 10 
```

To render a long video:
```bash
# in the folder
pointersect --input_point_cloud pcd.ply --output_dir out --th_hit_prob 0 --output_camera_setting '{"fov": 30, "width_px": 300, "height_px": 300}' --output_camera_trajectory state_dict.pt --n_output_imgs 200 
```