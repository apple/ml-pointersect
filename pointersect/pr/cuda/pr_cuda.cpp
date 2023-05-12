//
// Copyright (C) 2022 Apple Inc. All rights reserved.
//

#include <torch/extension.h>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <chrono>
#include <math.h>
using namespace std::chrono;
using namespace torch::indexing;

// CUDA function declarations


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> get_grid_idx_cuda(
        const torch::Tensor & points,  // (b, n, 3), float
        const torch::Tensor & grid_size,  // (b, 3), long
        const torch::Tensor & center,  // (b, 3), float
        const torch::Tensor & grid_width,  // (b, 3), float
        const std::string & mode = "ind"  
);

std::vector<std::vector<torch::Tensor> > gather_points_cuda(
        const torch::Tensor & grid_idxs,  // (b, n), long
        const torch::Tensor & total_cells, // (b,) int64_t
        const torch::Tensor & valid_mask,  // (b, n), bool
        const torch::Tensor & cell_counts  // (b, n_cells), int32
);

std::tuple<torch::Tensor, torch::Tensor> gather_points_cuda_v2(
        const torch::Tensor & grid_idxs,  // (b, n), long
        const torch::Tensor & total_cells, // (b,) int64_t
        const torch::Tensor & valid_mask,  // (b, n), bool
        const torch::Tensor & cell_counts  // (b, n_cells), int32
);

std::tuple<torch::Tensor, torch::Tensor> grid_ray_intersection_cuda(
        const torch::Tensor & ray_origins,  // (b, m, 3), float
        const torch::Tensor & ray_directions,  // (b, m, 3), float
        const torch::Tensor & ray_radius,   //  (b, )m float
        const torch::Tensor & grid_size,  // (b, 3), long
        const torch::Tensor & grid_center,  // (b, 3) float
        const torch::Tensor & grid_width  // (b, 3) float
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> collect_points_on_ray_cuda(
        const torch::Tensor & ray2gidx,  // (b, m, n_gidxs)  long
        const torch::Tensor & n_ray2gidx,  // (b, m) long
        const torch::Tensor & gidx2pidx_bank,  // (b, n) long
        const torch::Tensor & gidx_start_idx,  // (b, n_cells+1)  long
        const torch::Tensor & points,  // (b, n, 3), float
        const torch::Tensor & ray_origins,  // (b, m, 3), float
        const torch::Tensor & ray_directions,  // (b, m, 3), float
        const torch::Tensor & ray_radius,   //  (b, ) float
        float t_min = 0.0,
        float t_max = 1.0e12
);


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> find_neighbor_points_of_rays_cuda(
        torch::Tensor points,  // (b, n, 3), float
        torch::Tensor ray_origins, // (b, m, 3), float
        torch::Tensor ray_directions,  // (b, m, 3), float
        torch::Tensor ray_radius,  // (b,), float
        torch::Tensor grid_size,  // (b, 3), long
        torch::Tensor grid_center,  // (b, 3), float
        torch::Tensor grid_width,  // (b, 3), float
        float t_min = 0.,
        float t_max = 1.e12
);


std::tuple<torch::Tensor, torch::Tensor> find_k_neighbor_points_of_rays_cuda(
        torch::Tensor points,  // (b, n, 3), float
        torch::Tensor ray_origins, // (b, m, 3), float
        torch::Tensor ray_directions,  // (b, m, 3), float
        torch::Tensor ray_radius,  // (b,), float
        torch::Tensor grid_size,  // (b, 3), long
        torch::Tensor grid_center,  // (b, 3), float
        torch::Tensor grid_width,  // (b, 3), float
        int64_t k, 
        float t_min = 0.,
        float t_max = 1.e12
);

torch::Tensor keep_min_k_values_cuda(
        torch::Tensor values,  // (b, n), float
        int64_t k
);


std::tuple<torch::Tensor, torch::Tensor> find_k_neighbor_points_of_rays_cuda_v2(
        torch::Tensor points,  // (b, n, 3), float
        torch::Tensor ray_origins, // (b, m, 3), float
        torch::Tensor ray_directions,  // (b, m, 3), float
        torch::Tensor ray_radius,  // (b,), float
        torch::Tensor grid_size,  // (b, 3), long
        torch::Tensor grid_center,  // (b, 3), float
        torch::Tensor grid_width,  // (b, 3), float
        int64_t k, 
        float t_min = 0.,
        float t_max = 1.e12
);


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> find_k_neighbor_points_of_rays_cuda_v3(
        torch::Tensor points,  // (b, n, 3), float
        torch::Tensor ray_origins, // (b, m, 3), float
        torch::Tensor ray_directions,  // (b, m, 3), float
        torch::Tensor ray_radius,  // (b,), float
        torch::Tensor grid_size,  // (b, 3), long
        torch::Tensor grid_center,  // (b, 3), float
        torch::Tensor grid_width,  // (b, 3), float
        torch::Tensor gidx2pidx_bank,  // (b, n) long or empty (,)
        torch::Tensor gidx_start_idx,  // (b, n_cell+1) int32 or empty (,)
        int64_t k, 
        float t_min = 0.,
        float t_max = 1.e12,
        bool refresh_cache = true
);


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> find_k_neighbor_points_of_rays_cuda_v4(
        torch::Tensor points,  // (b, n, 3), float
        torch::Tensor ray_origins, // (b, m, 3), float
        torch::Tensor ray_directions,  // (b, m, 3), float
        torch::Tensor ray_radius,  // (b,), float
        torch::Tensor grid_size,  // (b, 3), long
        torch::Tensor grid_center,  // (b, 3), float
        torch::Tensor grid_width,  // (b, 3), float
        torch::Tensor gidx2pidx_bank,  // (b, n) long or empty (,)
        torch::Tensor gidx_start_idx,  // (b, n_cell+1) int32 or empty (,)
        torch::Tensor valid_mask,  // (b, n) bool
        int64_t k, 
        float t_min = 0.,
        float t_max = 1.e12,
        bool refresh_cache = true
);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Compute the grid index given xyz_w.
// 
// Args:
//     points:
//         (b, n, 3)
//     grid_size:
//         (b, 3) long. number of grid cells in x y z.
//     center:
//         (b, 3)  center of the grid
//     grid_width:
//         (b, 3)  length (full width) of the grid in xyz
//     mode:
//         'subidx': return sub idx
//         'ind': return linear index
// 
// Returns:
//     grid_idx:
//         if mode == 'subidx':  (*, n, 3) long
//         elif mode == 'ind':   (*, n) long
//     valid_mask:
//         (b, n) bool
//     cell_counts:
//         (b, max_num_cells) int32.  number of points in each cell.  Note that
//         we use the max num_cells within the batch.  Since grid_size can be different 
//         for each b, some cells at the end will be zero.
// 
// Algorithm:
//     Let
//         x_from = center_x - grid_length_x / 2
//         x_to = center_x + grid_length_x / 2
//         cell_width_x = grid_length / grid_size_x
// 
//     We divide x = [x_from, x_to] into grid_size cells, each cell is of width x_cell.
// 
//         x_idx = ((x - x_from) / cell_width_x).floor().clamp(0, grid_size_x-1)
//         y_idx = ((y - y_from) / cell_width_y).floor().clamp(0, grid_size_y-1)
//         z_idx = ((z - z_from) / cell_width_z).floor().clamp(0, grid_size_z-1)
// 
//         grid_idx = z_idx + y_dix * grid_size_z + x_idx * (grid_size_y * grid_size_z)
// 
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> get_grid_idx(
        const torch::Tensor & points,  // (*, n, 3), float
        const torch::Tensor & grid_size,  // (*, 3), long
        const torch::Tensor & grid_center,  // (*, 3), float
        const torch::Tensor & grid_width,  // (*, 3), float
        const std::string & mode = "ind"  
) { 

    CHECK_INPUT(points);
    CHECK_INPUT(grid_size);
    CHECK_INPUT(grid_center);
    CHECK_INPUT(grid_width);

    return get_grid_idx_cuda(points, grid_size, grid_center, grid_width, mode);
}



// Gather the points belonging to each grid cell.
// 
// Args:
//     grid_idxs:
//         (b, n), grid linear index of each point
//     total_cells:
//         (b,) total grid cells
//     valid_mask:
//         (b, n), whether to record the point 
//     cell_counts: 
//         (b, max_num_cells), number of points in a cell
// 
// Returns:
//     vector of vector of tensors, b -> cell_idx -> pidx of points in the cell
// 
std::vector<std::vector<torch::Tensor> > gather_points_v1(
        const torch::Tensor & grid_idxs,  // (b, n), long
        const torch::Tensor & total_cells, // (b,) int64_t
        const torch::Tensor & valid_mask,  // (b, n), bool
        const torch::Tensor & cell_counts  // (b, n_cells), int32
) { 

    CHECK_INPUT(grid_idxs);
    CHECK_INPUT(total_cells);
    CHECK_INPUT(valid_mask);
    CHECK_INPUT(cell_counts);

    return gather_points_cuda(grid_idxs, total_cells, valid_mask, cell_counts);
}


// Gather the points belonging to each grid cell.
// 
// Args:
//     grid_idxs:
//         (b, n), grid linear index of each point
//     total_cells:
//         (b,) total grid cells
//     valid_mask:
//         (b, n), whether to record the point 
//     cell_counts: 
//         (b, max_num_cells), number of points in a cell
// 
// Returns:
//     membank: 
//         (b, n), should use the pidx of cell_idx is 
//         from cell_start_idx[b][cell_idx] to cell_counts[b][cell_idx+1] (excluded).
//     cell_start_idx:
//         (b, n_cells+1), 
// 
std::tuple<torch::Tensor, torch::Tensor> gather_points_v2(
        const torch::Tensor & grid_idxs,  // (b, n), long
        const torch::Tensor & total_cells, // (b,) int64_t
        const torch::Tensor & valid_mask,  // (b, n), bool
        const torch::Tensor & cell_counts  // (b, n_cells), int32
) { 

    CHECK_INPUT(grid_idxs);
    CHECK_INPUT(total_cells);
    CHECK_INPUT(valid_mask);
    CHECK_INPUT(cell_counts);


    return gather_points_cuda_v2(grid_idxs, total_cells, valid_mask, cell_counts);
}





// Compute the intersection between grid cells and the ray.
// 
// Args:
//     ray_origins:
//         (b, m, 3)
//     ray_directions:
//         (b, m, 3)
//     ray_radius:
//         (b,) 
//     grid_size:
//         (b, 3) long, xyz
//     grid_center:
//         (b, 3), xyz
//     grid_width:
//         (b, 3), xyz
// 
// Returns:
//     ray2gidx: 
//          (b, m, n_gidxs), gidx can be outside the grid
//     n_ray2gidx:
//          (b, m), number of neighbors each ray
// 
std::tuple<torch::Tensor, torch::Tensor> grid_ray_intersection(
        const torch::Tensor & ray_origins,  // (b, m, 3), float
        const torch::Tensor & ray_directions,  // (b, m, 3), float
        const torch::Tensor & ray_radius,   //  (b, )m float
        const torch::Tensor & grid_size,  // (b, 3), long
        const torch::Tensor & grid_center,  // (b, 3) float
        const torch::Tensor & grid_width  // (b, 3) float
) {
    CHECK_INPUT(ray_origins);
    CHECK_INPUT(ray_directions);
    CHECK_INPUT(ray_radius);
    CHECK_INPUT(grid_size);
    CHECK_INPUT(grid_center);
    CHECK_INPUT(grid_width);

    return grid_ray_intersection_cuda(
        ray_origins, 
        ray_directions, 
        ray_radius, 
        grid_size,
        grid_center,
        grid_width
    );
}





// Given ray2gidx and gidx2pidx, construct ray2pidxs.
// 
// Args:
//     ray2gidx: 
//          (b, m, n_gidxs), gidx can be outside the grid
//     n_ray2gidx:
//          (b, m), number of neighbors each ray
//     gidx2pidx_bank:
//         (b, n), should use the pidx of cell_idx is 
//         from cell_start_idx[b][cell_idx] to cell_counts[b][cell_idx+1] (excluded).
//     gidx_start_idx:
//         (b, n_cells+1), cell starts at gidx_start_idx[i] and ends at gidx_start_idx[i+1]
//     ray_origins:
//         (b, m, 3)
//     ray_directions:
//         (b, m, 3)
//     ray_radius:
//         (b,) 
//     max_num_points:
//         int
//     sort: 
//         bool, whether to sort the points based on the distance to the ray
// 
// Returns:
//     ray2pidx: 
//          (b, M), all the points. The pidxs of a ray is found via ray2pidx[b][ray_start_idx[m]:ray_start_idx[m+1]]
//     ray_start_idx:
//          (b, m+1), 
// 
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> collect_points_on_ray(
        const torch::Tensor & ray2gidx,  // (b, m, n_gidxs)  long
        const torch::Tensor & n_ray2gidx,  // (b, m) long
        const torch::Tensor & gidx2pidx_bank,  // (b, n) long
        const torch::Tensor & gidx_start_idx,  // (b, n_cells+1)  int32
        const torch::Tensor & points,  // (b, n, 3), float
        const torch::Tensor & ray_origins,  // (b, m, 3), float
        const torch::Tensor & ray_directions,  // (b, m, 3), float
        const torch::Tensor & ray_radius,   //  (b, )m float
        const float t_min = 0.,
        const float t_max = 1.0e12
) {

    CHECK_INPUT(ray2gidx);
    CHECK_INPUT(n_ray2gidx);
    CHECK_INPUT(gidx2pidx_bank);
    CHECK_INPUT(gidx_start_idx);
    CHECK_INPUT(points);
    CHECK_INPUT(ray_origins);
    CHECK_INPUT(ray_directions);
    CHECK_INPUT(ray_radius);

    return collect_points_on_ray_cuda(
        ray2gidx,
        n_ray2gidx,
        gidx2pidx_bank,
        gidx_start_idx,
        points,
        ray_origins, 
        ray_directions, 
        ray_radius,
        t_min,
        t_max
    );
}






// Find the points within `radius` of a ray, ie, the vertical distance from the point to ray <= radius.
// 
// Returns:
//     ray2pidx: 
//          (b, M), all the points. The pidxs of a ray is found via ray2pidx[b][ray_start_idx[m]:ray_end_idx[m]]
//     ray_start_idx:
//          (b, m), 
//     ray_end_idx:
//          (b, m)
// 
// Note:
//     Our algorithm is very simple. In order to be parallelized on gpu easily, we want to every thread to have
//     as few branching conditions (if/else) as possible.
// 
//     We will:
//     - parallelize on ray
//     - do the same thing for all rays.
//     - ignore all points outside the grid boundary
// 
// Algorithm:
// 
//     1. determine a grid
//     2. discretize the xyz of points -> calculate grid indices of each point
//     3. for each grid cell, gather points belonging to the cell
//     4. for each ray (px, py, pz, dx, dy, dz)
//             if dy.abs() < 1e-8:
//                 # use x=c plane
//             else:
//                 # use y=c plane
// 
//             - find plane-ray intersection point on each grid plane (xi, yi, zi)
//             - for each (xi, yi, zi)
//                 x_from = discretize(xi - radius)
//                 x_to = discretize(xi + radius)
//                 (same for y and z)
// 
//                 find grid idx for all of combinations, ie, get all grid cells
//                 surrounding the point.
// 
//     5. given grid idxs for each ray, gather point idxs
// 
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> find_neighbor_points_of_rays(
        torch::Tensor points,  // (b, n, 3), float
        torch::Tensor ray_origins, // (b, m, 3), float
        torch::Tensor ray_directions,  // (b, m, 3), float
        torch::Tensor ray_radius,  // (b,), float
        torch::Tensor grid_size,  // (b, 3), long
        torch::Tensor grid_center,  // (b, 3), float
        torch::Tensor grid_width,  // (b, 3), float
        float t_min = 0.,
        float t_max = 1.e12
) {
    
    CHECK_INPUT(points);
    CHECK_INPUT(ray_origins);
    CHECK_INPUT(ray_directions);
    CHECK_INPUT(ray_radius);
    CHECK_INPUT(grid_size);
    CHECK_INPUT(grid_center);
    CHECK_INPUT(grid_width);

    return find_neighbor_points_of_rays_cuda(
        points,  // (b, n, 3), 
        ray_origins, // (b, m, 3), float
        ray_directions,  // (b, m, 3), float
        ray_radius,  // (b,), float
        grid_size,  // (b, 3), long
        grid_center,  // (b, 3), float
        grid_width,  // (b, 3), float
        t_min,
        t_max
    );
}

// Find the k nearest points within `radius` of a ray, ie, the vertical distance from the point to ray <= radius.
// If not enough points found, return dummy index
// 
// Returns:
//     ray2pidx: 
//          (b, M), all the points. The pidxs of a ray is found via ray2pidx[b][ray_start_idx[m]:ray_end_idx[m]]
//     ray_start_idx:
//          (b, m), 
//     ray_end_idx:
//          (b, m)
// 
// Note:
//     Our algorithm is very simple. In order to be parallelized on gpu easily, we want to every thread to have
//     as few branching conditions (if/else) as possible.
// 
//     We will:
//     - parallelize on ray
//     - do the same thing for all rays.
//     - ignore all points outside the grid boundary
// 
// Algorithm:
// 
//     1. determine a grid
//     2. discretize the xyz of points -> calculate grid indices of each point
//     3. for each grid cell, gather points belonging to the cell
//     4. for each ray (px, py, pz, dx, dy, dz)
//             if dy.abs() < 1e-8:
//                 # use x=c plane
//             else:
//                 # use y=c plane
// 
//             - find plane-ray intersection point on each grid plane (xi, yi, zi)
//             - for each (xi, yi, zi)
//                 x_from = discretize(xi - radius)
//                 x_to = discretize(xi + radius)
//                 (same for y and z)
// 
//                 find grid idx for all of combinations, ie, get all grid cells
//                 surrounding the point.
// 
//     5. given grid idxs for each ray, gather point idxs
// 
std::tuple<torch::Tensor, torch::Tensor> find_k_neighbor_points_of_rays(
        torch::Tensor points,  // (b, n, 3), float
        torch::Tensor ray_origins, // (b, m, 3), float
        torch::Tensor ray_directions,  // (b, m, 3), float
        torch::Tensor ray_radius,  // (b,), float
        torch::Tensor grid_size,  // (b, 3), long
        torch::Tensor grid_center,  // (b, 3), float
        torch::Tensor grid_width,  // (b, 3), float
        int64_t k,
        float t_min = 0.,
        float t_max = 1.e12
) {
    
    CHECK_INPUT(points);
    CHECK_INPUT(ray_origins);
    CHECK_INPUT(ray_directions);
    CHECK_INPUT(ray_radius);
    CHECK_INPUT(grid_size);
    CHECK_INPUT(grid_center);
    CHECK_INPUT(grid_width);

    return find_k_neighbor_points_of_rays_cuda(
        points,  // (b, n, 3), 
        ray_origins, // (b, m, 3), float
        ray_directions,  // (b, m, 3), float
        ray_radius,  // (b,), float
        grid_size,  // (b, 3), long
        grid_center,  // (b, 3), float
        grid_width,  // (b, 3), float
        k,
        t_min,
        t_max
    );
}


torch::Tensor keep_min_k_values(
        torch::Tensor values,  // (b, n), float
        int64_t k
) {
//    return values;
    return keep_min_k_values_cuda(
        values,  // (b, n,
        k
    );
}


// Find the k nearest points within `radius` of a ray, ie, the vertical distance from the point to ray <= radius.
// If not enough points found, return dummy index
// 
// Returns:
//     ray2pidx: 
//          (b, M), all the points. The pidxs of a ray is found via ray2pidx[b][ray_start_idx[m]:ray_end_idx[m]]
//     ray_start_idx:
//          (b, m), 
//     ray_end_idx:
//          (b, m)
// 
// Note:
//     Our algorithm is very simple. In order to be parallelized on gpu easily, we want to every thread to have
//     as few branching conditions (if/else) as possible.
// 
//     We will:
//     - parallelize on ray
//     - do the same thing for all rays.
//     - ignore all points outside the grid boundary
// 
// Algorithm:
// 
//     1. determine a grid
//     2. discretize the xyz of points -> calculate grid indices of each point
//     3. for each grid cell, gather points belonging to the cell
//     4. for each ray (px, py, pz, dx, dy, dz)
//             if dy.abs() < 1e-8:
//                 # use x=c plane
//             else:
//                 # use y=c plane
// 
//             - find plane-ray intersection point on each grid plane (xi, yi, zi)
//             - for each (xi, yi, zi)
//                 x_from = discretize(xi - radius)
//                 x_to = discretize(xi + radius)
//                 (same for y and z)
// 
//                 find grid idx for all of combinations, ie, get all grid cells
//                 surrounding the point.
// 
//     5. given grid idxs for each ray, gather point idxs
// 
std::tuple<torch::Tensor, torch::Tensor> find_k_neighbor_points_of_rays_v2(
        torch::Tensor points,  // (b, n, 3), float
        torch::Tensor ray_origins, // (b, m, 3), float
        torch::Tensor ray_directions,  // (b, m, 3), float
        torch::Tensor ray_radius,  // (b,), float
        torch::Tensor grid_size,  // (b, 3), long
        torch::Tensor grid_center,  // (b, 3), float
        torch::Tensor grid_width,  // (b, 3), float
        int64_t k,
        float t_min = 0.,
        float t_max = 1.e12
) {
    
    CHECK_INPUT(points);
    CHECK_INPUT(ray_origins);
    CHECK_INPUT(ray_directions);
    CHECK_INPUT(ray_radius);
    CHECK_INPUT(grid_size);
    CHECK_INPUT(grid_center);
    CHECK_INPUT(grid_width);

    return find_k_neighbor_points_of_rays_cuda_v2(
        points,  // (b, n, 3), 
        ray_origins, // (b, m, 3), float
        ray_directions,  // (b, m, 3), float
        ray_radius,  // (b,), float
        grid_size,  // (b, 3), long
        grid_center,  // (b, 3), float
        grid_width,  // (b, 3), float
        k,
        t_min,
        t_max
    );
}



std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> find_k_neighbor_points_of_rays_v3(
        torch::Tensor points,  // (b, n, 3), float
        torch::Tensor ray_origins, // (b, m, 3), float
        torch::Tensor ray_directions,  // (b, m, 3), float
        torch::Tensor ray_radius,  // (b,), float
        torch::Tensor grid_size,  // (b, 3), long
        torch::Tensor grid_center,  // (b, 3), float
        torch::Tensor grid_width,  // (b, 3), float
        torch::Tensor gidx2pidx_bank,  // (b, n) long or empty (,)
        torch::Tensor gidx_start_idx,  // (b, n_cell+1) int32 or empty (,)
        int64_t k,
        float t_min = 0.,
        float t_max = 1.e12,
        bool refresh_cache = true
) {
    
    CHECK_INPUT(points);
    CHECK_INPUT(ray_origins);
    CHECK_INPUT(ray_directions);
    CHECK_INPUT(ray_radius);
    CHECK_INPUT(grid_size);
    CHECK_INPUT(grid_center);
    CHECK_INPUT(grid_width);
    CHECK_INPUT(gidx2pidx_bank);
    CHECK_INPUT(gidx_start_idx);

    return find_k_neighbor_points_of_rays_cuda_v3(
        points,  // (b, n, 3), 
        ray_origins, // (b, m, 3), float
        ray_directions,  // (b, m, 3), float
        ray_radius,  // (b,), float
        grid_size,  // (b, 3), long
        grid_center,  // (b, 3), float
        grid_width,  // (b, 3), float
        gidx2pidx_bank,  // (b, n) long or empty (,)
        gidx_start_idx,   // (b, n_cell+1) int32 or empty (,)
        k,
        t_min,
        t_max,
        refresh_cache
    );
}




std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> find_k_neighbor_points_of_rays_v4(
        torch::Tensor points,  // (b, n, 3), float
        torch::Tensor ray_origins, // (b, m, 3), float
        torch::Tensor ray_directions,  // (b, m, 3), float
        torch::Tensor ray_radius,  // (b,), float
        torch::Tensor grid_size,  // (b, 3), long
        torch::Tensor grid_center,  // (b, 3), float
        torch::Tensor grid_width,  // (b, 3), float
        torch::Tensor gidx2pidx_bank,  // (b, n) long or empty (,)
        torch::Tensor gidx_start_idx,  // (b, n_cell+1) int32 or empty (,)
        torch::Tensor valid_mask,  // (b, n) bool
        int64_t k,
        float t_min = 0.,
        float t_max = 1.e12,
        bool refresh_cache = true
) {
    
    CHECK_INPUT(points);
    CHECK_INPUT(ray_origins);
    CHECK_INPUT(ray_directions);
    CHECK_INPUT(ray_radius);
    CHECK_INPUT(grid_size);
    CHECK_INPUT(grid_center);
    CHECK_INPUT(grid_width);
    CHECK_INPUT(gidx2pidx_bank);
    CHECK_INPUT(gidx_start_idx);
    CHECK_INPUT(valid_mask);

    return find_k_neighbor_points_of_rays_cuda_v4(
        points,  // (b, n, 3), 
        ray_origins, // (b, m, 3), float
        ray_directions,  // (b, m, 3), float
        ray_radius,  // (b,), float
        grid_size,  // (b, 3), long
        grid_center,  // (b, 3), float
        grid_width,  // (b, 3), float
        gidx2pidx_bank,  // (b, n) long or empty (,)
        gidx_start_idx,   // (b, n_cell+1) int32 or empty (,)
        valid_mask,  // (b, n), bool
        k,
        t_min,
        t_max,
        refresh_cache
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("get_grid_idx", &get_grid_idx, "compute the grid index for each point");
    m.def("gather_points_v1", &gather_points_v1, "gather the points in each cell");
    m.def("gather_points_v2", &gather_points_v2, "gather the points in each cell");
    m.def("grid_ray_intersection", &grid_ray_intersection, "intersect ray with a grid");
    m.def("collect_points_on_ray", &collect_points_on_ray, "collect points along a ray using compiled structure");
    m.def("find_neighbor_points_of_rays", &find_neighbor_points_of_rays, "find the neighboring points of a ray");
    m.def("find_k_neighbor_points_of_rays", &find_k_neighbor_points_of_rays, "find the k_neighboring points of a ray");
    m.def("keep_min_k_values", &keep_min_k_values, "find k-min values with max heap");
    m.def("find_k_neighbor_points_of_rays_v2", &find_k_neighbor_points_of_rays_v2, "find the k_neighboring points of a ray (version 2)");
    m.def("find_k_neighbor_points_of_rays_v3", &find_k_neighbor_points_of_rays_v3, "find the k_neighboring points of a ray (version 3)");
    m.def("find_k_neighbor_points_of_rays_v4", &find_k_neighbor_points_of_rays_v4, "find the k_neighboring points of a ray (version 4)");
}
