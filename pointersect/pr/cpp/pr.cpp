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

typedef std::unordered_map<int64_t, std::unordered_set<int64_t> > Table;
typedef std::vector<std::vector<int64_t> > llist;


struct idxmap {
    Table table;  // grid_idx -> set of point_idx
                  // ray_idx -> set of grid_idx

    idxmap() {}

    // reserve n capacity in the table
    idxmap(size_t n) {
        table.reserve(n);
    }

    // initialize with table
    idxmap(const Table& t) {
        table = t;
    }

    // associate pidx with gidx
    void insert(int64_t gidx, int64_t pidx) {
        auto iter = table.find(gidx);
        if (iter == table.end()) {
            table[gidx] = std::unordered_set<int64_t>({pidx});
        }
        else {
            table[gidx].insert(pidx);
        }
    }

    // get the pointer to the point index set
    std::unordered_set<int64_t> * get_pidx(int64_t gidx) {
        auto iter = table.find(gidx);
        if (iter == table.end()) {
            return nullptr;
        }
        else {
            return &(table[gidx]);
        }
    }

    // get the underlying table
    Table get_table() {
        return table;
    }
};



inline std::vector<int64_t> ind2sub_c(
    const int64_t & ind,
    const int64_t & sx,
    const int64_t & sy,
    const int64_t & sz
) {
    auto yz_size = sy * sz;
    auto x = ind / yz_size;  
    auto _ind = ind - x * yz_size;
    auto y = _ind / sz; 
    auto z = _ind - y * sz;
    return {x, y, z};
}

inline int64_t sub2ind_c(
    const int64_t & x,
    const int64_t & y,
    const int64_t & z,
    const int64_t & sx,
    const int64_t & sy,
    const int64_t & sz
) {
    return z + y * sz + x * (sy * sz);
}


// Given x y z index, change to the linear index.  (matlab's sub2ind)
// 
// Args:
//     idx:
//         (*, n, 3) int64_t
//     size:
//         (*, 3) int64_t
// 
// Returns:
//     (*, n) int64_t
// 
torch::Tensor sub2ind(
    const torch::Tensor & idx,  // (*, n, 3) int64_t 
    const torch::Tensor & size  // (*, 3) int64_t
) {
    
    assert(idx.dtype() == torch::kLong);
    assert(size.dtype() == torch::kLong);

    auto linear_idx = 
        idx.index({"...", 2}) + 
        idx.index({"...", 1}) * size.index({"...", Slice(2, 3)}) + 
        idx.index({"...", 0}) * (size.index({"...", Slice(1, 2)}) * size.index({"...", Slice(2, 3)}));  // (*, n)
    return linear_idx;
}



// Given linear index, return (i,j,k)
// 
// Args:
//     ind:
//         (*, n) int64_t,  z + y * sz + x * sy * sz
//     size:
//         (*, 3) int64_t
// 
// Returns:
//     (*, n, 3) int64_t
// 
torch::Tensor ind2sub(
    const torch::Tensor & ind,  // (*, n) int64_t
    const torch::Tensor & size  // (*, 3) int64_t
) {

    assert(ind.dtype() == torch::kLong);
    assert(size.dtype() == torch::kLong);
    
    auto yz_size = size.index({"...", Slice(1, 2)}) * size.index({"...", Slice(2, 3)});  // (*, 1)
    auto xs = torch::div(ind, yz_size, "floor");  // (*, n)
    auto _ind = ind - xs * yz_size;  // (*, n)
    auto ys = torch::div(_ind, size.index({"...", Slice(2, 3)}), "floor");  // (*, n)
    auto zs = _ind - ys * size.index({"...", Slice(2, 3)});  // (*, n)
    auto idx = torch::stack({xs, ys, zs}, -1);  // (*, n, 3)
    return idx;
}


// Gather the points belonging to each grid cell.
// 
// Args:
//     grid_idxs:
//         (b, n), grid index of each point
//     total_cells:
//         (b,) total grid cells
//     valid_mask:
//         (b, n), whether to record the point 
// 
// Returns:
//     vector of idxmap, b -> (total_cells[bidx] -> n_idx of points in the cell)
// 
std::vector<idxmap> gather_points(
    // input
    const torch::Tensor & grid_idxs,   // (b, n), int64_t
    const torch::Tensor & total_cells, // (b,) int64_t
    const torch::Tensor & valid_mask   // (b, n), bool
    // // output
    // std::vector<idxmap> & all_cell2pidx  // b -> gidx -> pidx
) {

    assert(grid_idxs.dtype() == torch::kLong);
    assert(total_cells.dtype() == torch::kLong);
    assert(valid_mask.dtype() == torch::kBool);

    auto batch_size = grid_idxs.size(0);
    auto n_points = grid_idxs.size(1);

    // accessor 
    auto a_grid_idxs = grid_idxs.accessor<int64_t, 2>();
    auto a_total_cells = total_cells.accessor<int64_t, 1>();
    auto a_valid_mask = valid_mask.accessor<bool, 2>();

    std::vector<idxmap> all_cell2pidx;
    all_cell2pidx.reserve(batch_size);

    // loop through each batch element
    for (auto b = 0; b < batch_size; ++b) {    
        all_cell2pidx.push_back(idxmap(a_total_cells[b]));
        idxmap& cell2pidx = all_cell2pidx[b];

        // loop through each point
        for (auto i = 0; i < n_points; ++i) {
            if (a_valid_mask[b][i]) {
                cell2pidx.insert(a_grid_idxs[b][i], i);
            }
        }
    }
    return all_cell2pidx;
}






// Compute the grid index given xyz_w.
// 
// Args:
//     points:
//         (*, n, 3)
//     grid_size:
//         (*, 3) long. number of grid cells in x y z.
//     center:
//         (*, 3)  center of the grid
//     grid_width:
//         (*, 3)  length (full width) of the grid in xyz
//     mode:
//         'subidx': return sub idx
//         'ind': return linear index
// 
// Returns:
//     grid_idx:
//         if mode == 'subidx':  (*, n, 3) long
//         elif mode == 'ind':   (*, n) long
//     valid_mask:
//         (*, n) bool
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
std::vector<torch::Tensor> get_grid_idx(
        const torch::Tensor & points,  // (*, n, 3), float
        const torch::Tensor & grid_size,  // (*, 3), long
        const torch::Tensor & center,  // (*, 3), float
        const torch::Tensor & grid_width,  // (*, 3), float
        const std::string & mode = "ind"  
) { 

    assert(grid_size.dtype() == torch::kLong);
    
    auto grid_from = (center - grid_width / 2).unsqueeze(-2);  // (*, 1, 3)
    auto grid_to = (center + grid_width / 2).unsqueeze(-2);  // (*, 1, 3)
    auto cell_width = (grid_width / grid_size).unsqueeze(-2);  // (*, 1, 3)
    auto p_idx = ((points - grid_from) / cell_width).floor().to(torch::kLong);  // (*, n, 3), sub_idx on the grid

    // we will mark any out-of-bound points invalid
    auto valid_mask = torch::logical_and(
        points >= grid_from,
        points <= grid_to
    ).all(-1);  // (*, n),  bool

    if (mode.compare("ind") == 0) {
        auto grid_idx = sub2ind(
            p_idx,
            grid_size
        );  // (*, n) long
        return {grid_idx, valid_mask};
    }
    else if (mode.compare("subidx") == 0) {
        return {p_idx, valid_mask};
    }
    else {
        throw;
    }

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
//     list of list of list:  b -> ray_idx -> grid_idx (including -1, outside the grid)
// 
// Algorithm:
//     for each ray (px, py, pz, dx, dy, dz)
//         if dx.abs() < 1e-8:
//             # use y=c plane
//         else:
//             # use x=c plane
// 
//         - find plane-ray intersection point on each grid plane (xi, yi, zi)
//         - for each (xi, yi, zi)
//             x_from = discretize(xi - radius)
//             x_to = discretize(xi + radius)
//             (same for y and z)
// 
//             find grid idx for all of combinations, ie, get all grid cells
//             surrounding the point.
// 
std::vector<idxmap> grid_ray_intersection(
        const torch::Tensor & ray_origins,  // (b, m, 3), float
        const torch::Tensor & ray_directions,  // (b, m, 3), float
        const torch::Tensor & ray_radius,   //  (b, )m float
        const torch::Tensor & grid_size,  // (b, 3), long
        const torch::Tensor & grid_center,  // (b, 3) float
        const torch::Tensor & grid_width  // (b, 3) float
) {

    assert(grid_size.dtype() == torch::kLong);

    auto batch_size = ray_origins.size(0);
    auto n_rays = ray_origins.size(1);
    auto device = ray_origins.device();

    auto grid_from = grid_center - grid_width * 0.5;  // (b, 3)
    auto cell_width = grid_width / grid_size;  // (b, 3)
    auto total_cells = torch::prod(grid_size, -1);  // (b,)

    // determine whether to intersect with x=c plane, y=c plane, or z=c plane.
    // It is important to select a direction so that the intersection point on two nearby planes
    // do not exceed one grid in one of the rest of the two directions.
    // Fortunately, it can be shown that it will always happen when we select wisely.

    // Let inv_tx = |dx / wx|  (1 / time to travel one x grid)
    //     inv_ty = |dy / wy|  (1 / time to travel one y grid)
    //     inv_tz = |dz / wz|  (1 / time to travel one z grid)
    // If inv_tx is the largest of (inv_tx, inv_ty, inv_tz) -> we will intersect with x=c planes.
    // vise versa for inv_ty and inv_tz.
    // When we select inv_tx, it ensures we move in the x direction in 1 grid, y and z in <= 1 grid.

    auto inv_t = (ray_directions / cell_width.unsqueeze(-2)).abs();  // (b, m, 3)
    auto idx_to_use = torch::argmax(inv_t, -1);  // (b, m)  (0, 1, 2)

    // create accessors to each tensor
    // auto start = high_resolution_clock::now();
    auto a_grid_from = grid_from.accessor<float, 2>();  // (b, 3)
    auto a_cell_width = cell_width.accessor<float, 2>();  // (b, 3)
    auto a_grid_size = grid_size.accessor<int64_t, 2>();  // (b, 3)
    auto a_idx_to_use = idx_to_use.accessor<int64_t, 2>();  // (b, m)  0, 1, 2
    // auto stop = high_resolution_clock::now();
    // auto duration = duration_cast<microseconds>(stop - start);
    // std::cout<<"accessor_time: "<<duration.count()<<std::endl;

    auto l_options = torch::TensorOptions().dtype(torch::kLong).device(device);
    auto f_options = torch::TensorOptions().dtype(torch::kFloat32).device(device);


    std::vector<idxmap> all_grid_idxs;
    all_grid_idxs.reserve(batch_size);
    for (auto b = 0; b < batch_size; ++b) {

        auto a_grid_size_b = a_grid_size[b];

        // build a meshgrid for local grid idx.  This will determine the neighboring grids to gather.
        // the local grid width = 2 * local_grid_radius
        auto local_grid_radius = (torch::ceil(ray_radius.index({b, None}) / cell_width[b] + 1)).to(torch::kLong);  // (3,) long,  xyz
        auto grid_idx_from = -1 * local_grid_radius;  // (3,) long, xyz
        auto grid_idx_to = local_grid_radius;  // (3,) long, xyz

        auto a_local_grid_radius = local_grid_radius.accessor<int64_t, 1>();  // (3,) 
        auto a_grid_idx_from = grid_idx_from.accessor<int64_t, 1>();  // (3,) 
        auto a_grid_idx_to = grid_idx_to.accessor<int64_t, 1>();  // (3,) 

        auto xs = torch::arange(a_grid_idx_from[0], a_grid_idx_to[0] + 1, l_options);
        auto ys = torch::arange(a_grid_idx_from[1], a_grid_idx_to[1] + 1, l_options);
        auto zs = torch::arange(a_grid_idx_from[2], a_grid_idx_to[2] + 1, l_options);
        auto XYZs = torch::meshgrid({xs, ys, zs}, "ij");
        auto local_grid_subidx = torch::cat({
            XYZs[0].reshape({-1, 1}),
            XYZs[1].reshape({-1, 1}),
            XYZs[2].reshape({-1, 1}),
        }, -1);  // (n_local_grid, 3)  xyz in grid

        // since the local grid has a certain width, we need to intersect with more planes
        auto xplane_idxs = torch::arange(
            -1 * a_local_grid_radius[0],
            a_grid_size_b[0] + a_local_grid_radius[0],
            f_options
        );  // (num_xplanes+2r,),  -r, -r+1, ..., -1, | 0, 1, ..., n_xplanes-1 |, n_xplanes, ..., n_xplanes+r-1
        auto yplane_idxs = torch::arange(
            -1 * a_local_grid_radius[1],
            a_grid_size_b[1] + a_local_grid_radius[1],
            f_options
        );  // (num_yplanes+2r,)
        auto zplane_idxs = torch::arange(
            -1 * a_local_grid_radius[2],
            a_grid_size_b[2] + a_local_grid_radius[2],
            f_options
        );  // (num_zplanes+2r,)
        auto xplane_cs = a_grid_from[b][0] + (xplane_idxs + 0.5) * a_cell_width[b][0];  // (num_xplanes+2r,)
        auto yplane_cs = a_grid_from[b][1] + (yplane_idxs + 0.5) * a_cell_width[b][1];  // (num_yplanes+2r,)
        auto zplane_cs = a_grid_from[b][2] + (zplane_idxs + 0.5) * a_cell_width[b][2];  // (num_zplanes+2r,)
        auto xplane_ts = (xplane_cs.unsqueeze(0) - ray_origins.index({b, Slice(), Slice(0,1)})) / ray_directions.index({b, Slice(), Slice(0,1)});  // (m, num_xplanes+2r)
        auto yplane_ts = (yplane_cs.unsqueeze(0) - ray_origins.index({b, Slice(), Slice(1,2)})) / ray_directions.index({b, Slice(), Slice(1,2)});  // (m, num_yplanes+2r)
        auto zplane_ts = (zplane_cs.unsqueeze(0) - ray_origins.index({b, Slice(), Slice(2,3)})) / ray_directions.index({b, Slice(), Slice(2,3)});  // (m, num_zplanes+2r)


        // compute ray-plane intersection
        idxmap ray2gidxs(n_rays);  // ray_idx -> grid_idxs
        for (auto m = 0; m < n_rays; ++m) {

            torch::Tensor ts;            
            if (a_idx_to_use[b][m] == 0)
                ts = xplane_ts[m];  // (num_xplanes+2rx, )
            else if (a_idx_to_use[b][m] == 1)
                ts = yplane_ts[m];  // (num_yplanes+2ry, )
            else if (a_idx_to_use[b][m] == 2)
                ts = zplane_ts[m];  // (num_zplanes+2rz, )
            else
                throw;

            // for each intersection point, gather all neighboring cells
            // start = high_resolution_clock::now();
            auto ps = ray_origins.index({b, m}).unsqueeze(0) + 
                      ts.unsqueeze(-1) * ray_directions.index({b, m}).unsqueeze(0);  // (num_planes+2r, 3)
            // stop = high_resolution_clock::now();
            // duration = duration_cast<microseconds>(stop - start);
            // std::cout<<"ps time: "<<duration.count()<<std::endl;

            // start = high_resolution_clock::now();
            auto ps_grid_ind = get_grid_idx(
                ps,  // points (n_plane+4r, 3)
                grid_size[b],  // grid_size (3,)
                grid_center[b],  // center (3,)
                grid_width[b],  // grid_width (3,)
                "subidx"   // mode
            )[0];  // (n_plane+2r, 3),  ps_grid_ind can < 0 or >= grid_size
            // stop = high_resolution_clock::now();
            // duration = duration_cast<microseconds>(stop - start);
            // std::cout<<"get grid idx time: "<<duration.count()<<std::endl;
            
            // get all neighboring cells (some will have invalid grid_idx)
            auto gidxs = (ps_grid_ind.unsqueeze(1) + local_grid_subidx.unsqueeze(0)).reshape({-1, 3});  // ((n_plane+2r* n_local_grid), 3) subidx
            // std::cout<<"gidxs shape = "<<gidxs.sizes()<<std::endl; 
            auto a_gidxs = gidxs.accessor<int64_t, 2>();  // (n_grid, 3) subidx

            // start = high_resolution_clock::now();
            for (auto i = 0; i < a_gidxs.size(0); ++i) {
                // check gidx is valid
                auto a_gidx = a_gidxs[i];  // (3,) 
                if (a_gidx[0] < 0 || a_gidx[0] >= a_grid_size_b[0] ||
                    a_gidx[1] < 0 || a_gidx[1] >= a_grid_size_b[1] ||
                    a_gidx[2] < 0 || a_gidx[2] >= a_grid_size_b[2]
                )
                    continue;

                auto ind = sub2ind_c(
                    a_gidx[0], a_gidx[1], a_gidx[2], 
                    a_grid_size_b[0], a_grid_size_b[1], a_grid_size_b[2]
                );
                ray2gidxs.insert(m, ind);
            }
            // stop = high_resolution_clock::now();
            // duration = duration_cast<microseconds>(stop - start);
            // std::cout<<"new time: "<<duration.count()<<std::endl;



            // // get all neighboring cells (some will have invalid grid_idx)
            // auto gidxs = ps_grid_ind.unsqueeze(1) + local_grid_subidx.unsqueeze(0);  // (n_plane+2r, n_local_grid, 3) subidx
            
            // // handle invalid gidxs
            // start = high_resolution_clock::now();
            // auto valid_mask = torch::logical_and(
            //     gidxs >= 0,
            //     gidxs < grid_size[b].reshape({1, 1, 3})
            // ).all(-1);  // (n_plane+2r, n_local_grid)

            // gidxs = gidxs.index({valid_mask});
            // stop = high_resolution_clock::now();
            // duration = duration_cast<microseconds>(stop - start);
            // std::cout<<"invalid time: "<<duration.count()<<std::endl;


            // start = high_resolution_clock::now();
            // gidxs = sub2ind(
            //     gidxs.view({-1, 3}),  // (n_plane*n_local_grid, 3)
            //     grid_size[b]  // (3,)
            // );  // (n_cells,)
            // stop = high_resolution_clock::now();
            // duration = duration_cast<microseconds>(stop - start);
            // std::cout<<"sub2ind time: "<<duration.count()<<std::endl;

            // start = high_resolution_clock::now();
            // auto a_gidxs = gidxs.accessor<int64_t, 1>();  // (n_cells,)
            // for (auto i = 0; i < a_gidxs.size(0); ++i) {
            //     ray2gidxs.insert(m, a_gidxs[i]);
            // }
            // stop = high_resolution_clock::now();
            // duration = duration_cast<microseconds>(stop - start);
            // std::cout<<"set time: "<<duration.count()<<std::endl;
        }

        all_grid_idxs.push_back(std::move(ray2gidxs));

    }
    return all_grid_idxs;

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
//     list of list of list:  b -> ray_idx -> grid_idx (including -1, outside the grid)
// 
// Algorithm:
//     for each ray (px, py, pz, dx, dy, dz)
//         if dx.abs() < 1e-8:
//             # use y=c plane
//         else:
//             # use x=c plane
// 
//         - find plane-ray intersection point on each grid plane (xi, yi, zi)
//         - for each (xi, yi, zi)
//             x_from = discretize(xi - radius)
//             x_to = discretize(xi + radius)
//             (same for y and z)
// 
//             find grid idx for all of combinations, ie, get all grid cells
//             surrounding the point.
// 
std::vector<idxmap> grid_ray_intersection_v2(
        const torch::Tensor & ray_origins,  // (b, m, 3), float
        const torch::Tensor & ray_directions,  // (b, m, 3), float
        const torch::Tensor & ray_radius,   //  (b, )m float
        const torch::Tensor & grid_size,  // (b, 3), long
        const torch::Tensor & grid_center,  // (b, 3) float
        const torch::Tensor & grid_width  // (b, 3) float
) {

    assert(grid_size.dtype() == torch::kLong);

    auto batch_size = ray_origins.size(0);
    auto n_rays = ray_origins.size(1);
    auto device = ray_origins.device();

    auto grid_from = grid_center - grid_width * 0.5;  // (b, 3)
    auto cell_width = grid_width / grid_size;  // (b, 3)
    auto total_cells = torch::prod(grid_size, -1);  // (b,)

    // determine whether to intersect with x=c plane, y=c plane, or z=c plane.
    // It is important to select a direction so that the intersection point on two nearby planes
    // do not exceed one grid in one of the rest of the two directions.
    // Fortunately, it can be shown that it will always happen when we select wisely.

    // Let inv_tx = |dx / wx|  (1 / time to travel one x grid)
    //     inv_ty = |dy / wy|  (1 / time to travel one y grid)
    //     inv_tz = |dz / wz|  (1 / time to travel one z grid)
    // If inv_tx is the largest of (inv_tx, inv_ty, inv_tz) -> we will intersect with x=c planes.
    // vise versa for inv_ty and inv_tz.
    // When we select inv_tx, it ensures we move in the x direction in 1 grid, y and z in <= 1 grid.

    auto inv_t = (ray_directions / cell_width.unsqueeze(-2)).abs();  // (b, m, 3)
    auto idx_to_use = torch::argmax(inv_t, -1);  // (b, m)  (0, 1, 2)

    // std::cout<<"idx_to_use:"<<std::endl;
    // std::cout<<idx_to_use<<std::endl;


    // create accessors to each tensor
    // auto start = high_resolution_clock::now();
    auto a_ray_directions = ray_directions.accessor<float, 3>();  // (b, m, 3)
    auto a_ray_radius = ray_radius.accessor<float, 1>();  // (b,)
    auto a_grid_from = grid_from.accessor<float, 2>();  // (b, 3)
    auto a_cell_width = cell_width.accessor<float, 2>();  // (b, 3)
    auto a_grid_size = grid_size.accessor<int64_t, 2>();  // (b, 3)
    auto a_idx_to_use = idx_to_use.accessor<int64_t, 2>();  // (b, m)  0, 1, 2
    // auto stop = high_resolution_clock::now();
    // auto duration = duration_cast<microseconds>(stop - start);
    // std::cout<<"accessor_time: "<<duration.count()<<std::endl;

    auto l_options = torch::TensorOptions().dtype(torch::kLong).device(device);
    auto f_options = torch::TensorOptions().dtype(torch::kFloat32).device(device);


    std::vector<idxmap> all_grid_idxs;
    all_grid_idxs.reserve(batch_size);
    for (auto b = 0; b < batch_size; ++b) {

        // auto a_grid_size_b = a_grid_size[b];
        int64_t grid_size_b[3] = {
            a_grid_size[b][0],
            a_grid_size[b][1],
            a_grid_size[b][2]
        };

        // since the local grid has a certain width, we need to intersect with more planes
        auto xplane_idxs = torch::arange(
            0,
            grid_size_b[0],
            f_options
        );  // (num_xplanes,),  0, 1, ..., n_xplanes-1
        auto yplane_idxs = torch::arange(
            0,
            grid_size_b[1],
            f_options
        );  // (num_yplanes,)
        auto zplane_idxs = torch::arange(
            0,
            grid_size_b[2],
            f_options
        );  // (num_zplanes,)
        auto xplane_cs = a_grid_from[b][0] + (xplane_idxs + 0.5) * a_cell_width[b][0];  // (num_xplanes+2r,)
        auto yplane_cs = a_grid_from[b][1] + (yplane_idxs + 0.5) * a_cell_width[b][1];  // (num_yplanes+2r,)
        auto zplane_cs = a_grid_from[b][2] + (zplane_idxs + 0.5) * a_cell_width[b][2];  // (num_zplanes+2r,)
        auto xplane_ts = (xplane_cs.unsqueeze(0) - ray_origins.index({b, Slice(), Slice(0,1)})) / ray_directions.index({b, Slice(), Slice(0,1)});  // (m, num_xplanes+2r)
        auto yplane_ts = (yplane_cs.unsqueeze(0) - ray_origins.index({b, Slice(), Slice(1,2)})) / ray_directions.index({b, Slice(), Slice(1,2)});  // (m, num_yplanes+2r)
        auto zplane_ts = (zplane_cs.unsqueeze(0) - ray_origins.index({b, Slice(), Slice(2,3)})) / ray_directions.index({b, Slice(), Slice(2,3)});  // (m, num_zplanes+2r)


        // compute ray-plane intersection
        idxmap ray2gidxs(n_rays);  // ray_idx -> grid_idxs
        for (auto m = 0; m < n_rays; ++m) {

            // Determine ts and build a meshgrid for neighboring grids.
            // If we trace along x, we will build a meshgrid on yz-plane.
            torch::Tensor ts;     
            int64_t rx, ry, rz;       
            if (a_idx_to_use[b][m] == 0) {
                ts = xplane_ts[m];  // (num_xplanes+2rx, )
                rx = 0;
                ry = 1 + int64_t(ceil(a_ray_radius[b] / (sqrt(1. - (a_ray_directions[b][m][1] * a_ray_directions[b][m][1])) * a_cell_width[b][1])));
                rz = 1 + int64_t(ceil(a_ray_radius[b] / (sqrt(1. - (a_ray_directions[b][m][2] * a_ray_directions[b][m][2])) * a_cell_width[b][2])));
            }
            else if (a_idx_to_use[b][m] == 1) {
                ts = yplane_ts[m];  // (num_yplanes+2ry, )
                rx = 1 + int64_t(ceil(a_ray_radius[b] / (sqrt(1. - (a_ray_directions[b][m][0] * a_ray_directions[b][m][0])) * a_cell_width[b][0])));
                ry = 0;
                rz = 1 + int64_t(ceil(a_ray_radius[b] / (sqrt(1. - (a_ray_directions[b][m][2] * a_ray_directions[b][m][2])) * a_cell_width[b][2])));
            }
            else if (a_idx_to_use[b][m] == 2) {
                ts = zplane_ts[m];  // (num_zplanes+2rz, )
                rx = 1 + int64_t(ceil(a_ray_radius[b] / (sqrt(1. - (a_ray_directions[b][m][0] * a_ray_directions[b][m][0])) * a_cell_width[b][0])));
                ry = 1 + int64_t(ceil(a_ray_radius[b] / (sqrt(1. - (a_ray_directions[b][m][1] * a_ray_directions[b][m][1])) * a_cell_width[b][1])));
                rz = 0;
            }
            else
                throw;

            
            // for each intersection point, gather all neighboring cells
            // start = high_resolution_clock::now();
            auto ps = ray_origins.index({b, m}).unsqueeze(0) + 
                      ts.unsqueeze(-1) * ray_directions.index({b, m}).unsqueeze(0);  // (num_planes+2r, 3)
            // stop = high_resolution_clock::now();
            // duration = duration_cast<microseconds>(stop - start);
            // std::cout<<"ps time: "<<duration.count()<<std::endl;

            // start = high_resolution_clock::now();
            auto ps_grid_subidx = get_grid_idx(
                ps,  // points (n_plane+4r, 3)
                grid_size[b],  // grid_size (3,)
                grid_center[b],  // center (3,)
                grid_width[b],  // grid_width (3,)
                "subidx"   // mode
            )[0];  // (n_plane+2r, 3),  ps_grid_ind can < 0 or >= grid_size
            // stop = high_resolution_clock::now();
            // duration = duration_cast<microseconds>(stop - start);
            // std::cout<<"get grid idx time: "<<duration.count()<<std::endl;
            
            auto a_ps_grid_subidx = ps_grid_subidx.accessor<int64_t, 2>();  // (n_plane, 3)

            // start = high_resolution_clock::now();
            for (auto ip = 0; ip < a_ps_grid_subidx.size(0); ++ip) {
                auto px = a_ps_grid_subidx[ip][0];
                auto py = a_ps_grid_subidx[ip][1];
                auto pz = a_ps_grid_subidx[ip][2];

                for (auto ix = -rx; ix <= rx; ++ix) {
                    for (auto iy = -ry; iy <= ry; ++iy) {
                        for (auto iz = -rz; iz <= rz; ++iz) {
                            auto x = px + ix;
                            auto y = py + iy;
                            auto z = pz + iz;

                            // ignore out-of-bound cells
                            if (x < 0 || x >= grid_size_b[0] ||
                                y < 0 || y >= grid_size_b[1] ||
                                z < 0 || z >= grid_size_b[2]
                            )
                                continue;

                            auto ind = sub2ind_c(
                                x, y, z, 
                                grid_size_b[0], grid_size_b[1], grid_size_b[2]
                            );
                            ray2gidxs.insert(m, ind);
                        }   
                    }
                }
            }
        }

        all_grid_idxs.push_back(std::move(ray2gidxs));

    }
    return all_grid_idxs;
}



// Compute the distance between a point and its projection on a ray. 
// 
// Args:
//      p_p: (3, ) float pointer to xyz
//      p_ro: (3, ) float pointer to xyz of ray origin
//      p_rd: (3, ) float pointer to ray direction (normalized)
// 
// Returns:
//      dist:  float
//      t:     float
// 
inline std::tuple<float, float> compute_point_ray_distance_c(
    const float * p_p,  // (3,)
    const float * p_ro,  // (3,)
    const float * p_rd  // (3,)
) {
    float dv[3] = {
        p_p[0] - p_ro[0], 
        p_p[1] - p_ro[1], 
        p_p[2] - p_ro[2] 
    };

    float t = dv[0] * p_rd[0] + dv[1] * p_rd[1] + dv[2] * p_rd[2];

    float dd[3] = {
        p_p[0] - (p_ro[0] + t * p_rd[0]),
        p_p[1] - (p_ro[1] + t * p_rd[1]),
        p_p[2] - (p_ro[2] + t * p_rd[2])
    };

    float dist = sqrt(dd[0] * dd[0] + dd[1] * dd[1] + dd[2] * dd[2]);

    return std::make_tuple(dist, t);
}



// Find the points within `radius` of a ray, ie, the vertical distance from the point to ray <= radius.
// 
// Returns:
//     list of list of list:  b -> m -> n_idx
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
std::vector<llist> find_neighbor_points_of_rays(
        torch::Tensor points,  // (b, n, 3), float
        torch::Tensor ray_origins, // (b, m, 3), float
        torch::Tensor ray_directions,  // (b, m, 3), float
        torch::Tensor ray_radius,  // (b,), float
        torch::Tensor grid_size,  // (b, 3), long
        torch::Tensor grid_center,  // (b, 3), float
        torch::Tensor grid_width,  // (b, 3), float
        float t_min = 0.,
        float t_max = 1.e12,
        std::string version = "v2"  // v2 is faster
) {
    
    assert(grid_size.dtype() == torch::kLong);

    auto batch_size = ray_origins.size(0);
    auto n_points = points.size(1);
    auto n_rays = ray_origins.size(1);
    auto device = ray_origins.device();
    auto total_cells = torch::prod(grid_size, -1);  // (b,) long

    // get grid idx of each point
    auto start = high_resolution_clock::now();
    auto tmp_grid_out = get_grid_idx(
        points,  
        grid_size,
        grid_center,
        grid_width,
        "ind"
    );  
    torch::Tensor &  grid_idxs = tmp_grid_out[0];  // (b, n), long  [0, total_cells-1]
    torch::Tensor & valid_mask = tmp_grid_out[1];  // (b, n), bool 
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    // std::cout<<"get_grid_idx: "<<duration.count()<<std::endl;
    

    // gather points belonging to each cell
    start = high_resolution_clock::now();
    std::vector<idxmap> all_cell2pidx = gather_points(
        grid_idxs,  // (b, n)
        total_cells,  // (b,)
        valid_mask  // (b, n)
    );  // b -> gidx -> point_idxs
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    // std::cout<<"gather points: "<<duration.count()<<std::endl;


    // gather the cells intersected by the ray
    start = high_resolution_clock::now();
    std::vector<idxmap> all_ray2gidxs;
    if (version.compare("v1") == 0) {
        all_ray2gidxs = std::move(grid_ray_intersection(
            ray_origins,
            ray_directions,
            ray_radius,
            grid_size,
            grid_center,
            grid_width
        ));  // b -> ray_idx -> grid_idx 
    }
    else if (version.compare("v2") == 0) {
        all_ray2gidxs = std::move(grid_ray_intersection_v2(
            ray_origins,
            ray_directions,
            ray_radius,
            grid_size,
            grid_center,
            grid_width
        ));  // b -> ray_idx -> grid_idx 
    }
    else
        throw;
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    // std::cout<<"grid intersection: "<<duration.count()<<std::endl;


    // accessor to points and rays
    start = high_resolution_clock::now();

    auto a_points = points.accessor<float, 3>();  // (b, n, 3)
    auto a_ray_origins = ray_origins.accessor<float, 3>();  // (b, n, 3)
    auto a_ray_directions = ray_directions.accessor<float, 3>();  // (b, n, 3)
    auto a_ray_radius = ray_radius.accessor<float, 1>();  // (b,)


    // gather points in the intersected cells
    std::vector<llist> all_ray2pidxs;  // b -> ray_idx -> point_idx
    all_ray2pidxs.reserve(batch_size);

    for (auto b = 0; b < batch_size; ++b) {
        llist ray2pidxs;  // vector of vector 
        ray2pidxs.reserve(n_rays);
        idxmap & ray2gidxs = all_ray2gidxs[b];  // ray_idx -> grid_idx 
        idxmap & cell2pidxs = all_cell2pidx[b];  // grid_idx -> point idxs

        auto a_points_b = a_points[b];  // (n, 3)
        auto a_ray_origins_b = a_ray_origins[b];  // (m, 3)
        auto a_ray_directions_b = a_ray_directions[b];  // (m, 3)
        auto ray_r = a_ray_radius[b];

        for (auto m = 0; m < n_rays; ++m) {
            std::unordered_set<int64_t> * p_gidxs = ray2gidxs.get_pidx(m);
            
            // if no cell is intersect, record a empty list
            if (!p_gidxs) {
                ray2pidxs.push_back(std::vector<int64_t>());
                continue;
            }

            std::vector<int64_t> pidxs; 
            for (auto iter_g = p_gidxs->begin(); iter_g != p_gidxs->end(); ++iter_g) {
                auto gidx = *iter_g;
                std::unordered_set<int64_t> * p_pidxs = cell2pidxs.get_pidx(gidx);

                if (!p_pidxs) 
                    continue;

                for (auto iter_p = p_pidxs->begin(); iter_p != p_pidxs->end(); ++iter_p) {
                    auto pidx = *iter_p;

                    // check if the point is actually within ray_radius
                    float dist, t; 
                    float pp[3] = {a_points_b[pidx][0], a_points_b[pidx][1], a_points_b[pidx][2]};
                    float ro[3] = {a_ray_origins_b[m][0], a_ray_origins_b[m][1], a_ray_origins_b[m][2]};
                    float rd[3] = {a_ray_directions_b[m][0], a_ray_directions_b[m][1], a_ray_directions_b[m][2]};
                    std::tie(dist, t) = compute_point_ray_distance_c(
                        pp,  // (3,)
                        ro,  // (3,)
                        rd  // (3,)
                    );
                    
                    if (dist > ray_r || t < t_min || t > t_max)
                        continue;

                    pidxs.push_back(*iter_p);
                }
            }
            ray2pidxs.push_back(std::move(pidxs));
        }
        all_ray2pidxs.push_back(std::move(ray2pidxs));
    }
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    // std::cout<<"collect result: "<<duration.count()<<std::endl;
    
    return all_ray2pidxs;
}







namespace py = pybind11;
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    // The following functions should not be called in python.
    // They are exposed for unit-testing.
    py::class_<idxmap>(m, "idxmap")
        .def(py::init())
        .def(py::init<size_t>())
        .def(py::init<const Table &>())
        .def("insert", &idxmap::insert)
        .def("get_pidx", &idxmap::get_pidx)
        .def("get_table", &idxmap::get_table);

    m.def("sub2ind", &sub2ind, "Sub index to linear index");
    m.def("ind2sub", &ind2sub, "linear index to sub index");
    m.def("gather_points", &gather_points, "Gather points in each grid cell");
    m.def("get_grid_idx", &get_grid_idx, "Compute the grid index a xyz point belongs to");
    m.def("grid_ray_intersection", &grid_ray_intersection, "Find out what are the grid cells a ray intersects with");
    m.def("grid_ray_intersection_v2", &grid_ray_intersection_v2, "Find out what are the grid cells a ray intersects with");
    m.def("find_neighbor_points_of_rays", &find_neighbor_points_of_rays, "Find the neighboring points of a ray");
}