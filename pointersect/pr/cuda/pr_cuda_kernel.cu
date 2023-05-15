// 
// Copyright (C) 2022 Apple Inc. All rights reserved.
// 

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <chrono>
using namespace torch::indexing;

namespace {

const int MAX_THREADS_PER_BLOCK = 1024;
const int MAX_K = 200;

__device__ __forceinline__ void ind2sub_d(
    const int64_t ind,
    const int64_t & sx,
    const int64_t & sy,
    const int64_t & sz,
    int64_t & x,
    int64_t & y,
    int64_t & z
) {
    auto yz_size = sy * sz;
    x = ind / yz_size;  
    auto _ind = ind - x * yz_size;
    y = _ind / sz; 
    z = _ind - y * sz;
}


__device__ __forceinline__ int64_t sub2ind_d(
    const int & x,
    const int & y,
    const int & z,
    const int & sx,
    const int & sy,
    const int & sz
) {
    return z + y * sz + x * (sy * sz);
}


template <typename scalar_t>
__device__ __forceinline__ void discretize(
    const scalar_t point[3],  // (3,)
    const scalar_t grid_from[3],   // (3,
    const scalar_t cell_width[3],  // (3,
    int64_t sub_idx[3]  // (3,
) {
    for (auto d = 0; d < 3; ++d) {
        sub_idx[d] = int64_t(floor((point[d] - grid_from[d]) / cell_width[d]));
    }
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
template <typename scalar_t>
__device__ __forceinline__ void compute_point_ray_distance_d(
    const scalar_t p_p[3],  // (3,)
    const scalar_t p_ro[3],  // (3,)
    const scalar_t p_rd[3],  // (3,)
    scalar_t & dist,
    scalar_t & t
) {
    scalar_t dv[3] = {
        p_p[0] - p_ro[0], 
        p_p[1] - p_ro[1], 
        p_p[2] - p_ro[2] 
    };

    t = dv[0] * p_rd[0] + dv[1] * p_rd[1] + dv[2] * p_rd[2];

    scalar_t dd[3] = {
        p_p[0] - (p_ro[0] + t * p_rd[0]),
        p_p[1] - (p_ro[1] + t * p_rd[1]),
        p_p[2] - (p_ro[2] + t * p_rd[2])
    };

    dist = sqrt(dd[0] * dd[0] + dd[1] * dd[1] + dd[2] * dd[2]);
}

template <typename T>
__device__ __forceinline__ void swap(
    T* arr,  // (3,)
    const int i,
    const int j
) {
    T temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;
}


template <typename scalar_t>
__device__ __forceinline__ void retain_k_smallest_values_in_max_heap(
    const scalar_t value,  // value to be inserted
    const int64_t id,     // id to be inserted
    const int k,    // heap max size
    int& n,           // current heap size, would change if a new element inserted
    scalar_t* value_heap,  // value max heap
    int64_t* id_heap     // id max heap
) {
    int i;
    if ( n < k ) {
        // if the heap is not full, insert the new element into end of the heap,
        id_heap[n] = id;
        value_heap[n] = value;

        // buttom up heapify
        i = n;
        n = n + 1;
        int parent;
        while(true) {
            parent = (i-1) /2;
            if (parent < 0)
                break;

            if ( value_heap[i] > value_heap[parent] ){
                swap(id_heap, i, parent);
                swap(value_heap, i, parent);
                i = parent;
            }else{
                break;
            }
        }
    } 
    else if ( n == k  && value < value_heap[0]) {
        // if the heap is full and a new element is coming
        // replace the root element (in 0) with new one
        id_heap[0] = id;
        value_heap[0] = value;

        // top down heapify
        // using while loop rather then recursive function to improve efficiency
        i = 0;
        int l;
        int r;
        int largest;
        while(true){
            largest = i;
            l = 2 * i + 1;
            r = 2 * i + 2;

            // finding the largest child
            if (  l < n && value_heap[l] > value_heap[largest] )
                largest = l;

            if (  r < n && value_heap[r] > value_heap[largest] )
                largest = r;

            // swap parent and largest child
            if (largest != i){
                swap(id_heap, i, largest);
                swap(value_heap, i, largest);
                i = largest;
            }else
                break;
        }
    }
}


// print current memory usage of current gpu context
void print_gpu_memory() {
    int num_gpus, gpu_id;
    size_t free, total;
    cudaGetDeviceCount( &num_gpus );
    cudaGetDevice(&gpu_id);
    cudaMemGetInfo(&free, &total);
    float free_mb = float(free) / (1024. * 1024.);
    float total_mb = float(total) / (1024. * 1024.);
    float used_mb = total_mb - free_mb;
    std::cout << "GPU " << gpu_id << " memory: used=" << used_mb << " MB, total=" << total_mb << " MB" << std::endl;

    // auto m1 = torch::cuda::memory_allocated();
    
    
}

} // namespace





template <typename scalar_t>
__global__ void get_grid_subidx_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> points,  // (b, n, 3)
    const torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> grid_size,  // (b, 3)
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grid_center,  // (b, 3)
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grid_width,  // (b, 3)
    torch::PackedTensorAccessor32<int64_t, 3, torch::RestrictPtrTraits> sub_idxs,  // (b, n, 3)
    torch::PackedTensorAccessor32<bool, 2, torch::RestrictPtrTraits> valid_mask,  // (b, n)
    torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> cell_counts  // (b, total_cells)
) {

    __shared__ int64_t gs[3];
    __shared__ scalar_t gc[3];
    __shared__ scalar_t gf[3];
    __shared__ scalar_t gw[3];
    __shared__ scalar_t cw[3];


    // batch index
    const int b = blockIdx.y;
    // point index
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    
    auto batch_size = points.size(0);
    auto n_points = points.size(1);

    if (p >= n_points || b >= batch_size)
        return;

    if (threadIdx.x == 0) {
        for (auto i = 0; i < 3; ++i) {
            gs[i] = grid_size[b][i];
            gc[i] = grid_center[b][i];
            gw[i] = grid_width[b][i];
            gf[i] = gc[i] - 0.5 * gw[i];
            cw[i] = grid_width[b][i] / scalar_t(gs[i]);
        }
    }
    __syncthreads(); 
    

    int64_t total_cells = gs[0] * gs[1] * gs[2];
    
    int64_t sub_idx[3];
    scalar_t point[3]; 
    point[0] = points[b][p][0]; 
    point[1] = points[b][p][1]; 
    point[2] = points[b][p][2]; 
    discretize<scalar_t>(point, gf, cw, sub_idx);

    // for (auto d = 0; d < 3; ++d) {
    //     grid_from[d] = grid_center[b][d] - 0.5 * grid_width[b][d];
    //     cell_width[d] = grid_width[b][d] / scalar_t(grid_size[b][d]);
    //     sub_idx[d] = int64_t(floor((points[b][p][d] - grid_from[d]) / cell_width[d]));
    // }

    bool valid = true;
    for (auto d = 0; d < 3; ++d) {
        if (sub_idx[d] < 0 || sub_idx[d] >= gs[d]) {
            valid = false;
            break;
        }
    }
    sub_idxs[b][p][0] = sub_idx[0];
    sub_idxs[b][p][1] = sub_idx[1];
    sub_idxs[b][p][2] = sub_idx[2];
    valid_mask[b][p] = valid;

    if (valid) {
        // record cell_counts
        auto ind = sub2ind_d(
            sub_idx[0],
            sub_idx[1],
            sub_idx[2],
            gs[0],
            gs[1],
            gs[2]
        );
        assert(ind >= 0 && ind < total_cells);
        atomicAdd(&(cell_counts[b][ind]), 1);
    }
}



template <typename scalar_t>
__global__ void get_grid_ind_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> points,  // (b, n, 3)
    const torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> grid_size,  // (b, 3)
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grid_center,  // (b, 3)
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grid_width,  // (b, 3)
    torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> inds,  // (b, n)
    torch::PackedTensorAccessor32<bool, 2, torch::RestrictPtrTraits> valid_mask,  // (b, n)
    torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> cell_counts  // (b, total_cells)
) {

    __shared__ int64_t gs[3];
    __shared__ scalar_t gc[3];
    __shared__ scalar_t gf[3];
    __shared__ scalar_t gw[3];
    __shared__ scalar_t cw[3];

    // batch index
    const int b = blockIdx.y;
    // point index
    const int p = blockIdx.x * blockDim.x + threadIdx.x;

    auto batch_size = points.size(0);
    auto n_points = points.size(1);

    if (p >= n_points || b >= batch_size)
        return;

    if (threadIdx.x == 0) {
        for (auto i = 0; i < 3; ++i) {
            gs[i] = grid_size[b][i];
            gc[i] = grid_center[b][i];
            gw[i] = grid_width[b][i];
            gf[i] = gc[i] - 0.5 * gw[i];
            cw[i] = grid_width[b][i] / scalar_t(gs[i]);
        }
    }
    __syncthreads(); 

    int64_t total_cells = gs[0] * gs[1] * gs[2];

    int64_t sub_idx[3];
    scalar_t point[3]; 
    point[0] = points[b][p][0]; 
    point[1] = points[b][p][1]; 
    point[2] = points[b][p][2]; 
    discretize<scalar_t>(point, gf, cw, sub_idx);

    // for (auto d = 0; d < 3; ++d) {
    //     grid_from[d] = grid_center[b][d] - 0.5 * grid_width[b][d];
    //     cell_width[d] = grid_width[b][d] / scalar_t(grid_size[b][d]);
    //     sub_idx[d] = int64_t(floor((points[b][p][d] - grid_from[d]) / cell_width[d]));
    // }

    auto ind = sub2ind_d(
        sub_idx[0],
        sub_idx[1],
        sub_idx[2],
        gs[0],
        gs[1],
        gs[2]
    );
    inds[b][p] = ind;
    
    bool valid = true;
    for (auto d = 0; d < 3; ++d) {
        if (sub_idx[d] < 0 || sub_idx[d] >= gs[d]) {
            valid = false;
            break;
        }
    }
    valid_mask[b][p] = valid;

    // record cell_counts
    if (valid) {
        assert(ind >= 0 && ind < total_cells);
        atomicAdd(&(cell_counts[b][ind]), 1);
    }
}


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
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> get_grid_idx_cuda(
        const torch::Tensor & points,  // (b, n, 3), float
        const torch::Tensor & grid_size,  // (b, 3), long
        const torch::Tensor & grid_center,  // (b, 3), float
        const torch::Tensor & grid_width,  // (b, 3), float
        const std::string & mode = "ind"  
) { 

    assert(grid_size.dtype() == torch::kLong);
    auto batch_size = points.size(0);
    auto n_points = points.size(1);
    auto device = points.device();

    auto long_options = torch::TensorOptions()
        .dtype(torch::kLong)
        .device(device);
    auto int_options = torch::TensorOptions()
        .dtype(torch::kInt32)
        .device(device);
    auto bool_options = torch::TensorOptions()
        .dtype(torch::kBool)
        .device(device);

    // allocate memory to record cell_count
    int64_t max_total_cells = torch::prod(grid_size, -1).max().item<int64_t>();  
    torch::Tensor cell_counts = torch::zeros({batch_size, max_total_cells}, int_options);
    

    // Parallelization strategy: 
    // we use one thread to handle the discretization of a point.
    const int threads = MAX_THREADS_PER_BLOCK;
    const dim3 blocks((n_points + threads - 1) / threads, batch_size);

    torch::Tensor grid_idx; 
    torch::Tensor valid_mask;
    if (mode.compare("ind") == 0) {
        // allocate memory
        grid_idx = torch::zeros({batch_size, n_points}, long_options);
        valid_mask = torch::zeros({batch_size, n_points}, bool_options);

        AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "get_grid_ind_cuda", ([&] {
            get_grid_ind_kernel<scalar_t><<<blocks, threads>>>(
                points.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                grid_size.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
                grid_center.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                grid_width.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                grid_idx.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
                valid_mask.packed_accessor32<bool, 2, torch::RestrictPtrTraits>(),
                cell_counts.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>()
            );
        }));
    }
    else if (mode.compare("subidx") == 0) {
        // allocate memory
        grid_idx = torch::zeros({batch_size, n_points, 3}, long_options);
        valid_mask = torch::zeros({batch_size, n_points}, bool_options);

        AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "get_grid_subidx_cuda", ([&] {
            get_grid_subidx_kernel<scalar_t><<<blocks, threads>>>(
                points.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                grid_size.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
                grid_center.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                grid_width.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                grid_idx.packed_accessor32<int64_t, 3, torch::RestrictPtrTraits>(),
                valid_mask.packed_accessor32<bool, 2, torch::RestrictPtrTraits>(),
                cell_counts.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>()
            );
        }));
    }
    else {
        throw;
    }

    return {grid_idx, valid_mask, cell_counts};
}



template <typename scalar_t>
__global__ void gather_points_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grid_idxs,  // (b, n)
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> total_cells,  // (b, )
    const torch::PackedTensorAccessor32<bool, 2, torch::RestrictPtrTraits> valid_mask,  // (b, n)
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> cell_counts,  // (b, total_cells)
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> ** a_all_cell2pidx  // (b, cell_idx, pidx_in_cell)
) {
    // strategy:
    // 1. divide grid_idxs into blocks that fit shared memory
    // 2. for each block, use all threads to load grid_idxs into shared memory
    // 3. each thread handles writing into a_all_cell2pidx

    const auto memsize = MAX_THREADS_PER_BLOCK;
    assert(blockDim.x == memsize);
    __shared__ int shared_mem[memsize];  // we use int to catch int64_t (assuming not that many points)
    __shared__ bool shared_mem_valid[memsize];


    // batch index
    const int b = blockIdx.y;
    // cell index
    const int cell_idx = blockIdx.x * blockDim.x + threadIdx.x;

    auto batch_size = grid_idxs.size(0);
    auto n_points = grid_idxs.size(1);
    auto max_num_cells = total_cells[b];
    auto num_blocks = (n_points + memsize - 1) / memsize;

    int32_t max_count = 0;
    if (cell_idx < total_cells[b] && b < batch_size) {
        max_count = cell_counts[b][cell_idx];
    }

    int count = 0;  // num of points collected
    for (auto block_idx = 0; block_idx < num_blocks; ++block_idx) {
        // use all threads to load each block
        auto p = block_idx * memsize + threadIdx.x;
        if (p < n_points) {
            shared_mem[threadIdx.x] = grid_idxs[b][p]; 
            shared_mem_valid[threadIdx.x] = valid_mask[b][p]; 
        }
        else {
            shared_mem[threadIdx.x] = -1;
            shared_mem_valid[threadIdx.x] = false;
        }
        __syncthreads(); 

        // each thread go through the entire shared_mem
        for (auto i = 0; i < memsize; ++i) {
            if (shared_mem[i] < 0)
                break;
            if (shared_mem[i] == cell_idx && shared_mem_valid[i]) {
                auto pidx = block_idx * memsize + i;
                assert(count < max_count);
                a_all_cell2pidx[b][cell_idx][count] = pidx;
                ++count;
            }
        }
        __syncthreads(); 
    }
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
// 
// Returns:
//     vector of vector of tensors, b -> cell_idx -> pidx of points in the cell
// 
std::vector<std::vector<torch::Tensor> > gather_points_cuda(
        const torch::Tensor & grid_idxs,  // (b, n), long
        const torch::Tensor & total_cells, // (b,) int64_t
        const torch::Tensor & valid_mask,  // (b, n), bool
        const torch::Tensor & cell_counts  // (b, n_cells), int32
) {
    assert(grid_idxs.dtype() == torch::kLong);
    assert(total_cells.dtype() == torch::kLong);
    assert(valid_mask.dtype() == torch::kBool);

    auto batch_size = grid_idxs.size(0);
    auto n_points = grid_idxs.size(1);
    auto max_num_cells = cell_counts.size(1);
    auto device = grid_idxs.device();

    auto long_options = torch::TensorOptions()
        .dtype(torch::kLong)
        .device(device);
    auto int_options = torch::TensorOptions()
        .dtype(torch::kInt32)
        .device(device);
    auto bool_options = torch::TensorOptions()
        .dtype(torch::kBool)
        .device(device);

    // accessor 
    auto total_cells_cpu = total_cells.cpu();
    auto cell_counts_cpu = cell_counts.cpu();
    auto a_total_cells = total_cells_cpu.accessor<int64_t, 1>();
    auto a_cell_counts = cell_counts_cpu.accessor<int32_t, 2>();

    // allocate memory to record cell2pidx
    std::vector<std::vector<torch::Tensor> > all_cell2pidx;
    all_cell2pidx.reserve(batch_size);

    for (auto b = 0; b < batch_size; ++b) {
        all_cell2pidx.push_back(std::vector<torch::Tensor>());
        std::vector<torch::Tensor> & cell2pidx = all_cell2pidx[b];
        cell2pidx.reserve(a_total_cells[b]);
        for (auto gidx = 0; gidx < a_total_cells[b]; ++gidx) {
            cell2pidx.push_back(torch::zeros({a_cell_counts[b][gidx]}, long_options));
        }
    }

    // create packed_accessor for kernel
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> ** a_all_cell2pidx;  // (b, n_cell, n_points_in_cell)
    cudaMallocManaged(&a_all_cell2pidx, batch_size * sizeof(torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits>*));
    for (auto b = 0; b < batch_size; ++b) {
        cudaMallocManaged(&(a_all_cell2pidx[b]), a_total_cells[b] * sizeof(torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits>));
        for (auto gidx = 0; gidx < a_total_cells[b]; ++gidx) {
            a_all_cell2pidx[b][gidx] = all_cell2pidx[b][gidx].packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>();
        }
    }

    // Parallelization strategy: 
    // we use one thread to handle a cell (putting pidx into the corresponding cell in a_all_cell2pidx)
    const int threads = MAX_THREADS_PER_BLOCK;
    const dim3 blocks((max_num_cells + threads - 1) / threads, batch_size);


    AT_DISPATCH_INTEGRAL_TYPES(grid_idxs.scalar_type(), "gather_points_cuda", ([&] {
        gather_points_kernel<scalar_t><<<blocks, threads>>>(
            grid_idxs.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            total_cells.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            valid_mask.packed_accessor32<bool, 2, torch::RestrictPtrTraits>(),
            cell_counts.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            a_all_cell2pidx
        );
    }));
    // cudaDeviceSynchronize();

    // free cuda memory
    for (auto b = 0; b < batch_size; ++b) {
        cudaFree(a_all_cell2pidx[b]); 
    }
    cudaFree(a_all_cell2pidx);

    return all_cell2pidx;
}


template <typename scalar_t>
__global__ void gather_points_kernel_v2(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grid_idxs,  // (b, n)
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> total_cells,  // (b, )
    const torch::PackedTensorAccessor32<bool, 2, torch::RestrictPtrTraits> valid_mask,  // (b, n)
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> cell_counts,  // (b, total_cells)
    torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> membank,  // (b, n)
    torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> cell_current_idx  // (b, n_cells)
) {
    
    // batch index
    const int b = blockIdx.y;
    // point index
    const int pidx = blockIdx.x * blockDim.x + threadIdx.x;

    auto batch_size = grid_idxs.size(0);
    auto n_points = grid_idxs.size(1);

    if (pidx >= n_points || b >= batch_size || valid_mask[b][pidx] == false) {
        return;
    }

    auto gidx = grid_idxs[b][pidx];

    // write pidx to membank
    int32_t current_idx = atomicAdd(&(cell_current_idx[b][gidx]), 1);  
    assert(current_idx < n_points);
    membank[b][current_idx] = pidx;
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
// 
// Returns:
//     membank: 
//         (b, n), should use the pidx of cell_idx is 
//         from cell_start_idx[b][cell_idx] to cell_counts[b][cell_idx+1] (excluded).
//     cell_start_idx:
//         (b, n_cells+1), 
// 
std::tuple<torch::Tensor, torch::Tensor> gather_points_cuda_v2(
        const torch::Tensor & grid_idxs,  // (b, n), long
        const torch::Tensor & total_cells, // (b,) int64_t
        const torch::Tensor & valid_mask,  // (b, n), bool
        const torch::Tensor & cell_counts  // (b, n_cells), int32
) {
    assert(grid_idxs.dtype() == torch::kLong);
    assert(total_cells.dtype() == torch::kLong);
    assert(valid_mask.dtype() == torch::kBool);

    auto batch_size = grid_idxs.size(0);
    auto n_points = grid_idxs.size(1);
    auto max_num_cells = cell_counts.size(1);
    auto device = grid_idxs.device();

    auto long_options = torch::TensorOptions()
        .dtype(torch::kLong)
        .device(device);
    auto int_options = torch::TensorOptions()
        .dtype(torch::kInt32)
        .device(device);
    auto bool_options = torch::TensorOptions()
        .dtype(torch::kBool)
        .device(device);

    // allocate memory for all_cell2pix
    auto membank = torch::zeros({batch_size, n_points}, long_options);  // (b, n)
    auto cell_current_idx = torch::cat({
        torch::zeros({batch_size, 1}, int_options),
        torch::cumsum(cell_counts, -1).to(int_options)
    }, -1);  // (b, n_cells+1)
    auto cell_start_idx = cell_current_idx.clone();

    
    // Parallelization strategy: 
    // we use one thread to handle a point 
    const int threads = MAX_THREADS_PER_BLOCK;
    const dim3 blocks((n_points + threads - 1) / threads, batch_size);

    // std::cout<<"grid_idxs: "<<grid_idxs.options()<<std::endl;
    // std::cout<<"total_cells: "<<total_cells.options()<<std::endl;
    // std::cout<<"valid_mask: "<<valid_mask.options()<<std::endl;
    // std::cout<<"cell_counts: "<<cell_counts.options()<<std::endl;
    // std::cout<<"membank: "<<membank.options()<<std::endl;
    // std::cout<<"cell_current_idx: "<<cell_current_idx.options()<<std::endl;

    // std::cout<<"(before) cell_current_idx:  "<<std::endl;
    // std::cout<<cell_current_idx;

    AT_DISPATCH_INTEGRAL_TYPES(grid_idxs.scalar_type(), "gather_points_cuda_v2", ([&] {
        gather_points_kernel_v2<scalar_t><<<blocks, threads>>>(
            grid_idxs.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            total_cells.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            valid_mask.packed_accessor32<bool, 2, torch::RestrictPtrTraits>(),
            cell_counts.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            membank.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
            cell_current_idx.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>()
        );
    }));

    // std::cout<<"(after) cell_current_idx: "<<std::endl;
    // std::cout<<cell_current_idx;


    return {membank, cell_start_idx};
}


// calculate whether to use x-plane, y-plane, or z-plane as the main direction
template <typename scalar_t>
__global__ void determine_main_direction(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> ray_directions,  // (b, m, 3)
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> ray_radius,  // (b, )
    const torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> grid_size,  // (b, 3)
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grid_width,  // (b, 3)
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> inv_ts,  // (b, m, 3)
    torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> idx_to_uses,  // (b, m)  0,1,2
    torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> local_grid_sizes,  // (b, m)
    torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> local_rs  // (b, m, 3)
) {
    
    // //debug
    // printf("in determine_main_direction, threadIdx.x = %d\n", threadIdx.x);

    __shared__ scalar_t rr;
    __shared__ scalar_t cw[3];
    __shared__ int64_t gs[3];
     __shared__ scalar_t gw[3];

    // batch index
    const int b = blockIdx.y;
    // ray index
    const int ridx = blockIdx.x * blockDim.x + threadIdx.x;

    auto batch_size = ray_directions.size(0);
    auto n_rays = ray_directions.size(1);

    if (ridx >= n_rays || b >= batch_size) {
        return;
    }

    // gather information
    scalar_t rd[3] = {
        ray_directions[b][ridx][0], 
        ray_directions[b][ridx][1],
        ray_directions[b][ridx][2]
    };
    if (threadIdx.x == 0) {
        rr = ray_radius[b];
        for (auto i = 0 ; i < 3; ++i) {
            gs[i] = grid_size[b][i];
            gw[i] = grid_width[b][i];
            cw[i] = gw[i] / scalar_t(gs[i]);
        }
    }
    __syncthreads(); 

    scalar_t inv_t[3]; 
    int32_t idx_to_use = 0;
    scalar_t current_max = -1;
    for (auto i = 0; i < 3; ++i) {
        inv_t[i] = abs(rd[i] / cw[i]);
        if (inv_t[i] > current_max) {
            current_max = inv_t[i];
            idx_to_use = i;
        }
    }

    int32_t local_r[3]; 
    for (auto i = 0; i < 3; ++i) {
        // drawing a larger square to contain nearby grid is more robust
        // original implementation fails in some cases, and seems not to be the boundary issue
        // note that abs(rd[idx_to_use]) < sqrt(1. - (rd[i] * rd[i]) for i != idx_to_use
        // local_r[i] = 1 + int32_t(ceil(rr / (abs(rd[idx_to_use]) * cw[i])));
        local_r[i] = 1 + int32_t(ceil(rr / (sqrt(1. - (rd[i] * rd[i])) * cw[i])));
    }

    local_r[idx_to_use] = 0;
    int32_t local_grid_size = 1;
    for (auto i = 0; i < 3; ++i) {
        local_grid_size = local_grid_size * (local_r[i] * 2 + 1);
    }
    local_grid_size = local_grid_size * (int32_t)gs[idx_to_use];
   
    // write result to global memory
    inv_ts[b][ridx][0] = inv_t[0];  
    inv_ts[b][ridx][1] = inv_t[1];
    inv_ts[b][ridx][2] = inv_t[2];
    idx_to_uses[b][ridx] = idx_to_use; 
    local_grid_sizes[b][ridx] = local_grid_size;
    local_rs[b][ridx][0] = local_r[0];  
    local_rs[b][ridx][1] = local_r[1];
    local_rs[b][ridx][2] = local_r[2];
}


// intersect a ray with the grid, get its neighbors
template <typename scalar_t>
__global__ void traverse_grid(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> ray_origins,  // (b, m, 3)
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> ray_directions,  // (b, m, 3)
    const torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> grid_size,  // (b, 3)
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grid_center,  // (b, 3)
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grid_width,  // (b, 3)
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> idx_to_uses,  // (b, m)  0,1,2
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> local_grid_sizes,  // (b, m)
    const torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> local_rs,  // (b, m, 3)
    torch::PackedTensorAccessor32<int64_t, 3, torch::RestrictPtrTraits> ray2gidxs,  // (b, m, max_n_gidx)
    torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> n_ray2gidxs  // (b, m)
) {

    // //debug
    // printf("in traverse_grid, threadIdx.x = %d\n", threadIdx.x);

    __shared__ int gs[3];
    __shared__ scalar_t gc[3];
    __shared__ scalar_t gf[3];
    __shared__ scalar_t cw[3];
    __shared__ scalar_t gw[3];

    // batch index
    const int b = blockIdx.y;
    // ray index
    const int ridx = blockIdx.x * blockDim.x + threadIdx.x;

    auto batch_size = ray_directions.size(0);
    auto n_rays = ray_directions.size(1);

    
    if (ridx >= n_rays || b >= batch_size) {
        return;
    }

    if (threadIdx.x == 0) {
        for (auto i = 0; i < 3; ++i) {
            gs[i] = int(grid_size[b][i]);
            gw[i] = grid_width[b][i];
            gc[i] = grid_center[b][i];
            gf[i] = gc[i] - 0.5 * gw[i];
            cw[i] = gw[i] / scalar_t(gs[i]);
        }
    }
    __syncthreads(); 


    // // debug
    // printf("gs = (%d, %d, %d), grid_size[%ld] = (%ld, %ld, %ld)\n", 
    //     gs[0], gs[1], gs[2],
    //     b,
    //     grid_size[b][0], grid_size[b][1], grid_size[b][2]
    // );

    const int32_t idx_to_use = idx_to_uses[b][ridx];
    scalar_t ro[3], rd[3];
    for (auto i = 0; i < 3; ++i) {
        ro[i] = ray_origins[b][ridx][i];
        rd[i] = ray_directions[b][ridx][i];
    }

    
    scalar_t plane_c = gf[idx_to_use] + 0.5 * cw[idx_to_use];
    scalar_t plane_t = (plane_c - ro[idx_to_use]) / rd[idx_to_use];
    scalar_t t_step = cw[idx_to_use] / rd[idx_to_use]; 

    const int rx = local_rs[b][ridx][0];
    const int ry = local_rs[b][ridx][1];
    const int rz = local_rs[b][ridx][2];
    const int max_num_gidxs = local_grid_sizes[b][ridx];
    int x, y, z;

    // compute ray-plane intersection for each plane
   
    scalar_t p[3];
    int64_t p_subidx[3];

    // //debug
    // printf("ridx / nray = %d / %d,  b / batch_size = %d / %d\n", (int)ridx, (int)n_rays, (int)b, (int)batch_size);
    // printf("max_num_gidx = %d\n", (int)max_num_gidxs);
    // printf("gs[%d] = %d\n", (int)idx_to_use, (int)gs[idx_to_use]);

    int32_t current_idx = 0;
    for (auto plane_idx = 0; plane_idx < gs[idx_to_use]; ++plane_idx) {
        // for each intersection point, gather all neighboring cells
        for (auto i = 0; i < 3; ++i) {
            p[i] = ro[i] + plane_t * rd[i];
        }
        // convert p to grid's sub_idx, p_subidx can < 0 or >= grid_size
        discretize<scalar_t>(p, gf, cw, p_subidx); 

        //debug
        // printf("plane_idx = %d, current_idx = %d, p = (%.2f, %.2f, %.2f),  p_sub_idx = (%d, %d, %d), local_r = (%d, %d, %d)\n", 
        //        (int)plane_idx,
        //        (int)current_idx,
        //        (float)p[0], (float)p[1], (float)p[2],
        //        (int)p_subidx[0], (int)p_subidx[1], (int)p_subidx[2],
        //        (int)rx, (int)ry, (int)rz
        // );

        // add neighbor grids
        for (int ix = -rx; ix <= rx; ++ix) {
            x = int(p_subidx[0]) + ix;
            if (x < 0 || x >= gs[0]) 
                continue;

            for (int iy = -ry; iy <= ry; ++iy) {
                y = int(p_subidx[1]) + iy;
                if (y < 0 || y >= gs[1]) 
                    continue;

                for (int iz = -rz; iz <= rz; ++iz) {
                    z = int(p_subidx[2]) + iz;
                    if (z < 0 || z >= gs[2]) 
                        continue;

                    ray2gidxs[b][ridx][current_idx] = sub2ind_d(
                        x, y, z, 
                        (int)gs[0], (int)gs[1], (int)gs[2]
                    );
                    ++current_idx;
                }
            }
        }

        plane_t += t_step;
    }
    n_ray2gidxs[b][ridx] = current_idx;

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
//          (b, m), number of index used in ray2gidx
// 
std::tuple<torch::Tensor, torch::Tensor> grid_ray_intersection_cuda(
        const torch::Tensor & ray_origins,  // (b, m, 3), float
        const torch::Tensor & ray_directions,  // (b, m, 3), float
        const torch::Tensor & ray_radius,   //  (b, ) float
        const torch::Tensor & grid_size,  // (b, 3), long
        const torch::Tensor & grid_center,  // (b, 3) float
        const torch::Tensor & grid_width  // (b, 3) float
) {
    assert(grid_size.dtype() == torch::kLong);

    auto batch_size = ray_origins.size(0);
    auto n_rays = ray_origins.size(1);
    auto device = ray_origins.device();
    auto dtype = ray_origins.dtype();

    auto long_options = torch::TensorOptions()
        .dtype(torch::kLong)
        .device(device);
    auto int_options = torch::TensorOptions()
        .dtype(torch::kInt32)
        .device(device);
    auto float_options = torch::TensorOptions()
        .dtype(dtype)
        .device(device);

    // Parallelization strategy: 
    // we use one thread to handle a ray 
    const int threads = MAX_THREADS_PER_BLOCK;
    const dim3 blocks((n_rays + threads - 1) / threads, batch_size);

    // calculate the main direction and max number of neighbor grids among all rays
    torch::Tensor inv_ts = torch::zeros({batch_size, n_rays, 3}, float_options);  // (b, m, 3)
    torch::Tensor idx_to_uses = torch::zeros({batch_size, n_rays}, int_options);  // (b, m)
    torch::Tensor local_grid_sizes = torch::zeros({batch_size, n_rays}, int_options);  // (b, m)
    torch::Tensor local_rs = torch::zeros({batch_size, n_rays, 3}, int_options);  // (b, m, 3)
    // auto start = std::chrono::high_resolution_clock::now();

    AT_DISPATCH_FLOATING_TYPES(ray_directions.scalar_type(), "determine_main_direction_", ([&] {
        determine_main_direction<scalar_t><<<blocks, threads>>>(
            ray_directions.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            ray_radius.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            grid_size.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
            grid_width.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            inv_ts.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            idx_to_uses.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            local_grid_sizes.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            local_rs.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>()
        );
    }));
    // auto stop = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    // std::cout<<"--determine_main_direction_: "<<duration.count()<<" us"<<std::endl;
    // torch::cuda::synchronize();

    // std::cout<<"local_grid_sizes:"<<std::endl;
    // std::cout<<local_grid_sizes<<std::endl;

    // std::cout<<"inv_ts:"<<std::endl;
    // std::cout<<inv_ts<<std::endl;

    // std::cout<<"idx_to_uses:"<<std::endl;
    // std::cout<<idx_to_uses<<std::endl;

    // std::cout<<"local_rs:"<<std::endl;
    // std::cout<<local_rs<<std::endl;

    

    // start = std::chrono::high_resolution_clock::now();
    // allocate memory for neighbor grid index for each ray
    int64_t max_local_grid_size = local_grid_sizes.max().detach().cpu().item<int64_t>();
    //int64_t max_local_grid_size = local_grid_sizes.max().item<int64_t>();

    // stop = std::chrono::high_resolution_clock::now();
    // duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    // std::cout<<"--max_local_grid_size: "<<duration.count()<<" us"<<std::endl;

    // start = std::chrono::high_resolution_clock::now();
    torch::Tensor ray2gidxs = torch::empty({
        batch_size, 
        n_rays, 
        max_local_grid_size  // nplane * num_grids
    }, long_options);  // (b, m, max_n_gidx)
    ray2gidxs.fill_(-1);
    torch::Tensor n_ray2gidxs = torch::zeros({
        batch_size, 
        n_rays
    }, long_options);  // (b, m)
    // stop = std::chrono::high_resolution_clock::now();
    // duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    // std::cout<<"--allocate memory for neighbor grid index for each ray: "<<duration.count()<<" us"<<std::endl;


    // torch::cuda::synchronize();

    // std::cout<<"before calling traverse_grid..."<<std::endl;
    // cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 2<<9);

    // traverse_grid<float><<<blocks, threads>>>(
    //     ray_origins.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
    //     ray_directions.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
    //     grid_size.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
    //     grid_center.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
    //     grid_width.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
    //     idx_to_uses.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
    //     local_grid_sizes.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
    //     local_rs.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
    //     ray2gidxs.packed_accessor32<int64_t, 3, torch::RestrictPtrTraits>(),
    //     n_ray2gidxs.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>()
    // );

    // start = std::chrono::high_resolution_clock::now();
    AT_DISPATCH_FLOATING_TYPES(ray_directions.scalar_type(), "traverse_grid_", ([&] {
        traverse_grid<scalar_t><<<blocks, threads>>>(
            ray_origins.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            ray_directions.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            grid_size.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
            grid_center.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            grid_width.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            idx_to_uses.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            local_grid_sizes.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            local_rs.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
            ray2gidxs.packed_accessor32<int64_t, 3, torch::RestrictPtrTraits>(),
            n_ray2gidxs.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>()
        );
    }));
    // stop = std::chrono::high_resolution_clock::now();
    // duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    // std::cout<<"--traverse_grid_: "<<duration.count()<<" us"<<std::endl;
    // torch::cuda::synchronize();
    
    // std::cout<<"blocks:"<<std::endl;
    // printf("%d, %d, %d\n", blocks.x, blocks.y, blocks.z);

    // std::cout<<"threads:"<<std::endl;
    // std::cout<<threads<<std::endl;

    // std::cout<<"ray2gidxs:"<<std::endl;
    // std::cout<<ray2gidxs<<std::endl;

    // std::cout<<"n_ray2gidxs:"<<std::endl;
    // std::cout<<n_ray2gidxs<<std::endl;
    // fflush(NULL);

    return {ray2gidxs, n_ray2gidxs};
}





// compute the max number of points per ray
template <typename scalar_t>
__global__ void compute_max_num_points(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> ray2gidx,  // (b, m, n_gidxs)  long
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> n_ray2gidx,  // (b, m)
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> gidx2pidx_bank,  // (b, n)
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> gidx_start_idx,  // (b, n_cells+1)
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> ray2npoints  // (b, m)
) {
   // batch index
    const int b = blockIdx.y;
    // ray index
    const int ridx = blockIdx.x * blockDim.x + threadIdx.x;

    auto batch_size = ray2gidx.size(0);
    auto n_rays = ray2gidx.size(1);

    if (ridx >= n_rays || b >= batch_size) {
        return;
    }

    scalar_t count = 0;
    for (auto i = 0; i < n_ray2gidx[b][ridx]; ++i) {
        auto gidx = ray2gidx[b][ridx][i];
        auto n_points_in_gidx = gidx_start_idx[b][gidx+1] - gidx_start_idx[b][gidx];
        count += (scalar_t)n_points_in_gidx;
    }
    ray2npoints[b][ridx] = count;
}



// fill in pidx that is within ray_radius
template <typename scalar_t>
__global__ void filter_and_fill_in_pidx(
    const torch::PackedTensorAccessor32<int64_t, 3, torch::RestrictPtrTraits> ray2gidx,  // (b, m, n_gidxs)  long
    const torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> n_ray2gidx,  // (b, m)
    const torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> gidx2pidx_bank,  // (b, n)
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> gidx_start_idx,  // (b, n_cells+1)
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> points,  // (b, n, 3)
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> ray_origins,  // (b, m, 3)
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> ray_directions,  // (b, m, 3)
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> ray_radius,  // (b,)
    const torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> ray_start_idx,  // (b, m)
    torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> ray2pidx_bank,  // (b, M)
    torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> ray_end_idx,  // (b, m)
    const float t_min = 0.0,
    const float t_max = 1.0e12
) {

    __shared__ scalar_t ray_r;

    // batch index
    const int b = blockIdx.y;
    // ray index
    const int ridx = blockIdx.x * blockDim.x + threadIdx.x;
    
    auto batch_size = ray_origins.size(0);
    auto n_rays = ray_origins.size(1);
    auto n_points = points.size(1);


    if (ridx >= n_rays || b >= batch_size) {
        return;
    }

    if (threadIdx.x == 0) {
        ray_r = ray_radius[b];
    }
    __syncthreads(); 

    scalar_t ro[3];
    scalar_t rd[3];
    for (auto i = 0; i < 3; ++i) {
        ro[i] = ray_origins[b][ridx][i];
        rd[i] = ray_directions[b][ridx][i];
    }

    int current_idx = ray_start_idx[b][ridx];
    const int max_idx = ray_start_idx[b][ridx+1];
    int64_t gidx;
    int64_t pidx;
    scalar_t pp[3];
    scalar_t dist;
    scalar_t t;
    for (auto ir = 0; ir < n_ray2gidx[b][ridx]; ++ir) {
        gidx = ray2gidx[b][ridx][ir];
        // printf("(%d, %d) gidx = %d \n", (int)b, (int)ridx, (int)(gidx));
        auto ip_start = gidx_start_idx[b][gidx];
        auto ip_end = gidx_start_idx[b][gidx+1];

        for (auto ip = ip_start; ip < ip_end; ++ip) {
            pidx = gidx2pidx_bank[b][ip];
            if (pidx < 0 || pidx >= n_points) { // -1 only happen at the end
                printf("(%d, %d) num points = %d\n", (int)b, (int)ridx, (int)(current_idx - ray_start_idx[b][ridx]));
                break;
            }
            // compute point-ray distance
            pp[0] = points[b][pidx][0];
            pp[1] = points[b][pidx][1];
            pp[2] = points[b][pidx][2];

            compute_point_ray_distance_d(
                pp,  // (3,)
                ro,  // (3,)
                rd,  // (3,)
                dist,
                t
            );

            if (dist > ray_r || t < t_min || t > t_max)
                continue;

            assert(current_idx < max_idx);
            ray2pidx_bank[b][current_idx] = pidx;
            ++current_idx;   
        }
    }
    ray_end_idx[b][ridx] = current_idx;
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
//     ray2pidx_bank: 
//          (b, M), all the points. The pidxs of a ray is found via ray2pidx_bank[b][ray_start_idx[m]:ray_end_idx[m]]
//     ray_start_idx:
//          (b, m),
//     ray_end_idx: 
//          (b, m),
// 
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> collect_points_on_ray_cuda(
        const torch::Tensor & ray2gidx,  // (b, m, n_gidxs)  long
        const torch::Tensor & n_ray2gidx,  // (b, m) long
        const torch::Tensor & gidx2pidx_bank,  // (b, n) long
        const torch::Tensor & gidx_start_idx,  // (b, n_cells+1)  int32
        const torch::Tensor & points,  // (b, n, 3), float
        const torch::Tensor & ray_origins,  // (b, m, 3), float
        const torch::Tensor & ray_directions,  // (b, m, 3), float
        const torch::Tensor & ray_radius,   //  (b, ) float
        float t_min = 0.0,
        float t_max = 1.0e12
) {

    assert(ray2gidx.dtype() == torch::kLong);
    assert(n_ray2gidx.dtype() == torch::kLong);
    assert(gidx2pidx_bank.dtype() == torch::kLong);
    assert(gidx_start_idx.dtype() == torch::kInt32);


    auto batch_size = ray_origins.size(0);
    auto n_rays = ray_origins.size(1);
    auto device = ray_origins.device();
    auto dtype = ray_origins.dtype();

    auto long_options = torch::TensorOptions()
        .dtype(torch::kLong)
        .device(device);
    auto int_options = torch::TensorOptions()
        .dtype(torch::kInt32)
        .device(device);
    auto float_options = torch::TensorOptions()
        .dtype(dtype)
        .device(device);


    // compute max amount of memory needed for each ray
    torch::Tensor ray2npoints = torch::zeros({batch_size, n_rays}, long_options);

    // Parallelization strategy: 
    // we use one thread to handle a ray 
    const int threads = 512;  // MAX_THREADS_PER_BLOCK;
    const dim3 blocks((n_rays + threads - 1) / threads, batch_size);

    AT_DISPATCH_INTEGRAL_TYPES(ray2gidx.scalar_type(), "compute_max_num_points_", ([&] {
        compute_max_num_points<scalar_t><<<blocks, threads>>>(
            ray2gidx.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            n_ray2gidx.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            gidx2pidx_bank.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            gidx_start_idx.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            ray2npoints.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()
        );
    }));
    // torch::cuda::synchronize();
    auto max_count = ray2npoints.sum(-1).max().detach().cpu().item<int64_t>();
    torch::Tensor ray2pidx_bank = torch::empty({batch_size, max_count}, long_options);  // (b, M)
    ray2pidx_bank.fill_(-1);


    // std::cout<<"ray2npoints:"<<std::endl;
    // std::cout<<ray2npoints<<std::endl;
    // std::cout<<"max_count:"<<std::endl;
    // std::cout<<max_count<<std::endl;


    auto ray_start_idx = torch::cat({
        torch::zeros({batch_size, 1}, long_options),
        torch::cumsum(ray2npoints, -1).to(long_options)
    }, -1);  // (b, n_rays+1)

    auto ray_end_idx = torch::zeros({batch_size, n_rays}, long_options); // (b, n_rays)
    
    // std::cout<<"finish allocating ray_start ray_end idx:"<<std::endl;

    // torch::cuda::synchronize();

    // std::cout<<"starting filter_and_fill_in_pidx:"<<std::endl;
    
    // record the pidx into the membank (ignore points not within ray_radius)
    AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "filter_and_fill_in_pidx_", ([&] {
        filter_and_fill_in_pidx<scalar_t><<<blocks, threads>>>(
            ray2gidx.packed_accessor32<int64_t, 3, torch::RestrictPtrTraits>(),
            n_ray2gidx.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
            gidx2pidx_bank.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
            gidx_start_idx.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            points.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            ray_origins.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            ray_directions.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            ray_radius.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            ray_start_idx.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
            ray2pidx_bank.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
            ray_end_idx.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
            t_min,
            t_max
        );
    }));

    // torch::cuda::synchronize();
    ray_start_idx = ray_start_idx.index({"...", Slice(0,-1)});  // (b, n_rays)

    // auto num_points = ray_end_idx - ray_start_idx;
    // std::cout<<"num_points:"<<std::endl;
    // std::cout<<num_points<<std::endl;
    
    return {ray2pidx_bank, ray_start_idx, ray_end_idx};
}





// Find the points within `radius` of a ray, ie, the vertical distance from the point to ray <= radius.
// 
// Returns:
//     ray2pidx: 
//          (b, M), long.  all the points. The pidxs of a ray is found via ray2pidx[b][ray_start_idx[m]:ray_start_idx[m+1]]
//     ray_start_idx:
//          (b, m+1), long
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
) {

    assert(grid_size.dtype() == torch::kLong);
    
    auto batch_size = ray_origins.size(0);
    auto n_rays = ray_origins.size(1);
    auto n_points = points.size(1);
    auto device = ray_origins.device();
    auto dtype = ray_origins.dtype();

    auto long_options = torch::TensorOptions()
        .dtype(torch::kLong)
        .device(device);
    auto int_options = torch::TensorOptions()
        .dtype(torch::kInt32)
        .device(device);
    auto float_options = torch::TensorOptions()
        .dtype(dtype)
        .device(device);

    auto total_cells = torch::prod(grid_size, -1);  // (b,) long

    // get grid idx of each point
    torch::Tensor grid_idxs, valid_mask, cell_counts;
    // auto start = std::chrono::high_resolution_clock::now();
    std::tie(grid_idxs, valid_mask, cell_counts) = get_grid_idx_cuda(
        points,  
        grid_size,
        grid_center,
        grid_width,
        "ind"
    );   // grid_idx: (b, n) long,  valid_mask: (b, n) bool,  cell_counts: (b, n_cells) int
    // cudaDeviceSynchronize();
    // auto stop = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    // std::cout<<"get_grid_idx: "<<duration.count()<<" us"<<std::endl;
    

    // gather points belonging to each cell
    // start = std::chrono::high_resolution_clock::now();
    torch::Tensor gidx2pidx_bank, gidx_start_idx;
    std::tie(gidx2pidx_bank, gidx_start_idx) = gather_points_cuda_v2(
        grid_idxs,  // (b, n)
        total_cells,  // (b,)
        valid_mask,  // (b, n)
        cell_counts  // (b, n_cell)
    );  // gidx2pidx_bank: (b, n) long,  gidx_start_idx: (b, n_cell+1) int
    // cudaDeviceSynchronize();
    // stop = std::chrono::high_resolution_clock::now();
    // duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    // std::cout<<"gather points: "<<duration.count()<<" us"<<std::endl;


    // gather the cells intersected by the ray
    // start = std::chrono::high_resolution_clock::now();
    torch::Tensor ray2gidx, n_ray2gidx;
    std::tie(ray2gidx, n_ray2gidx) = grid_ray_intersection_cuda(
        ray_origins,
        ray_directions,
        ray_radius,
        grid_size,
        grid_center,
        grid_width
    );  // ray2gidx: (b, m, n_gidxs) long,  n_ray2gidx: (b, m)
    // cudaDeviceSynchronize();
    // stop = std::chrono::high_resolution_clock::now();
    // duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    // std::cout<<"grid intersection: "<<duration.count()<<" us"<<std::endl;

    // gather points in the intersected cells
    // start = std::chrono::high_resolution_clock::now();
    torch::Tensor ray2pidx, ray_start_idx, ray_end_idx;
    std::tie(ray2pidx, ray_start_idx, ray_end_idx) = collect_points_on_ray_cuda(
        ray2gidx,  // (b, m, n_gidxs)  long
        n_ray2gidx,  // (b, m) long
        gidx2pidx_bank,  // (b, n) long
        gidx_start_idx,  // (b, n_cells+1)  int32
        points,  // (b, n, 3), float
        ray_origins,  // (b, m, 3), float
        ray_directions,  // (b, m, 3), float
        ray_radius,   //  (b, ) float
        t_min,
        t_max
    );  // ray2pidx: (b, M) long,  ray_start_idx (b, m+1) long
    // cudaDeviceSynchronize();
    // stop = std::chrono::high_resolution_clock::now();
    // duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    // std::cout<<"collect result: "<<duration.count()<<" us"<<std::endl;

    return {ray2pidx, ray_start_idx, ray_end_idx};
}


// New function tbd
// fill in pidx that is within ray_radius
template <typename scalar_t>
__global__ void filter_k_point_and_fill_in_pidx(
    const torch::PackedTensorAccessor32<int64_t, 3, torch::RestrictPtrTraits> ray2gidx,  // (b, m, n_gidxs)  long
    const torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> n_ray2gidx,  // (b, m)
    const torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> gidx2pidx_bank,  // (b, n)
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> gidx_start_idx,  // (b, n_cells+1)
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> points,  // (b, n, 3)
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> ray_origins,  // (b, m, 3)
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> ray_directions,  // (b, m, 3)
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> ray_radius,  // (b,)
    //const torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> ray_start_idx,  // (b, m)
    torch::PackedTensorAccessor32<int64_t, 3, torch::RestrictPtrTraits> ray2pidx_heap,  // (b, m, k)
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> ray2dist_heap,  // (b, m, k)
    torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> ray_neighbor_num,  // (b, m)
    //const int64_t k, // pre-defined to create heap array
    const float t_min = 0.0,
    const float t_max = 1.0e12
) {
    __shared__ scalar_t ray_r;

    // batch index
    const int b = blockIdx.y;
    // ray index
    const int ridx = blockIdx.x * blockDim.x + threadIdx.x;
    
    auto batch_size = ray_origins.size(0);
    auto n_rays = ray_origins.size(1);
    auto n_points = points.size(1);
    auto k = ray2pidx_heap.size(2);

    if (ridx >= n_rays || b >= batch_size) {
        return;
    }

    if (threadIdx.x == 0) {
        ray_r = ray_radius[b];
    }
    __syncthreads(); 

    scalar_t ro[3];
    scalar_t rd[3];
    for (auto i = 0; i < 3; ++i) {
        ro[i] = ray_origins[b][ridx][i];
        rd[i] = ray_directions[b][ridx][i];
    }

    //int current_idx = ray_start_idx[b][ridx];
    int current_idx = 0;
    //const int max_idx = ray_start_idx[b][ridx+1];
    int64_t gidx;
    int64_t pidx;
    scalar_t pp[3];
    scalar_t dist;
    scalar_t t;

    // heap_size = 200 actually may lead to performance drop
    // directly set heap_size = 40 is the fastest, but will crush if k>40
    int64_t pidx_heap[200];
    scalar_t dist_heap[200];
    if (k > 200)
        printf(" In kernel function filter_k_point_and_fill_in_pidx, max heap size is set to be 200\n");
    //scalar_t heap[k];

    //scalar_t* pidx_heap = new scalar_t[k];
    //scalar_t* dist_heap = new scalar_t[k];

    int n = 0; // current size of the max heap

    for (auto ir = 0; ir < n_ray2gidx[b][ridx]; ++ir) {
        gidx = ray2gidx[b][ridx][ir];
        // printf("(%d, %d) gidx = %d \n", (int)b, (int)ridx, (int)(gidx));
        auto ip_start = gidx_start_idx[b][gidx];
        auto ip_end = gidx_start_idx[b][gidx+1];

        for (auto ip = ip_start; ip < ip_end; ++ip) {
            pidx = gidx2pidx_bank[b][ip];
            if (pidx < 0 || pidx >= n_points) { // -1 only happen at the end
                printf("(%d, %d) num points = %d\n", (int)b, (int)ridx, (int)(current_idx));
                break;
            }
            // compute point-ray distance
            pp[0] = points[b][pidx][0];
            pp[1] = points[b][pidx][1];
            pp[2] = points[b][pidx][2];

            compute_point_ray_distance_d(
                pp,  // (3,)
                ro,  // (3,)
                rd,  // (3,)
                dist,
                t
            );

            if (dist > ray_r || t < t_min || t > t_max)
                continue;

            // retain a max heap to store smallest k dist
            // insert new element if
            // (1) n < k or
            // (2) n== k but new dist is smaller than root (max) dist
            retain_k_smallest_values_in_max_heap(
                dist,
                pidx,
                k,
                n,
                dist_heap,
                pidx_heap
            );

        }
    }
    ray_neighbor_num[b][ridx] = n;

    for (auto i = 0; i < k; ++i) {
        ray2pidx_heap[b][ridx][i] = pidx_heap[i];
        ray2dist_heap[b][ridx][i] = dist_heap[i];
    }
}



// New function TBD
// Given ray2gidx and gidx2pidx, construct []
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
//     ray2pidx_bank: 
//          (b, M), all the points. The pidxs of a ray is found via ray2pidx_bank[b][ray_start_idx[m]:ray_end_idx[m]]
//     ray_start_idx:
//          (b, m),
//     ray_end_idx: 
//          (b, m),
// 
std::tuple<torch::Tensor, torch::Tensor> collect_k_points_on_ray_cuda(
        const torch::Tensor & ray2gidx,  // (b, m, n_gidxs)  long
        const torch::Tensor & n_ray2gidx,  // (b, m) long
        const torch::Tensor & gidx2pidx_bank,  // (b, n) long
        const torch::Tensor & gidx_start_idx,  // (b, n_cells+1)  int32
        const torch::Tensor & points,  // (b, n, 3), float
        const torch::Tensor & ray_origins,  // (b, m, 3), float
        const torch::Tensor & ray_directions,  // (b, m, 3), float
        const torch::Tensor & ray_radius,   //  (b, ) float
        const int64_t k,
        float t_min = 0.0,
        float t_max = 1.0e12
) {

    assert(ray2gidx.dtype() == torch::kLong);
    assert(n_ray2gidx.dtype() == torch::kLong);
    assert(gidx2pidx_bank.dtype() == torch::kLong);
    assert(gidx_start_idx.dtype() == torch::kInt32);


    auto batch_size = ray_origins.size(0);
    auto n_rays = ray_origins.size(1);
    auto device = ray_origins.device();
    auto dtype = ray_origins.dtype();

    auto long_options = torch::TensorOptions()
        .dtype(torch::kLong)
        .device(device);
    auto int_options = torch::TensorOptions()
        .dtype(torch::kInt32)
        .device(device);
    auto float_options = torch::TensorOptions()
        .dtype(dtype)
        .device(device);


    // compute max amount of memory needed for each ray
    // torch::Tensor ray2npoints = torch::zeros({batch_size, n_rays}, long_options);

    // Parallelization strategy: 
    // we use one thread to handle a ray 
    const int threads = 512;  // MAX_THREADS_PER_BLOCK;
    const dim3 blocks((n_rays + threads - 1) / threads, batch_size);

    auto ray_neighbor_num = torch::zeros({batch_size, n_rays}, long_options); // (b, n_rays)

    torch::Tensor ray2pidx_heap = torch::empty({batch_size, n_rays ,k}, long_options);  // (b, n_rays, k)
    torch::Tensor ray2dist_heap = torch::empty({batch_size, n_rays ,k}, float_options);  // (b, n_rays, k)

    ray2pidx_heap.fill_(-1);

    // record the pidx into the membank (ignore points not within ray_radius)
    AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "filter_k_and_fill_in_pidx_", ([&] {
        filter_k_point_and_fill_in_pidx<scalar_t><<<blocks, threads>>>(
            ray2gidx.packed_accessor32<int64_t, 3, torch::RestrictPtrTraits>(),
            n_ray2gidx.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
            gidx2pidx_bank.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
            gidx_start_idx.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            points.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            ray_origins.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            ray_directions.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            ray_radius.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            //ray_start_idx.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
            ray2pidx_heap.packed_accessor32<int64_t, 3, torch::RestrictPtrTraits>(),
            ray2dist_heap.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            ray_neighbor_num.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
            //k,
            t_min,
            t_max
        );
    }));

    // torch::cuda::synchronize();
    //ray_start_idx = ray_start_idx.index({"...", Slice(0,-1)});  // (b, n_rays)

    // auto num_points = ray_end_idx - ray_start_idx;
    // std::cout<<"num_points:"<<std::endl;
    // std::cout<<num_points<<std::endl;
    
    //return {ray2pidx_bank, ray_start_idx, ray_end_idx};
    return {ray2pidx_heap, ray_neighbor_num};
}


// See function find_neighbor_points_of_rays_cuda
// replace function collect_points_on_ray_cuda by collect_k_points_on_ray_cuda
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
) {
    // auto print_cuda_memory = true;
    // if (print_cuda_memory)
    //     std::cout << "beginning" << std::endl;
    //     print_gpu_memory();

    assert(grid_size.dtype() == torch::kLong);
    
    auto batch_size = ray_origins.size(0);
    auto n_rays = ray_origins.size(1);
    auto n_points = points.size(1);
    auto device = ray_origins.device();
    auto dtype = ray_origins.dtype();
    // std::cout<<"device: "<< device << std::endl;

    auto long_options = torch::TensorOptions()
        .dtype(torch::kLong)
        .device(device);
    auto int_options = torch::TensorOptions()
        .dtype(torch::kInt32)
        .device(device);
    auto float_options = torch::TensorOptions()
        .dtype(dtype)
        .device(device);

    auto total_cells = torch::prod(grid_size, -1);  // (b,) long

    // get grid idx of each point
    torch::Tensor grid_idxs, valid_mask, cell_counts;
    // auto start = std::chrono::high_resolution_clock::now();
    std::tie(grid_idxs, valid_mask, cell_counts) = get_grid_idx_cuda(
        points,  
        grid_size,
        grid_center,
        grid_width,
        "ind"
    );   // grid_idx: (b, n) long,  valid_mask: (b, n) bool,  cell_counts: (b, n_cells) int
    // cudaDeviceSynchronize();
    // auto stop = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    // std::cout<<"get_grid_idx: "<<duration.count()<<" us"<<std::endl;
    
    // if (print_cuda_memory)
    //     std::cout << "after get_grid_idx_cuda" << std::endl;
    //     print_gpu_memory();


    // gather points belonging to each cell
    // start = std::chrono::high_resolution_clock::now();
    torch::Tensor gidx2pidx_bank, gidx_start_idx;
    std::tie(gidx2pidx_bank, gidx_start_idx) = gather_points_cuda_v2(
        grid_idxs,  // (b, n)
        total_cells,  // (b,)
        valid_mask,  // (b, n)
        cell_counts  // (b, n_cell)
    );  // gidx2pidx_bank: (b, n) long,  gidx_start_idx: (b, n_cell+1) int
    // cudaDeviceSynchronize();
    // stop = std::chrono::high_resolution_clock::now();
    // duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    // std::cout<<"gather points: "<<duration.count()<<" us"<<std::endl;

    // if (print_cuda_memory)
    //     std::cout << "after gather_points_cuda_v2" << std::endl;
    //     print_gpu_memory();


    // gather the cells intersected by the ray
    // start = std::chrono::high_resolution_clock::now();
    torch::Tensor ray2gidx, n_ray2gidx;
    std::tie(ray2gidx, n_ray2gidx) = grid_ray_intersection_cuda(
        ray_origins,
        ray_directions,
        ray_radius,
        grid_size,
        grid_center,
        grid_width
    );  // ray2gidx: (b, m, n_gidxs) long,  n_ray2gidx: (b, m)
    // cudaDeviceSynchronize();
    // stop = std::chrono::high_resolution_clock::now();
    // duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    // std::cout<<"grid intersection: "<<duration.count()<<" us"<<std::endl;

    // if (print_cuda_memory)
    //     std::cout << "after grid_ray_intersection_cuda" << std::endl;
    //     print_gpu_memory();

    // gather k points in the intersected cells
    // start = std::chrono::high_resolution_clock::now();
    torch::Tensor ray2pidx_heap, ray_neighbor_num;
    std::tie(ray2pidx_heap, ray_neighbor_num) = collect_k_points_on_ray_cuda(
        ray2gidx,  // (b, m, n_gidxs)  long
        n_ray2gidx,  // (b, m) long
        gidx2pidx_bank,  // (b, n) long
        gidx_start_idx,  // (b, n_cells+1)  int32
        points,  // (b, n, 3), float
        ray_origins,  // (b, m, 3), float
        ray_directions,  // (b, m, 3), float
        ray_radius,   //  (b, ) float
        k,
        t_min,
        t_max
    );  // ray2pidx: (b, M) long,  ray_start_idx (b, m+1) long
    // cudaDeviceSynchronize();
    // stop = std::chrono::high_resolution_clock::now();
    // duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    // std::cout<<"collect result: "<<duration.count()<<" us"<<std::endl;

    // if (print_cuda_memory)
    //     std::cout << "after collect_k_points_on_ray_cuda" << std::endl;
    //     print_gpu_memory();

    return {ray2pidx_heap, ray_neighbor_num};
}




//
// fill in pidx that is within ray_radius
template <typename scalar_t>
__global__ void keep_min_k_values_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> values,  // (b, m, n)  long
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> min_k_values,  // (b, m, n)  long
    const int64_t k // pre-defined to create heap array
) {
    // batch index
    const int b = blockIdx.y;
    // ray index
    const int m = blockIdx.x * blockDim.x + threadIdx.x;

    auto id_size = values.size(2);

    __syncthreads();

    scalar_t value;
    scalar_t t;

    int64_t id_heap[40];
    scalar_t value_heap[40];

    //scalar_t* id_heap = new scalar_t[k];
    //scalar_t* value_heap = new scalar_t[k];

    int k_current = 0; // current size of the max heap

    for (auto id = 0; id < id_size; ++id) {
        value = values[b][m][id];
        // retain a max heap to store smallest k dist
        // insert new element if
        // (1) n < k or
        // (2) n== k but new dist is smaller than root (max) dist
        retain_k_smallest_values_in_max_heap(
            value,
            id,
            k,
            k_current,
            value_heap,
            id_heap
        );
    }

    for (auto i = 0; i < k; ++i) {
        min_k_values[b][m][i] = value_heap[i];
    }
}


// Test function to keep min k values
torch::Tensor keep_min_k_values_cuda(
        torch::Tensor values,  // (b, m, n), float
        int64_t k
) {
    auto batch_size = values.size(0);
    auto m_size = values.size(1);

    auto device = values.device();
    auto dtype = values.dtype();

    auto float_options = torch::TensorOptions()
        .dtype(dtype)
        .device(device);

    const int threads = 512;  // MAX_THREADS_PER_BLOCK;
    const dim3 blocks((m_size + threads - 1) / threads, batch_size);



    torch::Tensor min_k_values = torch::empty({batch_size, m_size, k}, float_options);  // (b, m, k)
    AT_DISPATCH_FLOATING_TYPES(values.scalar_type(), "keep_min_k_values_kernel", ([&] {
        keep_min_k_values_kernel<scalar_t><<<blocks, threads>>>(
            values.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            min_k_values.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            k
        );
    }));

    return min_k_values;
}





// traverse_grid, filter_k_point_and_fill_in_pidx
//
// Args:
//     ray_origins:
//         (b, m, 3)
//     ray_directions:
//         (b, m, 3)
//     grid_size:
//         (b, 3) long, xyz
//     grid_center:
//         (b, 3), xyz
//     grid_width:
//         (b, 3), xyz
//     idx_to_uses:
//         (b, m), grid direction to use as the main direction, (0, 1, 2)
//     local_grid_sizes:
//         (b, m), total number of grid cells for each ray
//     local_rs:
//         (b, m, 3), grid cell to include in each direction
//     gidx2pidx_bank:
//         (b, n), long, point index for each cell.  To find the pidxs, use the pidx of cell_idx is 
//         from cell_start_idx[b][cell_idx] to cell_counts[b][cell_idx+1] (excluded).
//     gidx_start_idx:
//         (b, n_cells+1), int32, cell starts at gidx_start_idx[i] and ends at gidx_start_idx[i+1]
//     k:
//         int, number of nearest points 
//     t_min:
//         float, min t to start tracing
//     t_max:
//         float, max t to stop tracing
template <typename scalar_t>
__global__ void march_and_collect(
    // for traverse_grid
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> ray_origins,  // (b, m, 3)
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> ray_directions,  // (b, m, 3)
    const torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> grid_size,  // (b, 3)
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grid_center,  // (b, 3)
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grid_width,  // (b, 3)
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> idx_to_uses,  // (b, m)  0,1,2
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> local_grid_sizes,  // (b, m)
    const torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> local_rs,  // (b, m, 3)
    // for collecting points with a heap
    const torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> gidx2pidx_bank,  // (b, n)
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> gidx_start_idx,  // (b, n_cells+1)
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> points,  // (b, n, 3)
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> ray_radius,  // (b,)
    torch::PackedTensorAccessor32<int64_t, 3, torch::RestrictPtrTraits> ray2pidx_heap,  // (b, m, k)
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> ray2dist_heap,  // (b, m, k)
    torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> ray_neighbor_num,  // (b, m)
    const float t_min = 0.0,
    const float t_max = 1.0e12
) {
    __shared__ scalar_t ray_r;  // ray radius
    __shared__ int gs[3];  // grid size
    __shared__ scalar_t gc[3];  // grid center
    __shared__ scalar_t gf[3];  // grid from
    __shared__ scalar_t cw[3];  // cell width
    __shared__ scalar_t gw[3];  // grid width

    // batch index
    const int b = blockIdx.y;
    // ray index
    const int ridx = blockIdx.x * blockDim.x + threadIdx.x;
    
    auto batch_size = ray_origins.size(0);
    auto n_rays = ray_origins.size(1);
    auto n_points = points.size(1);
    auto k = ray2pidx_heap.size(2);

    if (ridx >= n_rays || b >= batch_size) {
        return;
    }

    if (threadIdx.x == 0) {
        ray_r = ray_radius[b];
        for (auto i = 0; i < 3; ++i) {
            gs[i] = int(grid_size[b][i]);
            gw[i] = grid_width[b][i];
            gc[i] = grid_center[b][i];
            gf[i] = gc[i] - 0.5 * gw[i];
            cw[i] = gw[i] / scalar_t(gs[i]);
        }
    }
    __syncthreads(); 

    // --------------------------------------
    // traverse grid
    // --------------------------------------

    const int32_t idx_to_use = idx_to_uses[b][ridx];
    scalar_t ro[3], rd[3];
    for (auto i = 0; i < 3; ++i) {
        ro[i] = ray_origins[b][ridx][i];
        rd[i] = ray_directions[b][ridx][i];
    }

    scalar_t plane_c = gf[idx_to_use] + 0.5 * cw[idx_to_use];
    scalar_t plane_t = (plane_c - ro[idx_to_use]) / rd[idx_to_use];
    scalar_t t_step = cw[idx_to_use] / rd[idx_to_use]; 

    const int rx = local_rs[b][ridx][0];
    const int ry = local_rs[b][ridx][1];
    const int rz = local_rs[b][ridx][2];
    const int max_num_gidxs = local_grid_sizes[b][ridx];
    int x, y, z;

    // compute ray-plane intersection for each plane
   
    scalar_t p[3];  // current sample point on the ray
    int64_t p_subidx[3];  // p's grid sub_idx
    int32_t ip_start, ip_end;  
    int64_t gidx;
    int64_t pidx;
    scalar_t pp[3];
    scalar_t dist;
    scalar_t t;

    // heap_size = 200 actually may lead to performance drop
    // directly set heap_size = 40 is the fastest, but will crush if k>40
    int64_t pidx_heap[MAX_K];
    scalar_t dist_heap[MAX_K];
    if (k > MAX_K)
        printf(" k should be less than max heap size (%d)\n", MAX_K);

    int32_t n_heap = 0;  // current size of the max heap

    int32_t current_idx = 0;
    for (auto plane_idx = 0; plane_idx < gs[idx_to_use]; ++plane_idx) {
        // for each intersection point, gather all neighboring cells
        for (auto i = 0; i < 3; ++i) {
            p[i] = ro[i] + plane_t * rd[i];
        }
        // convert p to grid's sub_idx, p_subidx can < 0 or >= grid_size
        discretize<scalar_t>(p, gf, cw, p_subidx); 

        // add neighbor grids
        for (int ix = -rx; ix <= rx; ++ix) {
            x = int(p_subidx[0]) + ix;
            if (x < 0 || x >= gs[0]) 
                continue;

            for (int iy = -ry; iy <= ry; ++iy) {
                y = int(p_subidx[1]) + iy;
                if (y < 0 || y >= gs[1]) 
                    continue;

                for (int iz = -rz; iz <= rz; ++iz) {
                    z = int(p_subidx[2]) + iz;
                    if (z < 0 || z >= gs[2]) 
                        continue;

                    // compute the grid cell's ind
                    gidx = sub2ind_d(
                        x, y, z, 
                        (int)gs[0], (int)gs[1], (int)gs[2]
                    );
                    
                    // -----------------------------------
                    // start collecting points in the cell
                    // -----------------------------------
                    ip_start = gidx_start_idx[b][gidx];
                    ip_end = gidx_start_idx[b][gidx+1];

                    for (auto ip = ip_start; ip < ip_end; ++ip) {
                        pidx = gidx2pidx_bank[b][ip];
                        if (pidx < 0 || pidx >= n_points) { 
                            // -1 only happen at the end
                            printf("(%d, %d) num points = %d\n", (int)b, (int)ridx, (int)(current_idx));
                            break;
                        }
                        // compute point-ray distance
                        pp[0] = points[b][pidx][0];
                        pp[1] = points[b][pidx][1];
                        pp[2] = points[b][pidx][2];

                        compute_point_ray_distance_d(
                            pp,  // (3,)
                            ro,  // (3,)
                            rd,  // (3,)
                            dist,  // vertical distance from point to ray
                            t  // t of the point
                        );

                        if (dist > ray_r || t < t_min || t > t_max)
                            continue;

                        // retain a max heap to store smallest k dist
                        // insert new element if
                        // (1) n < k or
                        // (2) n== k but new dist is smaller than root (max) dist
                        retain_k_smallest_values_in_max_heap(
                            dist,
                            pidx,
                            k,
                            n_heap,
                            dist_heap,
                            pidx_heap
                        );

                    }
                    // end collecting points in the cell
                }
            }
        }

        plane_t += t_step;
    }

    ray_neighbor_num[b][ridx] = n_heap;

    // store the heap in local memory to global memory
    for (auto i = 0; i < k; ++i) {
        ray2pidx_heap[b][ridx][i] = pidx_heap[i];
        ray2dist_heap[b][ridx][i] = dist_heap[i];
    }

}





// March each ray along the grid, collecting k nearest points along the ray on the 
// intersected grid cells.
// 
// Args:
//     points:
//         (b, n, 3)
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
//     gidx2pidx_bank:
//         (b, n), long, point index for each cell.  To find the pidxs, use the pidx of cell_idx is 
//         from cell_start_idx[b][cell_idx] to cell_counts[b][cell_idx+1] (excluded).
//     gidx_start_idx:
//         (b, n_cells+1), int32, cell starts at gidx_start_idx[i] and ends at gidx_start_idx[i+1]
//     k:
//         int, number of nearest points 
//     t_min:
//         float, min t to start tracing
//     t_max:
//         float, max t to stop tracing
// 
// Returns:
//     ray2pidx_heap:
//          (b, m, k),  the memory storing at most k nearest points along each ray
//     ray_neighbor_num:
//          (b, m),  the valid number of points for each ray (not all rays have k points)
//     ray2dist_heap:
//          (b, m, k), float, the memory storing the ray-point distance of the 
//          corresponding points in ray2pidx_heap
// 
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> march_ray_on_grid_cuda(
        const torch::Tensor & points,  // (b, n, 3), float
        const torch::Tensor & ray_origins,  // (b, m, 3), float
        const torch::Tensor & ray_directions,  // (b, m, 3), float
        const torch::Tensor & ray_radius,   //  (b, ) float
        const torch::Tensor & grid_size,  // (b, 3), long
        const torch::Tensor & grid_center,  // (b, 3) float
        const torch::Tensor & grid_width,  // (b, 3) float
        const torch::Tensor & gidx2pidx_bank,  // (b, n) long
        const torch::Tensor & gidx_start_idx,  // (b, n_cells+1)  int32
        const int64_t k,
        float t_min = 0.0,
        float t_max = 1.0e12
) {
    assert(grid_size.dtype() == torch::kLong);
    assert(gidx2pidx_bank.dtype() == torch::kLong);
    assert(gidx_start_idx.dtype() == torch::kInt32);

    auto batch_size = ray_origins.size(0);
    auto n_rays = ray_origins.size(1);
    auto device = ray_origins.device();
    auto dtype = ray_origins.dtype();

    auto long_options = torch::TensorOptions()
        .dtype(torch::kLong)
        .device(device);
    auto int_options = torch::TensorOptions()
        .dtype(torch::kInt32)
        .device(device);
    auto float_options = torch::TensorOptions()
        .dtype(dtype)
        .device(device);

    // Parallelization strategy: 
    // we use one thread to handle a ray 
    const int threads = MAX_THREADS_PER_BLOCK;
    const dim3 blocks((n_rays + threads - 1) / threads, batch_size);

    // calculate the main direction and max number of neighbor grids among all rays
    torch::Tensor inv_ts = torch::zeros({batch_size, n_rays, 3}, float_options);  // (b, m, 3)
    torch::Tensor idx_to_uses = torch::zeros({batch_size, n_rays}, int_options);  // (b, m)
    torch::Tensor local_grid_sizes = torch::zeros({batch_size, n_rays}, int_options);  // (b, m)
    torch::Tensor local_rs = torch::zeros({batch_size, n_rays, 3}, int_options);  // (b, m, 3)
    // auto start = std::chrono::high_resolution_clock::now();

    AT_DISPATCH_FLOATING_TYPES(ray_directions.scalar_type(), "determine_main_direction_", ([&] {
        determine_main_direction<scalar_t><<<blocks, threads>>>(
            ray_directions.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            ray_radius.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            grid_size.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
            grid_width.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            inv_ts.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            idx_to_uses.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            local_grid_sizes.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            local_rs.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>()
        );
    }));
    // auto stop = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    // std::cout<<"--determine_main_direction_: "<<duration.count()<<" us"<<std::endl;
    // torch::cuda::synchronize();


    // start = std::chrono::high_resolution_clock::now();
    // allocate memory for neighbor grid index for each ray
    const int threads2 = 512;  // reduced to 512 since we use lot of local memory to store the heap for each ray
    const dim3 blocks2((n_rays + threads2 - 1) / threads2, batch_size);

    auto ray_neighbor_num = torch::zeros({batch_size, n_rays}, long_options);  // (b, m)
    torch::Tensor ray2pidx_heap = torch::empty({batch_size, n_rays, k}, long_options);  // (b, m, k)
    torch::Tensor ray2dist_heap = torch::empty({batch_size, n_rays, k}, float_options);  // (b, m, k)
    // initialize the heap to -1
    ray2pidx_heap.fill_(-1);

    // record the pidx into the membank (ignore points not within ray_radius)
    AT_DISPATCH_FLOATING_TYPES(ray_directions.scalar_type(), "march_and_collect_", ([&] {
        march_and_collect<scalar_t><<<blocks2, threads2>>>(
            // traverse grid
            ray_origins.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            ray_directions.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            grid_size.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
            grid_center.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            grid_width.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            idx_to_uses.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            local_grid_sizes.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            local_rs.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
            // collecting points
            gidx2pidx_bank.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
            gidx_start_idx.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            points.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            ray_radius.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            ray2pidx_heap.packed_accessor32<int64_t, 3, torch::RestrictPtrTraits>(),
            ray2dist_heap.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            ray_neighbor_num.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
            t_min,
            t_max
        );
    }));

    return {ray2pidx_heap, ray_neighbor_num, ray2dist_heap};
}




// See function find_neighbor_points_of_rays_cuda
// 1) replace function collect_points_on_ray_cuda by collect_k_points_on_ray_cuda
// 2) simultaneously performing grid_ray_intersection and collect k points (since grid-ray intersection consumes lots of memory)
//    Specifically, for each ray, we perform (i) determine_main_direction, (ii) traverse_grid, and (iii) filter_k_point_and_fill_in_pidx
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
) {
    // auto print_cuda_memory = true;
    // if (print_cuda_memory)
    //     std::cout << "beginning" << std::endl;
    //     print_gpu_memory();

    assert(grid_size.dtype() == torch::kLong);
    
    auto batch_size = ray_origins.size(0);
    auto n_rays = ray_origins.size(1);
    auto n_points = points.size(1);
    auto device = ray_origins.device();
    auto dtype = ray_origins.dtype();

    auto long_options = torch::TensorOptions()
        .dtype(torch::kLong)
        .device(device);
    auto int_options = torch::TensorOptions()
        .dtype(torch::kInt32)
        .device(device);
    auto float_options = torch::TensorOptions()
        .dtype(dtype)
        .device(device);

    auto total_cells = torch::prod(grid_size, -1);  // (b,) long

    // get grid idx of each point
    // auto start = std::chrono::high_resolution_clock::now();
    torch::Tensor grid_idxs, valid_mask, cell_counts;
    std::tie(grid_idxs, valid_mask, cell_counts) = get_grid_idx_cuda(
        points,  
        grid_size,
        grid_center,
        grid_width,
        "ind"
    );   // grid_idx: (b, n) long,  valid_mask: (b, n) bool,  cell_counts: (b, n_cells) int
    // cudaDeviceSynchronize();
    // auto stop = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    // std::cout<<"get_grid_idx: "<<duration.count()<<" us"<<std::endl;
    
    // if (print_cuda_memory)
    //     std::cout << "after get_grid_idx_cuda" << std::endl;
    //     print_gpu_memory();


    // gather points belonging to each cell
    // start = std::chrono::high_resolution_clock::now();
    torch::Tensor gidx2pidx_bank, gidx_start_idx;
    std::tie(gidx2pidx_bank, gidx_start_idx) = gather_points_cuda_v2(
        grid_idxs,  // (b, n)
        total_cells,  // (b,)
        valid_mask,  // (b, n)
        cell_counts  // (b, n_cell)
    );  // gidx2pidx_bank: (b, n) long,  gidx_start_idx: (b, n_cell+1) int
    // cudaDeviceSynchronize();
    // stop = std::chrono::high_resolution_clock::now();
    // duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    // std::cout<<"gather points: "<<duration.count()<<" us"<<std::endl;

    // if (print_cuda_memory)
    //     std::cout << "after gather_points_cuda_v2" << std::endl;
    //     print_gpu_memory();


    // march ray on grid and collect k nearest points
    // start = std::chrono::high_resolution_clock::now();
    torch::Tensor ray2pidx_heap, ray_neighbor_num, ray2dist_heap;
    std::tie(ray2pidx_heap, ray_neighbor_num, ray2dist_heap) = march_ray_on_grid_cuda(
        points,  // (b, n, 3) float
        ray_origins,  // (b, m, 3), float
        ray_directions,  // (b, m, 3), float
        ray_radius,   //  (b, ) float
        grid_size,  // (b, 3), long
        grid_center,  // (b, 3) float
        grid_width,  // (b, 3) float
        gidx2pidx_bank,  // (b, n) long
        gidx_start_idx,  // (b, n_cells+1)  int32
        k,
        t_min,
        t_max
    );  // ray2pidx: (b, M) long,  ray_start_idx (b, m+1) long
    // cudaDeviceSynchronize();
    // stop = std::chrono::high_resolution_clock::now();
    // duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    // std::cout<<"collect result: "<<duration.count()<<" us"<<std::endl;

    // if (print_cuda_memory)
    //     std::cout << "after collect_k_points_on_ray_cuda" << std::endl;
    //     print_gpu_memory();

    return {ray2pidx_heap, ray_neighbor_num};
}



//  Returns:
//     ray2pidx_heap:
//          (b, m, k),  the memory storing at most k nearest points along each ray
//     ray_neighbor_num:
//          (b, m),  the valid number of points for each ray (not all rays have k points)
//     ray2dist_heap:
//          (b, m, k), float, the memory storing the ray-point distance of the 
//          corresponding points in ray2pidx_heap
//     gidx2pidx_bank:
//          (b, n) long, stores the point indexes in each grid cell
//     gidx_start_idx:
//          (b, max_total_cells+1), int32, the pidxs of gidx starts at gidx2pidx_bank[gidx_start_idx[b, gidx]]
//          and ends at gidx2pidx_bank[gidx_start_idx[b, gidx+1]]
//
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> find_k_neighbor_points_of_rays_cuda_v3(
        torch::Tensor points,  // (b, n, 3), float
        torch::Tensor ray_origins, // (b, m, 3), float
        torch::Tensor ray_directions,  // (b, m, 3), float
        torch::Tensor ray_radius,  // (b,), float
        torch::Tensor grid_size,  // (b, 3), long
        torch::Tensor grid_center,  // (b, 3), float
        torch::Tensor grid_width,  // (b, 3), float
        torch::Tensor gidx2pidx_bank,  // (b, n) long or empty (,)
        torch::Tensor gidx_start_idx,  // // (b, n_cell+1) int32 or empty (,)
        int64_t k,
        float t_min = 0.,
        float t_max = 1.e12,
        bool refresh_cache = true
) {
    // auto print_cuda_memory = true;
    // if (print_cuda_memory)
    //     std::cout << "beginning" << std::endl;
    //     print_gpu_memory();

    assert(grid_size.dtype() == torch::kLong);

    auto batch_size = ray_origins.size(0);
    auto n_rays = ray_origins.size(1);
    auto n_points = points.size(1);
    auto device = ray_origins.device();
    auto dtype = ray_origins.dtype();

    auto long_options = torch::TensorOptions()
        .dtype(torch::kLong)
        .device(device);
    auto int_options = torch::TensorOptions()
        .dtype(torch::kInt32)
        .device(device);
    auto float_options = torch::TensorOptions()
        .dtype(dtype)
        .device(device);

    auto total_cells = torch::prod(grid_size, -1);  // (b,) long
    int64_t max_total_cells = total_cells.max().item<int64_t>();  

    if (refresh_cache) {
        
        // get grid idx of each point
        // auto start = std::chrono::high_resolution_clock::now();
        torch::Tensor grid_idxs, valid_mask, cell_counts;
        std::tie(grid_idxs, valid_mask, cell_counts) = get_grid_idx_cuda(
            points,  
            grid_size,
            grid_center,
            grid_width,
            "ind"
        );   // grid_idx: (b, n) long,  valid_mask: (b, n) bool,  cell_counts: (b, n_cells) int
        // cudaDeviceSynchronize();
        // auto stop = std::chrono::high_resolution_clock::now();
        // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        // std::cout<<"get_grid_idx: "<<duration.count()<<" us"<<std::endl;
        
        // if (print_cuda_memory)
        //     std::cout << "after get_grid_idx_cuda" << std::endl;
        //     print_gpu_memory();


        // gather points belonging to each cell
        // start = std::chrono::high_resolution_clock::now();
        // torch::Tensor gidx2pidx_bank, gidx_start_idx;
        std::tie(gidx2pidx_bank, gidx_start_idx) = gather_points_cuda_v2(
            grid_idxs,  // (b, n)
            total_cells,  // (b,)
            valid_mask,  // (b, n)
            cell_counts  // (b, n_cell)
        );  // gidx2pidx_bank: (b, n) long,  gidx_start_idx: (b, n_cell+1) int
        // cudaDeviceSynchronize();
        // stop = std::chrono::high_resolution_clock::now();
        // duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        // std::cout<<"gather points: "<<duration.count()<<" us"<<std::endl;

        // if (print_cuda_memory)
        //     std::cout << "after gather_points_cuda_v2" << std::endl;
        //     print_gpu_memory();
    }
    else {
        assert(gidx2pidx_bank.size(0) == batch_size);
        assert(gidx2pidx_bank.size(1) == n_points);
        assert(gidx_start_idx.size(0) == batch_size);
        assert(gidx_start_idx.size(1) == max_total_cells + 1);
    }

    // march ray on grid and collect k nearest points
    // start = std::chrono::high_resolution_clock::now();
    torch::Tensor ray2pidx_heap, ray_neighbor_num, ray2dist_heap;
    std::tie(ray2pidx_heap, ray_neighbor_num, ray2dist_heap) = march_ray_on_grid_cuda(
        points,  // (b, n, 3) float
        ray_origins,  // (b, m, 3), float
        ray_directions,  // (b, m, 3), float
        ray_radius,   //  (b, ) float
        grid_size,  // (b, 3), long
        grid_center,  // (b, 3) float
        grid_width,  // (b, 3) float
        gidx2pidx_bank,  // (b, n) long
        gidx_start_idx,  // (b, n_cells+1)  int32
        k,
        t_min,
        t_max
    );  // ray2pidx: (b, M) long,  ray_start_idx (b, m+1) long
    // cudaDeviceSynchronize();
    // stop = std::chrono::high_resolution_clock::now();
    // duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    // std::cout<<"collect result: "<<duration.count()<<" us"<<std::endl;

    // if (print_cuda_memory)
    //     std::cout << "after collect_k_points_on_ray_cuda" << std::endl;
    //     print_gpu_memory();

    return {
        ray2pidx_heap, 
        ray_neighbor_num, 
        ray2dist_heap, 
        gidx2pidx_bank, 
        gidx_start_idx
        };
}




//  Returns:
//     ray2pidx_heap:
//          (b, m, k),  the memory storing at most k nearest points along each ray
//     ray_neighbor_num:
//          (b, m),  the valid number of points for each ray (not all rays have k points)
//     ray2dist_heap:
//          (b, m, k), float, the memory storing the ray-point distance of the 
//          corresponding points in ray2pidx_heap
//     gidx2pidx_bank:
//          (b, n) long, stores the point indexes in each grid cell
//     gidx_start_idx:
//          (b, max_total_cells+1), int32, the pidxs of gidx starts at gidx2pidx_bank[gidx_start_idx[b, gidx]]
//          and ends at gidx2pidx_bank[gidx_start_idx[b, gidx+1]]
//
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> find_k_neighbor_points_of_rays_cuda_v4(
        torch::Tensor points,  // (b, n, 3), float
        torch::Tensor ray_origins, // (b, m, 3), float
        torch::Tensor ray_directions,  // (b, m, 3), float
        torch::Tensor ray_radius,  // (b,), float
        torch::Tensor grid_size,  // (b, 3), long
        torch::Tensor grid_center,  // (b, 3), float
        torch::Tensor grid_width,  // (b, 3), float
        torch::Tensor gidx2pidx_bank,  // (b, n) long or empty (,)
        torch::Tensor gidx_start_idx,  // // (b, n_cell+1) int32 or empty (,)
        torch::Tensor valid_mask,  // (b, n) bool
        int64_t k,
        float t_min = 0.,
        float t_max = 1.e12,
        bool refresh_cache = true
) {
    // auto print_cuda_memory = true;
    // if (print_cuda_memory)
    //     std::cout << "beginning" << std::endl;
    //     print_gpu_memory();

    assert(grid_size.dtype() == torch::kLong);

    auto batch_size = ray_origins.size(0);
    auto n_rays = ray_origins.size(1);
    auto n_points = points.size(1);
    auto device = ray_origins.device();
    auto dtype = ray_origins.dtype();

    auto long_options = torch::TensorOptions()
        .dtype(torch::kLong)
        .device(device);
    auto int_options = torch::TensorOptions()
        .dtype(torch::kInt32)
        .device(device);
    auto float_options = torch::TensorOptions()
        .dtype(dtype)
        .device(device);
    auto bool_options = torch::TensorOptions()
        .dtype(torch::kBool)
        .device(device);

    auto total_cells = torch::prod(grid_size, -1);  // (b,) long
    int64_t max_total_cells = total_cells.max().item<int64_t>();  

    if (refresh_cache) {
        
        // get grid idx of each point
        // auto start = std::chrono::high_resolution_clock::now();
        torch::Tensor grid_idxs, valid_mask_oob, cell_counts;
        std::tie(grid_idxs, valid_mask_oob, cell_counts) = get_grid_idx_cuda(
            points,  
            grid_size,
            grid_center,
            grid_width,
            "ind"
        );   // grid_idx: (b, n) long,  valid_mask_oob: (b, n) bool,  cell_counts: (b, n_cells) int
        // cudaDeviceSynchronize();
        // auto stop = std::chrono::high_resolution_clock::now();
        // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        // std::cout<<"get_grid_idx: "<<duration.count()<<" us"<<std::endl;
        
        // if (print_cuda_memory)
        //     std::cout << "after get_grid_idx_cuda" << std::endl;
        //     print_gpu_memory();

        // merge valid_mask_oob and valid_mask
        valid_mask = torch::logical_and(
            valid_mask_oob,
            valid_mask
        );  // (b, n)

        // gather points belonging to each cell
        // start = std::chrono::high_resolution_clock::now();
        // torch::Tensor gidx2pidx_bank, gidx_start_idx;
        std::tie(gidx2pidx_bank, gidx_start_idx) = gather_points_cuda_v2(
            grid_idxs,  // (b, n)
            total_cells,  // (b,)
            valid_mask,  // (b, n)
            cell_counts  // (b, n_cell)
        );  // gidx2pidx_bank: (b, n) long,  gidx_start_idx: (b, n_cell+1) int
        // cudaDeviceSynchronize();
        // stop = std::chrono::high_resolution_clock::now();
        // duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        // std::cout<<"gather points: "<<duration.count()<<" us"<<std::endl;

        // if (print_cuda_memory)
        //     std::cout << "after gather_points_cuda_v2" << std::endl;
        //     print_gpu_memory();
    }
    else {
        assert(gidx2pidx_bank.size(0) == batch_size);
        assert(gidx2pidx_bank.size(1) == n_points);
        assert(gidx_start_idx.size(0) == batch_size);
        assert(gidx_start_idx.size(1) == max_total_cells + 1);
    }

    // march ray on grid and collect k nearest points
    // start = std::chrono::high_resolution_clock::now();
    torch::Tensor ray2pidx_heap, ray_neighbor_num, ray2dist_heap;
    std::tie(ray2pidx_heap, ray_neighbor_num, ray2dist_heap) = march_ray_on_grid_cuda(
        points,  // (b, n, 3) float
        ray_origins,  // (b, m, 3), float
        ray_directions,  // (b, m, 3), float
        ray_radius,   //  (b, ) float
        grid_size,  // (b, 3), long
        grid_center,  // (b, 3) float
        grid_width,  // (b, 3) float
        gidx2pidx_bank,  // (b, n) long
        gidx_start_idx,  // (b, n_cells+1)  int32
        k,
        t_min,
        t_max
    );  // ray2pidx: (b, M) long,  ray_start_idx (b, m+1) long
    // cudaDeviceSynchronize();
    // stop = std::chrono::high_resolution_clock::now();
    // duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    // std::cout<<"collect result: "<<duration.count()<<" us"<<std::endl;

    // if (print_cuda_memory)
    //     std::cout << "after collect_k_points_on_ray_cuda" << std::endl;
    //     print_gpu_memory();

    return {
        ray2pidx_heap, 
        ray_neighbor_num, 
        ray2dist_heap, 
        gidx2pidx_bank, 
        gidx_start_idx
        };
}
