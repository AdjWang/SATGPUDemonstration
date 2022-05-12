#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include "sat.h"

// GPU utilities
#include <cuda_runtime.h>
#include "dependence.h"

// vector ---------------------------------------------------------------------

static vector_t vector_sub(const vector_t vec1, const vector_t vec2){
    return (vector_t){
        .x = vec1.x - vec2.x,
        .y = vec1.y - vec2.y,
    };
}

static double vector_dot(const vector_t vec1, const vector_t vec2){
    return vec1.x*vec2.x + vec1.y*vec2.y;
}

static double vecotr_magnitude(const vector_t vec){
    return sqrt(vec.x*vec.x + vec.y*vec.y);
}

static vector_t vector_normalize(const vector_t vec){
    double mag = vecotr_magnitude(vec);
    return (vector_t){
        .x = vec.x / mag,
        .y = vec.y / mag,
    };
}

static vector_t vector_perpendicular(const vector_t vec){
    // or (y, -x)
    return (vector_t){
        .x = -vec.y,
        .y = vec.x,
    };
}

// projection -----------------------------------------------------------------

static BOOL projection_is_overlap(const projection_t projection1, const projection_t projection2){
    return (projection1.left < projection2.right
        && projection1.right > projection2.left);
}

// ploygon --------------------------------------------------------------------

polygon_t* new_polygon(int n_vertex){
    if(n_vertex <= 2){
        return NULL;
    }
    polygon_t* polygon = (polygon_t*)malloc(sizeof(polygon_t));
    if(!polygon){
        RAISE("malloc failed\n");
    }
    polygon->n = n_vertex;
    polygon->vertices = (point_t*)malloc(sizeof(point_t)*n_vertex);
    if(!polygon->vertices){
        RAISE("malloc failed\n");
    }
    polygon->axes = (vector_t*)malloc(sizeof(vector_t)*n_vertex);
    if(!polygon->axes){
        RAISE("malloc failed\n");
    }
    return polygon;
}

void del_polygon(polygon_t* polygon){
    if(!polygon){
        return;
    }
    if(polygon->vertices){
        free(polygon->vertices);
        polygon->vertices = NULL;
    }
    if(polygon->axes){
        free(polygon->axes);
        polygon->axes = NULL;
    }
    free(polygon);
}

static int polygon_print_point_list(FILE* stream, const point_t* point_list, int n){
    int c = 0;
    c += fprintf(stream, "[");
    for(int i=0; i<n; i++){
        c += fprintf(stream, "(%.16lf,%.16lf),", point_list[i].x, point_list[i].y);
    }
    c += fprintf(stream, "]\n");
    return c;
}

int polygon_print(FILE* stream, const polygon_t* polygon){
    int c = 0;
    c += fprintf(stream, "%d\n", polygon->n);
    c += polygon_print_point_list(stream, polygon->vertices, polygon->n);
    return c;
}

static void polygon_get_axes(polygon_t* polygon){
    for(int i=0; i<polygon->n; i++){
        point_t p1 = polygon->vertices[i];
        point_t p2 = polygon->vertices[(i+1) == polygon->n ? 0 : (i+1)];
        vector_t edge = vector_sub(*(vector_t*)&p1, *(vector_t*)&p2);
        vector_t norm = vector_normalize(edge);
        vector_t perp = vector_perpendicular(norm);
        polygon->axes[i] = perp;    // copy by value
    }
}

static projection_t polygon_project(const polygon_t* polygon, const vector_t axis){
    double proj_min=INFINITY, proj_max=-INFINITY;
    for(int i=0; i<polygon->n; i++){
        double proj_num = vector_dot(axis, *(vector_t*)&(polygon->vertices[i]));
        if(proj_num < proj_min){
            proj_min = proj_num;
        }
        if(proj_num > proj_max){
            proj_max = proj_num;
        }
    }
    return (projection_t){
        .left = proj_min,
        .right = proj_max,
    };
}

BOOL polygon_is_overlap(const polygon_t* polygon1, const polygon_t* polygon2){
    // loop over the axes1
    vector_t* axes1 = polygon1->axes;
    for(int i=0; i<polygon1->n; i++){
        vector_t axis = axes1[i];
        // project both shapes onto the axis
        projection_t p1 = polygon_project(polygon1, axis);
        projection_t p2 = polygon_project(polygon2, axis);
        // do the projections overlap?
        if (!projection_is_overlap(p1, p2)) {
            // then we can guarantee that the shapes do not overlap
            return FALSE;
        }
    }

    // loop over the axes2
    vector_t* axes2 = polygon2->axes;
    for(int i=0; i<polygon2->n; i++){
        vector_t axis = axes2[i];
        // project both shapes onto the axis
        projection_t p1 = polygon_project(polygon1, axis);
        projection_t p2 = polygon_project(polygon2, axis);
        // do the projections overlap?
        if (!projection_is_overlap(p1, p2)) {
            // then we can guarantee that the shapes do not overlap
            return FALSE;
        }
    }

    return TRUE;
}

// cpu implemention -----------------------------------------------------------

void detect_overlap(polygon_t** polygon_list, int** result, int n){
    ASSERT(result != NULL, "invalid result: NULL\n");
    ASSERT(n > 0, "empty list: n <= 0\n");

    // calculate axes
    for(int i=0; i<n; i++){
        polygon_t* polygon = polygon_list[i];
        polygon_get_axes(polygon);
    }

    // calculate overlap
    for(int i=0; i<n-1; i++){
        for(int j=i+1; j<n; j++){
            if(polygon_is_overlap(polygon_list[i], polygon_list[j])){
                result[i][j] = result[j][i] = 1;
            }
        }
    }
}

// gpu implemention -----------------------------------------------------------

// device utilities

__device__ static vector_t vector_sub_gpu(const vector_t vec1, const vector_t vec2){
    return (vector_t){
        .x = vec1.x - vec2.x,
        .y = vec1.y - vec2.y,
    };
}

__device__ static double vector_dot_gpu(const vector_t vec1, const vector_t vec2){
    return vec1.x*vec2.x + vec1.y*vec2.y;
}

__device__ static double vecotr_magnitude_gpu(const vector_t vec){
    return sqrt(vec.x*vec.x + vec.y*vec.y);
}

__device__ static vector_t vector_normalize_gpu(const vector_t vec){
    double mag = vecotr_magnitude_gpu(vec);
    return (vector_t){
        .x = vec.x / mag,
        .y = vec.y / mag,
    };
}

__device__ static vector_t vector_perpendicular_gpu(const vector_t vec){
    // or (y, -x)
    return (vector_t){
        .x = -vec.y,
        .y = vec.x,
    };
}

// Get minimal and maximal value of endpoints.
__device__ static projection_t min_max_gpu(const double* endpoints, int n){
    double proj_min=INFINITY, proj_max=-INFINITY;
    // printf("endpoings: [%d, %d)\n", start, end);
    for(int i=0; i<n; i++){
        double proj_num = endpoints[i];
        // printf("%lf ", proj_num);
        if(proj_num < proj_min){
            proj_min = proj_num;
        }
        if(proj_num > proj_max){
            proj_max = proj_num;
        }
    }
    // printf("\n");
    return (projection_t){
        .left = proj_min,
        .right = proj_max,
    };
}

__device__ static BOOL projection_is_overlap_gpu(const projection_t* projection1, const projection_t* projection2){
    return (projection1->left < projection2->right
        && projection1->right > projection2->left);
}

// Flatten vertices of polygons for further parallelizing.
// description:
//   1. vertices of polygon i is vertices[i_polygon_map[i] : i_polygon_map[i] + polygon_n_map[i]]
//   2. owner_map is the flattened indices.
//   e.g. polygons(a triangle and a regtangle) whose number of vertices is {3, 4}, its:
//        vertices is {point0, point1, point2, point3, point4, point5, point6}
//        i_polygon_map is {0, 3}
//        polygon_n_map is {3, 4}
//        owner_map is {0, 0, 0, 1, 1, 1, 1}
static int util_flatten(polygon_t** polygon_list, int n_polygon,
                        /*out*/int** p_i_polygon_map_gpu, /*out*/int** p_polygon_n_map_gpu,
                        /*out*/int** p_owner_map_gpu, /*out*/point_t** p_vertices_gpu){
    ASSERT(*p_i_polygon_map_gpu == NULL, "*p_i_polygon_map_gpu should be NULL\n");
    ASSERT(*p_polygon_n_map_gpu == NULL, "*p_polygon_n_map_gpu should be NULL\n");
    ASSERT(*p_owner_map_gpu == NULL, "*p_owner_map_gpu should be NULL\n");
    ASSERT(*p_vertices_gpu == NULL, "*p_vertices_gpu should be NULL\n");

    // allocate local temp buf
    int* i_polygon_map = (int*)malloc(n_polygon * sizeof(int));
    if(!i_polygon_map){
        RAISE("malloc failed.\n");
    }
    int* polygon_n_map = (int*)malloc(n_polygon * sizeof(int));
    if(!polygon_n_map){
        RAISE("malloc failed.\n");
    }

    // get n_vertex to flatten vertices
    int n_vertex = 0;
    for(int i=0; i<n_polygon; i++){
        // prefix sum
        i_polygon_map[i] = n_vertex;

        int polygon_n_vertex = polygon_list[i]->n;
        // just every n
        polygon_n_map[i] = polygon_n_vertex;
        // count
        n_vertex += polygon_n_vertex;
    }

    CHECK(cudaMalloc(p_i_polygon_map_gpu, n_polygon * sizeof(int)));
    CHECK(cudaMemcpy(*p_i_polygon_map_gpu, i_polygon_map, n_polygon * sizeof(int), cudaMemcpyHostToDevice));

    CHECK(cudaMalloc(p_polygon_n_map_gpu, n_polygon * sizeof(int)));
    CHECK(cudaMemcpy(*p_polygon_n_map_gpu, polygon_n_map, n_polygon * sizeof(int), cudaMemcpyHostToDevice));

    free(polygon_n_map);
    free(i_polygon_map);

    int* owner_map = (int*)malloc(n_vertex * sizeof(int));
    if(!owner_map){
        RAISE("malloc failed.\n");
    }

    int* p_owner = owner_map;
    for(int i=0; i<n_polygon; i++){
        int polygon_n_vertex = polygon_list[i]->n;
        for(int j=0; j<polygon_n_vertex; j++){
            *p_owner = i;
            p_owner++;
        }
    }

    CHECK(cudaMalloc(p_owner_map_gpu, n_vertex * sizeof(int)));
    CHECK(cudaMemcpy(*p_owner_map_gpu, owner_map, n_vertex * sizeof(int), cudaMemcpyHostToDevice));

    free(owner_map);

    // alloc point_t vertices_gpu[] on device
    CHECK(cudaMalloc(p_vertices_gpu, n_vertex * sizeof(point_t)));

    // copy vertices from host to device
    point_t* p_vertex = *p_vertices_gpu;
    for(int i=0; i<n_polygon; i++){
        int polygon_n_vertex = polygon_list[i]->n;
        point_t* vertices = polygon_list[i]->vertices;
        CHECK(cudaMemcpy(p_vertex, vertices, polygon_n_vertex * sizeof(point_t), cudaMemcpyHostToDevice));
        p_vertex += polygon_n_vertex;
    }

    return n_vertex;
}

// kernel functions

// Calculate axes of polygons.
// nx = n_vertex, ny = 1
// input:
//   point_t vertices[n_vertex]
//   int n_vertex
//   int owner_map_gpu[n_vertex]
//   int polygon_n_map_gpu[n_vertex]
// output:
//   vector_t axes[n_vertex]
__global__ static void kernel_get_axis(point_t* vertices, int n_vertex, int* owner_map_gpu, int* polygon_n_map_gpu, /*out*/vector_t* axes){
    int ix = threadIdx.x + blockIdx.x * blockDim.x;

    if(ix < n_vertex){
        // e.g.
        // ix:              0, 1, 2, 3, 4, 5, 6
        // owmer_map:       0, 0, 0, 1, 1, 1, 1
        // polygon_n_map:   3, 4
        // i_vertex_a:      0, 1, 2, 3, 4, 5, 6
        // i_vertex_b:      1, 2, 0, 4, 5, 6, 3

        // get ia and ib
        int i_vertex_a = ix;
        int i_vertex_b = (ix + 1) >= n_vertex ? 0 : (ix + 1);
        if(owner_map_gpu[i_vertex_a] != owner_map_gpu[i_vertex_b]){
            int i_polygon_a = owner_map_gpu[i_vertex_a];
            int polygon_n_vertex = polygon_n_map_gpu[i_polygon_a];
            i_vertex_b = i_vertex_a + 1 - polygon_n_vertex;
        }
        
        // calculate axis
        point_t p1 = vertices[i_vertex_a];
        point_t p2 = vertices[i_vertex_b];
        vector_t edge = vector_sub_gpu(*(vector_t*)&p1, *(vector_t*)&p2);
        vector_t norm = vector_normalize_gpu(edge);
        vector_t perp = vector_perpendicular_gpu(norm);

        axes[i_vertex_a] = perp;
    }
}
void calculate_axes(point_t* vertices_gpu, int n_vertex, int* owner_map_gpu, int* polygon_n_map_gpu,
                    /*out*/vector_t** p_axes_gpu){
    ASSERT(*p_axes_gpu == NULL, "*p_axes_gpu should be NULL\n");

    // calculate on the device
    CHECK(cudaMalloc(p_axes_gpu, n_vertex * sizeof(vector_t)));

    const int dimx = 128;
    dim3 block(dimx);
    dim3 grid((n_vertex - 1)/block.x + 1);

    kernel_get_axis<<<grid, block>>>(vertices_gpu, n_vertex, owner_map_gpu, polygon_n_map_gpu, *p_axes_gpu);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
}

// Dot to get every single projection endpoint.
// nx = n_vertex, ny = n_vertex
// input:
//   point_t vertices[n_vertex]
//   vector_t axes[n_vertex]
//   int owner_map[n_vertex]
//   int n_vertices
// output:
//   double projection_endpoint[n_vertex * n_vertex]
//     actually should be projection_endpoints[n_axis][n_vertex]
__global__ static void kernel_get_projection_endpoints(point_t* vertices, vector_t* axes, int* owner_map, int n_vertex,
                                                       /*out*/double* projection_endpoints){
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    // int idx = iy * gridDim.x * blockDim.x + ix;

    // printf("thread_id(%d,%d) block_id(%d,%d) coordinate(%d,%d) global index %2d\n", \
    //     threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, ix, iy, idx);
    int i_vertex = ix;
    int i_axis = iy;
    int i_proj = i_axis * n_vertex + i_vertex;
    if(i_vertex < n_vertex && i_axis < n_vertex){
        // if(owner_map[i_vertex] != owner_map[i_axis]){
            double projection = vector_dot_gpu(*(vector_t*)&vertices[i_vertex], axes[i_axis]);
            projection_endpoints[i_proj] = projection;
        // }
    }
}
void calculate_projection_endpoints(point_t* vertices_gpu, vector_t* axes_gpu, int* owner_map_gpu, int n_vertex,
                                           /*out*/double** p_projection_endpoints_gpu){
    ASSERT(*p_projection_endpoints_gpu == NULL, "*p_projection_endpoint should be NULL\n");

    // allocate projection return value on the device
    // all projection endpoints of vertices and axes, not [min, max] segment
    CHECK(cudaMalloc(p_projection_endpoints_gpu, n_vertex*n_vertex * sizeof(double)));

    const int dimx = 32;
    const int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((n_vertex - 1)/block.x + 1, (n_vertex - 1)/block.y + 1);

    // printf("%d\n", n_vertex);
    kernel_get_projection_endpoints<<<grid, block>>>(vertices_gpu, axes_gpu, owner_map_gpu, n_vertex, /*out*/*p_projection_endpoints_gpu);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
}

// Aggregate projection endpoints to projections([min, max]).
// nx = n_vertex, ny = n_polygon
// input:
//   double projection_endpoints[n_vertex * n_vertex]
//     actually should be projection_endpoints[n_axis][n_vertex]
//   int i_polygon_map[n_polygon]
//     map i_polygon to the start index of vertices.
//   int polygon_n_map[n_vertex]
//     get polygon_n_vertex of the i_vertex of vertices.
//   int n_vertex
//   int n_polygon
// output:
//   projection_t projection_map[n_polygon * n_vertex]
//     actually should be projection_map[n_vertex][n_polygon]
// description:
//   1. for a certain i_polygon, its vertices is
//      projection_endpoints[i_axis][i_polygon_map[i_polygon] : i_polygon_map[i_polygon] + polygon_n_vertex[i_polygon]]
__global__ static void kernel_get_projection_segments(double* projection_endpoints, int* i_polygon_map, int* polygon_n_map,
                                             int n_vertex, int n_polygon, /*out*/projection_t* projection_map){
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    // int idx = iy * gridDim.x * blockDim.x + ix;

    // printf("thread_id(%d,%d) block_id(%d,%d) coordinate(%d,%d) global index %2d\n", \
    //     threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, ix, iy, idx);

    int i_axis = ix;
    int i_polygon = iy;
    if(i_axis < n_vertex && i_polygon < n_polygon){
        // index of a polygon in flattened vertices
        int polygon_idx = i_polygon_map[i_polygon];
        // n vertices of a polygon
        int polygon_n_vertex = polygon_n_map[i_polygon];
        // calculate projection segment
        double* vertices_slice = &projection_endpoints[i_axis*n_vertex + polygon_idx];
        projection_t projection_segment = min_max_gpu(vertices_slice, polygon_n_vertex);
        // return
        projection_map[i_axis*n_polygon + i_polygon] = projection_segment;
    }
}
void calculate_projection_segments(double* projection_endpoints_gpu, int* i_polygon_map_gpu, int* polygon_n_map_gpu,
                                   int n_vertex, int n_polygon, /*out*/projection_t** p_projection_map){
    ASSERT(*p_projection_map == NULL, "*p_projection_map should be NULL\n");

    CHECK(cudaMalloc(p_projection_map, n_vertex*n_polygon * sizeof(projection_t)));

    const int dimx = 32;
    const int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((n_vertex - 1)/block.x + 1, (n_polygon - 1)/block.y + 1);

    kernel_get_projection_segments<<<grid, block>>>(projection_endpoints_gpu, i_polygon_map_gpu, polygon_n_map_gpu, n_vertex, n_polygon, *p_projection_map);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
}

// Get overlapping map of polygons.
// nx = n_polygon, ny = n_polygon, nz = n_vertex
// input:
//   projection_t projection_map[n_vertex * n_polygon]
//     actually should be projection_map[n_vertex][n_polygon]
//   int owner_map[n_vertex]
// output:
//   int result[n_polygon * n_polygon]
// description:
//   check if polygon at ix and at iy overlapping on i_axis(iz)
__global__ static void kernel_get_overlapping(projection_t* projection_map, int n_vertex, int n_polygon, /*out*/int* result){
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int iz = threadIdx.z + blockIdx.z * blockDim.z;

    int i_polygon_a = ix;
    int i_polygon_b = iy;
    int i_axis = iz;
    if(i_polygon_a < n_polygon && i_polygon_b < n_polygon && i_axis < n_vertex){
        if(result[i_polygon_a*n_polygon + i_polygon_b] != 0){
            projection_t* p_projection_a = &projection_map[i_axis*n_polygon + i_polygon_a];
            projection_t* p_projection_b = &projection_map[i_axis*n_polygon + i_polygon_b];
            if(projection_is_overlap_gpu(p_projection_a, p_projection_b) == FALSE){
                // NOTE: test race conditions
                result[i_polygon_a*n_polygon + i_polygon_b] &= 0;   // set not overlapping
            }
        }// else the 2 polygons can not be overlapped, no need to calculate
    }
}
void calculate_is_overlapping(projection_t* projection_map_gpu, int n_vertex, int n_polygon, /*out*/int** p_result){
    ASSERT(*p_result == NULL, "*p_result should be NULL\n");

    CHECK(cudaMalloc(p_result, n_polygon*n_polygon * sizeof(int)));
    CHECK(cudaMemset(*p_result, 0xFF, n_polygon*n_polygon * sizeof(int)));

    const int dimx = 4;
    const int dimy = 4;
    const int dimz = 32;
    dim3 block(dimx, dimy, dimz);
    dim3 grid((n_polygon - 1)/block.x + 1, (n_polygon - 1)/block.y + 1, (n_vertex - 1)/block.z + 1);
    // printf("grid (%d, %d, %d)\n", grid.x, grid.y, grid.z);

    kernel_get_overlapping<<<grid, block>>>(projection_map_gpu, n_vertex, n_polygon, *p_result);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
}

// global interfaces and export for user

void detect_overlap_gpu(polygon_t** polygon_list, int** result, int n_polygon){
    ASSERT(result != NULL, "result should not be NULL\n");

    // 1. flatten vertices of all polygons to device
    point_t* vertices_gpu = NULL;   // destructor: cudaFree
    // vertices of polygon i is vertices[i_polygon_map[i] : i_polygon_map[i] + polygon_n_map[i]]
    int* i_polygon_map_gpu = NULL;   // destructor: cudaFree
    int* polygon_n_map_gpu = NULL;   // destructor: cudaFree
    int* owner_map_gpu = NULL;   // destructor: cudaFree
    int n_vertex = util_flatten(polygon_list, n_polygon, /*out*/&i_polygon_map_gpu, /*out*/&polygon_n_map_gpu, /*out*/&owner_map_gpu, /*out*/&vertices_gpu);

    // 2. calculate axes
    vector_t* axes_gpu = NULL;   // destructor: cudaFree
    calculate_axes(vertices_gpu, n_vertex, owner_map_gpu, polygon_n_map_gpu, /*out*/&axes_gpu);

    // 3. calculate projection endpoints
    double* projection_endpoints_gpu = NULL;   // destructor: cudaFree
    calculate_projection_endpoints(vertices_gpu, axes_gpu, owner_map_gpu, n_vertex, /*out*/&projection_endpoints_gpu);

    // 4. calculate projection segments
    projection_t* projection_map_gpu = NULL;   // destructor: cudaFree
    calculate_projection_segments(projection_endpoints_gpu, i_polygon_map_gpu, polygon_n_map_gpu, n_vertex, n_polygon, /*out*/&projection_map_gpu);

    // 5. calculate overlapping
    int* result_gpu = NULL;
    calculate_is_overlapping(projection_map_gpu, n_vertex, n_polygon, /*out*/&result_gpu);

    // set result
    int* result_host = (int*)malloc(n_polygon*n_polygon * sizeof(int));
    if(!result_host){
        RAISE("malloc failed.\n");
    }
    CHECK(cudaMemcpy(result_host, result_gpu, n_polygon*n_polygon * sizeof(int), cudaMemcpyDeviceToHost));
    for(int i_polygon_a=0; i_polygon_a<n_polygon; i_polygon_a++){
        for(int i_polygon_b=i_polygon_a; i_polygon_b<n_polygon; i_polygon_b++){
            if(i_polygon_a == i_polygon_b){
                result[i_polygon_a][i_polygon_b] = 0;
                continue;
            }
            int i = i_polygon_a*n_polygon + i_polygon_b;
            result[i_polygon_a][i_polygon_b] = (result_host[i] != 0);
            result[i_polygon_b][i_polygon_a] = (result_host[i] != 0);
        }
    }
    free(result_host);
    
    // recycle resources
    // from 5.
    cudaFree(result_gpu);
    // from 4.
    cudaFree(projection_map_gpu);
    // from 3.
    cudaFree(projection_endpoints_gpu);
    // from 2.
    cudaFree(axes_gpu);
    // from 1.
    cudaFree(owner_map_gpu);
    cudaFree(polygon_n_map_gpu);
    cudaFree(i_polygon_map_gpu);
    cudaFree(vertices_gpu);

    // typedef unsigned long long int uint64_t;
    // uint64_t nv = (uint64_t)n_vertex;
    // uint64_t np = (uint64_t)n_polygon;
    // uint64_t n_kernel = max(nv*nv, np*np*nv);
    // printf("kernel num: %llu\n", n_kernel);
}
