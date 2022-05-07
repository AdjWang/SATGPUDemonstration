#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include "sat.h"

// GPU utilities
#include <cuda_runtime.h>
#include "dependence.h"

// deprecated -----------------------------------------------------------------

// point_t* new_point(double x, double y){
//     point_t* point = (point_t*)malloc(sizeof(point_t));
//     if(!point){
//         RAISE("malloc failed\n");
//     }
//     point->x = x;
//     point->y = y;
//     return point;
// }

// void del_point(point_t* point){
//     if(point){
//         free(point);
//         point = NULL;
//     }
// }

// vector_t* new_vector(double x, double y){
//     vector_t* vector = (vector_t*)malloc(sizeof(vector_t));
//     if(!vector){
//         RAISE("malloc failed\n");
//     }
//     vector->x = x;
//     vector->y = y;
//     return vector;
// }

// void del_vector(vector_t* vector){
//     if(vector){
//         free(vector);
//         vector = NULL;
//     }
// }

// vector_t* vector_sub(const vector_t* vec1, const vector_t* vec2, vector_t* ret){
//     ret->x = vec1->x - vec2->x;
//     ret->y = vec1->y - vec2->y;
//     return ret;
// }

// double vector_dot(const vector_t* vec1, const vector_t* vec2){
//     return vec1->x*vec2->x + vec1->y*vec2->y;
// }

// double vecotr_magnitude(const vector_t* vec){
//     return sqrt(vec->x*vec->x + vec->y*vec->y);
// }

// vector_t* vector_normalize(const vector_t* vec, vector_t* ret){
//     double mag = vecotr_magnitude(vec);
//     ret->x = vec->x / mag;
//     ret->y = vec->y / mag;
//     return ret;
// }

// BOOL projection_is_overlap(const projection_t* projection1, const projection_t* projection2){
//     return (projection1->left < projection2->right
//         && projection1->right > projection2->left);
// }

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

static int polygon_print_point_list(const point_t* point_list, int n){
    int c = 0;
    c += printf("[");
    for(int i=0; i<n; i++){
        c += printf("(%.16lf,%.16lf),", point_list[i].x, point_list[i].y);
    }
    c += printf("]\n");
    return c;
}

int polygon_print(const polygon_t* polygon){
    int c = 0;
    c += printf("%d\n", polygon->n);
    c += polygon_print_point_list(polygon->vertices, polygon->n);
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

// __device__ static vector_t vector_sub_gpu(const vector_t vec1, const vector_t vec2){
//     return (vector_t){
//         .x = vec1.x - vec2.x,
//         .y = vec1.y - vec2.y,
//     };
// }

__device__ static double vector_dot_gpu(const vector_t vec1, const vector_t vec2){
    return vec1.x*vec2.x + vec1.y*vec2.y;
}

// __device__ static double vecotr_magnitude_gpu(const vector_t vec){
//     return sqrt(vec.x*vec.x + vec.y*vec.y);
// }

// __device__ static vector_t vector_normalize_gpu(const vector_t vec){
//     double mag = vecotr_magnitude(vec);
//     return (vector_t){
//         .x = vec.x / mag,
//         .y = vec.y / mag,
//     };
// }

// __device__ static vector_t vector_perpendicular_gpu(const vector_t vec){
//     // or (y, -x)
//     return (vector_t){
//         .x = -vec.y,
//         .y = vec.x,
//     };
// }

// __global__ static void sat_get_axes(point_t* vertices, vector_t* axes){
//     int ix = threadIdx.x + blockIdx.x * blockDim.x;
//     int iy = threadIdx.y + blockIdx.y * blockDim.y;
//     int idx = iy * gridDim.x * blockDim.x + ix;

//     // point_t p1 = vertices[idx].
//     // vector_t edge = vector_sub(*(vector_t*)&p1, *(vector_t*)&p2);
//     // vector_t norm = vector_normalize(edge);
//     // vector_t perp = vector_perpendicular(norm);
// }

__global__ static void sat_get_projects(point_t* vertices, vector_t* axes, int* owner_map, /*out*/double* projections, const int n_vertex){
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = iy * gridDim.x * blockDim.x + ix;

    printf("thread_id(%d,%d) block_id(%d,%d) coordinate(%d,%d) global index %2d\n", \
        threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, ix, iy, idx);
    int i_vertex = ix;
    int i_axis = iy;
    int i_proj = i_axis * n_vertex + i_vertex;
    if(i_vertex < nx && i_axis < ny){
        if(owner_map[i_vertex] != owner_map[i_axis]){
            double projection = vector_dot_gpu(*(vector_t*)&vertices[i_vertex], axes[i_axis]);
            projections[i_proj] = projection;
        }
    }
}

void detect_overlap_gpu(polygon_t** polygon_list, int** result, int n){
    // TODO: use GPU to calculate
    // calculate axes
    for(int i=0; i<n; i++){
        polygon_t* polygon = polygon_list[i];
        polygon_get_axes(polygon);
    }

    initDevice(0);

    // get n to flatten vertices
    int n_vertex = 0;
    for(int i=0; i<n; i++){
        n_vertex += polygon_list[i]->n;
    }
    
    // malloc on device
    // argument
    point_t* vertices_gpu = NULL;
    vector_t* axes_gpu = NULL;
    int* owner_map_gpu = NULL;
    CHECK(cudaMalloc((point_t**)&vertices_gpu, n_vertex * sizeof(point_t)));
    CHECK(cudaMalloc((vector_t**)&axes_gpu, n_vertex * sizeof(vector_t)));
    CHECK(cudaMalloc((int**)&owner_map_gpu, n_vertex * sizeof(int)));

    // copy vertices from host to device
    point_t* p_vertex = vertices_gpu;
    for(int i=0; i<n; i++){
        int polygon_n_vertex = polygon_list[i]->n;
        point_t* vertices = polygon_list[i]->vertices;
        CHECK(cudaMemcpy(p_vertex, vertices, polygon_n_vertex * sizeof(point_t), cudaMemcpyHostToDevice));
        p_vertex += polygon_n_vertex;
    }
    // TODO: use GPU to calculate
    // copy axes from host to device
    vector_t* p_axis = axes_gpu;
    for(int i=0; i<n; i++){
        int polygon_n_axis = polygon_list[i]->n;
        vector_t* axes = polygon_list[i]->axes;
        CHECK(cudaMemcpy(p_axis, axes, polygon_n_axis * sizeof(vector_t), cudaMemcpyHostToDevice));
        p_axis += polygon_n_axis;
    }
    // set and copy owner_map from host to device
    int* owner_map = (int*)malloc(n_vertex * sizeof(int));
    if(!owner_map){
        RAISE("malloc failed.\n");
    }
    int* p_owner = owner_map;
    for(int i=0; i<n; i++){
        int polygon_n_vertex = polygon_list[i]->n;
        for(int j=0; j<polygon_n_vertex; j++){
            *p_owner = i;
        }
    }
    CHECK(cudaMemcpy(owner_map_gpu, owner_map, n_vertex * sizeof(int), cudaMemcpyHostToDevice));

    // projection return value
    double* projections_gpu = NULL; // all projections of vertices and axes, not [min, max]
    CHECK(cudaMalloc((double**)&projections_gpu, n_vertex*n_vertex * sizeof(double)));

    const int dimx = 4;
    const int dimy = 4;
    dim3 block(dimx, dimy);
    dim3 grid((nx - 1)/block.x + 1, (ny - 1)/block.y + 1);

    printf("%d\n", n_vertex);
    sat_get_projects<<<grid, block>>>(vertices_gpu, axes_gpu, owner_map_gpu, projections_gpu, n_vertex);
    CHECK(cudaDeviceSynchronize());

    // cpu logics
    // copy projections from device to host
    double* projections = (double*)malloc(n_vertex*n_vertex * sizeof(double));
    if(!projections){
        RAISE("malloc failed.\n");
    }
    CHECK(cudaMemcpy(projections, projections_gpu, n_vertex*n_vertex * sizeof(double), cudaMemcpyDeviceToHost));
    
    projection_t* projection_group = (projection_t*)malloc(n*n)
    for(int i_axis=0; i_axis<n_vertex; i_axis++){
        for(int i_vertex=0; i_vertex<n_vertex; i_vertex++){

        }
    }

    cudaFree(vertices_gpu);
    cudaFree(axes_gpu);
    cudaFree(owner_map_gpu);

    cudaFree(projections_gpu);

    free(owner_map);
    free(projections);
    cudaDeviceReset();
}
