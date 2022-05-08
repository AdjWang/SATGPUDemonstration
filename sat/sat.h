#ifndef __SAT_H__
#define __SAT_H__
#include "config.h"

typedef struct point_t {
    double x;
    double y;
}point_t;

typedef struct vector_t {
    double x;
    double y;
}vector_t;

typedef struct projection_t {
    double left;
    double right;
}projection_t;

typedef struct polygon_t {
    point_t* vertices;
    vector_t* axes;
    int n;
}polygon_t;

extern polygon_t* new_polygon(int n_vertex);
extern void del_polygon(polygon_t* polygon);
extern int polygon_print(const polygon_t* polygon);
extern void detect_overlap(polygon_t** polygon_list, int** result, int n);
extern void detect_overlap_gpu(polygon_t** polygon_list, int** result, int n);

// only for test use
extern BOOL polygon_is_overlap(const polygon_t* polygon1, const polygon_t* polygon2);

extern void calculate_axes(point_t* vertices_gpu, int n_vertex, int* owner_map_gpu, int* polygon_n_map_gpu,
                           /*out*/vector_t** p_axes_gpu);
extern void make_owner_map(polygon_t** polygon_list, int n_polygon, int n_vertex, /*out*/int** p_owner_map, /*out*/int** p_owner_map_gpu);
extern void calculate_projection_endpoints(point_t* vertices_gpu, vector_t* axes_gpu, int* owner_map, int n_vertex,
                                           /*out*/double** p_projection_endpoints_gpu);
extern void calculate_projection_segments(double* projection_endpoints_gpu, polygon_t** polygon_list, int n_polygon, int n_vertex,
                                          /*out*/projection_t** p_projection_map);
extern void calculate_is_overlapping(projection_t* projection_map_gpu, int* owner_map_gpu, int n_vertex, /*output*/int** result);

#endif