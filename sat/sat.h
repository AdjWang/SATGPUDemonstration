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
extern void polygon_print(const polygon_t* polygon);
extern void polygon_get_axes(polygon_t* polygon);
extern BOOL polygon_is_overlap(const polygon_t* polygon1, const polygon_t* polygon2);

#endif