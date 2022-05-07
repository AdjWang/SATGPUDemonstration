#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include "sat.h"

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

void polygon_get_axes(polygon_t* polygon){
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
