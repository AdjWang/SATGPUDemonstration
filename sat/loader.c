#include <stdlib.h>
#include "sat.h"
#include "loader.h"
#include "config.h"

// static int load_n(FILE* fp){
//     int n_polygon = 0;
//     fscanf(fp, "%d", &n_polygon);
//     return n_polygon;
// }


static void load_vertices(FILE* fp, polygon_t* polygon){
    ASSERT(polygon != NULL, "polygon is NULL\n");

    char temp;
    fscanf(fp, "%c", &temp);    // skip '['

    // read points
    for(int j=0; j<polygon->n; j++){
        double x, y;
        fscanf(fp, "(%lf,%lf)%*[,]", &x, &y);
        DBG_PRINT("x, y: (%lf, %lf)\n", x, y);

        polygon->vertices[j].x = x;
        polygon->vertices[j].y = y;
    }

    fscanf(fp, "%c\n", &temp);    // skip ']'
}

static polygon_t* load_polygon(FILE* fp){
    int n_vertex = 0;
    fscanf(fp, "%d\n", &n_vertex);
    if(n_vertex <= 2){
        RAISE("n_vertex: %d <= 2\n", n_vertex);
    }
    DBG_PRINT("vertex num: %d\n", n_vertex);

    // malloc
    polygon_t* polygon = new_polygon(n_vertex);
    load_vertices(fp, polygon);

    return polygon;
}

int load_polygons(FILE* fp, /*out*/ polygon_t*** p_polygon_list){
    int n_polygon = 0;
    fscanf(fp, "%d\n", &n_polygon);
    if(n_polygon <= 0){
        return 0;
    }
    DBG_PRINT("polygon num: %d\n", n_polygon);

    *p_polygon_list = (polygon_t**)malloc(sizeof(polygon_t*) * n_polygon);
    if(!(*p_polygon_list)){
        RAISE("malloc failed.\n");
    }
    for(int i=0; i<n_polygon; i++){
        // malloc
        polygon_t* polygon = load_polygon(fp);
        // pre-calculate axes
        polygon_get_axes(polygon);

        (*p_polygon_list)[i] = polygon;
    }

    return n_polygon;
}
