#include <stdlib.h>
#include "loader.h"
#include "sat.h"


int main(int argc, char* argv[]){
    FILE* fp = fopen("./test_input.txt", "r");
    polygon_t** polygon_list = NULL;
    int n = load_polygons(fp, &polygon_list);
    ASSERT(polygon_list != NULL, "polygon_list is NULL, initialization failed");
    fclose(fp);

    // print polygons
    for(int i=0; i<n; i++){
        polygon_print(polygon_list[i]);
    }

    // free polygons
    for(int i=0; i<n; i++){
        del_polygon(polygon_list[i]);
    }

    return 0;
}