#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include "loader.h"
#include "sat.h"

int print_overlap_result(const int* overlap_indices, int n){
    int print_count = 0;
    int count = 0;
    for(int i=0; i<n; i++){
        if(overlap_indices[i] == 1){
            count++;
        }
    }
    print_count += printf("%d\n", count);
    print_count += printf("[");
    for(int i=0; i<n; i++){
        if(overlap_indices[i] == 1){
            print_count += printf("%d,", i);
        }
    }
    print_count += printf("]\n");
    return print_count;
}

int main(int argc, char* argv[]){
    ASSERT(argc == 2, "invalid argument. usage: sat.out polygon_input.txt\n");
    const char* input_file = argv[1];

    FILE* ifp = fopen(input_file, "r");
    if(ifp == NULL){
        printf("Error opening file %s: %s\n", input_file, strerror(errno));
        return errno;
    }
    polygon_t** polygon_list = NULL;
    int n = load_polygons(ifp, &polygon_list);
    ASSERT(polygon_list != NULL, "polygon_list is NULL, initialization failed");
    fclose(ifp);

    // sat detection
    int** result = (int**)malloc(n*sizeof(int*));
    for(int i=0; i<n; i++){
        result[i] = (int*)calloc(n, sizeof(int));
    }
    detect_overlap_gpu(polygon_list, result, n);

    // // print total number of polygons
    // printf("%d\n", n);
    // // print polygons
    // for(int i=0; i<n; i++){
    //     polygon_print(polygon_list[i]);
    //     print_overlap_result(result[i], n);
    // }

    // free polygons
    for(int i=0; i<n; i++){
        del_polygon(polygon_list[i]);
    }

    for(int i=0; i<n; i++){
        free(result[i]);
    }
    free(result);

    // test();

    return 0;
}