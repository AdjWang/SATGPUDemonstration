#include <stdlib.h>
#include <errno.h>
#include <math.h>
#include "loader.h"
#include "dependence.h"
#include "sat.h"

#define     TEMP_TEST_FILE      "./temp_test.txt"
#define     FAILED              "\e[31mFAILED\e[0m\n"
#define     PASSED              "\e[32mPASSED\e[0m\n"

// utilities ------------------------------------------------------------------

#define     LEN(x)  (sizeof(x)/sizeof(x[0]))

static BOOL isclose(double a, double b){
    return fabs(a-b) < 1e-16;
}

static void generate_temp_test_file(const char* test_input){
    FILE* fp = fopen(TEMP_TEST_FILE, "w");
    if(fp == NULL){
        printf("Error opening file %s: %s\n", TEMP_TEST_FILE, strerror(errno));
        exit(1);
    }
    fputs(test_input, fp);
    fclose(fp);
}

static void remove_temp_test_file(){
    remove(TEMP_TEST_FILE);
}

static int test_load_polygons(polygon_t*** p_polygon_list){
    ASSERT(*p_polygon_list == NULL, "polygon_list is not NULL, initialize it as NULL\n");

    FILE* fp = fopen(TEMP_TEST_FILE, "r");
    if(fp == NULL){
        printf("Error opening file %s: %s\n", TEMP_TEST_FILE, strerror(errno));
        exit(1);
    }
    int n = load_polygons(fp, p_polygon_list);
    if(n == 0){
        ASSERT(*p_polygon_list == NULL, "polygon_list is not NULL, should not be initialized");
    }else{
        ASSERT(*p_polygon_list != NULL, "polygon_list is NULL, initialization failed");
    }
    fclose(fp);
    return n;
}

static void test_free_polygons(polygon_t** polygon_list, int n){
    for(int i=0; i<n; i++){
        del_polygon(polygon_list[i]);
        polygon_list[i] = NULL;
    }
    free(polygon_list);
}

// test cases -----------------------------------------------------------------

static void test_empty_polygon(){
    printf("empty_polygon...");

    // the number of polygons is 0
    generate_temp_test_file("0");
    int n_truth = 0;

    polygon_t** temp = NULL;
    int n = test_load_polygons(&temp);

    ASSERT(n == n_truth, "wrong number of polygon. n: %d, expected: %d\n", n, n_truth);

    test_free_polygons(temp, n);
    remove_temp_test_file();

    printf(PASSED);
}

static void test_n_polygon(){
    printf("n_polygon...");

    generate_temp_test_file("\
2\n\
6\n\
[(0.9343917934863109,0.13045144291207023),(0.3496595677721136,0.029371321654252047),(0.17376032399107255,0.19791525658619602),(0.06493821266996724,0.904690432299808),(0.5443052896643373,0.7616839755359994),(0.6764463372540498,0.5474568981060993)]\n\
3\n\
[(0.5443052896643373,0.7616839755359994),(0.9315831160478467,0.30600531904742123),(0.2843560333040439,0.5658618493916742)]");
    int n_truth = 2;

    polygon_t** temp = NULL;
    int n = test_load_polygons(&temp);

    ASSERT(n == n_truth, "wrong number of polygon. n: %d, expected: %d\n", n, n_truth);

    test_free_polygons(temp, n);
    remove_temp_test_file();

    printf(PASSED);
}

static void test_n_vertices(){
    printf("n_vertices...");

    generate_temp_test_file("\
2\n\
6\n\
[(0.9343917934863109,0.13045144291207023),(0.3496595677721136,0.029371321654252047),(0.17376032399107255,0.19791525658619602),(0.06493821266996724,0.904690432299808),(0.5443052896643373,0.7616839755359994),(0.6764463372540498,0.5474568981060993)]\n\
3\n\
[(0.5443052896643373,0.7616839755359994),(0.9315831160478467,0.30600531904742123),(0.2843560333040439,0.5658618493916742)]");
    int n_vertex_list[] = {6, 3};

    polygon_t** polygon_list = NULL;
    int n = test_load_polygons(&polygon_list);

    for(int i=0; i<n; i++){
        ASSERT(polygon_list[i]->n == n_vertex_list[i], \
        "wrong number of vertices of polygon %d. n: %d, expected: %d\n", i, polygon_list[i]->n, n_vertex_list[i]);
    }

    test_free_polygons(polygon_list, n);
    remove_temp_test_file();

    printf(PASSED);
}

static void test_load_points(){
    printf("load_points...");

    generate_temp_test_file("\
2\n\
6\n\
[(0.9343917934863109,0.13045144291207023),(0.3496595677721136,0.029371321654252047),(0.17376032399107255,0.19791525658619602),(0.06493821266996724,0.904690432299808),(0.5443052896643373,0.7616839755359994),(0.6764463372540498,0.5474568981060993)]\n\
3\n\
[(0.5443052896643373,0.7616839755359994),(0.9315831160478467,0.30600531904742123),(0.2843560333040439,0.5658618493916742)]");
    double point_list[] = {
        0.9343917934863109,
        0.13045144291207023,
        0.3496595677721136,
        0.029371321654252047,
        0.17376032399107255,
        0.19791525658619602,
        0.06493821266996724,
        0.904690432299808,
        0.5443052896643373,
        0.7616839755359994,
        0.6764463372540498,
        0.5474568981060993,
        0.5443052896643373,
        0.7616839755359994,
        0.9315831160478467,
        0.30600531904742123,
        0.2843560333040439,
        0.5658618493916742
    };

    polygon_t** polygon_list = NULL;
    int n = test_load_polygons(&polygon_list);

    int count = 0;
    for(int i=0; i<n; i++){
        for(int j=0; j<polygon_list[i]->n; j++){
            double a, b;
            a = polygon_list[i]->vertices[j].x;
            b = point_list[count];
            ASSERT(isclose(a, b), \
            "different point.x polygon: %d, point: %d, val: %lf, expected: %lf\n", i, j, a, b);
            count++;

            a = polygon_list[i]->vertices[j].y;
            b = point_list[count];
            ASSERT(isclose(a, b), \
            "different point.y polygon: %d, point: %d, val: %lf, expected: %lf\n", i, j, a, b);
            count++;
        }
    }

    test_free_polygons(polygon_list, n);
    remove_temp_test_file();

    printf(PASSED);
}

static void test_polygon_is_overlap(){
    printf("polygon_is_overlap...");

    generate_temp_test_file("\
2\n\
3\n\
[(0.1268362978956069,0.41049566540979554),(0.3667669280136283,0.9061009375103477),(0.3205326471661416,0.9157739930630437)]\n\
3\n\
[(0.6464897380663633,0.5765302172591958),(0.3205326471661416,0.9157739930630437),(0.4121879820748392,0.40086738367448793)]");

    polygon_t** case2 = NULL;
    int n = test_load_polygons(&case2);
    BOOL ret = polygon_is_overlap(case2[0], case2[1]);
    ASSERT(ret == TRUE, "test case2 failed. should overlap\n");
    test_free_polygons(case2, n);

    remove_temp_test_file();

    printf(PASSED);
}

// gpu ------------------------------------------------------------------------

// utilities ------------------------------------------------------------------

static void* cudaMallocBy(int n_bytes, void* data_source){
    void* ptr_output = NULL;
    CHECK(cudaMalloc(&ptr_output, n_bytes));
    CHECK(cudaMemcpy(ptr_output, data_source, n_bytes, cudaMemcpyHostToDevice));
    return ptr_output;
}

static void* hostMallocBy(int n_bytes, void* data_source){
    void* ptr_output = malloc(n_bytes);
    if(!ptr_output){
        RAISE("malloc failed.\n");
    }
    CHECK(cudaMemcpy(ptr_output, data_source, n_bytes, cudaMemcpyDeviceToHost));
    return ptr_output;
}

// test cases -----------------------------------------------------------------

static void test_calculate_axes(){
    printf("calculate_axes...");

    const int n_vertex = 7;
    point_t vertices[7] = {
        (point_t){.x = 0.0, .y = 0.0},
        (point_t){.x = 1.0, .y = 0.0},
        (point_t){.x = 0.5, .y = 1.0},

        (point_t){.x = 0.0, .y = 0.0},
        (point_t){.x = 1.0, .y = 0.0},
        (point_t){.x = 1.0, .y = 1.0},
        (point_t){.x = 0.0, .y = 1.0},
    };
    int owner_map[7] = {0, 0, 0, 1, 1, 1, 1};
    int polygon_n_map[2] = {3, 4};

    point_t* vertices_gpu = (point_t*)cudaMallocBy(n_vertex * sizeof(point_t), vertices);
    int* owner_map_gpu = (int*)cudaMallocBy(n_vertex * sizeof(int), owner_map);
    int* polygon_n_map_gpu = (int*)cudaMallocBy(2 * sizeof(int), polygon_n_map);

    vector_t* axes_gpu = NULL;
    calculate_axes(vertices_gpu, n_vertex, owner_map_gpu, polygon_n_map_gpu, &axes_gpu);
    vector_t* axes = (vector_t*)hostMallocBy(n_vertex * sizeof(vector_t), axes_gpu);

    // assert axes
    vector_t axes_truth[7] = {
        (vector_t){.x = 0, .y = -1},
        (vector_t){.x = 0.89442719099991586, .y = 0.44721359549995793},
        (vector_t){.x = -0.89442719099991586, .y = 0.44721359549995793},

        (vector_t){.x = 0, .y = -1},
        (vector_t){.x = 1, .y = 0},
        (vector_t){.x = 0, .y = 1},
        (vector_t){.x = -1, .y = 0},
    };
    for(int i=0; i<n_vertex; i++){
        ASSERT(isclose(axes[i].x, axes_truth[i].x),
               "axis %d mismatch. got x=%lf, should be x=%lf\n", i, axes[i].x, axes_truth[i].x);
        ASSERT(isclose(axes[i].y, axes_truth[i].y),
               "axis %d mismatch. got y=%lf, should be y=%lf\n", i, axes[i].y, axes_truth[i].y);
    }

    // recycle resources
    free(axes);
    cudaFree(axes_gpu);
    cudaFree(polygon_n_map_gpu);
    cudaFree(owner_map_gpu);
    cudaFree(vertices_gpu);

    printf(PASSED);
}

static void test_calculate_projection_endpoints(){
    printf("calculate_projection_endpoints...");

    const int n_vertex = 7;
    point_t vertices[7] = {
        (point_t){.x = 0.0, .y = 0.0},
        (point_t){.x = 1.0, .y = 0.0},
        (point_t){.x = 0.5, .y = 1.0},

        (point_t){.x = 0.0, .y = 0.0},
        (point_t){.x = 1.0, .y = 0.0},
        (point_t){.x = 1.0, .y = 1.0},
        (point_t){.x = 0.0, .y = 1.0},
    };
    vector_t axes[7] = {
        (vector_t){.x = 0, .y = -1},
        (vector_t){.x = 0.89442719099991586, .y = 0.44721359549995793},
        (vector_t){.x = -0.89442719099991586, .y = 0.44721359549995793},

        (vector_t){.x = 0, .y = -1},
        (vector_t){.x = 1, .y = 0},
        (vector_t){.x = 0, .y = 1},
        (vector_t){.x = -1, .y = 0},
    };
    int owner_map[7] = {0, 0, 0, 1, 1, 1, 1};
    int polygon_n_map[2] = {3, 4};

    point_t* vertices_gpu = (point_t*)cudaMallocBy(n_vertex * sizeof(point_t), vertices);
    vector_t* axes_gpu = (vector_t*)cudaMallocBy(n_vertex * sizeof(vector_t), axes);
    int* owner_map_gpu = (int*)cudaMallocBy(n_vertex * sizeof(int), owner_map);
    int* polygon_n_map_gpu = (int*)cudaMallocBy(2 * sizeof(int), polygon_n_map);

    double* projection_endpoints_gpu = NULL;
    calculate_projection_endpoints(vertices_gpu, axes_gpu, owner_map_gpu, n_vertex, &projection_endpoints_gpu);
    double* projection_endpoints = (double*)hostMallocBy(n_vertex*n_vertex * sizeof(double), projection_endpoints_gpu);

    // assert projection endpoints
    double projection_endpoints_truth[7*7] = {
        0.0000000000000000, 0.0000000000000000, -1.0000000000000000, 0.0000000000000000, 0.0000000000000000, -1.0000000000000000, -1.0000000000000000, 
        0.0000000000000000, 0.8944271909999159, 0.8944271909999159, 0.0000000000000000, 0.8944271909999159, 1.3416407864998738, 0.4472135954999579, 
        0.0000000000000000, -0.8944271909999159, 0.0000000000000000, 0.0000000000000000, -0.8944271909999159, -0.4472135954999579, 0.4472135954999579, 
        0.0000000000000000, 0.0000000000000000, -1.0000000000000000, 0.0000000000000000, 0.0000000000000000, -1.0000000000000000, -1.0000000000000000, 
        0.0000000000000000, 1.0000000000000000, 0.5000000000000000, 0.0000000000000000, 1.0000000000000000, 1.0000000000000000, 0.0000000000000000, 
        0.0000000000000000, 0.0000000000000000, 1.0000000000000000, 0.0000000000000000, 0.0000000000000000, 1.0000000000000000, 1.0000000000000000, 
        0.0000000000000000, -1.0000000000000000, -0.5000000000000000, 0.0000000000000000, -1.0000000000000000, -1.0000000000000000, 0.0000000000000000
    };
    for(int i_axis=0; i_axis<n_vertex; i_axis++){
        for(int i_vertex=0; i_vertex<n_vertex; i_vertex++){
            int i = i_axis*n_vertex + i_vertex;
            // printf("%.16lf ", projection_endpoints[i]);
            ASSERT(isclose(projection_endpoints[i], projection_endpoints_truth[i]),
                   "projection endpoint %d mismatch. got %lf, should be %lf\n", i, projection_endpoints[i], projection_endpoints_truth[i]);
        }
        // printf("\n");
    }

    // recycle resources
    free(projection_endpoints);
    cudaFree(projection_endpoints_gpu);
    cudaFree(polygon_n_map_gpu);
    cudaFree(owner_map_gpu);
    cudaFree(axes_gpu);
    cudaFree(vertices_gpu);

    printf(PASSED);
}

static void test_calculate_projection_segments(){
    printf("calculate_projection_segments...");

    const int n_vertex = 7;
    const int n_polygon = 2;
    double projection_endpoints[7*7] = {
        0.0000000000000000, 0.0000000000000000, -1.0000000000000000, 0.0000000000000000, 0.0000000000000000, -1.0000000000000000, -1.0000000000000000, 
        0.0000000000000000, 0.8944271909999159, 0.8944271909999159, 0.0000000000000000, 0.8944271909999159, 1.3416407864998738, 0.4472135954999579, 
        0.0000000000000000, -0.8944271909999159, 0.0000000000000000, 0.0000000000000000, -0.8944271909999159, -0.4472135954999579, 0.4472135954999579, 
        0.0000000000000000, 0.0000000000000000, -1.0000000000000000, 0.0000000000000000, 0.0000000000000000, -1.0000000000000000, -1.0000000000000000, 
        0.0000000000000000, 1.0000000000000000, 0.5000000000000000, 0.0000000000000000, 1.0000000000000000, 1.0000000000000000, 0.0000000000000000, 
        0.0000000000000000, 0.0000000000000000, 1.0000000000000000, 0.0000000000000000, 0.0000000000000000, 1.0000000000000000, 1.0000000000000000, 
        0.0000000000000000, -1.0000000000000000, -0.5000000000000000, 0.0000000000000000, -1.0000000000000000, -1.0000000000000000, 0.0000000000000000
    };
    // polygon 0 from vertices[0] and polygon 1 from vertices[3]
    int i_polygon_map[2] = {0, 3};
    // polygon 0 has 3 vertices and polygon 1 has 4 vertices
    int polygon_n_map[2] = {3, 4};

    double* projection_endpoints_gpu = (double*)cudaMallocBy(n_vertex*n_vertex * sizeof(double), projection_endpoints);
    int* i_polygon_map_gpu = (int*)cudaMallocBy(2 * sizeof(int), i_polygon_map);
    int* polygon_n_map_gpu = (int*)cudaMallocBy(2 * sizeof(int), polygon_n_map);

    projection_t* projection_map_gpu = NULL;
    // calculate_projection_endpoints(vertices_gpu, axes_gpu, owner_map_gpu, n_vertex, &projection_endpoints_gpu);
    // double* projection_endpoints = (double*)hostMallocBy(n_vertex*n_vertex * sizeof(double), projection_endpoints_gpu);
    calculate_projection_segments(projection_endpoints_gpu, i_polygon_map_gpu, polygon_n_map_gpu, n_vertex, n_polygon, &projection_map_gpu);
    projection_t* projection_map = (projection_t*)hostMallocBy(n_vertex*n_polygon * sizeof(projection_t), projection_map_gpu);
    

    // assert projection map
    projection_t projection_map_truth[7*2] = {
        (projection_t){.left=-1.0000000000000000, .right=0.0000000000000000}, (projection_t){.left=-1.0000000000000000, .right=0.0000000000000000},
        (projection_t){.left=0.0000000000000000, .right=0.8944271909999159}, (projection_t){.left=0.0000000000000000, .right=1.3416407864998738},
        (projection_t){.left=-0.8944271909999159, .right=0.0000000000000000}, (projection_t){.left=-0.8944271909999159, .right=0.4472135954999579},
        (projection_t){.left=-1.0000000000000000, .right=0.0000000000000000}, (projection_t){.left=-1.0000000000000000, .right=0.0000000000000000},
        (projection_t){.left=0.0000000000000000, .right=1.0000000000000000}, (projection_t){.left=0.0000000000000000, .right=1.0000000000000000},
        (projection_t){.left=0.0000000000000000, .right=1.0000000000000000}, (projection_t){.left=0.0000000000000000, .right=1.0000000000000000},
        (projection_t){.left=-1.0000000000000000, .right=0.0000000000000000}, (projection_t){.left=-1.0000000000000000, .right=0.0000000000000000},
    };
    for(int i_axis=0; i_axis<n_vertex; i_axis++){
        for(int i_polygon=0; i_polygon<n_polygon; i_polygon++){
            int i = i_axis*n_polygon + i_polygon;
            projection_t projection = projection_map[i];
            projection_t projection_truth = projection_map_truth[i];
            // printf(".left=%.16lf, .right=%.16lf ", projection.left, projection.right);
            ASSERT(isclose(projection.left, projection_truth.left),
                   "projection %d mismatch. got left=%lf, should be left=%lf\n", i, projection.left, projection_truth.left);
            ASSERT(isclose(projection.right, projection_truth.right),
                   "projection %d mismatch. got right=%lf, should be right=%lf\n", i, projection.right, projection_truth.right);
        }
        // printf("\n");
    }

    // recycle resources
    free(projection_map);
    cudaFree(projection_map_gpu);
    cudaFree(polygon_n_map_gpu);
    cudaFree(i_polygon_map_gpu);
    cudaFree(projection_endpoints_gpu);

    printf(PASSED);
}

static void test_calculate_is_overlapping(){
    printf("calculate_is_overlapping...");

    const int n_vertex = 7;
    const int n_polygon = 2;
    projection_t projection_map[7*2] = {
        (projection_t){.left=-1.0000000000000000, .right=0.0000000000000000}, (projection_t){.left=-1.0000000000000000, .right=0.0000000000000000},
        (projection_t){.left=0.0000000000000000, .right=0.8944271909999159}, (projection_t){.left=0.0000000000000000, .right=1.3416407864998738},
        (projection_t){.left=-0.8944271909999159, .right=0.0000000000000000}, (projection_t){.left=-0.8944271909999159, .right=0.4472135954999579},
        (projection_t){.left=-1.0000000000000000, .right=0.0000000000000000}, (projection_t){.left=-1.0000000000000000, .right=0.0000000000000000},
        (projection_t){.left=0.0000000000000000, .right=1.0000000000000000}, (projection_t){.left=0.0000000000000000, .right=1.0000000000000000},
        (projection_t){.left=0.0000000000000000, .right=1.0000000000000000}, (projection_t){.left=0.0000000000000000, .right=1.0000000000000000},
        (projection_t){.left=-1.0000000000000000, .right=0.0000000000000000}, (projection_t){.left=-1.0000000000000000, .right=0.0000000000000000},
    };

    projection_t* projection_map_gpu = (projection_t*)cudaMallocBy(n_vertex*n_polygon * sizeof(projection_t), projection_map);

    int* result_gpu = NULL;
    calculate_is_overlapping(projection_map_gpu, n_vertex, n_polygon, /*out*/&result_gpu);
    int* result = (int*)hostMallocBy(n_polygon*n_polygon * sizeof(int), result_gpu);

    // assert result
    for(int i_polygon_a=0; i_polygon_a<n_polygon; i_polygon_a++){
        for(int i_polygon_b=0; i_polygon_b<n_polygon; i_polygon_b++){
            int i = i_polygon_a*n_polygon + i_polygon_b;
            int is_overlapping = result[i];

            if(i_polygon_a == i_polygon_b){
		ASSERT(is_overlapping != 0,
                       "polygon(%d, %d) mismatch. got not overlapping, should be overlapping\n", i_polygon_a, i_polygon_b);
                continue;
	    }
            // is_overlapping != 0 => overlapping
            // printf("a is overlapping b: %d\n", is_overlapping != 0);
            ASSERT(is_overlapping != 0,
                   "polygon(%d, %d) mismatch. got not overlapping, should be overlapping\n", i_polygon_a, i_polygon_b);
        }
        // printf("\n");
    }

    // recycle resources
    free(result);
    cudaFree(result_gpu);
    cudaFree(projection_map_gpu);

    printf(PASSED);
}

static void test_calculate_is_overlapping_2(){
    printf("calculate_is_overlapping_2...");

    const int n_vertex = 8;
    const int n_polygon = 2;
    projection_t projection_map[8*2] = {
        (projection_t){.left=-1.0000000000000000, .right=0.0000000000000000}, (projection_t){.left=-1.0000000000000000, .right=0.0000000000000000},
        (projection_t){.left=0.0000000000000000, .right=1.0000000000000000}, (projection_t){.left=2.0000000000000000, .right=3.0000000000000000},
        (projection_t){.left=0.0000000000000000, .right=1.0000000000000000}, (projection_t){.left=0.0000000000000000, .right=1.0000000000000000},
        (projection_t){.left=-1.0000000000000000, .right=0.0000000000000000}, (projection_t){.left=-3.0000000000000000, .right=-2.0000000000000000},

        (projection_t){.left=-1.0000000000000000, .right=0.0000000000000000}, (projection_t){.left=-1.0000000000000000, .right=0.0000000000000000},
        (projection_t){.left=0.0000000000000000, .right=1.0000000000000000}, (projection_t){.left=2.0000000000000000, .right=3.0000000000000000},
        (projection_t){.left=0.0000000000000000, .right=1.0000000000000000}, (projection_t){.left=0.0000000000000000, .right=1.0000000000000000},
        (projection_t){.left=-1.0000000000000000, .right=0.0000000000000000}, (projection_t){.left=-3.0000000000000000, .right=-2.0000000000000000},
    };

    projection_t* projection_map_gpu = (projection_t*)cudaMallocBy(n_vertex*n_polygon * sizeof(projection_t), projection_map);

    int* result_gpu = NULL;
    calculate_is_overlapping(projection_map_gpu, n_vertex, n_polygon, /*out*/&result_gpu);
    int* result = (int*)hostMallocBy(n_polygon*n_polygon * sizeof(int), result_gpu);

    // assert result
    for(int i_polygon_a=0; i_polygon_a<n_polygon; i_polygon_a++){
        for(int i_polygon_b=0; i_polygon_b<n_polygon; i_polygon_b++){
            int i = i_polygon_a*n_polygon + i_polygon_b;
            int is_overlapping = result[i];

            if(i_polygon_a == i_polygon_b){
		ASSERT(is_overlapping != 0,
                       "polygon(%d, %d) mismatch. got not overlapping, should be overlapping\n", i_polygon_a, i_polygon_b);
                continue;
	    }
            // is_overlapping != 0 => overlapping
            // printf("a is overlapping b: %d\n", is_overlapping != 0);
            ASSERT(is_overlapping == 0,
                   "polygon(%d, %d) mismatch. got overlapping, should be not overlapping\n", i_polygon_a, i_polygon_b);
        }
        // printf("\n");
    }

    // recycle resources
    free(result);
    cudaFree(result_gpu);
    cudaFree(projection_map_gpu);

    printf(PASSED);
}

int main(int argc, char* argv[]){
    // cpu
    test_empty_polygon();
    test_n_polygon();
    test_n_vertices();
    test_load_points();
    test_polygon_is_overlap();

    // gpu
    initDevice(0);

    test_calculate_axes();
    test_calculate_projection_endpoints();
    test_calculate_projection_segments();
    test_calculate_is_overlapping();
    test_calculate_is_overlapping_2();

    cudaDeviceReset();

    return 0;
}
