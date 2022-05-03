#include <stdlib.h>
#include <math.h>
#include "loader.h"
#include "sat.h"

#define     TEMP_TEST_FILE      "./temp_test.txt"

// utils ----------------------------------------------------------------------

#define     LEN(x)  (sizeof(x)/sizeof(x[0]))

static BOOL isclose(double a, double b){
    return fabs(a-b) < 1e-16;
}

static void generate_temp_test_file(char* test_input){
    FILE* fp = fopen(TEMP_TEST_FILE, "w");
    fputs(test_input, fp);
    fclose(fp);
}

static void remove_temp_test_file(){
    remove(TEMP_TEST_FILE);
}

static int test_load_polygons(polygon_t*** p_polygon_list){
    ASSERT(*p_polygon_list == NULL, "polygon_list is not NULL, initialize it as NULL\n");

    FILE* fp = fopen(TEMP_TEST_FILE, "r");
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

static void test_n_polygon(int n_truth){
    polygon_t** temp = NULL;
    int n = test_load_polygons(&temp);

    ASSERT(n == n_truth, "wrong number of polygon. n: %d, expected: %d\n", n, n_truth);

    test_free_polygons(temp, n);
}

static void test_n_vertices(int* n_truth_list, int _){
    polygon_t** polygon_list = NULL;
    int n = test_load_polygons(&polygon_list);

    for(int i=0; i<n; i++){
        ASSERT(polygon_list[i]->n == n_truth_list[i], \
        "wrong number of vertices of polygon %d. n: %d, expected: %d\n", i, polygon_list[i]->n, n_truth_list[i]);
    }

    test_free_polygons(polygon_list, n);
}

static void test_points(double* point_list, int _){
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
}

int main(int argc, char* argv[]){
    if(argc > 1){
        // assertion fail cases, in which temp_test.txt passed by test.sh
        polygon_t** temp = NULL;
        int n = test_load_polygons(&temp);
        test_free_polygons(temp, n);
        return 0;
    }

    // normal cases
    generate_temp_test_file("\
2\n\
6\n\
[(0.9343917934863109,0.13045144291207023),(0.3496595677721136,0.029371321654252047),(0.17376032399107255,0.19791525658619602),(0.06493821266996724,0.904690432299808),(0.5443052896643373,0.7616839755359994),(0.6764463372540498,0.5474568981060993)]\n\
3\n\
[(0.5443052896643373,0.7616839755359994),(0.9315831160478467,0.30600531904742123),(0.2843560333040439,0.5658618493916742)]");

    int n_truth = 2;
    test_n_polygon(n_truth);

    int n_vertices_list[] = {6, 3};
    test_n_vertices(n_vertices_list, LEN(n_vertices_list));

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
    test_points(point_list, LEN(point_list));

    remove_temp_test_file();


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

    // edge cases
    // the number of polygons is 0
    generate_temp_test_file("0");

    n_truth = 0;
    test_n_polygon(n_truth);

    remove_temp_test_file();

    return 0;
}
