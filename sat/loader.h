#ifndef __LOADER_H__
#define __LOADER_H__
#include <stdio.h>
#include "sat.h"

extern int load_polygons(FILE* fp, /*out*/ polygon_t*** polygon_list);

#endif