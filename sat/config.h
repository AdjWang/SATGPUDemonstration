#ifndef __CONFIG_H__
#define __CONFIG_H__

// user configurations
// #define DEBUG

// utils(DO NOT EDIT)
#define BOOL int
#define TRUE    1
#define FALSE   0

#define RAISE(format, ...)  do{\
                            fprintf(stderr, "%s:%d, raise exception:\n", __FILE__, __LINE__);\
                            fprintf(stderr, format, ##__VA_ARGS__);\
                            exit(1);\
                        }while(0)

#define ASSERT(cond, format, ...)   do{\
                                        if(!(cond)){\
                                            fprintf(stderr, "%s:%d, assertation failed.\n", __FILE__, __LINE__);\
                                            fprintf(stderr, format, ##__VA_ARGS__);\
                                            exit(1);\
                                        }\
                                    }while(0)

#ifdef DEBUG
#define DBG_PRINT(format, ...)    printf(format, ##__VA_ARGS__)
#else
#define DBG_PRINT(format, ...)
#endif

#endif