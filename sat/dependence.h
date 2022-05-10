#ifndef DEPENDENCE_H
#define DEPENDENCE_H
#define CHECK(call)\
do{\
  const cudaError_t error=call;\
  if(error!=cudaSuccess)\
  {\
      printf("ERROR: %s:%d,",__FILE__,__LINE__);\
      printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));\
      exit(1);\
  }\
}while(0)

#ifdef _WIN32
extern int gettimeofday(struct timeval *tp, void *tzp);
#endif

extern double cpuSecond();
extern void initialData(float* ip,int size);
extern void initialData_int(int* ip, int size);
extern void printMatrix(float * C,const int nx,const int ny);
extern void initDevice(int devNum);
extern void checkResult(float * hostRef,float * gpuRef,const int N);

#endif//FRESHMAN_H
