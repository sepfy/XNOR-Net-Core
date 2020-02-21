#include <math.h>
void add(int N, int M, float *A, float *B, float *C);
void row_sum(int N, int M, float *A, float *B);
void col_sum(int N, int M, float *A, float *B);
bool compare(int N, int M, float *A, float *B);
void mat_minus(int N, int M, float *mat1, float *mat2, float* mat_out);
void mat_scalar(int N, int M, float *mat1, float scalar, float* mat_out);
float cross_entropy(int batch, int N, float *output, float *target);
float L1_norm(int N, int M, float *A);
float L2_norm(int N, int M, float *A);
float Linf_norm(int N, int M, float *A);
void transpose(int N, int M, float *A, float *B);
void scalar(int N, float s, float *A, float *B);
int* argmax(int batch, int N, float *A);
float accuracy(int batch, int N, float *A, float *B);

#ifdef GPU
void row_sum_gpu(int N, int M, float *A, float *B);
void col_sum_gpu(int N, int M, float *A, float *B);
void bias_add_gpu(float *output, float *bias, int batch, int size, int c);
#endif
