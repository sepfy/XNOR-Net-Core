#include <math.h>
void bias_add(int N, int M, float *A, float *bias);
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
