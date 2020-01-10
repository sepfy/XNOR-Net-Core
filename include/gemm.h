
void gemm(int M, int N, int P,
  float alpha, float *A, float *B, float *C);

void gemm_ta(int M, int N, int P,
  float alpha, float *A, float *B, float *C); 

void gemm_tb(int M, int N, int P,
  float alpha, float *A, float *B, float *C);

void gemm_gpu(int M, int N, int P,
  float alpha, float *A, float *B, float *C);


