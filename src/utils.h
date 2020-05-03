
void im2col(int W, int H, int C, int FW, int FH, int FC,
            int stride, int pad, float *im, float *col);

void im2col_xnor(int W, int H, int C, int FW, int FH, int FC,
            int stride, int pad, float *im, int8_t *col);

void col2im(int W, int H, int C, int FW, int FH, int FC,
            int stride, int pad, float *im, float *col);

void im2col_gpu(int W, int H, int C, int FW, int FH, int FC,
            int stride, int pad, float *im, float *col);

void col2im_gpu(int W, int H, int C, int FW, int FH, int FC,
            int stride, int pad, float *im, float *col);

void random_normal(int size, float *mat);
typedef unsigned long long ms_t;
unsigned long long getms();
