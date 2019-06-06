
void im2col(int W, int H, int C, int FW, int FH, int FC,
            int stride, int pad, float *im, float *col);

void col2im(int W, int H, int C, int FW, int FH, int FC,
            int stride, int pad, float *im, float *col);

typedef unsigned long long ms_t;
unsigned long long getms();
