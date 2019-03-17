#include <iostream>
#include <stdlib.h>
#include <string.h>
#include "blas.h"

using namespace std;

int* argmax(int batch, int N, float *A);
float accuracy(int batch, int N, float *A, float *B);

