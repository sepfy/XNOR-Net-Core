#include <stdio.h>
#include <stdlib.h>

void read_param(const char *filename, int size, float *buf) {
  FILE *fp;
  fp = fopen(filename, "r");
  fread(buf, 4, size, fp);
  fclose(fp);
}
