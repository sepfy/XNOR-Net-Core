#include <iostream>
#include <random>
#include <sys/time.h>
#include <math.h>
#include <bitset>
using namespace std;



unsigned long long getms() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec*1.0e+6 + tv.tv_usec;
}

uint32_t countSetBits(uint32_t n) { 
  uint32_t count = 0; 
  while (n) { 
    count += n & 1; 
    n >>= 1; 
  } 
  return count; 
} 

class Bitset {

  public:
    uint32_t *bits;
    int size;
    int bit_num;
    uint32_t complement;
    Bitset(int _size) {
      bit_num = _size;
      size = ceil((float)_size/(8*sizeof(uint32_t)));
      bits = new uint32_t[size];
      int offset = _size%(8*sizeof(uint32_t));
      complement = (UINT32_MAX << (32 - offset)) >> (32-offset);
    }

    void set(int idx, uint32_t value) {
      int bidx = idx/(8*sizeof(uint32_t));
      int offset = idx%(8*sizeof(uint32_t));
      //cout << (value << offset)<<endl;
      bits[bidx] |= (value << offset);
    }

    void sign_to_bin(int size, float *input) {
      for(int i = 0; i < size; i++) {
        input[i] >= 0 ? set(size-1-i, 1) : set(size-1-i, 0);
      }

    }

    uint32_t count() {

      int count = 0;
      for(int i = 0; i < size; i++) {
        count += countSetBits(bits[i]);
      }
      return count;
    }

    int count2value() {
      int c = count();
      //int value = (bit_num - count())*(-1) + count();
      int value = 2*c - bit_num;
      return value;
    }

};


void xnor(Bitset *b1, Bitset *b2, Bitset *b3) {

  if(b1->bit_num != b2->bit_num || b1->bit_num != b3->bit_num) {
    printf("Bit num not equal");
  }
  int i = 0;
  for(i = 0; i < b3->size - 1; i++) {
    b3->bits[i] = ~(b1->bits[i]^b2->bits[i]);
  }

  b3->bits[b3->size-1] = ~(b1->bits[b1->size-1]^b2->bits[b2->size-1]) & b3->complement;
  

}


int main() {

  int i;
  int size = 6;
  random_device rd;
  default_random_engine generator = default_random_engine(rd());
  normal_distribution<float> distribution(0, 0.5);
  float *mat1 = new float[size];
  float *mat2 = new float[size];
  for(i = 0; i < size; i++) {
    mat1[i] = distribution(generator);
    mat2[i] = distribution(generator);
  }
  int count = 0;

  bitset<32> bbits("10101001100101001111011010100001");
  //cout << bbits.to_ullong() << endl;

  unsigned long long start = getms();




  for(int i = 0; i < size; i++) {
    //mat[i] >= 0 ? mat[i] = 0 : mat[i] = 1;
    mat1[i] >= 0 ? cout << 1 : cout << 0;
  }
  cout << endl;

  for(int i = 0; i < size; i++) {
    //mat[i] >= 0 ? mat[i] = 0 : mat[i] = 1;
    mat2[i] >= 0 ? cout << 1 : cout << 0;
  }
  cout << endl;
  Bitset bitset1(size);
  Bitset bitset2(size);
  Bitset bitset3(size);
  bitset1.sign_to_bin(size, mat1);
  bitset2.sign_to_bin(size, mat2);


  xnor(&bitset1, &bitset2, &bitset3);
  cout << bitset1.bits[0] << endl;
  cout << bitset2.bits[0] << endl;
  cout << bitset3.bits[0] << endl;
  cout << "count" << endl;
  cout << bitset1.count2value() << endl;
  cout << bitset2.count2value() << endl;
  cout << bitset3.count2value() << endl;
  return 0;
}
