#include <stdio.h>

int main() {
  int num = 10;
  float fnum = 3.14;

  // This is a pointer of any type
  void* vptr;
  vptr = &num;

  // Type cast
  printf("Value: %d\n", *(int*)vptr);

  vptr = &fnum;
  printf("Value: %.2f\n", *(float*)vptr);
}
