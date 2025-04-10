#include <stdio.h>
#include <stdlib.h>

int main() {
  int arr[] = {1, 2, 3, 4, 5};
  int* ptr = arr;

  printf("Position 0: %p\n", (void*) ptr);

  for (int i = 0; i < 5; i++) {
    // Pointer can be dereferenced back in a array style
    printf("Dereferenced: %d\n", *ptr);
    printf("Just using array indexing: %d\n", arr[i]);
    printf("Addresss: %p\n", (void*) ptr);
    ptr++;
  }
}
