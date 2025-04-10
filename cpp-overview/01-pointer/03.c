#include <stdio.h>
#include <stdlib.h>

int main() {
  int* ptr = NULL;

  printf("Pointer value: %p\n", (void*)ptr);

  ptr = malloc(sizeof(int));

  if (ptr == NULL) {
    printf("pointer is null!\n");
    return 0;
  }

  printf("Pointer value after allocation: %p\n", (void*)ptr);
  return 0;
}
