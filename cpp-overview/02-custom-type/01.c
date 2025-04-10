#include <stdio.h>
#include <stdlib.h>

typedef struct {
  int* arr;
  int m;
  int n;
} Matrix;

Matrix *create_matrix(int **original, int m, int n);
void print_matrix(Matrix*);

int main() {
  int row_1[] = {1, 2 ,3};
  int row_2[] = {4, 5, 6};
  int* columns[] = {row_1, row_2};
  Matrix* matrix = create_matrix(columns, 2, 3);

  print_matrix(matrix);

  free(matrix->arr);
  free(matrix);

  return 0;
}

Matrix* create_matrix(int** original, int m, int n) {
  int* arr = malloc(sizeof(int) * m * n);

  if (arr == NULL) {
    return NULL;
  }

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      arr[i * n + j] = original[i][j];
    }
  }

  Matrix* result = malloc(sizeof(Matrix));
  if (result == NULL) {
    free(arr);
    return NULL;
  }

  result->arr = arr;
  result->m = m;
  result->n = n;

  return result;
}

void print_matrix(Matrix* matrix) {
  printf("[\n");
  for (int i = 0; i < matrix->m; i++) {
    printf(" [ ");
    for (int j = 0; j < matrix->n; j++) {
      printf("%d, ", matrix->arr[i * matrix->n + j]);
    }
    printf("]\n");
  }
  printf("]\n");
}

