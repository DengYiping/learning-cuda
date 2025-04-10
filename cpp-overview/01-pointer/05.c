# include <stdio.h>

void print_matrix(int **matrix, int m, int n);

int main() {
  int arr1[] = {1, 2, 3, 4};
  int arr2[] = {5, 6, 7, 8};
  int* matrix[] = {arr1, arr2};
  print_matrix(matrix, 2, 4);
}

void print_matrix(int** matrix, int m, int n) {
  printf("[\n");
  for (int i = 0; i < m; i++) {
    printf("[");
    for (int j = 0; j < n; j++) {
      int element = matrix[i][j];
      printf("%d ", element);
    }
    printf("]");
    printf("\n");
  }
  printf("]\n");
}
