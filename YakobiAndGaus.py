import numpy as np



def is_dominant(matrix):
    for i in range(len(matrix)):
        row_sum = sum(abs(matrix[i][j]) for j in range(len(matrix)) if i != j)
        if abs(matrix[i][i]) < row_sum:
            return False
    return True


def make_dominant(matrix, vector):
    n = len(matrix)
    for i in range(n):
        max_row = i + max(range(n - i), key=lambda k: abs(matrix[i + k][i]), default=0)
        if i != max_row:
            matrix[[i, max_row]] = matrix[[max_row, i]]
            vector[[i, max_row]] = vector[[max_row, i]]
    return is_dominant(matrix)


def jacobi_method(matrix, vector, tol=1e-5, max_iterations=1000):
    matrix = np.array(matrix, dtype=float)
    vector = np.array(vector, dtype=float)
    n = len(matrix)
    x = np.zeros(n)

    if not is_dominant(matrix):
        if make_dominant(matrix, vector):
            print("Matrix was rearranged to be diagonally dominant.")
        else:
            print("Matrix is not diagonally dominant. Results may not converge.")

    for iteration in range(1, max_iterations + 1):
        x_new = np.zeros_like(x)
        for i in range(n):
            s = sum(matrix[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (vector[i] - s) / matrix[i][i]

        print(f"Iteration {iteration}: {x_new}")
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            print(f"Jacobi converged in {iteration} iterations.")
            return x_new
        x = x_new

    print("Jacobi did not converge.")
    return x


def gauss_seidel_method(matrix, vector, tol=1e-5, max_iterations=1000):
    matrix = np.array(matrix, dtype=float)
    vector = np.array(vector, dtype=float)
    n = len(matrix)
    x = np.zeros(n)

    if not is_dominant(matrix):
        if make_dominant(matrix, vector):
            print("Matrix was rearranged to be diagonally dominant.")
        else:
            print("Matrix is not diagonally dominant. Results may not converge.")

    for iteration in range(1, max_iterations + 1):
        x_new = np.copy(x)
        for i in range(n):
            s1 = sum(matrix[i][j] * x_new[j] for j in range(i))
            s2 = sum(matrix[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (vector[i] - s1 - s2) / matrix[i][i]

        print(f"Iteration {iteration}: {x_new}")
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            print(f"Gauss-Seidel converged in {iteration} iterations.")
            return x_new
        x = x_new

    print("Gauss-Seidel did not converge.")
    return x


def main():
    matrixA = np.array([[4, 2, 0], [2, 10, 4], [0, 4, 5]])
    vectorB = np.array([2, 6, 5])

    print("Choose method:\n1. Jacobi\n2. Gauss-Seidel")
    choice = input("Enter choice (1 or 2): ")

    if choice == '1':
        print("\nRunning Jacobi Method...")
        solution = jacobi_method(matrixA, vectorB)
        print(f"Solution: {solution}")
    elif choice == '2':
        print("\nRunning Gauss-Seidel Method...")
        solution = gauss_seidel_method(matrixA, vectorB)
        print(f"Solution: {solution}")
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
