import numpy as np


def jacobi_method(A, b, x0, epsilon):
    n = len(A)
    D = np.diag(np.diag(A))
    L = np.tril(A, k=-1)
    U = np.triu(A, k=1)

    x = x0
    iteration = 0

    while True:
        x_new = -np.linalg.inv(D) @ (L + U) @ x + np.linalg.inv(D) @ b
        residual = b - A @ x_new
        error = np.linalg.norm(residual)

        if error < epsilon:
            break

        x = x_new
        iteration += 1

    return x_new, iteration


def gauss_seidel_method(A, b, x0, epsilon):
    n = len(A)
    L_D = np.tril(A)
    U = np.triu(A, k=1)

    x = x0
    iteration = 0

    while True:
        x_new = np.linalg.inv(L_D) @ (-U) @ x + np.linalg.inv(L_D) @ b
        residual = b - A @ x_new
        error = np.linalg.norm(residual)

        if error < epsilon:
            break

        x = x_new
        iteration += 1

    return x_new, iteration


# Example usage
A = np.array([[4, -1, 1], [2, 5, 2], [1, 2, 4]], dtype=float)
b = np.array([8, 3, 11], dtype=float)
x0 = np.array([0, 0, 0], dtype=float)
epsilon = 1e-10

# Jacobi method
solution_jacobi, iterations_jacobi = jacobi_method(A, b, x0, epsilon)
print("Jacobi method:")
print("Solution:", solution_jacobi)
print("Iterations:", iterations_jacobi)

# Gauss-Seidel method
solution_gauss_seidel, iterations_gauss_seidel = gauss_seidel_method(A, b, x0, epsilon)
print("\nGauss-Seidel method:")
print("Solution:", solution_gauss_seidel)
print("Iterations:", iterations_gauss_seidel)
