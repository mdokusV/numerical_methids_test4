import numpy as np


def gaussian_elimination_no_selection(A, b):
    n = len(A)

    # Create L and U matrices
    L = np.zeros((n, n))
    for diagonal in range(n):
        L[diagonal][diagonal] = 1

    # Perform Gaussian elimination without main element selection
    for i in range(n - 1):
        L_temp = np.zeros((n, n))
        L_temp = np.zeros((n, n))
        for diagonal in range(n):
            L_temp[diagonal][diagonal] = 1

        for j in range(i + 1, n):
            if A[i][i] == 0:
                raise ValueError(
                    "Zero division encountered. The system may be singular or have no unique solution."
                )

            factor = A[j][i] / A[i][i]
            L_temp[j][i] = factor

            for k in range(i, n):
                A[j][k] -= factor * A[i][k]
            b[j] -= factor * b[i]

        L = np.dot(L, L_temp)

        print("A matrix after elimination:")
        print(A)
        print("b vector after elimination:")
        print(b)
        print("\n")

    return A, b, L


def gaussian_elimination_partial_selection(A, b):
    n = len(A)

    L = np.zeros((n, n))
    for diagonal in range(n):
        L[diagonal][diagonal] = 1

    # Perform Gaussian elimination with partial main element selection
    for i in range(n - 1):
        L_temp = np.zeros((n, n))
        for diagonal in range(n):
            L_temp[diagonal][diagonal] = 1

        max_row = i
        for j in range(i + 1, n):
            if abs(A[j][i]) > abs(A[max_row][i]):
                max_row = j
        A[[i, max_row]] = A[[max_row, i]]
        b[[i, max_row]] = b[[max_row, i]]

        L[[i, max_row]] = L[[max_row, i]]

        L[i, i], L[max_row, i] = L[max_row, i], L[i, i]
        L[max_row, max_row], L[i, max_row] = L[i, max_row], L[max_row, max_row]

        for j in range(i + 1, n):
            if A[i][i] == 0:
                raise ValueError(
                    "Zero division encountered. The system may be singular or have no unique solution."
                )

            factor = A[j][i] / A[i][i]
            L_temp[j][i] = factor

            for k in range(i, n):
                A[j][k] -= factor * A[i][k]
            b[j] -= factor * b[i]

        L = np.dot(L, L_temp)

        print("A matrix after elimination:")
        print(A)
        print("b vector after elimination:")
        print(b)
        print("\n")

    return A, b, L


def back_substitution(U, b):
    n = len(U)
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        sum1 = sum(U[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (b[i] - sum1) / U[i][i]

    return x


# Example usage
A = np.array(
    [[2, -3, -4, 1], [-1, 2, 1, 1], [3, -2, -1, 2], [1, -1, 2, -2]], dtype=float
)

b = np.array([-5, -1, 5, -2], dtype=float)

print("A matrix:")
print(A)
print("b vector:")
print(b)
print("\n")

print("Gaussian elimination without main element selection:")
A1, b1, L1 = gaussian_elimination_no_selection(A.copy(), b.copy())

print("------------------------")

print("A matrix after elimination:")
print(A1)
print("L matrix:")
print(L1)
print("b vector after elimination:")
print(b1)
print("Solution:")
x = back_substitution(A1, b1)
print(x)
print("______________________________________________________")

print("\n\nGaussian elimination with partial main element selection:")
A2, b2, L2 = gaussian_elimination_partial_selection(A.copy(), b.copy())

print("------------------------")

print("A matrix after elimination:")
print(A2)
print("L matrix:")
print(L2)
print("b vector after elimination:")
print(b2)
print("Solution:")
x = back_substitution(A2, b2)
print(x)

print(np.dot(L2, A2))
