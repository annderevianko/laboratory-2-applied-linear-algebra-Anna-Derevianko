import numpy as np

def eigenval_eigenvec(A):
    A = np.array(A)
    eigenvalues, eigenvectors = np.linalg.eig(A)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)

    print(f'eigenvalues: \n {eigenvalues}\n')
    print(f'eigenvectors: \n{eigenvectors}\n')
    return eigenvalues, eigenvectors

def verify_equality(A, eigenvalues, eigenvectors):
    A = np.array(A)
    for i in range(len(eigenvalues)):
        lambdaa = eigenvalues[i]
        v = eigenvectors[:, i]

        left = A @ v
        right = lambdaa * v

        equality = np.allclose(left, right)
        if (equality == True):
            print(f'A⋅v=λ⋅v is equal: \n{A} ⋅ {v} = {lambdaa} ⋅ {v}\n')
        else:
            print(f'A⋅v=λ⋅v isnt equal: \n{A} ⋅ {v} ≠ {lambdaa} ⋅ {v}\n')


# test 1
print('Test 1')
A = [[1,5],[9,4]]
eigenvalues, eigenvectors = eigenval_eigenvec(A)
verify_equality(A, eigenvalues, eigenvectors)


# test 2
print(f'\nTest 2')
A = [[12,81,55],[5,34,101],[8,36,12]]
eigenvalues, eigenvectors = eigenval_eigenvec(A)
verify_equality(A, eigenvalues, eigenvectors)

# test 3
print(f'\nTest 3')
A = [[12,81,55],[5,34,101]]
eigenvalues, eigenvectors = eigenval_eigenvec(A)
verify_equality(A, eigenvalues, eigenvectors)

