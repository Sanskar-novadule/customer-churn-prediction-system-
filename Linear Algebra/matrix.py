import numpy as np

A = np.array([
    [2, 4],
    [6, 8]
])

B = np.array([
    [1, 3],
    [5, 7]
])

result = A + B

print(result)

#subtraction

result = A - B

print(result)



A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print(A)
print(A.T)

print(A.shape)
print(A.T.shape)