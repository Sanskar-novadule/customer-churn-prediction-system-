import numpy as np

student = np.array([5, 80, 70])

print(student)
print(student.ndim)

#we can also use the shape attribute to get the dimensions of the array
print(student.shape)

#vectot addition

A = np.array([2, 3, 4])
B = np.array([1, 5, 2])

result = A + B
print(result)

#vectpr subtraction
result = A - B
print(result)

#vector multiplication
result = 3 * A

print(result)

#dot product
A = np.array([2, 3, 4])
B = np.array([1, 5, 2])

result = np.dot(A, B)

print(result)