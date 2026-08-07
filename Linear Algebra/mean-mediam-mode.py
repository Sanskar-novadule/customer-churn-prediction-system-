#mean, median and mode of marks
import numpy as np

marks = np.array([10, 20, 20, 30, 40])

mean = np.mean(marks)
median = np.median(marks)
mode = np.argmax(np.bincount(marks))

print(mean)
print(median)
print(mode)