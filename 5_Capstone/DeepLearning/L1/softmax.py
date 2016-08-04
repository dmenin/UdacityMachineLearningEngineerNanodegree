"""Softmax."""
import numpy as np

scores = np.array([3.0, 1.0, 0.2])
x = [3.0, 1.0, 0.2]
x = np.array([[1, 2, 3, 6],
              [2, 4, 5, 6],
              [3, 8, 7, 6]])




def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


print(softmax(scores ))

# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()



x.shape
scores.shape


###################
foo = 10e9
bar = 10e-6
for i in range(0,1000000):
    foo +=  bar
foo = foo - 10e9
foo 


a = 1000000000
for i in xrange(1000000):
    a +=  1e-6
print a - 1000000000
