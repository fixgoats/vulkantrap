import matplotlib.pyplot as plt
import numpy as np


def nsqr(a, b):
    return a + b.real * b.real + b.imag * b.imag


a = np.frompyfunc(nsqr, 2, 1)

x = np.array(
    [
        [[1, 0.5 + 0.5j], [0.1j, 3]],
        [
            [0.1j, 3],
            [1, 0.5 + 0.5j],
        ],
    ]
)

b = a.reduce(x, axis=2, initial=0)
print(b)
