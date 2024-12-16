import numpy as np
from scipy.integrate import solve_ivp

a = 60


def nsqr(x):
    return x.real * x.real + x.imag * x.imag


def f(t, y):
    return np.array(
        [
            (1 - (1 + a * 1j) * (nsqr(y[0]) + 2 * nsqr(y[1]))) * y[0],
            (1 - (1 + a * 1j) * (nsqr(y[1]) + 2 * nsqr(y[0]))) * y[1],
        ]
    )


y0 = np.array([0.1 + 0.03j, -0.02 - 0.1j])
b = solve_ivp(f, (0, 600), y0)

print(b)


def S1(v):
    return np.dot(v.conj(), np.array([v[1], v[0]]))


def S2(v):
    return np.dot(v.conj(), np.array([-1j * v[1], 1j * v[0]]))


def S3(v):
    return np.dot(v.conj(), np.array([v[0], -v[1]]))


print(S3(b.y[:, -1]))
print(S2(b.y[:, -1]))
print(S1(b.y[:, -1]))
