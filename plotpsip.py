import matplotlib.pyplot as plt
import numpy as np
from numba import njit

a = np.loadtxt("build/psip.csv", dtype=complex)


@njit
def nsqr(x):
    return x.real**2 + x.imag**2


a = nsqr(a)
print(np.shape(a))
fig, ax = plt.subplots()
im = ax.imshow(
    a,
    extent=(1.0, 1.8, -5.0, 5.0),
    aspect=0.8 / 10.0,
    interpolation="none",
)
ax.set_xlabel(r"$P/P_{th}$")
ax.set_ylabel(r"$B_{ext}$ [T]")
ax.set_title(r"$|\psi_+|$")
cb = plt.colorbar(im)
minv = np.min(a)
maxv = np.max(a)
# cb.set_ticks([minv, maxv], labels=[f"{minv:.1f} µeV", f"{maxv:.1f} µeV"])
fig.savefig("psip.pdf")
