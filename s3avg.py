import tomllib
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np

parser = ArgumentParser()
parser.add_argument(
    "-f", "--format", help="The format to ouput with specified by file ending."
)
args = parser.parse_args()


a = np.loadtxt("build/data/24-12-16/18-57/psip.csv", dtype=complex)
b = np.loadtxt("build/data/24-12-16/18-57/psim.csv", dtype=complex)
with open("build/data/24-12-16/18-57/simconf.toml", "rb") as f:
    conf = tomllib.load(f)["constants"]

z1 = np.dstack((a, b))

a = np.loadtxt("build/data/24-12-16/19-1/psip.csv", dtype=complex)
b = np.loadtxt("build/data/24-12-16/19-1/psim.csv", dtype=complex)

z2 = np.dstack((a, b))


def S1(v):
    return np.dot(np.conj(v), np.array([v[1], v[0]]))


def S2(v):
    return np.dot(np.conj(v), np.array([-1j * v[1], 1j * v[0]]))


def S3(v):
    return np.dot(np.conj(v), np.array([v[0], -v[1]]))


Snorm1 = (
    np.apply_along_axis(S1, 2, z1).real ** 2
    + np.apply_along_axis(S2, 2, z1).real ** 2
    + np.apply_along_axis(S3, 2, z1).real ** 2
)
Snorm2 = (
    np.apply_along_axis(S1, 2, z2).real ** 2
    + np.apply_along_axis(S2, 2, z2).real ** 2
    + np.apply_along_axis(S3, 2, z2).real ** 2
)

s3avg = (
    np.apply_along_axis(S3, 2, z1).real ** 2 / Snorm1
    + np.apply_along_axis(S3, 2, z2).real ** 2 / Snorm2
) / 2

fig, ax = plt.subplots()
im = ax.imshow(
    s3avg,
    extent=(conf["xstart"], conf["xend"], conf["eend"], conf["estart"]),
    aspect="auto",
    interpolation="none",
)
ax.set_title(r"$S_3^2/|\vec{S}|^2$")
ax.set_xlabel("p")
ax.set_ylabel(r"$\epsilon$")
cb = plt.colorbar(im)
fig.savefig(f"S3.{args.format}")
