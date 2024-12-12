import tomllib
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from numba.core.config import os

parser = ArgumentParser()
parser.add_argument("-d", "--directory", help="directory where data files are stored")
args = parser.parse_args()

parts = args.directory.split("/")
graphdir = os.path.join("graphs", parts[-3], parts[-2])
Path(graphdir).mkdir(parents=True, exist_ok=True)


a = np.loadtxt(os.path.join(args.directory, "psip.csv"), dtype=complex)
b = np.loadtxt(os.path.join(args.directory, "psim.csv"), dtype=complex)
with open(os.path.join(args.directory, "simconf.toml"), "rb") as f:
    conf = tomllib.load(f)["constants"]

z = np.dstack((a, b))


def S1(v):
    return np.dot(np.conj(v), np.array([v[1], v[0]]))


def S2(v):
    return np.dot(np.conj(v), np.array([-1j * v[1], 1j * v[0]]))


def S3(v):
    return np.dot(np.conj(v), np.array([v[0], -v[1]]))


def nsqr(v):
    return v.real * v.real + v.imag + v.imag


fig, ax = plt.subplots()
im = ax.imshow(
    np.apply_along_axis(S1, 2, z).real,
    interpolation="none",
)
ax.set_title(r"S1")
cb = plt.colorbar(im)
fig.savefig(os.path.join(graphdir, "S1.pdf"))
fig.clear()
ax.clear()

fig, ax = plt.subplots()
im = ax.imshow(
    np.apply_along_axis(S2, 2, z).real,
    interpolation="none",
)
ax.set_title(r"S2")
cb = plt.colorbar(im)
fig.savefig(os.path.join(graphdir, "S2.pdf"))
fig.clear()
ax.clear()

fig, ax = plt.subplots()
im = ax.imshow(
    np.apply_along_axis(S3, 2, z).real,
    interpolation="none",
)
ax.set_title(r"S3")
cb = plt.colorbar(im)
fig.savefig(os.path.join(graphdir, "S3.pdf"))
fig.clear()
ax.clear()

fig, ax = plt.subplots()
im = ax.imshow(
    nsqr(a),
    interpolation="none",
)
ax.set_title(r"$|\psi_+|$")
cb = plt.colorbar(im)
minv = np.min(a)
maxv = np.max(a)
fig.savefig(os.path.join(graphdir, "psip.png"))

a = nsqr(a)
fig, ax = plt.subplots()
im = ax.imshow(
    nsqr(a),
    interpolation="none",
)
ax.set_title(r"$|\psi_-|$")
cb = plt.colorbar(im)
minv = np.min(a)
maxv = np.max(a)
fig.savefig(os.path.join(graphdir, "psim.png"))