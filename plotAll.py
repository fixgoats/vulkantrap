import tomllib
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numba.core.config import os

parser = ArgumentParser()
parser.add_argument("-d", "--directory", help="directory where data files are stored.")
parser.add_argument(
    "-f", "--format", help="The format to ouput with specified by file ending."
)
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
    return v.real * v.real + v.imag * v.imag


Snorm = (
    np.apply_along_axis(S1, 2, z).real ** 2
    + np.apply_along_axis(S2, 2, z).real ** 2
    + np.apply_along_axis(S3, 2, z).real ** 2
)
# Snorm = np.clip(Snorm, a_min=0.01, a_max=None)

fig, ax = plt.subplots()
im = ax.imshow(
    np.apply_along_axis(S1, 2, z).real ** 2 / Snorm,
    extent=(conf["xstart"], conf["xend"], conf["eend"], conf["estart"]),
    aspect="auto",
    interpolation="none",
)
ax.set_title(r"$S_1^2/|\vec{S}|^2$")
ax.set_xlabel("p")
ax.set_ylabel(r"$\epsilon$")
cb = plt.colorbar(im)
fig.savefig(os.path.join(graphdir, f"S1.{args.format}"))
fig.clear()
ax.clear()

fig, ax = plt.subplots()
im = ax.imshow(
    np.apply_along_axis(S2, 2, z).real ** 2 / Snorm,
    extent=(conf["xstart"], conf["xend"], conf["eend"], conf["estart"]),
    aspect="auto",
    interpolation="none",
)
ax.set_title(r"$S_2^2/|\vec{S}|^2$")
ax.set_xlabel("p")
ax.set_ylabel(r"$\epsilon$")
cb = plt.colorbar(im)
fig.savefig(os.path.join(graphdir, f"S2.{args.format}"))
fig.clear()
ax.clear()

fig, ax = plt.subplots()
im = ax.imshow(
    np.apply_along_axis(S3, 2, z).real ** 2 / Snorm,
    extent=(conf["xstart"], conf["xend"], conf["eend"], conf["estart"]),
    aspect="auto",
    interpolation="none",
)
ax.set_title(r"$S_3^2/|\vec{S}|^2$")
ax.set_xlabel("p")
ax.set_ylabel(r"$\epsilon$")
cb = plt.colorbar(im)
fig.savefig(os.path.join(graphdir, f"S3.{args.format}"))
fig.clear()
ax.clear()

fig, ax = plt.subplots()
im = ax.imshow(
    nsqr(a),
    extent=(conf["xstart"], conf["xend"], conf["eend"], conf["estart"]),
    aspect="auto",
    interpolation="none",
)
ax.set_title(r"$|\psi_+|$")
ax.set_xlabel("p")
ax.set_ylabel(r"$\epsilon$")
cb = plt.colorbar(im)
fig.savefig(os.path.join(graphdir, f"psip.{args.format}"))
fig.clear()
ax.clear()

fig, ax = plt.subplots()
im = ax.imshow(
    nsqr(b),
    extent=(conf["xstart"], conf["xend"], conf["eend"], conf["estart"]),
    aspect="auto",
    interpolation="none",
)
ax.set_title(r"$|\psi_-|$")
ax.set_xlabel("p")
ax.set_ylabel(r"$\epsilon$")
cb = plt.colorbar(im)
fig.savefig(os.path.join(graphdir, f"psim.{args.format}"))
fig.clear()
ax.clear()


fig, ax = plt.subplots()
im = ax.imshow(
    nsqr(a) + nsqr(b),
    extent=(conf["xstart"], conf["xend"], conf["eend"], conf["estart"]),
    aspect="auto",
    interpolation="none",
)
ax.set_title(r"$|\psi|$")
ax.set_xlabel("p")
ax.set_ylabel(r"$\epsilon$")
cb = plt.colorbar(im)
fig.savefig(os.path.join(graphdir, f"psi.{args.format}"))
fig.clear()
ax.clear()
