import os

import tomllib
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numba import njit

parser = ArgumentParser()
parser.add_argument("-d", "--directory", help="directory where data files are stored")
parser.add_argument(
    "-f", "--format", help="format to save the figures as, file ending", default="png"
)
args = parser.parse_args()

parts = args.directory.split("/")
graphdir = os.path.join("graphs", parts[-3], parts[-2])
Path(graphdir).mkdir(parents=True, exist_ok=True)


a = np.loadtxt(os.path.join(args.directory, "fullrandom.csv"), dtype=complex)
a = np.reshape(a, (1000, 40, 2))[:, ::5, :]
b = np.loadtxt(os.path.join(args.directory, "randomic.csv"), dtype=complex)
b = np.reshape(b, (1000, 40, 2))[:, ::5, :]
c = np.loadtxt(os.path.join(args.directory, "escan.csv"), dtype=complex)
c = np.reshape(c, (1000, 40, 2))[:, ::5, :]
with open(os.path.join(args.directory, "simconf.toml"), "rb") as f:
    conf = tomllib.load(f)["constants"]


@njit
def S1(x):
    return 2 * (x[0].real * x[1].real + x[0].imag * x[1].imag)


@njit
def S2(x):
    return 2 * (x[0].real * x[1].imag - x[0].imag * x[1].real)


@njit
def nsqr(x):
    return x.real * x.real + x.imag * x.imag


@njit
def S3(x):
    return nsqr(x[0]) - nsqr(x[1])


@njit
def vnsqr(x):
    return nsqr(x[0]) + nsqr(x[1])


print(np.shape(a))
t = np.arange(1000)
anorm = np.apply_along_axis(vnsqr, 2, a)
bnorm = np.apply_along_axis(vnsqr, 2, b)
cnorm = np.apply_along_axis(vnsqr, 2, c)

as3 = np.apply_along_axis(S3, 2, a) / anorm
as2 = np.apply_along_axis(S2, 2, a) / anorm
as1 = np.apply_along_axis(S1, 2, a) / anorm

bs3 = np.apply_along_axis(S3, 2, b) / bnorm
bs2 = np.apply_along_axis(S2, 2, b) / bnorm
bs1 = np.apply_along_axis(S1, 2, b) / bnorm

cs3 = np.apply_along_axis(S3, 2, c) / cnorm
cs2 = np.apply_along_axis(S2, 2, c) / cnorm
cs1 = np.apply_along_axis(S1, 2, c) / cnorm

fig, ax = plt.subplots()
pl = ax.plot(t, as1)
ax.set_ylabel(r"S1")
fig.savefig(os.path.join(graphdir, f"fullrandomS1.{args.format}"))
fig.clear()
ax.clear()

fig, ax = plt.subplots()
pl = ax.plot(t, as2)
ax.set_ylabel(r"S2")
fig.savefig(os.path.join(graphdir, f"fullrandomS2.{args.format}"))
fig.clear()
ax.clear()

fig, ax = plt.subplots()
pl = ax.plot(t, as3)
ax.set_ylabel(r"S3")
fig.savefig(os.path.join(graphdir, f"fullrandomS3.{args.format}"))
fig.clear()
ax.clear()

fig, ax = plt.subplots()
pl = ax.plot(t, bs1)
ax.set_ylabel(r"S1")
fig.savefig(os.path.join(graphdir, f"randomicS1.{args.format}"))
fig.clear()
ax.clear()

fig, ax = plt.subplots()
pl = ax.plot(t, bs2)
ax.set_ylabel(r"S1")
fig.savefig(os.path.join(graphdir, f"randomicS2.{args.format}"))
fig.clear()
ax.clear()

fig, ax = plt.subplots()
pl = ax.plot(t, bs3)
ax.set_ylabel(r"S1")
fig.savefig(os.path.join(graphdir, f"randomicS3.{args.format}"))
fig.clear()
ax.clear()

fig, ax = plt.subplots()
pl = ax.plot(t, cs1)
ax.set_ylabel(r"S1")
fig.savefig(os.path.join(graphdir, f"escanS1.{args.format}"))
fig.clear()
ax.clear()

fig, ax = plt.subplots()
pl = ax.plot(t, cs2)
ax.set_ylabel(r"S2")
fig.savefig(os.path.join(graphdir, f"escanS2.{args.format}"))
fig.clear()
ax.clear()

fig, ax = plt.subplots()
pl = ax.plot(t, cs3)
ax.set_ylabel(r"S3")
fig.savefig(os.path.join(graphdir, f"escanS3.{args.format}"))
fig.clear()
ax.clear()

fig, ax = plt.subplots()
pl = ax.plot(np.atan2(as2, as1), np.atan2(np.sqrt(as2**2 + as1**2), as3))
ax.set_title(r"angles")
ax.set_xlabel(r"$\phi$")
ax.set_ylabel(r"$\theta$")
fig.savefig(os.path.join(graphdir, f"fullrandomphase.{args.format}"))
fig.clear()
ax.clear()

fig, ax = plt.subplots()
pl = ax.plot(np.atan2(bs2, bs1), np.atan2(np.sqrt(bs2**2 + bs1**2), bs3))
ax.set_title(r"angles")
ax.set_xlabel(r"$\phi$")
ax.set_ylabel(r"$\theta$")
fig.savefig(os.path.join(graphdir, f"randomicphase.{args.format}"))
fig.clear()
ax.clear()

fig, ax = plt.subplots()
pl = ax.plot(np.atan2(cs2, cs1), np.atan2(np.sqrt(cs2**2 + cs1**2), cs3))
ax.set_title(r"angles")
ax.set_xlabel(r"$\phi$")
ax.set_ylabel(r"$\theta$")
fig.savefig(os.path.join(graphdir, f"escanphase.{args.format}"))
fig.clear()
ax.clear()


fig, ax = plt.subplots()
pl = ax.plot(
    t,
    np.apply_along_axis(vnsqr, 2, c),
)
ax.set_title(r"$|\psi|^2$")
fig.savefig(os.path.join(graphdir, f"asize.{args.format}"))
fig.clear()
ax.clear()
