import tomllib
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

parser = ArgumentParser()
parser.add_argument("-d", "--directory", help="directory where data files are stored.")
parser.add_argument(
    "-f",
    "--format",
    help="The format to ouput with specified by file ending.",
    default="pdf",
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

s3 = np.loadtxt(os.path.join(args.directory, "S3avg.csv"))
s2 = np.loadtxt(os.path.join(args.directory, "S2avg.csv"))
s1 = np.loadtxt(os.path.join(args.directory, "S1avg.csv"))

fig, ax = plt.subplots()
im = ax.imshow(
    s3,
    extent=(conf["xstart"], conf["xend"], conf["eend"], conf["estart"]),
    aspect="auto",
    interpolation="none",
)
ax.set_title(r"$<S_3>$")
ax.set_xlabel("p")
ax.set_ylabel(r"$\epsilon$")
cb = plt.colorbar(im)
fig.savefig(f"S3.{args.format}")
ax.clear()
fig.clear()

fig, ax = plt.subplots()
im = ax.imshow(
    s2,
    extent=(conf["xstart"], conf["xend"], conf["eend"], conf["estart"]),
    aspect="auto",
    interpolation="none",
)
ax.set_title(r"$<S_2>$")
ax.set_xlabel("p")
ax.set_ylabel(r"$\epsilon$")
cb = plt.colorbar(im)
fig.savefig(f"S2.{args.format}")
ax.clear()
fig.clear()

fig, ax = plt.subplots()
im = ax.imshow(
    s1,
    extent=(conf["xstart"], conf["xend"], conf["eend"], conf["estart"]),
    aspect="auto",
    interpolation="none",
)
ax.set_title(r"$<S_1>$")
ax.set_xlabel("p")
ax.set_ylabel(r"$\epsilon$")
cb = plt.colorbar(im)
fig.savefig(f"S1.{args.format}")
ax.clear()
fig.clear()
