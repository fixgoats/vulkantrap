import tomllib
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np

parser = ArgumentParser()
parser.add_argument(
    "-f", "--format", help="The format to ouput with specified by file ending."
)
args = parser.parse_args()


s3 = np.loadtxt("build/data/25-1-7/11-56/S3avg.csv")
s2 = np.loadtxt("build/data/25-1-7/11-56/S2avg.csv")
s1 = np.loadtxt("build/data/25-1-7/11-56/S1avg.csv")
with open("build/data/24-12-16/18-57/simconf.toml", "rb") as f:
    conf = tomllib.load(f)["constants"]

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
