import tomllib
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numba.core.config import os

parser = ArgumentParser()
parser.add_argument("-d", "--directory", help="directory where data files are stored")
args = parser.parse_args()

parts = args.directory.split("/")
graphdir = os.path.join("graphs", parts[-3], parts[-2])
Path(graphdir).mkdir(parents=True, exist_ok=True)


a = np.loadtxt(os.path.join(args.directory, "S3.csv"))
with open(os.path.join(args.directory, "simconf.toml"), "rb") as f:
    conf = tomllib.load(f)["constants"]

print(a)


fig, ax = plt.subplots()
im = ax.imshow(
    a,
    extent=(conf["xstart"], conf["xend"], conf["eend"], conf["estart"]),
    aspect="auto",
    interpolation="none",
)
ax.set_title(r"S3")
ax.set_xlabel("p")
ax.set_ylabel(r"$\epsilon$")
cb = plt.colorbar(im)
fig.savefig(os.path.join(graphdir, "S3.pdf"))
fig.clear()
ax.clear()
