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

print(np.shape(a))
t = np.arange(300)

fig, ax = plt.subplots()
im = ax.plot(
    t,
    a,
)
ax.set_title(r"S3")
fig.savefig(os.path.join(graphdir, "S3.pdf"))
fig.clear()
ax.clear()
