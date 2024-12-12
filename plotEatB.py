import tomllib
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numba.core.config import os

parser = ArgumentParser()
parser.add_argument("-d", "--directory", help="directory where data files are stored")
parser.add_argument("-r", "--row", help="row of Es to plot", type=int)
args = parser.parse_args()

a = np.loadtxt(os.path.join(args.directory, "Es.csv"))
with open(os.path.join(args.directory, "simconf.toml"), "rb") as f:
    conf = tomllib.load(f)["constants"]
parts = args.directory.split("/")
graphdir = os.path.join("graphs", parts[-3], parts[-2])
Path(graphdir).mkdir(parents=True, exist_ok=True)

dt = conf["dt"]
emax = (
    np.pi * 658.2119569 / (100 * dt)
)  # 100 should ideally be configurable, figure out later


def idxtoe(i):
    return (i - 128) * emax / 128.0


a = idxtoe(a)
fig, ax = plt.subplots()
B = conf["Bstart"] + args.row / 256 * (conf["Bend"] - conf["Bstart"])
Bs = np.array([40, 50, 60, 70, 80])
print(conf["Bstart"] + Bs / 256 * (conf["Bend"] - conf["Bstart"]))
for b in Bs:
    ax.plot(np.linspace(1.0, 5.0, num=256, endpoint=False), a[b, :])
ax.set_xlabel(r"$P/P_{th}$")
ax.set_ylabel(r"E [µeV]")
ax.set_title(f"E cross-section, B={B} T")
fig.savefig(os.path.join(graphdir, "EatB.pdf"))
