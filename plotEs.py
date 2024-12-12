import tomllib
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
from numba import njit

parser = ArgumentParser()
parser.add_argument("-d", "--directory", help="directory where data files are stored")
parser.add_argument("-r", "--row", help="row of Es to plot", type=int)
args = parser.parse_args()

a = np.loadtxt(f"{args.directory}/Es.csv")
with open(f"{args.directory}/simconf.toml", "rb") as f:
    conf = tomllib.load(f)["constants"]

print(np.shape(a))
dt = conf["dt"]
emax = (
    np.pi * 658.2119569 / (100 * dt)
)  # 100 should ideally be configurable, figure out later


@njit
def idxtoe(i):
    return (i - 128) * emax / 128.0


a = idxtoe(a)
fig, ax = plt.subplots()
im = ax.imshow(
    a,
    extent=(1.0, 5.0, conf["Bend"], conf["Bstart"]),
    aspect=4.0 / (conf["Bstart"] - conf["Bend"]),
    interpolation="none",
)
ax.set_xlabel(r"$P/P_{th}$")
ax.set_ylabel(r"$B_{ext}$ [T]")
ax.set_title(r"$E$ [µeV]")
cb = plt.colorbar(im)
minv = np.min(a)
maxv = np.max(a)
# cb.set_ticks([minv, maxv], labels=[f"{minv:.1f} µeV", f"{maxv:.1f} µeV"])
fig.savefig("energies.pdf")

B = conf["Bstart"] + args.row / 256 * (conf["Bend"] - conf["Bstart"])
fig, ax = plt.subplots()
ax.plot(np.linspace(1.0, 5.0, num=256, endpoint=False), a[args.row, :])
ax.set_xlabel(r"$P/P_{th}$")
ax.set_ylabel(r"E [µeV]")
ax.set_title(f"E cross-section, B={B} T")
fig.savefig("EatB.pdf")
