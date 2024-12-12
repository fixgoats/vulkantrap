import tomllib
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np

parser = ArgumentParser()
parser.add_argument("-d", "--directory", help="directory where data files are stored")
parser.add_argument("-r", "--row", help="row of Es to plot", type=int)
args = parser.parse_args()

a = np.loadtxt(f"{args.directory}/Bint.csv")
with open(f"{args.directory}/simconf.toml", "rb") as f:
    conf = tomllib.load(f)["constants"]

fig, ax = plt.subplots()
im = ax.imshow(
    a,
    extent=(1.0, 5.0, conf["Bend"], conf["Bstart"]),
    aspect=4.0 / (conf["Bstart"] - conf["Bend"]),
    interpolation="none",
)
ax.set_xlabel(r"$P/P_{th}$")
ax.set_ylabel(r"$B$ [T]")
ax.set_title(r"$B_{int}$ [T]")
cb = plt.colorbar(im)
minv = np.min(a)
maxv = np.max(a)
# cb.set_ticks([minv, maxv], labels=[f"{minv:.1f} µeV", f"{maxv:.1f} µeV"])
fig.savefig("Bint.pdf")

fig, ax = plt.subplots()
b = a + a[::-1, :]
im = ax.imshow(
    b,
    extent=(1.0, 5.0, conf["Bend"], conf["Bstart"]),
    aspect=4.0 / (conf["Bstart"] - conf["Bend"]),
    interpolation="none",
)
ax.set_xlabel(r"$P/P_{th}$")
ax.set_ylabel(r"$B$ [T]")
ax.set_title(r"$B_{int}(P,B) - B_{int}(P,-B)$ [T]")
cb = plt.colorbar(im)
minv = np.min(b)
maxv = np.max(b)
# cb.set_ticks([minv, maxv], labels=[f"{minv:.1f} µeV", f"{maxv:.1f} µeV"])
fig.savefig("symtest.pdf")
