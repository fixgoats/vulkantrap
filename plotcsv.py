import matplotlib.pyplot as plt
import numpy as np

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-i", "--input", help="datafile.")
parser.add_argument("-o", "--out", help="output filename")
args = parser.parse_args()

a = np.loadtxt(args.input)

fig, ax = plt.subplots()
im = ax.imshow(
    a,
    extent=(1.0, 1.8, -5.0, 5.0),
    aspect=0.8 / 10.0,
    interpolation="none",
)
ax.set_xlabel(r"$p$")
ax.set_ylabel(r"$\epsilon$")
ax.set_title(r"eee")
fig.savefig(args.out)
