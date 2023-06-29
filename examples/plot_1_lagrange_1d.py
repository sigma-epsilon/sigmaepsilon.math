"""
Lagrange Polynomials in 1d
==========================

Generation of Lagrange polynomials for approximation on a grid.
"""

# %%
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import sympy as sy

from sigmaepsilon.math.function import Function
from sigmaepsilon.math.approx.lagrange import gen_Lagrange_1d

inds = [1, 2, 3]
data = gen_Lagrange_1d(i=inds)

fig = plt.figure(figsize=(4, 7))  # in inches
fig.patch.set_facecolor("white")
gs = gridspec.GridSpec(len(inds), 1)

xdata = np.linspace(-1, 1, 100)

for i, ind in enumerate(inds):
    ax = fig.add_subplot(gs[i])
    label = "$" + data[ind]["symbol"] + "$"
    ax.set_title(label)
    fnc = Function(data[ind][0])
    fdata = fnc([xdata])
    ax.plot(xdata, fdata)
    ax.hlines(y=0, xmin=-1, xmax=1, colors="k", zorder=-10, lw=1.0)

fig.tight_layout()
