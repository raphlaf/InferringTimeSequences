# Figure 2
import matplotlib
from cycler import cycler
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, InsetPosition
import numpy as np
import h5py
from scipy.special import gamma
from scipy.interpolate import UnivariateSpline

from matplotlib import patheffects

matplotlib.style.use("custom-style.mplstyle")


def panel_label(ax, label, loc="tr", pad=6, **kw):
    """
    Place a panel label like '(a)' at a consistent distance from a corner.

    loc: 'tl' (top-left), 'tr' (top-right), 'bl' (bottom-left), 'br' (bottom-right)
    pad: offset in points (~1/72 inch)
    """
    ha = 'left' if 'l' in loc else 'right'
    va = 'top'  if 't' in loc else 'bottom'
    x  = 0.0 if 'l' in loc else 1.0
    y  = 1.0 if 't' in loc else 0.0
    dx =  pad if ha == 'left'  else -pad
    dy = -pad if va == 'top'   else  pad

    defaults = dict(
        fontsize=8, fontweight='bold', color='black',
        path_effects=[patheffects.withStroke(linewidth=1.0, foreground='white')],
    )
    defaults.update(kw)

    ax.annotate(
        label, xy=(x, y), xycoords='axes fraction',
        xytext=(dx, dy), textcoords='offset points',
        ha=ha, va=va, **defaults
    )

data = h5py.File("Fig 2/data.jld2")

arange = np.array(data["arange"])
crange = np.array(data["crange"])
taurange = np.array(data["taurange"])
betarange = np.array(data["betarange"])
FI = np.array(data["FI"]).T
N = 1000  # number of cells
FI = N*FI

fig = plt.figure(figsize=(3.4252, 4.0))
ax = fig.add_subplot(111)
axes = fig.subplots(2, 2, sharey=True)

axes[0, 0].plot(arange, FI[0, :])
axes[0, 0].set_xlabel(r"$a$")
axes[0, 0].set_xticks([0, 5, 10])
# axes[0, 0].text(0.05, 0.96, r"(a)", transform=axes[0, 0].transAxes, fontsize=12, fontweight='bold', va='top', ha='left',
#                 color='black', path_effects=[patheffects.withStroke(linewidth=1,foreground="white")])
panel_label(axes[0, 0], "(a)", loc="tr", fontsize=12, pad=3)

axes[0, 1].plot(betarange, FI[3, :])
axes[0, 1].set_xlabel(r"$\beta$")
axes[0, 1].set_xticks([0, 0.5, 1])
# axes[0, 1].text(0.78, 0.96, r"(b)", transform=axes[0, 1].transAxes, fontsize=12, fontweight='bold', va='top', ha='left',
#                 color='black', path_effects=[patheffects.withStroke(linewidth=1,foreground="white")])
panel_label(axes[0, 1], "(b)", loc="tr", fontsize=12, pad=3)

axes[1, 0].plot(crange, FI[1, :])
axes[1, 0].set_xlabel(r"$c$")
axes[1, 0].set_xticks([0, 5, 10])
# axes[1, 0].text(0.78, 0.96, r"(c)", transform=axes[1, 0].transAxes, fontsize=12, fontweight='bold', va='top', ha='left',
#              color='black', path_effects=[patheffects.withStroke(linewidth=1,foreground="white")])
panel_label(axes[1, 0], "(c)", loc="tr", fontsize=12, pad=3)


axes[1, 1].plot(taurange, FI[2, :])
axes[1, 1].set_xlabel(r"$\tau$ (s)")
axes[1, 1].set_xticks([0, 15, 30])
# axes[1, 1].text(0.78, 0.96, r"(d)", transform=axes[1, 1].transAxes, fontsize=12, fontweight='bold', va='top', ha='left',
#                 color='black', path_effects=[patheffects.withStroke(linewidth=1,foreground="white")])
panel_label(axes[1, 1], "(d)", loc="tr", fontsize=12, pad=3)

yl = axes[0, 0].get_ylim()
axes[0, 0].set_ylim(0.0, 4)
axes[0, 0].set_xlim(0.0, None)
axes[0, 1].set_xlim(0.0, None)
axes[1, 0].set_xlim(0.0, None)
axes[1, 1].set_xlim(0.0, None)

ax.spines["top"].set_color("none")
ax.spines["bottom"].set_color("none")
ax.spines["right"].set_color("none")
ax.spines["left"].set_color("none")
ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

ax.set_ylabel(r"Fisher information $(s^{-2})$", labelpad=-5)
fig.subplots_adjust(left=0.13, bottom=0.17, right=0.97, top=0.95, hspace=0.5, wspace=0.3)



outdir = r"Fig 2"
import os
os.makedirs(outdir, exist_ok=True)
fname = os.path.basename(os.path.normpath(outdir))
fig.savefig(os.path.join(outdir, fname + ".png"), dpi=600, bbox_inches="tight")
fig.savefig(os.path.join(outdir, fname + ".pdf"), bbox_inches="tight")
plt.close(fig)

