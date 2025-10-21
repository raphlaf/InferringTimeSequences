# Figure 7

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

filepath = "Fig 7/data.jld2"
data = h5py.File(filepath, "r")


llikelihood = np.asarray(data["llikelihood"])
T = np.asarray(data["T"])
Tmin = np.asarray(data["Tmin"])
Tmax = np.asarray(data["Tmax"])

fig, axes = plt.subplots(2, 2, figsize=(3.4252, 3.4252), sharex=True, sharey=True)

axes[0, 0].imshow(np.exp((llikelihood[:, :, 0]) - np.max(llikelihood[:, :, 0])), origin="lower", extent=(Tmin, Tmax, Tmin, Tmax))
# axes[0, 0].text(0.80, 0.97, r"(a)", transform=axes[0, 0].transAxes, fontsize=12, fontweight='bold', va='top', ha='left', color='black')
panel_label(axes[0, 0], "(a)", loc="tr", fontsize=12, pad=3)
axes[0, 1].imshow(np.exp((llikelihood[:, :, 1]) - np.max(llikelihood[:, :, 1])), origin="lower", extent=(Tmin, Tmax, Tmin, Tmax))
# axes[0, 1].text(0.80, 0.97, r"(b)", transform=axes[0, 1].transAxes, fontsize=12, fontweight='bold', va='top', ha='left', color='black')
panel_label(axes[0, 1], "(b)", loc="tr", fontsize=12, pad=3)
axes[1, 0].imshow(np.exp((llikelihood[:, :, 2]) - np.max(llikelihood[:, :, 2])), origin="lower", extent=(Tmin, Tmax, Tmin, Tmax))
# axes[1, 0].text(0.80, 0.97, r"(c)", transform=axes[1, 0].transAxes, fontsize=12, fontweight='bold', va='top', ha='left', color='black')
panel_label(axes[1, 0], "(c)", loc="tr", fontsize=12, pad=3)
im = axes[1, 1].imshow(np.exp((llikelihood[:, :, 3]) - np.max(llikelihood[:, :, 3])), origin="lower", extent=(Tmin, Tmax, Tmin, Tmax))
# axes[1, 1].text(0.80, 0.97, r"(d)", transform=axes[1, 1].transAxes, fontsize=12, fontweight='bold', va='top', ha='left', color='black')
panel_label(axes[1, 1], "(d)", loc="tr", fontsize=12, pad=3)

axes[0, 0].set_xticks([0.0, 10.0, 20.0, 30.0])
axes[0, 0].set_yticks([0.0, 10.0, 20.0, 30.0])
axes[0, 1].set_xticks([0.0, 10.0, 20.0, 30.0])
axes[0, 1].set_yticks([0.0, 10.0, 20.0, 30.0])
axes[1, 0].set_xticks([0.0, 10.0, 20.0, 30.0])
axes[1, 0].set_yticks([0.0, 10.0, 20.0, 30.0])
axes[1, 1].set_xticks([0.0, 10.0, 20.0, 30.0])
axes[1, 1].set_yticks([0.0, 10.0, 20.0, 30.0])

axes[0, 0].set_ylabel(r"$T_2$ (s)")
axes[1, 0].set_ylabel(r"$T_2$ (s)")
axes[1, 0].set_xlabel(r"$T_1$ (s)")
axes[1, 1].set_xlabel(r"$T_1$ (s)")

fig.subplots_adjust(left=0.16, bottom=0.17, right=0.9, top=0.95, hspace=0.2, wspace=0.15)

ax_bbox1 = axes[0, 1].get_position()
ax_bounds1 = ax_bbox1.bounds
ax_bbox2 = axes[1, 1].get_position()
ax_bounds2 = ax_bbox2.bounds

y0 = ax_bounds2[1]
y1 = ax_bounds1[1] + ax_bounds1[3]
h = y1 - y0


cbar_ax = fig.add_axes([0.915, y0, 0.025, h])

cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_ticks([0, 1])
cbar.set_ticklabels(["Least likely", "Most likely"], rotation=90, va="center")
cbar_ax.xaxis.set_ticks_position("top")
cbar_ax.xaxis.set_label_position("top")
cbar_ax.tick_params(axis="y", length=0, pad=1.8)
labels = cbar_ax.yaxis.get_majorticklabels()

dy = 30/72
offset0 = matplotlib.transforms.ScaledTranslation(0, dy, fig.dpi_scale_trans)
offset1 = matplotlib.transforms.ScaledTranslation(0, -dy, fig.dpi_scale_trans)

labels[0].set_transform(labels[0].get_transform() + offset0)
labels[1].set_transform(labels[1].get_transform() + offset1)



outdir = r"Fig 7"
import os
os.makedirs(outdir, exist_ok=True)
fname = os.path.basename(os.path.normpath(outdir))
fig.savefig(os.path.join(outdir, fname + ".png"), dpi=600, bbox_inches="tight")
fig.savefig(os.path.join(outdir, fname + ".pdf"), bbox_inches="tight")
plt.close(fig)

