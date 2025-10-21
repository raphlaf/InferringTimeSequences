# Figure S1

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

data = h5py.File("Fig S1/data1.jld2")
data2 = h5py.File("Fig S1/data2.jld2")

tau_range = np.array(data["tau_range"])
starti = 0
FI = 1000*np.array(data["FI"]).T
FI[0, :] *= 10.0**2
FI[1, :] *= 15.0**2

tau_range2 = np.array(data2["tau_range"])
FI2 = 1000*np.array(data2["FI"]).T
FI2[0, :] *= 2.0**2
FI2[1, :] *= 20.0**2

res = tau_range.size

fig = plt.figure(figsize=(3.425, 3.35))
axes = fig.subplots(3, 2, sharex=True, sharey="col")

axes[0, 0].plot(tau_range[starti:-1], 1/np.sqrt(FI[0, starti:-1]))
# axes[0, 0].set_ylabel(r"rel-$CRLB$")
# axes[0, 0].text(0.80, 0.97, r"(a)", transform=axes[0, 0].transAxes, fontsize=12, fontweight='bold', va='top', ha='left',
#                 color='black', path_effects=[patheffects.withStroke(linewidth=1,foreground="white")])
panel_label(axes[0, 0], "(a)", loc="tr", fontsize=12, pad=3)

axes[1, 0].plot(tau_range[starti:-1], 0.5/np.sqrt(FI[0, starti:-1]) + 0.5/np.sqrt(FI[1, starti:-1]))
axes[1, 0].set_ylabel(r"rel-$CRLB$")
# axes[1, 0].text(0.80, 0.97, r"(b)", transform=axes[1, 0].transAxes, fontsize=12, fontweight='bold', va='top', ha='left',
#                 color='black', path_effects=[patheffects.withStroke(linewidth=1,foreground="white")])
panel_label(axes[1, 0], "(b)", loc="tr", fontsize=12, pad=3)

axes[2, 0].plot(tau_range[starti:-1], 0.5/np.sqrt(FI2[0, starti:-1]) + 0.5/np.sqrt(FI2[1, starti:-1]))
axes[2, 0].set_xlabel(r"$\tau^1$ $(s)$")
# axes[2, 0].set_ylabel(r"rel-$CRLB$")
# axes[2, 0].text(0.80, 0.97, r"(c)", transform=axes[2, 0].transAxes, fontsize=12, fontweight='bold', va='top', ha='left',
#                 color='black', path_effects=[patheffects.withStroke(linewidth=1,foreground="white")])
panel_label(axes[2, 0], "(c)", loc="tr", fontsize=12, pad=3)


RMSE12 = np.zeros((res, res))
RMSE22 = np.zeros((res, res))
RMSE122 = np.zeros((res, res))
RMSE222 = np.zeros((res, res))


for i in range(starti, res):
    for j in range(starti, res):
        if (np.sqrt(0.5*(FI[0, i] + FI[0, j])) != 0.0):
            RMSE12[i, j] = 1/np.sqrt(0.5*(FI[0, i] + FI[0, j]))
        if (np.sqrt(0.5*(FI2[0, i] + FI2[0, j])) != 0.0):
            RMSE122[i, j] = 1/np.sqrt(0.5*(FI2[0, i] + FI2[0, j]))
        if (np.sqrt(0.5*(FI[0, i] + FI[0, j])) != 0 and np.sqrt(0.5*(FI[1, i] + FI[1, j])) != 0):
            RMSE22[i, j] = 0.5/np.sqrt(0.5*(FI[0, i] + FI[0, j])) + 0.5/np.sqrt(0.5*(FI[1, i] + FI[1, j]))
        if (np.sqrt(0.5*(FI2[0, i] + FI2[0, j])) != 0 and np.sqrt(0.5*(FI2[1, i] + FI2[1, j])) != 0):
            RMSE222[i, j] = 0.5/np.sqrt(0.5*(FI2[0, i] + FI2[0, j])) + 0.5/np.sqrt(0.5*(FI2[1, i] + FI2[1, j]))

vmin = 0.2
vmax = 0.264



vmin, vmax = 0.03, 0.04
levels = np.linspace(vmin, vmax, 10)
im1 = axes[0, 1].imshow(RMSE12[starti:-1, starti:-1], origin="lower", 
                  extent=(tau_range[starti], tau_range[-1], tau_range[starti], tau_range[-1]),
                  vmin=vmin, vmax=vmax)
cont = axes[0, 1].contour(tau_range[starti:-1], tau_range[starti:-1],
                          RMSE12[starti:-1, starti:-1], colors="black",
                          linestyles=":", alpha=0.5, linewidths=1.0,
                          levels=levels)

axes[0, 1].set_ylabel(r"$\tau^2$ (s)")
# axes[0, 1].text(0.80, 0.97, r"(d)", transform=axes[0, 1].transAxes, fontsize=12, fontweight='bold', va='top', ha='left',
#                 color='black', path_effects=[patheffects.withStroke(linewidth=1,foreground="white")])
panel_label(axes[0, 1], "(d)", loc="tr", fontsize=12, pad=3)


vmin, vmax = 0.02, 0.03
levels = np.linspace(vmin, vmax, 10)
im2 = axes[1, 1].imshow(RMSE22[starti:-1, starti:-1], origin="lower",
                       extent=(tau_range[starti], tau_range[-1], tau_range[starti], tau_range[-1]),
                       vmin=vmin, vmax=vmax)

cont = axes[1, 1].contour(tau_range[starti:-1], tau_range[starti:-1],
                          RMSE22[starti:-1, starti:-1],colors="black",
                          linestyles=":", alpha=0.5, linewidths=1.0,
                          levels=levels)

axes[1, 1].set_ylabel(r"$\tau^2$ (s)")
# axes[1, 1].text(0.80, 0.97, r"(e)", transform=axes[1, 1].transAxes, fontsize=12, fontweight='bold', va='top', ha='left',
#                 color='black', path_effects=[patheffects.withStroke(linewidth=1,foreground="white")])
panel_label(axes[1, 1], "(e)", loc="tr", fontsize=12, pad=3)

vmin, vmax = 0.03, 0.04
levels = np.linspace(vmin, vmax, 10)
im3 = axes[2, 1].imshow(RMSE222[starti:-1, starti:-1], origin="lower",
                       extent=(tau_range[starti], tau_range[-1], tau_range[starti], tau_range[-1]),
                       vmin=vmin, vmax=vmax)

cont = axes[2, 1].contour(tau_range[starti:-1], tau_range[starti:-1],
                          RMSE222[starti:-1, starti:-1], colors="black",
                          linestyles=":", alpha=0.5, linewidths=1.0,
                          levels=levels)

axes[2, 1].set_xlabel(r"$\tau^1$ (s)")
axes[2, 1].set_ylabel(r"$\tau^2$ (s)")
# axes[2, 1].text(0.80, 0.97, r"(f)", transform=axes[2, 1].transAxes, fontsize=12, fontweight='bold', va='top', ha='left',
#                 color='black', path_effects=[patheffects.withStroke(linewidth=1,foreground="white")])
panel_label(axes[2, 1], "(f)", loc="tr", fontsize=12, pad=3)

axes[0, 1].set_xticks([0.0, 25, 50.0])
axes[1, 1].set_xticks([0.0, 25, 50.0])
axes[0, 1].set_yticks([0.0, 25, 50.0])
axes[1, 1].set_yticks([0.0, 25, 50.0])
axes[2, 1].set_yticks([0.0, 25, 50.0])
axes[0, 0].set_ylim(0.02, 0.05)

fig.subplots_adjust(left=0.165, right=0.78, bottom=0.165, top=0.95, wspace=0.7)

ax_bbox = axes[0, 1].get_position()
ax_bounds = ax_bbox.bounds
y0 = ax_bounds[1]
h = ax_bounds[3]
cbar_ax = fig.add_axes([0.79, y0, 0.025, h])
cbar = fig.colorbar(im1, cax=cbar_ax, orientation="vertical")
cbar_ax.set_yticks([0.028, 0.04])
cbar_ax.set_yticklabels([r"$\leq 0.03$", r"$\geq 0.04$"])
cbar_ax.tick_params(axis="y", which="both", length=0, pad=1.2)

ax_bbox = axes[1, 1].get_position()
ax_bounds = ax_bbox.bounds
y0 = ax_bounds[1]
h = ax_bounds[3]
cbar_ax = fig.add_axes([0.79, y0, 0.025, h])
cbar = fig.colorbar(im2, cax=cbar_ax, orientation="vertical")
cbar_ax.set_yticks([0.02, 0.03])
cbar_ax.set_yticklabels([r"$\leq 0.02$", r"$\geq 0.03$"])
cbar_ax.tick_params(axis="y", which="both", length=0, pad=1.2)

ax_bbox = axes[2, 1].get_position()
ax_bounds = ax_bbox.bounds
y0 = ax_bounds[1]
h = ax_bounds[3]
cbar_ax = fig.add_axes([0.79, y0, 0.025, h])
cbar = fig.colorbar(im3, cax=cbar_ax, orientation="vertical")
cbar_ax.set_yticks([0.03, 0.04])
cbar_ax.set_yticklabels([r"$\leq 0.03$", r"$\geq 0.04$"])
cbar_ax.tick_params(axis="y", which="both", length=0, pad=1.2)



outdir = r"Fig S1"
import os
os.makedirs(outdir, exist_ok=True)
fname = os.path.basename(os.path.normpath(outdir))
fig.savefig(os.path.join(outdir, fname + ".png"), dpi=600, bbox_inches="tight")
fig.savefig(os.path.join(outdir, fname + ".pdf"), bbox_inches="tight")
plt.close(fig)

