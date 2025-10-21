# Figure 9

import matplotlib
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from matplotlib import patheffects
import numpy as np
import h5py

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
data = h5py.File("Fig 9/data.jld2")

detFIT2 = np.asarray(data["detFIT2"]).T
detFIT3 = np.asarray(data["detFIT3"]).T
CRLBT2 = np.sqrt(np.asarray(data["CRLBT2"]).T)
CRLBT3 = np.sqrt(np.asarray(data["CRLBT3"]).T)
Trange = np.asarray(data["Trange"])



fig = plt.figure(figsize=(3.4252, 4.4252))
axes = fig.subplots(3, 2, sharex=True, sharey=True)

# DETERMINANT LOG

# vmin, vmax = min(np.min(detFIT3), np.min(detFIT2)), max(np.max(detFIT3), np.max(detFIT2))

# im1 = axes[0, 0].imshow(detFIT2[0, :, :].T, origin="lower",
#                   extent=(Trange[0], Trange[-1], Trange[0], Trange[-1]),
#                   norm=LogNorm(vmin=vmin, vmax=vmax))
# txt = axes[0, 0].text(0.05, 0.97, r"(a)", transform=axes[0, 0].transAxes, fontsize=12, fontweight='bold', va='top', ha='left', 
#                       color='black', path_effects=[patheffects.withStroke(linewidth=1,foreground="white")])

# im2 = axes[1, 0].imshow(detFIT2[1, :, :].T, origin="lower",
#                   extent=(Trange[0], Trange[-1], Trange[0], Trange[-1]),
#                   norm=LogNorm(vmin=vmin, vmax=vmax))
# txt = axes[1, 0].text(0.05, 0.97, r"(b)", transform=axes[1, 0].transAxes, fontsize=12, fontweight='bold', va='top', ha='left', 
#                       color='black', path_effects=[patheffects.withStroke(linewidth=1,foreground="white")])

# im3 = axes[2, 0].imshow(detFIT2[2, :, :].T, origin="lower",
#                   extent=(Trange[0], Trange[-1], Trange[0], Trange[-1]),
#                   norm=LogNorm(vmin=vmin, vmax=vmax))
# txt = axes[2, 0].text(0.05, 0.97, r"(c)", transform=axes[2, 0].transAxes, fontsize=12, fontweight='bold', va='top', ha='left', 
#                       color='black', path_effects=[patheffects.withStroke(linewidth=1,foreground="white")])

# ind = 66
# print(Trange[ind])

# # vmin, vmax = np.min(np.log10(detFIT3)), np.max(np.log10(detFIT3))
# # vmin, vmax = np.min(detFIT3), np.max(detFIT3)
# im4 = axes[0, 1].imshow(detFIT3[:, :, ind].T, origin="lower",
#                   extent=(Trange[0], Trange[-1], Trange[0], Trange[-1]),
#                   norm=LogNorm(vmin=vmin, vmax=vmax))
# txt = axes[0, 1].text(0.05, 0.97, r"(d)", transform=axes[0, 1].transAxes, fontsize=12, fontweight='bold', va='top', ha='left', 
#                       color='black', path_effects=[patheffects.withStroke(linewidth=1,foreground="white")])

# im5 = axes[1, 1].imshow(detFIT3[:, ind, :].T, origin="lower",
#                   extent=(Trange[0], Trange[-1], Trange[0], Trange[-1]),
#                   norm=LogNorm(vmin=vmin, vmax=vmax))
# txt = axes[1, 1].text(0.05, 0.97, r"(e)", transform=axes[1, 1].transAxes, fontsize=12, fontweight='bold', va='top', ha='left', 
#                       color='black', path_effects=[patheffects.withStroke(linewidth=1,foreground="white")])

# im6 = axes[2, 1].imshow(detFIT3[ind, :, :].T, origin="lower",
#                   extent=(Trange[0], Trange[-1], Trange[0], Trange[-1]),
#                   norm=LogNorm(vmin=vmin, vmax=vmax))
# txt = axes[2, 1].text(0.05, 0.97, r"(f)", transform=axes[2, 1].transAxes, fontsize=12, fontweight='bold', va='top', ha='left', 
#                       color='black', path_effects=[patheffects.withStroke(linewidth=1,foreground="white")])

# CRLB LOG

vmin, vmax = min(np.min(CRLBT3), np.min(CRLBT2)), max(np.max(CRLBT3), np.max(CRLBT2))

im1 = axes[1, 0].imshow(CRLBT2[0, :, :].T, origin="lower",
                  extent=(Trange[0], Trange[-1], Trange[0], Trange[-1]),
                  norm=LogNorm(vmin=vmin, vmax=vmax))
# txt = axes[0, 0].text(0.05, 0.97, r"(a)", transform=axes[0, 0].transAxes, fontsize=12, fontweight='bold', va='top', ha='left', 
#                       color='black', path_effects=[patheffects.withStroke(linewidth=1,foreground="white")])
panel_label(axes[1, 0], "(b)", loc="tl", fontsize=12, pad=3)

im2 = axes[2, 0].imshow(CRLBT2[1, :, :].T, origin="lower",
                  extent=(Trange[0], Trange[-1], Trange[0], Trange[-1]),
                  norm=LogNorm(vmin=vmin, vmax=vmax))
# txt = axes[1, 0].text(0.05, 0.97, r"(b)", transform=axes[1, 0].transAxes, fontsize=12, fontweight='bold', va='top', ha='left', 
#                       color='black', path_effects=[patheffects.withStroke(linewidth=1,foreground="white")])
panel_label(axes[2, 0], "(c)", loc="tl", fontsize=12, pad=3)

im3 = axes[0, 0].imshow(CRLBT2[2, :, :].T, origin="lower",
                  extent=(Trange[0], Trange[-1], Trange[0], Trange[-1]),
                  norm=LogNorm(vmin=vmin, vmax=vmax))
# txt = axes[2, 0].text(0.05, 0.97, r"(c)", transform=axes[2, 0].transAxes, fontsize=12, fontweight='bold', va='top', ha='left', 
#                       color='black', path_effects=[patheffects.withStroke(linewidth=1,foreground="white")])
panel_label(axes[0, 0], "(a)", loc="tl", fontsize=12, pad=3)

ind = 66
print(Trange[ind])

# vmin, vmax = np.min(np.log10(detFIT3)), np.max(np.log10(detFIT3))
# vmin, vmax = np.min(detFIT3), np.max(detFIT3)
im4 = axes[0, 1].imshow(CRLBT3[:, :, ind].T, origin="lower",
                  extent=(Trange[0], Trange[-1], Trange[0], Trange[-1]),
                  norm=LogNorm(vmin=vmin, vmax=vmax))
# txt = axes[0, 1].text(0.05, 0.97, r"(d)", transform=axes[0, 1].transAxes, fontsize=12, fontweight='bold', va='top', ha='left', 
#                       color='black', path_effects=[patheffects.withStroke(linewidth=1,foreground="white")])
panel_label(axes[0, 1], "(d)", loc="tl", fontsize=12, pad=3)

im5 = axes[1, 1].imshow(CRLBT3[:, ind, :].T, origin="lower",
                  extent=(Trange[0], Trange[-1], Trange[0], Trange[-1]),
                  norm=LogNorm(vmin=vmin, vmax=vmax))
# txt = axes[1, 1].text(0.05, 0.97, r"(e)", transform=axes[1, 1].transAxes, fontsize=12, fontweight='bold', va='top', ha='left', 
#                       color='black', path_effects=[patheffects.withStroke(linewidth=1,foreground="white")])
panel_label(axes[1, 1], "(e)", loc="tl", fontsize=12, pad=3)

im6 = axes[2, 1].imshow(CRLBT3[ind, :, :].T, origin="lower",
                  extent=(Trange[0], Trange[-1], Trange[0], Trange[-1]),
                  norm=LogNorm(vmin=vmin, vmax=vmax))
# txt = axes[2, 1].text(0.05, 0.97, r"(f)", transform=axes[2, 1].transAxes, fontsize=12, fontweight='bold', va='top', ha='left', 
#                       color='black', path_effects=[patheffects.withStroke(linewidth=1,foreground="white")])
panel_label(axes[2, 1], "(f)", loc="tl", fontsize=12, pad=3)

# CRLB LINEAR

# vmin, vmax = 0.0, 300.0

# im1 = axes[0, 0].imshow(CRLBT2[0, :, :].T, origin="lower",
#                   extent=(Trange[0], Trange[-1], Trange[0], Trange[-1]),
#                   vmin=vmin, vmax=vmax)
# txt = axes[0, 0].text(0.05, 0.97, r"(a)", transform=axes[0, 0].transAxes, fontsize=12, fontweight='bold', va='top', ha='left', 
#                       color='black', path_effects=[patheffects.withStroke(linewidth=1,foreground="white")])

# im2 = axes[1, 0].imshow(CRLBT2[1, :, :].T, origin="lower",
#                   extent=(Trange[0], Trange[-1], Trange[0], Trange[-1]),
#                   vmin=vmin, vmax=vmax)
# txt = axes[1, 0].text(0.05, 0.97, r"(b)", transform=axes[1, 0].transAxes, fontsize=12, fontweight='bold', va='top', ha='left', 
#                       color='black', path_effects=[patheffects.withStroke(linewidth=1,foreground="white")])

# im3 = axes[2, 0].imshow(CRLBT2[2, :, :].T, origin="lower",
#                   extent=(Trange[0], Trange[-1], Trange[0], Trange[-1]),
#                   vmin=vmin, vmax=vmax)
# txt = axes[2, 0].text(0.05, 0.97, r"(c)", transform=axes[2, 0].transAxes, fontsize=12, fontweight='bold', va='top', ha='left', 
#                       color='black', path_effects=[patheffects.withStroke(linewidth=1,foreground="white")])

# ind = 66
# print(Trange[ind])

# im4 = axes[0, 1].imshow(CRLBT3[:, :, ind].T, origin="lower",
#                   extent=(Trange[0], Trange[-1], Trange[0], Trange[-1]),
#                   vmin=vmin, vmax=vmax)
# txt = axes[0, 1].text(0.05, 0.97, r"(d)", transform=axes[0, 1].transAxes, fontsize=12, fontweight='bold', va='top', ha='left', 
#                       color='black', path_effects=[patheffects.withStroke(linewidth=1,foreground="white")])

# im5 = axes[1, 1].imshow(CRLBT3[:, ind, :].T, origin="lower",
#                   extent=(Trange[0], Trange[-1], Trange[0], Trange[-1]),
#                   vmin=vmin, vmax=vmax)
# txt = axes[1, 1].text(0.05, 0.97, r"(e)", transform=axes[1, 1].transAxes, fontsize=12, fontweight='bold', va='top', ha='left', 
#                       color='black', path_effects=[patheffects.withStroke(linewidth=1,foreground="white")])

# im6 = axes[2, 1].imshow(CRLBT3[ind, :, :].T, origin="lower",
#                   extent=(Trange[0], Trange[-1], Trange[0], Trange[-1]),
#                   vmin=vmin, vmax=vmax)
# txt = axes[2, 1].text(0.05, 0.97, r"(f)", transform=axes[2, 1].transAxes, fontsize=12, fontweight='bold', va='top', ha='left', 
#                       color='black', path_effects=[patheffects.withStroke(linewidth=1,foreground="white")])


axes[0, 0].set_xlabel(r"$T_1$ (s)")
axes[0, 0].set_ylabel(r"$T_2$ (s)")
axes[0, 0].set_xticks([0.0, 10.0, 20.0, 30.0])
axes[0, 0].set_yticks([0.0, 10.0, 20.0, 30.0])
axes[1, 0].set_xlabel(r"$T_1$ (s)")
axes[1, 0].set_ylabel(r"$T_2$ (s)")
axes[2, 0].set_xlabel(r"$T_1$ (s)")
axes[2, 0].set_ylabel(r"$T_2$ (s)")

axes[0, 1].set_xlabel(r"$T_1$ (s)")
axes[0, 1].set_ylabel(r"$T_2$ (s)")
axes[1, 1].set_xlabel(r"$T_1$ (s)")
axes[1, 1].set_ylabel(r"$T_3$ (s)")
axes[2, 1].set_xlabel(r"$T_2$ (s)")
axes[2, 1].set_ylabel(r"$T_3$ (s)")




fig.subplots_adjust(left=0.165, right=0.75, bottom=0.2, top=0.92, wspace=0.3, hspace=0.3)

ax_bbox = axes[2, 1].get_position()
ax_bounds = ax_bbox.bounds
x0 = ax_bounds[0]
y0 = ax_bounds[1]
w = ax_bounds[2]

ax_bbox = axes[0, 1].get_position()
ax_bounds = ax_bbox.bounds
h = ax_bounds[1] + ax_bounds[3] - y0

cbar_ax = fig.add_axes([x0+w+0.01, y0, 0.025, h])
cbar = fig.colorbar(im1, cax=cbar_ax)
cbar_ax.set_ylabel("CRLB"r"$_{Z_n}$"" (s)")
cbar_ax.tick_params(axis="y", which="both", length=3.0, pad=2)



outdir = r"Fig 9"
import os
os.makedirs(outdir, exist_ok=True)
fname = os.path.basename(os.path.normpath(outdir))
fig.savefig(os.path.join(outdir, fname + ".png"), dpi=600, bbox_inches="tight")
fig.savefig(os.path.join(outdir, fname + ".pdf"), bbox_inches="tight")
plt.close(fig)

