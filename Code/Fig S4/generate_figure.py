# Figure S4

import matplotlib
from matplotlib import patheffects
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import h5py

matplotlib.style.use("custom-style.mplstyle")

def panel_label(ax, label, loc="tl", pad=6, **kw):
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

data_re = h5py.File("Fig S4/data_rc.jld2")


beta_range_re = np.asarray(data_re["beta_range"]).T
ratio_range_re = np.asarray(data_re["ratio_range"]).T
rmse_values_re = np.asarray(data_re["rmse_values"]).T

# vmin = np.min(rmse_values_re)
# vmax = np.max(rmse_values_re)


vmin_re1 = np.min(rmse_values_re[1:-1, 1:-1, 0])
vmax_re1 = np.max(rmse_values_re[1:-1, 1:-1, 0])
avg_re1 = np.mean(rmse_values_re[1:-1, 1:-1, 0])
hom_avg_re1 = np.mean(np.concatenate((rmse_values_re[0, 1:-1, 0], rmse_values_re[:, 0, 0], rmse_values_re[:-1, -1, 0])))
beta0_avg_re1 = np.mean(np.concatenate((rmse_values_re[0, 1:, 0], rmse_values_re[:, 0, 0])))
vmin_re2 = np.min(rmse_values_re[1:-1, 1:-1, 1])
vmax_re2 = np.max(rmse_values_re[1:-1, 1:-1, 1])
avg_re2 = np.mean(rmse_values_re[1:, 1:-1, 1])
# hom_avg_re2 = np.mean(np.concatenate((rmse_values_re[0, 1:-1, 1], rmse_values_re[:, 0, 1], rmse_values_re[:-1, -1, 1])))
hom_avg_re2 = np.mean(rmse_values_re[1:, -1, 1])
beta0_avg_re2 = np.mean(np.concatenate((rmse_values_re[0, 1:, 1], rmse_values_re[:, 0, 1])))
print(vmin_re1, vmax_re1, avg_re1, rmse_values_re[-1, -1, 0], hom_avg_re1, beta0_avg_re1)
print(np.mean(rmse_values_re[:, :, 0]))
print(vmin_re2, vmax_re2, avg_re2, rmse_values_re[-1, -1, 1], hom_avg_re2, beta0_avg_re2)

min_idx = np.unravel_index(np.argmin(rmse_values_re[:, :, 1]), rmse_values_re[:, :, 1].shape)
min_beta = beta_range_re[min_idx[0]]
min_ratio = ratio_range_re[min_idx[1]]
print("Indices of minimum in rmse_values_re[:, :, 1]:", min_idx)
print("Corresponding beta:", min_beta)
print("Corresponding ratio:", min_ratio)

fig = plt.figure(figsize=(3.4252, 1.7))
axes = fig.subplots(1, 2, sharex="col", sharey="row")
fig.subplots_adjust(left=0.21, right=0.8, bottom=0.3, top=0.8, wspace=0.45)


im1 = axes[0].imshow(rmse_values_re[:, :, 0].T, origin="lower", aspect="auto", 
                        extent=(beta_range_re[0]-0.05, beta_range_re[-1]+0.05,
                                ratio_range_re[0]-0.05, ratio_range_re[-1]+0.05),
                        # norm=LogNorm())
)#vmin=vmin, vmax=vmax)
# txt = axes[0].text(0.05, 0.97, r"\textbf{A}", transform=axes[0].transAxes, fontsize=12, fontweight='bold', va='top', ha='left', 
#                       color='black', path_effects=[patheffects.withStroke(linewidth=1,foreground="white")])
panel_label(axes[0], "(a)", loc="tl", fontsize=12, pad=4)


im2 = axes[1].imshow(rmse_values_re[:, :, 1].T, origin="lower", aspect="auto", 
                        extent=(beta_range_re[0]-0.05, beta_range_re[-1]+0.05,
                                ratio_range_re[0]-0.05, ratio_range_re[-1]+0.05),
                        # norm=LogNorm())
)#vmin=vmin, vmax=vmax)
# txt = axes[1].text(0.05, 0.97, r"\textbf{B}", transform=axes[1].transAxes, fontsize=12, fontweight='bold', va='top', ha='left', 
#                       color='black', path_effects=[patheffects.withStroke(linewidth=1,foreground="white")])
panel_label(axes[1], "(b)", loc="tl", fontsize=12, pad=4)



axes[0].set_title(r"$T_n$ estimation", fontsize=8)
axes[0].set_xticks([0.0, 0.5, 1.0])
axes[0].set_xticklabels(["0", "0.5", "1"])
axes[0].set_xlabel(r"$\beta_m$")
axes[0].set_yticks([0.0, 0.5, 1.0])
axes[0].set_yticklabels(["0", "0.5", "1"])
axes[0].set_ylabel(r"$p_m$")

axes[1].set_title(r"$T_{n-1}$ estimation", fontsize=8)
axes[1].set_xticks([0.0, 0.5, 1.0])
axes[1].set_xticklabels(["0", "0.5", "1"])
axes[1].set_xlabel(r"$\beta_m$")

ax_bbox = axes[0].get_position()
ax_bounds = ax_bbox.bounds
x0 = ax_bounds[0]
y0 = ax_bounds[1]
w = ax_bounds[2]
ax_bbox = axes[1].get_position()
ax_bounds = ax_bbox.bounds
h = ax_bounds[1] + ax_bounds[3] - y0
cbar_ax1 = fig.add_axes([x0+w+0.01, y0, 0.025, h])
cbar = fig.colorbar(im1, cax=cbar_ax1)
# cbar_ax1.set_yticks([2.0, 4.0, 6.0, 8.0])
# cbar_ax1.set_yticklabels(["2", "4", "6", "8", "10", r"$\geq 12$"])
# cbar_ax1.set_ylabel("RMSE (s)", labelpad=2.0)

cbar_ax1.tick_params(axis="y", which="both", length=3.0, pad=2)

ax_bbox = axes[1].get_position()
ax_bounds = ax_bbox.bounds
x0 = ax_bounds[0]
y0 = ax_bounds[1]
w = ax_bounds[2]
ax_bbox = axes[1].get_position()
ax_bounds = ax_bbox.bounds
h = ax_bounds[1] + ax_bounds[3] - y0
cbar_ax2 = fig.add_axes([x0+w+0.01, y0, 0.025, h])
cbar = fig.colorbar(im2, cax=cbar_ax2)
# cbar_ax2.set_yticks([2.0, 4.0, 6.0, 8.0])
# cbar_ax2.set_yticklabels(["2", "4", "6", "8", "10", r"$\geq 12$"])
cbar_ax2.set_ylabel("RMSE (s)", labelpad=2.0)

cbar_ax2.tick_params(axis="y", which="both", length=3.0, pad=2)


outdir = r"Fig S4"
import os
os.makedirs(outdir, exist_ok=True)
fname = os.path.basename(os.path.normpath(outdir))
fig.savefig(os.path.join(outdir, fname + ".png"), dpi=600, bbox_inches="tight")
fig.savefig(os.path.join(outdir, fname + ".pdf"), bbox_inches="tight")
plt.close(fig)

