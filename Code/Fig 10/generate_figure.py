# Figure 10

import matplotlib
from matplotlib import patheffects
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
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

data_ff = h5py.File("Fig 10/data_mlp.jld2")
data_re = h5py.File("Fig 10/data_rc.jld2")

beta_range_ff = np.asarray(data_ff["beta_range"]).T
ratio_range_ff = np.asarray(data_ff["ratio_range"]).T
rmse_values_ff = np.asarray(data_ff["rmse_values"]).T

beta_range_re = np.asarray(data_re["beta_range"]).T
ratio_range_re = np.asarray(data_re["ratio_range"]).T
rmse_values_re = np.asarray(data_re["rmse_values"]).T


fig = plt.figure(figsize=(3.4252, 3.3))
axes = fig.subplots(2, 2, sharex="col", sharey="row")
fig.subplots_adjust(left=0.21, right=0.8, bottom=0.3, top=0.92, wspace=0.4)

vmin1, vmax1 = 1.0, 12.0
im1 = axes[0, 0].imshow(rmse_values_ff[:, :, 0].T, origin="lower", aspect="auto",
                        extent=(beta_range_ff[0]-0.05, beta_range_ff[-1]+0.05,
                                ratio_range_ff[0]-0.05, ratio_range_ff[-1]+0.05),
)#vmin=vmin1, vmax=vmax1)
# txt = axes[0, 0].text(0.05, 0.97, r"(a)", transform=axes[0, 0].transAxes, fontsize=12, fontweight='bold', va='top', ha='left', 
#                       color='black', path_effects=[patheffects.withStroke(linewidth=1,foreground="white")])
panel_label(axes[0, 0], "(a)", loc="tl", fontsize=12, pad=3)

chance_level = 8.6

mask = rmse_values_ff[:, :, 0].T < chance_level
for j in range(rmse_values_ff[:, :, 0].T.shape[1]-1):
    col1 = mask[:, j]
    col2 = mask[:, j+1]
    diff = col1 != col2
    if np.any(diff):
        ys = ratio_range_ff[diff]
        for y in ys:
            axes[0, 0].plot([beta_range_ff[j]+0.05, beta_range_ff[j]+0.05], [y-0.05, y+0.05], 'r-', alpha=0.8)

# horizontal edges
for i in range(rmse_values_ff[:, :, 0].T.shape[0]-1):
    row1 = mask[i, :]
    row2 = mask[i+1, :]
    diff = row1 != row2
    if np.any(diff):
        xs = beta_range_ff[diff]
        for x in xs:
            axes[0, 0].plot([x-0.05, x+0.05], [ratio_range_ff[i]+0.05, ratio_range_ff[i]+0.05], 'r-', alpha=0.8)


im2 = axes[0, 1].imshow(rmse_values_ff[:, :, 1].T, origin="lower", aspect="auto", 
                        extent=(beta_range_ff[0]-0.05, beta_range_ff[-1]+0.05,
                                ratio_range_ff[0]-0.05, ratio_range_ff[-1]+0.05),
                                vmax=12
)#vmin=vmin1, vmax=vmax1)
# txt = axes[0, 1].text(0.05, 0.97, r"(b)", transform=axes[0, 1].transAxes, fontsize=12, fontweight='bold', va='top', ha='left', 
#                       color='black', path_effects=[patheffects.withStroke(linewidth=1,foreground="white")])
panel_label(axes[0, 1], "(b)", loc="tl", fontsize=12, pad=3)
mask = rmse_values_ff[:, :, 1].T < chance_level
for j in range(rmse_values_ff[:, :, 1].T.shape[1]-1):
    col1 = mask[:, j]
    col2 = mask[:, j+1]
    diff = col1 != col2
    if np.any(diff):
        ys = ratio_range_ff[diff]
        for y in ys:
            axes[0, 1].plot([beta_range_ff[j]+0.05, beta_range_ff[j]+0.05], [y-0.05, y+0.05], 'r-', alpha=0.8)

# horizontal edges
for i in range(rmse_values_ff[:, :, 1].T.shape[0]-1):
    row1 = mask[i, :]
    row2 = mask[i+1, :]
    diff = row1 != row2
    if np.any(diff):
        xs = beta_range_ff[diff]
        for x in xs:
            axes[0, 1].plot([x-0.05, x+0.05], [ratio_range_ff[i]+0.05, ratio_range_ff[i]+0.05], 'r-', alpha=0.8)


vmin2, vmax2 = np.min(rmse_values_re), 8.0
im3 = axes[1, 0].imshow(rmse_values_re[:, :, 0].T, origin="lower", aspect="auto", 
                        extent=(beta_range_re[0]-0.05, beta_range_re[-1]+0.05,
                                ratio_range_re[0]-0.05, ratio_range_re[-1]+0.05),
                        # norm=LogNorm())
)#vmin=vmin2, vmax=vmax2)
# txt = axes[1, 0].text(0.05, 0.97, r"(c)", transform=axes[1, 0].transAxes, fontsize=12, fontweight='bold', va='top', ha='left', 
#                       color='black', path_effects=[patheffects.withStroke(linewidth=1,foreground="white")])
panel_label(axes[1, 0], "(c)", loc="tl", fontsize=12, pad=3)
mask = rmse_values_re[:, :, 0].T < chance_level
for j in range(rmse_values_re[:, :, 0].T.shape[1]-1):
    col1 = mask[:, j]
    col2 = mask[:, j+1]
    diff = col1 != col2
    if np.any(diff):
        ys = ratio_range_ff[diff]
        for y in ys:
            axes[1, 0].plot([beta_range_ff[j]+0.05, beta_range_ff[j]+0.05], [y-0.05, y+0.05], 'r-', alpha=0.8)

# horizontal edges
for i in range(rmse_values_re[:, :, 0].T.shape[0]-1):
    row1 = mask[i, :]
    row2 = mask[i+1, :]
    diff = row1 != row2
    if np.any(diff):
        xs = beta_range_ff[diff]
        for x in xs:
            axes[1, 0].plot([x-0.05, x+0.05], [ratio_range_ff[i]+0.05, ratio_range_ff[i]+0.05], 'r-', alpha=0.8)

im4 = axes[1, 1].imshow(rmse_values_re[:, :, 1].T, origin="lower", aspect="auto", 
                        extent=(beta_range_re[0]-0.05, beta_range_re[-1]+0.05,
                                ratio_range_re[0]-0.05, ratio_range_re[-1]+0.05),
                        # norm=LogNorm())
)#vmin=vmin2, vmax=vmax2)
# txt = axes[1, 1].text(0.05, 0.97, r"(d)", transform=axes[1, 1].transAxes, fontsize=12, fontweight='bold', va='top', ha='left', 
#                       color='black', path_effects=[patheffects.withStroke(linewidth=1,foreground="white")])
panel_label(axes[1, 1], "(d)", loc="tl", fontsize=12, pad=3)
mask = rmse_values_re[:, :, 1].T < chance_level
for j in range(rmse_values_re[:, :, 1].T.shape[1]-1):
    col1 = mask[:, j]
    col2 = mask[:, j+1]
    diff = col1 != col2
    if np.any(diff):
        ys = ratio_range_ff[diff]
        for y in ys:
            axes[1, 1].plot([beta_range_ff[j]+0.05, beta_range_ff[j]+0.05], [y-0.05, y+0.05], 'r-', alpha=0.8)

# horizontal edges
for i in range(rmse_values_re[:, :, 1].T.shape[0]-1):
    row1 = mask[i, :]
    row2 = mask[i+1, :]
    diff = row1 != row2
    if np.any(diff):
        xs = beta_range_ff[diff]
        for x in xs:
            axes[1, 1].plot([x-0.05, x+0.05], [ratio_range_ff[i]+0.05, ratio_range_ff[i]+0.05], 'r-', alpha=0.8)





axes[0, 0].set_title(r"$T_n$ estimation", fontsize=8)
axes[0, 1].set_title(r"$T_{n-1}$ estimation", fontsize=8)
axes[0, 0].set_yticks([0.0, 0.5, 1.0])
axes[0, 0].set_yticklabels(["0", "0.5", "1"])
axes[0, 0].set_ylabel("MLP\n" + r"$p_m$")

axes[1, 0].set_xticks([0.0, 0.5, 1.0])
axes[1, 0].set_xticklabels(["0", "0.5", "1"])
axes[1, 0].set_xlabel(r"$\beta_m$")
axes[1, 0].set_yticks([0.0, 0.5, 1.0])
axes[1, 0].set_yticklabels(["0", "0.5", "1"])
axes[1, 0].set_ylabel("RC\n" + r"$p_m$")

axes[1, 1].set_xticks([0.0, 0.5, 1.0])
axes[1, 1].set_xticklabels(["0", "0.5", "1"])
axes[1, 1].set_xlabel(r"$\beta_m$")


ax_bbox = axes[0, 0].get_position()
ax_bounds = ax_bbox.bounds
x0 = ax_bounds[0]
y0 = ax_bounds[1]
w = ax_bounds[2]
h = ax_bounds[1] + ax_bounds[3] - y0
cbar_ax1 = fig.add_axes([x0+w+0.01, y0, 0.025, h])
cbar = fig.colorbar(im1, cax=cbar_ax1)
cbar_ax1.set_yticks([2, 4, 6, 8, 10, 12])
cbar_ax1.set_yticklabels(["2", "4", "6", "8", "10", "12"])
# cbar_ax1.set_ylabel("RMSE (s)", labelpad=0.0)
xmin, xmax = cbar_ax1.get_xlim()
cbar_ax1.plot([xmin, xmax], [chance_level, chance_level], color="red")
cbar_ax1.tick_params(axis="y", which="both", length=3.0, pad=2)

print(np.max(rmse_values_ff[:, :, 1]))
ax_bbox = axes[0, 1].get_position()
ax_bounds = ax_bbox.bounds
x0 = ax_bounds[0]
y0 = ax_bounds[1]
w = ax_bounds[2]
h = ax_bounds[1] + ax_bounds[3] - y0
cbar_ax2 = fig.add_axes([x0+w+0.01, y0, 0.025, h])
cbar = fig.colorbar(im2, cax=cbar_ax2)
cbar_ax2.set_yticks([6, 8, 10, 12])
cbar_ax2.set_yticklabels(["6", "8", "10", "12"])
cbar_ax2.set_ylabel("RMSE (s)", labelpad=0.0)
xmin, xmax = cbar_ax2.get_xlim()
cbar_ax2.plot([xmin, xmax], [chance_level, chance_level], color="red")
cbar_ax2.tick_params(axis="y", which="both", length=3.0, pad=2)
xmin, xmax = cbar_ax1.get_xlim()

ax_bbox = axes[1, 0].get_position()
ax_bounds = ax_bbox.bounds
x0 = ax_bounds[0]
y0 = ax_bounds[1]
w = ax_bounds[2]
h = ax_bounds[1] + ax_bounds[3] - y0
cbar_ax3 = fig.add_axes([x0+w+0.01, y0, 0.025, h])
cbar = fig.colorbar(im3, cax=cbar_ax3)
cbar_ax3.set_yticks([2, 4])
cbar_ax3.set_yticklabels(["2", "4"])
# cbar_ax3.set_ylabel("RMSE (s)", labelpad=0.0)
cbar_ax3.tick_params(axis="y", which="both", length=3.0, pad=2)


ax_bbox = axes[1, 1].get_position()
ax_bounds = ax_bbox.bounds
x0 = ax_bounds[0]
y0 = ax_bounds[1]
w = ax_bounds[2]
h = ax_bounds[1] + ax_bounds[3] - y0
cbar_ax4 = fig.add_axes([x0+w+0.01, y0, 0.025, h])
cbar = fig.colorbar(im4, cax=cbar_ax4)
cbar_ax4.set_yticks([6, 8])
cbar_ax4.set_yticklabels(["6", "8"])
cbar_ax4.set_ylabel("RMSE (s)", labelpad=5.0)
cbar_ax4.tick_params(axis="y", which="both", length=3.0, pad=2)




outdir = r"Fig 10"
import os
os.makedirs(outdir, exist_ok=True)
fname = os.path.basename(os.path.normpath(outdir))
fig.savefig(os.path.join(outdir, fname + ".png"), dpi=600, bbox_inches="tight")
fig.savefig(os.path.join(outdir, fname + ".pdf"), bbox_inches="tight")
plt.close(fig)

