# Figure 8

import matplotlib
import matplotlib.patches
import matplotlib.pyplot as plt
from matplotlib import patheffects
import numpy as np
import h5py
from scipy.stats import linregress

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
data = h5py.File("Fig 8/data.jld2")

det_values_t1 = np.array(data["det_values_t1"]).T
crlb_values_t1 = np.array(data["crlb_values_t1"]).T
beta_values_t1 = np.array(data["beta_values_t1"])
tau_values_t1 = np.array(data["tau_values_t1"])
t1 = np.float64(data["t1"])

n, _, _ = det_values_t1.shape
max_det_values = np.zeros(n)
min_crlb_values = np.zeros(n)
for i in range(n):
    indices = np.unravel_index(np.argmax(det_values_t1[i, :, :], keepdims=True), det_values_t1[i, :, :].shape)
    max_det_values[i] = np.max(det_values_t1[i, :, :])
    min_crlb_values[i] = np.sqrt(crlb_values_t1[i, indices[0], indices[1]])

print(beta_values_t1)
print(tau_values_t1)
print(max_det_values)
print(min_crlb_values)




beta_range = np.array(data["beta_range"])
tau_range = np.array(data["tau_range"])


fig = plt.figure(figsize=(3.4252, 4.4252))
axes = fig.subplots(3, 2, sharex=True, sharey=True)

im1 = axes[0, 0].imshow(det_values_t1[0, :, :].T, origin="lower",
                  extent=(beta_range[0], beta_range[-1], tau_range[0], tau_range[-1]))
axes[0, 0].scatter(beta_values_t1[0], tau_values_t1[0], s=30, color="r", marker="x")
# txt = axes[0, 0].text(0.05, 0.97, r"(a)", transform=axes[0, 0].transAxes, fontsize=12, fontweight='bold', va='top', ha='left', 
#                       color='black', path_effects=[patheffects.withStroke(linewidth=1,foreground="white")])
panel_label(axes[0, 0], "(a)", loc="tl", fontsize=12, pad=4)

im2 = axes[1, 0].imshow(det_values_t1[1, :, :].T, origin="lower",
                  extent=(beta_range[0], beta_range[-1], tau_range[0], tau_range[-1]))
axes[1, 0].scatter(beta_values_t1[1], tau_values_t1[1], s=30, color="r", marker="x")
# txt = axes[1, 0].text(0.05, 0.97, r"(b)", transform=axes[1, 0].transAxes, fontsize=12, fontweight='bold', va='top', ha='left', 
#                       color='black', path_effects=[patheffects.withStroke(linewidth=1,foreground="white")])
panel_label(axes[1, 0], "(b)", loc="tl", fontsize=12, pad=4)
axes[1, 0].plot([1e-3, 1e-3], [0.0, 60.0], ls="--", color="black")

im3 = axes[2, 0].imshow(det_values_t1[2, :, :].T, origin="lower",
                  extent=(beta_range[0], beta_range[-1], tau_range[0], tau_range[-1]))
axes[2, 0].scatter(beta_values_t1[2], tau_values_t1[2], s=30, color="r", marker="x")
# txt = axes[2, 0].text(0.05, 0.97, r"(c)", transform=axes[2, 0].transAxes, fontsize=12, fontweight='bold', va='top', ha='left', 
#                       color='black', path_effects=[patheffects.withStroke(linewidth=1,foreground="white")])
panel_label(axes[2, 0], "(c)", loc="tl", fontsize=12, pad=4)
beta2_range = np.linspace(np.exp(-t1/tau_values_t1[1])*beta_values_t1[1]+1e-3, 1.0, 100)
print(beta2_range)
axes[2, 0].plot([1e-3, 1e-3], [0.0, 60.0], ls="--", color="black")
axes[2, 0].plot(beta2_range, t1/(t1/tau_values_t1[1] - np.log(beta_values_t1[1]/beta2_range)), ls="--", color="black")


im4 = axes[0, 1].imshow(det_values_t1[3, :, :].T, origin="lower",
                  extent=(beta_range[0], beta_range[-1], tau_range[0], tau_range[-1]))
axes[0, 1].scatter(beta_values_t1[3], tau_values_t1[3], s=30, color="r", marker="x")
# txt = axes[0, 1].text(0.78, 0.96, r"(d)", transform=axes[0, 1].transAxes, fontsize=12, fontweight='bold', va='top', ha='left', 
#                       color='black', path_effects=[patheffects.withStroke(linewidth=1,foreground="white")])
panel_label(axes[0, 1], "(d)", loc="tl", fontsize=12, pad=4)
axes[0, 1].plot([1e-3, 1e-3], [0.0, 60.0], ls="--", color="black")
axes[0, 1].plot(beta2_range, t1/(t1/tau_values_t1[1] - np.log(beta_values_t1[1]/beta2_range)), ls="--", color="black")
beta3_range = np.linspace(np.exp(-t1/tau_values_t1[2])*beta_values_t1[2]+1e-3, 1.0, 100)
axes[0, 1].plot(beta3_range, t1/(t1/tau_values_t1[2] - np.log(beta_values_t1[2]/beta3_range)), ls="--", color="black")

im5 = axes[1, 1].imshow(det_values_t1[4, :, :].T, origin="lower",
                  extent=(beta_range[0], beta_range[-1], tau_range[0], tau_range[-1]))
axes[1, 1].scatter(beta_values_t1[4], tau_values_t1[4], s=30, color="r", marker="x")
# txt = axes[1, 1].text(0.78, 0.96, r"(e)", transform=axes[1, 1].transAxes, fontsize=12, fontweight='bold', va='top', ha='left', 
#                       color='black', path_effects=[patheffects.withStroke(linewidth=1,foreground="white")])
panel_label(axes[1, 1], "(e)", loc="tl", fontsize=12, pad=4)
axes[1, 1].plot([1e-3, 1e-3], [0.0, 60.0], ls="--", color="black")
axes[1, 1].plot(beta2_range, t1/(t1/tau_values_t1[1] - np.log(beta_values_t1[1]/beta2_range)), ls="--", color="black")
axes[1, 1].plot(beta3_range, t1/(t1/tau_values_t1[2] - np.log(beta_values_t1[2]/beta3_range)), ls="--", color="black")
beta4_range = np.linspace(np.exp(-t1/tau_values_t1[3])*beta_values_t1[3], 1.0, 200)
axes[1, 1].plot(beta4_range, t1/(t1/tau_values_t1[3] - np.log(beta_values_t1[3]/beta4_range)), ls="--", color="black")

im6 = axes[2, 1].imshow(det_values_t1[5, :, :].T, origin="lower",
                  extent=(beta_range[0], beta_range[-1], tau_range[0], tau_range[-1]))
axes[2, 1].scatter(beta_values_t1[5], tau_values_t1[5], s=30, color="r", marker="x")
# txt = axes[2, 1].text(0.78, 0.96, r"(f)", transform=axes[2, 1].transAxes, fontsize=12, fontweight='bold', va='top', ha='left', 
#                       color='black', path_effects=[patheffects.withStroke(linewidth=1,foreground="white")])
panel_label(axes[2, 1], "(f)", loc="tl", fontsize=12, pad=4)
axes[2, 1].plot([1e-3, 1e-3], [0.0, 60.0], ls="--", color="black")
axes[2, 1].plot(beta2_range, t1/(t1/tau_values_t1[1] - np.log(beta_values_t1[1]/beta2_range)), ls="--", color="black")
axes[2, 1].plot(beta3_range, t1/(t1/tau_values_t1[2] - np.log(beta_values_t1[2]/beta3_range)), ls="--", color="black")
axes[2, 1].plot(beta4_range, t1/(t1/tau_values_t1[3] - np.log(beta_values_t1[3]/beta4_range)), ls="--", color="black")
beta5_range = np.linspace(np.exp(-t1/tau_values_t1[4])*beta_values_t1[4], 1.0, 100)
axes[2, 1].plot(beta5_range, t1/(t1/tau_values_t1[4] - np.log(beta_values_t1[4]/beta5_range)), ls="--", color="black")



axes[0, 0].set_xticks([0.0, 0.5, 1.0])
axes[0, 0].set_yticks([0.0, 20.0, 40.0, 60.0])
axes[0, 0].set_xlim(0.0, 1.0)
axes[0, 0].set_ylim(0.0, 60.0)

axes[0, 0].set_ylabel(r"$\tau_n$ (s)")
axes[1, 0].set_ylabel(r"$\tau_n$ (s)")
axes[2, 0].set_ylabel(r"$\tau_n$ (s)")
axes[2, 0].set_xlabel(r"$\beta_n$")
axes[2, 1].set_xlabel(r"$\beta_n$")

axes[2, 0].set_xticklabels(["0", "0.5", "1"])
axes[2, 1].set_xticklabels(["0", "0.5", "1"])



fig.subplots_adjust(left=0.165, right=0.8, bottom=0.15, top=0.92)

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
cbar_ax.set_yticks([0.0, np.max(det_values_t1[0, :, :])])
cbar_ax.set_yticklabels(["0", "Max"])
cbar_ax.set_ylabel("FI determinant", labelpad=-20.0)
cbar_ax.tick_params(axis="y", which="both", length=0.0, pad=2)



outdir = r"Fig 8"
import os
os.makedirs(outdir, exist_ok=True)
fname = os.path.basename(os.path.normpath(outdir))
fig.savefig(os.path.join(outdir, fname + ".png"), dpi=600, bbox_inches="tight")
fig.savefig(os.path.join(outdir, fname + ".pdf"), bbox_inches="tight")
plt.close(fig)

