# Figure S7

import matplotlib
from matplotlib import patheffects
import matplotlib.pyplot as plt
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

data_files = [h5py.File("Fig S7/calibration_tau_res=0.5.jld2"), 
              h5py.File("Fig S7/calibration_tau_res=1.0.jld2"),
              h5py.File("Fig S7/calibration_tau_res=2.0.jld2"),
              h5py.File("Fig S7/calibration_tau_res=5.0.jld2"),
              h5py.File("Fig S7/calibration_tau_res=10.0.jld2")]

tau_res_values = [0.5, 1.0, 2.0, 5.0, 10.0]

tau_s_range = np.asarray(data_files[0]["tau_s_range"])
scale_range = np.asarray(data_files[0]["scale_range"])

rmse_data = [np.asarray(f["rmse_values"]).T for f in data_files]

n = 1  # 0 for T_n ; 1 for T_{n-1}

fig = plt.figure(figsize=(3.425, 4.5))
axes = fig.subplots(3, 2, sharex=False, sharey=False)
fig.subplots_adjust(left=0.25, right=0.9, bottom=0.15, top=0.92, wspace=0.3, hspace=0.3)

combined = np.zeros((len(scale_range), len(tau_s_range)*len(data_files)))
tau_ratio = np.zeros(len(tau_s_range)*len(data_files))

for (i, rmse) in enumerate(rmse_data):
    xind = (i+1)//3
    yind = (i+1) % 3
    for (j, scale) in enumerate(scale_range):
        # axes[yind, xind].plot(tau_s_range, rmse[j, :, n], label="S="+str(scale))
        axes[yind, xind].plot(tau_s_range/tau_res_values[i], rmse[j, :, n], label="S="+str(scale))
    axes[yind, xind].set_xscale("log")
    # axes[yind, xind].legend()
    if xind == 1:
        axes[yind, xind].set_yticklabels([])
    # axes[yind, xind].set_title(r"$\tau_{R}=$"+str(tau_res_values[i])+" s")
    # axes[yind, xind].set_xlabel(r"$\tau_{in}/\tau_R$ (s)")
    lab = r"$T_{n-1}$" if n==1 else r"$T_n$"

    for (j, tau_s) in enumerate(tau_s_range):
        tau_ratio[i*len(tau_s_range)+j] = tau_s / tau_res_values[i]
        combined[:, i*len(tau_s_range)+j] = rmse[:, j, n]

print(tau_ratio)

for (i, scale) in enumerate(scale_range):
    axes[0, 0].plot(tau_ratio, combined[i, :], ".", label="S="+str(scale))


# axes[2, 0].set_xlabel(r"$\tau_{in}$ (s)")
# axes[2, 1].set_xlabel(r"$\tau_{in}$ (s)")
axes[2, 0].set_xlabel(r"$\tau_{in}/\tau_R$")
axes[2, 1].set_xlabel(r"$\tau_{in}/\tau_R$")
axes[0, 0].set_ylabel(r"RMSE " + lab +" (s)")
axes[1, 0].set_ylabel(r"RMSE " + lab +" (s)")
axes[2, 0].set_ylabel(r"RMSE " + lab +" (s)")

# axes[0, 0].set_title("Combined")
# axes[0, 0].set_xlabel(r"$\tau_{in}/\tau_R$")
lab = r"$T_{n-1}$" if n==1 else r"$T_n$"
yup = 15.0 if n==1 else 10.0
axes[0, 0].set_xlim(0.5e-2, 1.5)
# axes[0, 0].set_ylim(None, yup)
axes[0, 0].set_xscale("log")
# axes[0, 0].legend()
axes[0, 1].legend(bbox_to_anchor=[0.1, 0.8], loc="upper left")

ymin, ymax = 10.0, 0.0

for ax in axes.flat:
    if ax is not axes[0, 0]:
        new_ymin, new_ymax = ax.get_ylim()
        print(new_ymin, ymin)
        print(new_ymax, ymax)
        if (new_ymin < ymin): ymin = new_ymin
        if (new_ymax > ymax): ymax = new_ymax

for ax in axes.flat:
    if ax is not axes[0, 0]:
        ax.set_ylim(ymin, ymax)

panel_label(axes[0, 0], "(a)", loc="tl", fontsize=12, pad=4)
panel_label(axes[1, 0], "(b)", loc="tl", fontsize=12, pad=4)
panel_label(axes[2, 0], "(c)", loc="tl", fontsize=12, pad=4)
panel_label(axes[0, 1], "(d)", loc="tl", fontsize=12, pad=4)
panel_label(axes[1, 1], "(e)", loc="tl", fontsize=12, pad=4)
panel_label(axes[2, 1], "(f)", loc="tl", fontsize=12, pad=4)


outdir = r"Fig S7"
import os
os.makedirs(outdir, exist_ok=True)
fname = os.path.basename(os.path.normpath(outdir))
fig.savefig(os.path.join(outdir, fname + ".png"), dpi=600, bbox_inches="tight")
fig.savefig(os.path.join(outdir, fname + ".pdf"), bbox_inches="tight")
plt.close(fig)

