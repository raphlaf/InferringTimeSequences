# Figure 4

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


def lamb(t, a, c, tau, beta, x0=1.0):
    ret = a*(1.0 - np.exp(-t/tau)*(1.0 - beta*x0)) + c
    if (ret < 0.0): return 0.0 
    return ret

def fisher(t, a, c, tau, beta, x0=1.0):
    lam = lamb(t, a, c, tau, beta, x0)
    if (lam == 0.0): return 0.0
    if (tau == 0.0): return 0.0
    return (a + c - lam)**2/(tau**2*lam)


file_path_homogeneous = "Fig 4/data_homogeneous.jld2"
file_path_heterogeneous = "Fig 4/data_heterogeneous.jld2"

data_hom = h5py.File(file_path_homogeneous)
data_het = h5py.File(file_path_heterogeneous)

T_values = np.array(data_hom["T_values"]).T
N_values = np.array(data_hom["N_values"]).T
mean_results_hom = np.array(data_hom["mean_results"]).T
var_results_hom = np.array(data_hom["var_results"]).T
a_values_hom = np.array(data_hom["a_values"]).T
tau_values_hom = np.array(data_hom["tau_values"]).T

mean_results_het = np.array(data_het["mean_results"]).T
var_results_het = np.array(data_het["var_results"]).T
a_values_het = np.array(data_het["a_values"]).T
tau_values_het = np.array(data_het["tau_values"]).T

fig = plt.figure(figsize=(3.4252, 3.3))

axes = fig.subplots(2, 2, sharex=True, sharey="row")

nN = N_values.size
nT = T_values.size

b_hom = np.zeros((nN, nT))
rmse_hom = np.zeros((nN, nT))
crlb_hom = np.zeros((nN, nT))

b_het = np.zeros((nN, nT))
rmse_het = np.zeros((nN, nT))
crlb_het = np.zeros((nN, nT))

beta = 0.0
c = 0.0

axes[0, 0].set_yscale("log")
axes[0, 1].set_yscale("log")
axes[1, 0].set_yscale("log")
axes[1, 1].set_yscale("log")



axes[0, 0].set_ylabel(r"RMSE $(s)$")
axes[1, 0].set_ylabel(r"Relative RMSE")
axes[1, 0].set_xlabel(r"Time interval $(s)$")
axes[1, 1].set_xlabel(r"Time interval $(s)$")

# axes[0, 0].text(0.05, 0.96, r"(a)", transform=axes[0, 0].transAxes, fontsize=12, fontweight='bold', va='top', ha='left',
#                 color='black', path_effects=[patheffects.withStroke(linewidth=1,foreground="white")])
panel_label(axes[0, 0], "(a)", loc="tl", fontsize=12, pad=3)
# axes[0, 1].text(0.05, 0.96, r"(b)", transform=axes[0, 1].transAxes, fontsize=12, fontweight='bold', va='top', ha='left',
#                 color='black', path_effects=[patheffects.withStroke(linewidth=1,foreground="white")])
panel_label(axes[0, 1], "(b)", loc="tl", fontsize=12, pad=3)
# axes[1, 0].text(0.05, 0.96, r"(c)", transform=axes[1, 0].transAxes, fontsize=12, fontweight='bold', va='top', ha='left',
#                 color='black', path_effects=[patheffects.withStroke(linewidth=1,foreground="white")])
panel_label(axes[1, 0], "(c)", loc="tl", fontsize=12, pad=3)
# axes[1, 1].text(0.05, 0.96, r"(d)", transform=axes[1, 1].transAxes, fontsize=12, fontweight='bold', va='top', ha='left',
#                 color='black', path_effects=[patheffects.withStroke(linewidth=1,foreground="white")])
panel_label(axes[1, 1], "(d)", loc="tl", fontsize=12, pad=3)


for i in range(nN):
    b_hom[i, :] = T_values - mean_results_hom[i, :]
    rmse_hom[i, :] = np.sqrt( var_results_hom[i, :] + b_hom[i, :]**2 )
    crlb_hom[i, :] = [( np.sum( [fisher(T_values[j], a_values_hom[k], c, tau_values_hom[k], beta) for k in range(N_values[i])] ) )**(-0.5) for j in range(nT)]
    
    b_het[i, :] = T_values - mean_results_het[i, :]
    rmse_het[i, :] = np.sqrt( var_results_het[i, :] + b_het[i, :]**2 )
    crlb_het[i, :] = [( np.sum( [fisher(T_values[j], a_values_het[k], c, tau_values_het[k], beta) for k in range(N_values[i])] ) )**(-0.5) for j in range(nT)]

lw0 = 1.0
dlw = 0.5

labels = [r"$N = 1$", r"$N = 10$", r"$N = 100$", r"$N = 1,000$", r"$N = 10,000$"]

selected_N = [0, 2, 3]

for i in selected_N:
    lin = axes[0, 0].plot(T_values, rmse_hom[i, :], label=labels[i], lw=lw0+i*dlw)
    axes[0, 0].plot(T_values, crlb_hom[i, :], ls="--", color=lin[0].get_color(), lw=lw0+i*dlw)
    lin = axes[0, 1].plot(T_values, rmse_het[i, :], label=labels[i], lw=lw0+i*dlw)
    axes[0, 1].plot(T_values, crlb_het[i, :], ls="--", color=lin[0].get_color(), lw=lw0+i*dlw)

    lin = axes[1, 0].plot(T_values, rmse_hom[i, :]/T_values, label=labels[i], lw=lw0+i*dlw)
    axes[1, 0].plot(T_values, crlb_hom[i, :]/T_values, ls="--", color=lin[0].get_color(), lw=lw0+i*dlw)
    lin = axes[1, 1].plot(T_values, rmse_het[i, :]/T_values, label=labels[i], lw=lw0+i*dlw)
    axes[1, 1].plot(T_values, crlb_het[i, :]/T_values, ls="--", color=lin[0].get_color(), lw=lw0+i*dlw)

axes[0, 0].legend(loc="lower right")

axes[1, 0].set_xticks([0.0, 15.0, 30.0])
axes[1, 1].set_xticks([0.0, 15.0, 30.0])

axes[0, 0].set_ylim(1e-6, 5e2)
axes[0, 0].set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2])
axes[0, 0].set_yticklabels([r"$10^{-6}$", "", r"$10^{-4}$", "", r"$10^{-2}$", "", r"$10^{0}$", "", r"$10^{2}$"])
axes[0, 0].set_yticks([2e-6, 4e-6, 6e-6, 8e-6,
                       2e-5, 4e-5, 6e-5, 8e-5,
                       2e-4, 4e-4, 6e-4, 8e-4,
                       2e-3, 4e-3, 6e-3, 8e-3,
                       2e-2, 4e-2, 6e-2, 8e-2,
                       2e-1, 4e-1, 6e-1, 8e-1,
                       2e0, 4e0, 6e0, 8e0,
                       2e1, 4e1, 6e1, 8e1,
                       2e2, 4e2], minor=True)

axes[1, 0].set_ylim(5e-3, 2e1)
axes[1, 0].set_yticks([1e-2, 1e-1, 1e0, 1e1])
axes[1, 0].set_yticklabels([r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$", r"$10^{1}$"])
axes[1, 0].set_yticks([5e-3, 6e-3, 7e-3, 8e-3, 9e-3,
                       2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2,9e-2,
                       2e-1, 3e-1, 4e-1, 5e-1, 6e-1, 7e-1, 8e-1,9e-1,
                       2e0, 3e0, 4e0, 5e0, 6e0, 7e0, 8e0,9e0,
                       2e1], minor=True)

fig.subplots_adjust(left=0.2, right=0.95, bottom=0.165, top=0.95, wspace=0.2)




outdir = r"Fig 4"
import os
os.makedirs(outdir, exist_ok=True)
fname = os.path.basename(os.path.normpath(outdir))
fig.savefig(os.path.join(outdir, fname + ".png"), dpi=600, bbox_inches="tight")
fig.savefig(os.path.join(outdir, fname + ".pdf"), bbox_inches="tight")
plt.close(fig)

