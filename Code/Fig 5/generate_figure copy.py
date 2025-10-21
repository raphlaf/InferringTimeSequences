# Figure 5

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
        path_effects=[patheffects.withStroke(linewidth=0.8, foreground='white')],
    )
    defaults.update(kw)

    ax.annotate(
        label, xy=(x, y), xycoords='axes fraction',
        xytext=(dx, dy), textcoords='offset points',
        ha=ha, va=va, **defaults
    )

data = h5py.File("Fig 5/data.jld2")

fig, axes = plt.subplots(3, 2, figsize=(3.4252, 5.3))

res = data["res"]
tau_space = np.array(data["tau_space"]).T
T_space = np.array(data["T_space"]).T
T_min = data["T_min"]
T_max = data["T_max"]
k_values = np.array(data["k_values"])
RMSE_dirac_values = np.array(data["RMSE_dirac_values"]).T
RMSE_avg_dirac_values = np.array(data["RMSE_avg_dirac_values"]).T
mu_values_mu = np.array(data["mu_values_mu"]).T
sigm_values_mu = np.array(data["sigm_values_mu"]).T
RMSE_total_values_mu = np.array(data["RMSE_total_values_mu"]).T
mu_values_sigm = np.array(data["mu_values_sigm"]).T
sigm_values_sigm = np.array(data["sigm_values_sigm"]).T
RMSE_total_values_sigm = np.array(data["RMSE_total_values_sigm"]).T

starti, endi = 0, len(T_space)-1
startj, endj = 0, len(tau_space)-1

Tindex = 100
tauindex = 50
N = 1000  # number of cells

wanted_T = [6.0, 18.0, 30.0]
wanted_tau = [12.0, 36.0, 60.0]
Tindices = []
tauindices = []
for T in wanted_T:
    Tindices.append((np.abs(T_space - T)).argmin())
for tau in wanted_tau:
    tauindices.append((np.abs(tau_space - tau)).argmin())

tau_minima_indices = np.argmin(RMSE_dirac_values, axis=1)

levels = np.linspace(0.001, 1.0, 10)
con = axes[0, 0].contour(T_space[starti:endi], tau_space[startj:endj],
                         RMSE_dirac_values[starti:endi, startj:endj].T/np.sqrt(N),
                         origin="lower", levels=levels,
                         colors="black", linestyles=":", alpha=0.5)

D = RMSE_dirac_values[starti:endi, startj:endj].T/np.sqrt(N)
D = np.where(D > 1.0, 1.0, D)
im = axes[0, 0].imshow(D,
                       origin="lower",
                       extent=(T_space[starti], T_space[endi], tau_space[startj], tau_space[endj]),
                       vmin=0.0, vmax=1.0)
axes[0, 0].plot(T_space[starti:endi], 1.5533*T_space[starti:endi], "k--")
# axes[0, 0].text(0.08, 0.97, r"(a)", transform=axes[0, 0].transAxes, fontsize=12, fontweight='bold', va='top', ha='left',
#                 color='black', path_effects=[patheffects.withStroke(linewidth=1,foreground="white")])
panel_label(axes[0, 0], "(a)", loc="tl", fontsize=12, pad=3)

fig.subplots_adjust(left=0.2, right=0.95, bottom=0.15, top=0.85, wspace=0.7, hspace=0.45)

ax_bbox = axes[0, 0].get_position()

ax_bounds = ax_bbox.bounds
x0 = ax_bounds[0]
w = ax_bounds[2]

cbar_ax = fig.add_axes([x0, 0.86, w, 0.025])

cbar = fig.colorbar(im, cax=cbar_ax, label=r"CRLB $(s)$", orientation="horizontal")
cbar_ax.set_xticks([0.0, 0.5, 1.0])
cbar_ax.set_xticklabels([r"$0$", r"$0.5$", r"$\geq1$"])
cbar_ax.xaxis.set_ticks_position("top")
cbar_ax.xaxis.set_label_position("top")
cbar_ax.tick_params(axis="x", pad=0.5)


axes[1, 0].set_prop_cycle(cycler("color", plt.cm.tab10.colors))
labels = [r"$\tau = 12$ s", r"$\tau = 36$ s", r"$\tau = 60$ s"]
for (i, j) in enumerate(tauindices):
    axes[1, 0].plot(T_space[starti:endi], RMSE_dirac_values[starti:endi, j]/np.sqrt(N), label=labels[i])
# axes[1, 0].text(0.08, 0.97, r"(b)", transform=axes[1, 0].transAxes, fontsize=12, fontweight='bold', va='top', ha='left',
#                 color='black', path_effects=[patheffects.withStroke(linewidth=1,foreground="white")])
panel_label(axes[1, 0], "(b)", loc="tl", fontsize=12, pad=3)
axes[1, 0].legend(bbox_to_anchor=[1.025, 0.555], loc="upper right")

axes[2, 0].set_prop_cycle(cycler("color", plt.cm.tab10.colors[len(tauindices):]))
labels = [r"$T = 6$ s", r"$T = 18$ s", r"$T = 30$ s"]
for (j, i) in enumerate(Tindices):
    axes[2, 0].plot(tau_space[startj:endj], RMSE_dirac_values[i, startj:endj]/np.sqrt(N), label=labels[j])
# axes[2, 0].text(0.80, 0.97, r"(c)", transform=axes[2, 0].transAxes, fontsize=12, fontweight='bold', va='top', ha='left', zorder=40,
#                 color='black', path_effects=[patheffects.withStroke(linewidth=1,foreground="white")])
panel_label(axes[2, 0], "(c)", loc="tr", fontsize=12, pad=3)
axes[2, 0].legend(bbox_to_anchor=[1.025, 0.78], loc="upper right")
startj, endj = 10, -1

axes[0, 1].set_prop_cycle(cycler("color", plt.cm.Set1.colors))
labels = [r"$k=-1$", r"$k=0$", r"$k=1$"]
for (i, k) in enumerate(k_values):
    axes[0, 1].plot(tau_space[startj:endj], RMSE_avg_dirac_values[startj:endj, i]/np.sqrt(N), label=labels[i])
# axes[0, 1].text(0.80, 0.97, r"(d)", transform=axes[0, 1].transAxes, fontsize=12, fontweight='bold', va='top', ha='left',
#                 color='black', path_effects=[patheffects.withStroke(linewidth=1,foreground="white")])
panel_label(axes[0, 1], "(d)", loc="tr", fontsize=12, pad=3)
axes[0, 1].legend(bbox_to_anchor=[1.025, 0.8], loc="upper right")

axes[1, 1].set_prop_cycle(cycler("color", plt.cm.Set1.colors[len(k_values):]))
labels = [r"$\mu = 1$ s", r"$\mu = 10$ s", r"$\mu = 30$ s"]
for i in range(len(mu_values_mu)):
    axes[1, 1].plot(T_space, RMSE_total_values_mu[:, i]/np.sqrt(N), label=labels[i])
# axes[1, 1].text(0.08, 0.97, r"(e)", transform=axes[1, 1].transAxes, fontsize=12, fontweight='bold', va='top', ha='left',
#                 color='black', path_effects=[patheffects.withStroke(linewidth=1,foreground="white")])
panel_label(axes[1, 1], "(e)", loc="tr", fontsize=12, pad=3)
axes[1, 1].legend(bbox_to_anchor=[1.025, 0.555], loc="upper right")

axes[2, 1].set_prop_cycle(cycler("color", plt.cm.Set1.colors[len(k_values)+len(mu_values_mu):]))
labels = [r"$\sigma = 5$ s", r"$\sigma = 15$ s", r"$\sigma = 50$ s"]
for i in range(len(mu_values_sigm)):
    axes[2, 1].plot(T_space, RMSE_total_values_sigm[:, i]/np.sqrt(N), label=labels[i])
# axes[2, 1].text(0.08, 0.97, r"(f)", transform=axes[2, 1].transAxes, fontsize=12, fontweight='bold', va='top', ha='left',
#                 color='black', path_effects=[patheffects.withStroke(linewidth=1,foreground="white")])
panel_label(axes[2, 1], "(f)", loc="tl", fontsize=12, pad=3)
axes[2, 1].legend(bbox_to_anchor=[1.025, 0.555], loc="upper right")

axes[0, 0].set_xlabel(r"$T$ (s)")
axes[0, 0].set_ylabel(r"$\tau$ (s)")
axes[0, 0].set_xlim(0.0, 30.0)
axes[0, 0].set_ylim(0.0, 30.0)
axes[0, 0].set_xticks([0.0, 15.0, 30.0])
axes[0, 0].set_yticks([15.0, 30.0])
axes[0, 0].tick_params(axis="both", pad=2.0)

axes[0, 1].set_xlabel(r"$\tau$ $(s)$")
axes[0, 1].set_ylabel(r"$\left<\right.$" "CRLB" r"$\left.\right>_{T}$ $(s)$")
axes[0, 1].set_yscale("log")
axes[0, 1].set_xlim(0.0, 30.0)
axes[0, 1].set_ylim(8e-2, 5e2)
axes[0, 1].set_xticks([0, 15, 30])
axes[0, 1].set_yticks([1e-1, 1e0, 1e1, 1e2])
axes[0, 1].set_yticklabels([r"", "$10^{0}$", r"", r"$10^{2}$"])
axes[0, 1].set_yticks([9e-2,
                       2e-1, 3e-1, 4e-1, 5e-1, 6e-1, 7e-1, 8e-1, 9e-1,
                       2e0, 3e0, 4e0, 5e0, 6e0, 7e0, 8e0, 9e0,
                       2e1, 3e1, 4e1, 5e1, 6e1, 7e1, 8e1, 9e1,
                       2e2, 3e2, 4e2], minor=True)
axes[0, 1].tick_params(axis="both", pad=2.0)

axes[1, 0].set_xlabel(r"$T$ (s)")
axes[1, 0].set_ylabel(r"CRLB (s)")
axes[1, 0].set_yscale("log")
axes[1, 0].set_xlim(0.0, 30.0)
axes[1, 0].set_ylim(5e-3, 2e0)
axes[1, 0].set_xticks([0, 15, 30])
axes[1, 0].set_yticks([1e-2, 1e-1, 1e0])
axes[1, 0].set_yticklabels([r"$10^{-2}$", "$10^{-1}$", r"$10^{0}$"])
axes[1, 0].set_yticks([5e-3, 6e-3, 7e-3, 8e-3, 9e-3,
                       2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2,
                       2e-1, 3e-1, 4e-1, 5e-1, 6e-1, 7e-1, 8e-1, 9e-1,
                       2e0], minor=True)
axes[1, 0].tick_params(axis="both", pad=2.0)

axes[1, 1].set_xlabel(r"$T$ $(s)$")
axes[1, 1].set_ylabel(r"$\left<\right.$" "CRLB" r"$\left.\right>_{\tau}$ $(s)$")
axes[1, 1].set_yscale("log")
axes[1, 1].set_ylim(1.5e-4, 3e1)
axes[1, 1].set_xticks([0, 15, 30])
axes[1, 1].set_yticks([1e-3, 1e-2, 1e-1, 1e0, 1e1])
axes[1, 1].set_yticklabels([r"$10^{-3}$", r"", r"$10^{-1}$", r"", r"$10^{1}$"])
axes[1, 1].set_yticks([2e-4, 4e-4, 6e-4, 8e-4,
                       2e-3, 4e-3, 6e-3, 8e-3,
                       2e-2, 4e-2, 6e-2, 8e-2,
                       2e-1, 4e-1, 6e-1, 8e-1,
                       2e0, 4e0, 6e0, 8e0,
                       2e1], minor=True)
axes[1, 1].tick_params(axis="both", pad=2.0)

axes[2, 0].set_xlabel(r"$\tau$ (s)")
axes[2, 0].set_ylabel(r"CRLB (s)")
axes[2, 0].set_yscale("log")
axes[2, 0].set_xlim(0.0, 30.0)
axes[2, 0].set_ylim(9e-2, 1e4)
axes[2, 0].set_xticks([0, 15, 30])
axes[2, 0].set_yticks([1e-1, 1e0, 1e1, 1e2, 1e3])
axes[2, 0].set_yticklabels([r"$10^{-1}$", r"", r"$10^{1}$", r"", r"$10^{3}$"])
axes[2, 0].set_yticks([2e-1, 4e-1, 6e-1, 8e-1,
                       2e0, 4e0, 6e0, 8e0,
                       2e1, 4e1, 6e1, 8e1,
                       2e2, 4e2, 6e2, 8e2,
                       2e3, 4e3, 6e3, 8e3,], minor=True)
axes[2, 0].tick_params(axis="both", pad=2.0)

axes[2, 1].set_xlabel(r"$T$ $(s)$")
axes[2, 1].set_ylabel(r"$\left<\right.$" "CRLB" r"$\left.\right>_{\tau}$ $(s)$")
axes[2, 1].set_yscale("log")
axes[2, 1].set_xlim(0.0, 30.0)
axes[2, 1].set_xticks([0.0, 15.0, 30.0])
axes[2, 1].set_ylim(4e-3, 2e0)
axes[2, 1].set_xticks([0, 15, 30])
axes[2, 1].set_yticks([1e-2, 1e-1, 1e0])
axes[2, 1].set_yticklabels([r"$10^{-2}$", r"$10^{-1}$",r"$10^{0}$"])
axes[2, 1].set_yticks([2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2,
                       2e-1, 3e-1, 4e-1, 5e-1, 6e-1, 7e-1, 8e-1, 9e-1,], minor=True)
axes[2, 1].tick_params(axis="both", pad=2.0)



outdir = r"Fig 5"
import os
os.makedirs(outdir, exist_ok=True)
fname = os.path.basename(os.path.normpath(outdir))
fig.savefig(os.path.join(outdir, fname + ".png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(outdir, fname + ".pdf"), bbox_inches="tight")
plt.close(fig)

