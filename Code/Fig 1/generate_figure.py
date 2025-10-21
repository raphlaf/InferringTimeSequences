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
import matplotlib.patches
import os

matplotlib.style.use("custom-style.mplstyle")

def panel_label(ax, label, loc="tr", pad=6, **kw):
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

time_intervals = np.array([0.0, 10.0, 5.0, 20.0])

def dynamical_variable(t, beta, tau, x0):
    return 1.0 - np.exp(-t/tau)*(1.0 - beta*x0)

def rectified_linear(x, a, c):
    lam = a*x + c
    return 0.5*(np.abs(lam) + lam)

def poisson_dist(lam, k):
    return np.exp(-lam)*lam**(k)/gamma(k+1)

n = len(time_intervals)-1
res = 100
x1 = np.zeros(res*n)
x2 = np.zeros(res*n)
x0 = 1.0
x1[0] = x0
x2[0] = x0
beta1 = 0.0
beta2 = 0.5
tau1 = 10.0
tau2 = 10.0
full_trange = np.zeros(res*n)
for i in range(n):
    x10 = x1[i*res-1]
    x20 = x2[i*res-1]
    if (i == 0): x10 = x20 = x0
    T = time_intervals[i+1]
    trange = np.linspace(0.0, T, res)
    full_trange[i*res:(i+1)*res] = trange + np.sum(time_intervals[0:i+1])
    x1values = dynamical_variable(trange, beta1, tau1, x10)
    x2values = dynamical_variable(trange, beta2, tau2, x20)
    x1[i*res:(i+1)*res] = x1values
    x2[i*res:(i+1)*res] = x2values

fig, axes = plt.subplots(2, 3, figsize=(7.09, 4.2))

fig.subplots_adjust(left=0.2, right=0.95, bottom=0.15, top=0.85, wspace=0.7, hspace=0.45)

ax_a, ax_b, ax_c = axes[0]
ax_d, ax_e, ax_f = axes[1]

ax_a.plot(full_trange, x1, lw=2.0, zorder=10)
ax_a.plot(full_trange, x2, lw=2.0, ls="-")
ax_a.set_xlim(0.0, None)
ax_a.set_ylim(0.0, 1.0)
ax_a.set_yticks([0.0, 1.0])
ax_a.set_ylabel(r"$x_n(t)$", labelpad=-5)
ax_a.set_xlabel(r"Time $t$ (s)")
ax_a.set_xticks([0.0, 10.0, 20.0, 30.0])
panel_label(ax_a, "(a)", loc="tl", fontsize=12, pad=3)
ax_a.arrow(10.0, 0.93, 0.0, -0.05, facecolor="Black", edgecolor="Black", width=0.6, zorder=2, head_length=0.035)
ax_a.arrow(15.0, 0.76, 0.0, -0.05, facecolor="Black", edgecolor="Black", width=0.6, zorder=2, head_length=0.035)
ax_a.text(0.45, 0.97, r"\boldmath$\beta = 0.5$", transform=ax_a.transAxes, fontsize=8, va='top', ha='left', color="tab:orange", path_effects=[patheffects.withStroke(linewidth=0.0,foreground="black")])
ax_a.text(0.57, 0.45, r"\boldmath$\beta = 0.0$", transform=ax_a.transAxes, fontsize=8, va='top', ha='left', color="tab:blue", path_effects=[patheffects.withStroke(linewidth=0.0,foreground="black")])

xspace = np.linspace(0.0, 1.0, 100)
ax_b.plot(xspace, rectified_linear(xspace, 10.0, -5.0), lw=2.0)
ax_b.set_yticks([])
ax_b.set_yticklabels([])
ax_b.set_xticks([0.0, 0.5, 1.0])
ax_b.set_xticklabels(["0", "0.5", "1"])
ax_b.set_ylabel(r"$\lambda_n(x_n)$", labelpad=10)
ax_b.set_xlabel(r"$x_n$")
ax_b.set_xlim(0.0, None)
ax_b.set_ylim(-0.05, None)
panel_label(ax_b, "(b)", loc="tl", fontsize=12, pad=3)

kspace = range(11)
barheights = [poisson_dist(4.0, k) for k in kspace]
ax_c.bar(kspace, barheights)
ax_c.set_xticks([0, 5, 10])
ax_c.tick_params(axis="x", which="both", length=0)
ax_c.set_yticks([])
ax_c.set_yticklabels([])
ax_c.set_ylabel(r"$P(R_n=k)$", labelpad=10)
ax_c.set_xlabel(r"Spike count $k$")
panel_label(ax_c, "(c)", loc="tl", fontsize=12, pad=3)

data = h5py.File("Fig 1/data.jld2")
Trange = np.array(data["Trange"])
nll_values = np.array(data["nll"]).T
(n, res) = nll_values.shape
FI_values = np.array(data["FI"]).T
tau_sweep = np.array(data["tau_sweep"]).T

inset_ax2 = inset_axes(ax_e, width="45%", height="45%", loc="upper right")
inset_ax3 = inset_axes(ax_f, width="45%", height="45%", loc="lower right")

lw = 1.0
dx = 300
xind = (np.abs(Trange - 10.0).argmin())
labels = [r"$\tau = 2$ s", r"$\tau = 10$ s", r"$\tau = 15.53$ s", r"$\tau = 20$ s", r"$\tau = 40$ s"]

selected_n = [0, 1, 3, 4]

for i in selected_n:
    avg_xn = dynamical_variable(Trange, 0.0, tau_sweep[i], 1.0)
    avg_lambda = rectified_linear(avg_xn, 10.0, 0.0)
    actual_xn = dynamical_variable(10.0, 0.0, tau_sweep[i], 1.0)
    actual_lambda = rectified_linear(actual_xn, 10.0, 0.0)
    avg_nll = -1000*(actual_lambda*np.log(avg_lambda) - avg_lambda - np.log(gamma(actual_lambda + 1.0)))
    y_spl = UnivariateSpline(Trange, nll_values[i, :], s=0, k=4)
    y_spl_2d = y_spl.derivative(n=2)
    ax_d.plot(Trange, -nll_values[i, :], lw=lw, label=labels[i])
    ax_e.plot(Trange, FI_values[i, :], lw=lw)
    ax_f.plot(Trange, 1/FI_values[i, :], lw=lw)
    rescaled_likelihood = -nll_values[i, xind-dx:xind+dx] - np.max(-nll_values[i, xind-dx:xind+dx])
    rescaled_likelihood = np.exp(rescaled_likelihood)
    ndx = dx//2
    inset_ax2.plot(Trange[xind-ndx:xind+ndx], FI_values[i, xind-ndx:xind+ndx])
    inset_ax3.plot(Trange[xind-ndx:xind+ndx], 1/FI_values[i, xind-ndx:xind+ndx])

inset_ax2.set_yticks([])
inset_ax2.set_yticklabels([])
inset_ax2.set_xticks([])
inset_ax2.set_xticklabels([])
inset_ax3.set_yticks([])
inset_ax3.set_yticklabels([])
inset_ax3.set_xticks([])
inset_ax3.set_xticklabels([])
inset_ax2.set_ylim(10**(1), 0.5*10**(2))
inset_ax3.set_ylim(2*10**(-2), 0.9*10**(-1))

ax_d.set_yscale("symlog")
ax_d.set_ylim(-3e4, -9e2)
ax_d.set_yticks([-1e3, -1e4])
ax_d.set_yticks([-2e3, -3e3, -4e3, -5e3, -6e3, -7e3, -8e3, -9e3, -2e4, -3e4], minor=True)
ax_d.set_xticks([0.0, 10.0, 20.0, 30.0])
ax_d.set_ylabel(r"Log-likelihood")
ax_d.set_xlabel(r"Time interval (s)")
panel_label(ax_d, "(d)", loc="tl", fontsize=12, pad=3)
ax_d.legend(loc="lower right")

xlim_in = inset_ax2.get_xlim()
ylim_in = inset_ax2.get_ylim()
xlim = ax_e.get_xlim()
ylim = ax_e.get_ylim()
rect = matplotlib.patches.Rectangle((xlim_in[0], ylim_in[0]), xlim_in[1]-xlim_in[0], ylim_in[1]-ylim_in[0], edgecolor="black", ls="-", lw=1.0, fill=False, zorder=5)
l2 = matplotlib.patches.ConnectionPatch((xlim_in[0], ylim_in[1]), (xlim_in[0], ylim_in[1]), coordsA = ax_e.transData, coordsB = inset_ax2.transData, color="black", zorder=5, lw=1.0, alpha=0.8)
l3 = matplotlib.patches.ConnectionPatch((xlim_in[1], ylim_in[0]), (xlim_in[1], ylim_in[0]), coordsA = ax_e.transData, coordsB = inset_ax2.transData, color="black", zorder=5, lw=1.0, alpha=0.8)
ax_e.add_patch(rect)
ax_e.add_artist(l2)
ax_e.add_artist(l3)

xlim_in = inset_ax3.get_xlim()
ylim_in = inset_ax3.get_ylim()
xlim = ax_f.get_xlim()
ylim = ax_f.get_ylim()
rect = matplotlib.patches.Rectangle((xlim_in[0], ylim_in[0]), xlim_in[1]-xlim_in[0], ylim_in[1]-ylim_in[0], edgecolor="black", ls="-", lw=1.0, fill=False, zorder=5)
l1 = matplotlib.patches.ConnectionPatch((xlim_in[0], ylim_in[0]), (xlim_in[0], ylim_in[0]), coordsA = ax_f.transData, coordsB = inset_ax3.transData, color="black", zorder=5, lw=1.0, alpha=0.8)
l4 = matplotlib.patches.ConnectionPatch((xlim_in[1], ylim_in[1]), (xlim_in[1], ylim_in[1]), coordsA = ax_f.transData, coordsB = inset_ax3.transData, color="black", zorder=5, lw=1.0, alpha=0.8)
ax_f.add_patch(rect)
ax_f.add_artist(l1)
ax_f.add_artist(l4)

ax_e.set_yscale("log")
ax_e.set_ylim(1e-1, 1e4)
ax_e.set_yticks([1e-1, 1e0, 1e1, 1e2, 1e3])
ax_e.set_yticklabels([r"$10^{-1}$", "", r"$10^{1}$", "", r"$10^{3}$"])
ax_e.set_yticks([2e-1, 3e-1, 4e-1, 5e-1, 6e-1, 7e-1, 8e-1, 9e-1, 2e0, 3e0, 4e0, 5e0, 6e0, 7e0, 8e0, 9e0, 2e1, 3e1, 4e1, 5e1, 6e1, 7e1, 8e1, 9e1, 2e2, 3e2, 4e2, 5e2, 6e2, 7e2, 8e2, 9e2, 2e3, 3e3, 4e3, 5e3, 6e3, 7e3, 8e3, 9e3], minor=True)
ax_e.set_xticks([0.0, 10.0, 20.0, 30.0])
ax_e.set_ylabel(r"Fisher information $(s^{-2})$")
ax_e.set_xlabel(r"Time interval (s)")
panel_label(ax_e, "(e)", loc="tl", fontsize=12, pad=3)

ax_f.set_xticks([0.0, 10.0, 20.0, 30.0])
ax_f.set_yscale("log")
ax_f.set_ylim(0.5e-4, 5e0)
ax_f.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0])
ax_f.set_yticklabels([r"$10^{-4}$", "", r"$10^{-2}$", "", r"$10^{0}$"])
ax_f.set_yticks([2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3, 2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2, 2e-1, 3e-1, 4e-1, 5e-1, 6e-1, 7e-1, 8e-1, 9e-1, 2e0, 3e0, 4e0, 5e0], minor=True)
ax_f.set_ylabel(r"CRLB $(s)$")
ax_f.set_xlabel(r"Time interval (s)")
panel_label(ax_f, "(f)", loc="tl", fontsize=12, pad=3)


outdir = r"Fig 1"
os.makedirs(outdir, exist_ok=True)
fname = os.path.basename(os.path.normpath(outdir))
fig.savefig(os.path.join(outdir, fname + ".png"), dpi=600, bbox_inches="tight")
fig.savefig(os.path.join(outdir, fname + ".pdf"), bbox_inches="tight")
plt.close(fig)
