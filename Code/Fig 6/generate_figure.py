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

file_path = "Fig 6/data_abs.jld2"
file_pathrel = "Fig 6/data_rel.jld2"

data = h5py.File(file_path)
datarel = h5py.File(file_pathrel)

fig, axes = plt.subplots(2, 3, figsize=(7.09, 4.2))
fig.subplots_adjust(left=0.165, right=0.98, bottom=0.125, top=0.96, hspace=0.6, wspace=0.7)

ax_a, ax_b, ax_c = axes[0]
ax_d, ax_e, ax_f = axes[1]

def prior_power(t, Tmin, Tmax, k):
    a = 1/np.log(Tmax/Tmin)
    if (k != 1):
        a = (1-k)/(Tmax**(1-k) - Tmin**(1-k))
    return a*t**(-k)

def lognormal(x, mu, sigm):
    m = np.log(mu**2/np.sqrt(mu**2 + sigm**2))
    s2 = np.log(1 + sigm**2/mu**2)
    return np.exp(-(np.log(x)-m)**2/(2*s2))/(x*np.sqrt(s2*2*np.pi))

res = 400
tau_space = np.linspace(0.0, 60.0, res)
T_space = np.linspace(0.0, 30.0, res)
k_values = [1, 0.0, -1.0]
T_prior = np.zeros((len(k_values), res))

mean_tau = np.array(data["mean_tau"]).T
mu_values = mean_tau[:-1, mean_tau.shape[1]-1]
sigm_values = np.array(data["sigm_values"]).T
tau_prior = np.zeros((len(mu_values), res))

Tmin, Tmax = 0.1, 30.0
klabels = [r"$k=1$", r"$k=0$", r"$k=-1$"]

ax_a.set_prop_cycle(cycler("color", plt.cm.tab10.colors[3:]))
for (i, k) in enumerate(k_values):
    T_prior[i, :] = prior_power(T_space, Tmin, Tmax, k)
    ax_a.plot(T_space, T_prior[i, :], label=klabels[i])

for i in range(len(sigm_values)-1, -1, -1):
    tau_prior[i, :] = lognormal(tau_space, mu_values[i], sigm_values[i])
    ax_b.plot(tau_space, tau_prior[i, :])

tau_min = 0.1
tau_max = np.sqrt(12)*np.max(sigm_values) + tau_min
h = 1/(tau_max - tau_min)
ax_b.plot([0.0, tau_min, tau_min, tau_max, tau_max, tau_space[-1]], [0.0, 0.0, h, h, 0.0, 0.0], "k:")
ylim = ax_b.get_ylim()
ax_b.plot([mean_tau[-2, mean_tau.shape[1]-2], mean_tau[-2, mean_tau.shape[1]-2]], ylim, "k--")
ax_b.set_ylim(ylim)

ax_a.set_ylabel(r"$P_T(T)$")
ax_a.set_xlabel(r"$T$ $(s)$")
ax_a.set_xticks([0, 15, 30])
ax_a.set_ylim(-0.01, 0.2)
panel_label(ax_a, "(a)", loc="tr", fontsize=12, pad=3)
ax_a.legend(bbox_to_anchor=[0.46, 1.0], loc="upper center")

ax_b.set_ylabel(r"$P_\tau(\tau)$")
ax_b.set_xlabel(r"$\tau$ $(s)$")
panel_label(ax_b, "(b)", loc="tr", fontsize=12, pad=3)
ax_b.set_xticks([0.0, 30.0, 60.0])

k_values = np.array(data["k_values"]).T
xi_values = np.array(data["xi_values"]).T

for i in range(len(sigm_values)-1, -1, -1):
    mu_values = np.log(mean_tau[i, :]**2/np.sqrt(mean_tau[i, :]**2 + sigm_values[i]**2))
    sigma2 = np.log(1 + sigm_values[i]**2/mean_tau[i, :]**2)
    p = np.exp(mu_values - sigma2)
    med = np.exp(mu_values)
    ax_c.plot(k_values, mean_tau[i, :], label=r"$\sigma = "+str(sigm_values[i])+r"$ $s$")
    ax_d.plot(k_values, xi_values[i, :], label=r"$\sigma = "+str(sigm_values[i])+r"$ $s$")

ax_c.plot(k_values, mean_tau[-1, :], "k--", label=r"$\sigma = 0$ $s$")
ax_d.plot(k_values, xi_values[-1, :], "k--", label=r"$\sigma = 0$ $s$")
ax_d.plot(k_values, xi_values[-2, :], "k:", label=r"Uniform")

panel_label(ax_c, "(c)", loc="tr", fontsize=12, pad=3)
ax_c.set_ylabel(r"Optimal $\mu$ (s)")
ax_d.set_ylabel(r"$\left<\right.$""CRLB"r"$\left.\right>_{\tau,T}$ (s)")
ax_c.set_xlabel(r"$k$")
ax_d.set_xlabel(r"$k$")
panel_label(ax_d, "(d)", loc="tr", fontsize=12, pad=3)

k_values = np.array(datarel["k_values"]).T
xi_values = np.array(datarel["xi_values"]).T
mean_tau = np.array(datarel["mean_tau"]).T
sigm_labels=["$\sigma = 4$ $s$", "$\sigma = 8$ $s$", "$\sigma = 16$ $s$"]

for i in range(len(sigm_values)-1, -1, -1):
    mu_values = np.log(mean_tau[i, :]**2/np.sqrt(mean_tau[i, :]**2 + sigm_values[i]**2))
    sigma2 = np.log(1 + sigm_values[i]**2/mean_tau[i, :]**2)
    p = np.exp(mu_values - sigma2)
    med = np.exp(mu_values)
    ax_e.plot(k_values, mean_tau[i, :], label=sigm_labels[i])
    ax_f.plot(k_values, xi_values[i, :], label=sigm_labels[i])

ax_e.plot(k_values, mean_tau[-1, :], "k--", label=r"Single $\tau$")
ax_f.plot(k_values, xi_values[-1, :], "k--", label=r"Single $\tau$")
ax_f.plot(k_values, xi_values[-2, :], "k:", label=r"Uniform")

panel_label(ax_e, "(e)", loc="tr", fontsize=12, pad=3)
ax_e.set_ylabel(r"Optimal $\mu$ (s)")
ax_f.set_ylabel(r"rel-$\left<\right.$""CRLB"r"$\left.\right>_{\tau,T}$")
ax_e.set_xlabel(r"$k$")
ax_f.set_xlabel(r"$k$")
ax_f.set_ylim(0.6444855763502046, 1.6)
panel_label(ax_f, "(f)", loc="br", fontsize=12, pad=3)
ax_f.legend(bbox_to_anchor=[0.025, 1.0], loc="upper left")

outdir = r"Fig 6"
os.makedirs(outdir, exist_ok=True)
fname = os.path.basename(os.path.normpath(outdir))
fig.savefig(os.path.join(outdir, fname + ".png"), dpi=600, bbox_inches="tight")
fig.savefig(os.path.join(outdir, fname + ".pdf"), bbox_inches="tight")
plt.close(fig)
