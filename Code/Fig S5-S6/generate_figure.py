# Figure S5-S6True
# S6 is just a zoomed in version of S5

zoomed = True

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patheffects
import h5py

np.random.seed(1234)

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
    
file_name = "Fig S5-S6/data.jld2"  # beta = 0.5 ratio = 0.5

f = h5py.File(file_name, 'r')

train_output = np.asarray(f["train_output"]).T
print(train_output.shape)
test_output = np.asarray(f["test_output"]).T
print(test_output.shape)
train_rmse = np.asarray(f["train_rmse"])
print(train_rmse)
test_rmse = np.asarray(f["test_rmse"])
print(test_rmse)
train_pooled_states = np.asarray(f["train_pooled_states"]).T
print(train_pooled_states.shape)
test_pooled_states = np.asarray(f["test_pooled_states"]).T
print(test_pooled_states.shape)
train_rnn_states = np.asarray(f["train_rnn_states"]).T
print(train_rnn_states.shape)
test_rnn_states = np.asarray(f["test_rnn_states"]).T
print(test_rnn_states.shape)
output_data = np.asarray(f["output_data"]).T
print(output_data.shape)
input_data = np.asarray(f["input_data"]).T
print(input_data.shape)
events = np.asarray(f["events"]).T
print(events.shape)
event_ranges = np.asarray(f["event_ranges"])
time_intervals = np.asarray(f["time_intervals"])
memory_ratio = np.float64(f["memory_ratio"])
memory_value = np.float64(f["memory_value"])
Win = np.asarray(f["Win"]).T
print(Win.shape)
Wout = np.asarray(f["Wout"]).T
print(Wout.shape)

dt = 0.01

N_output, ntrain = train_output.shape
_, ntest = test_output.shape
nfull = time_intervals.size
ntransient = nfull - ntrain - ntest

N_input, total_steps = input_data.shape
N, train_steps = train_rnn_states.shape
_, test_steps = test_rnn_states.shape
transient_steps = total_steps - train_steps - test_steps

transient_trange = np.linspace(0.0, events[ntransient-1], transient_steps)
train_trange = np.linspace(events[ntransient-1], events[ntransient+ntrain-1], train_steps)
test_trange = np.linspace(events[ntransient+ntrain-1], events[-1], test_steps)

tstart, tend = 728.0, 762.0
if (zoomed):
    tstart, tend = 739.0, 744.0
indices_in_range = np.where((train_trange >= tstart) & (train_trange <= tend))[0]
wanted_train_trange = train_trange[indices_in_range]


subsampling = 5

train_input = input_data[:, transient_steps:transient_steps+train_steps]
test_input = input_data[:, transient_steps+train_steps:]
wanted_train_input = train_input[:, indices_in_range]
wanted_train_rnn_states = train_rnn_states[:, indices_in_range]

nrandom_cells = 2  # per population
random_cells = np.random.permutation(N//2)[:nrandom_cells]
random_cells = np.append(random_cells, np.random.permutation(N//2)[:nrandom_cells]+N//2)

train_start_i = 0
test_start_i = 0
train_show_len = train_steps-1
test_show_len = test_steps-1


train_output_trange = np.zeros(2*ntrain)
test_output_trange = np.zeros(2*ntest)

train_pooled_values = np.zeros((2*ntrain, N))
test_pooled_values = np.zeros((2*ntest, N))

train_output_values = np.zeros((2*ntrain, N_output))
test_output_values = np.zeros((2*ntest, N_output))

train_real_values = np.zeros((2*ntrain, N_output))
test_real_values = np.zeros((2*ntest, N_output))

train_output_trange[0] = events[ntransient-1]
train_output_trange[-1] = events[ntransient+ntrain-1]

train_pooled_values[0, :] = train_pooled_states[:, 0]
train_pooled_values[-1, :] = train_pooled_states[:, -1]

test_output_trange[0] = events[ntransient+ntrain-1]
test_output_trange[-1] = events[ntransient+ntrain+ntest-1]

test_pooled_values[0, :] = test_pooled_states[:, 0]
test_pooled_values[-1, :] = test_pooled_states[:, -1]

train_output_values[0, :] = train_output[:, 0]
train_output_values[-1, :] = train_output[:, -1]

test_output_values[0, :] = test_output[:, 0]
test_output_values[-1, :] = test_output[:, -1]

train_real_values[0, :] = output_data[:, 0]
train_real_values[-1, :] = output_data[:, ntrain-1]

test_real_values[0, :] = output_data[:, ntrain]
test_real_values[-1, :] = output_data[:, -1]

for i in range(ntrain-1):
    train_output_trange[2*i+1] = events[ntransient+i]
    train_output_trange[2*i+2] = events[ntransient+i]

    train_pooled_values[2*i+1, :] = train_pooled_states[:, i]
    train_pooled_values[2*i+2, :] = train_pooled_states[:, i+1]

    train_output_values[2*i+1, :] = train_output[:, i]
    train_output_values[2*i+2, :] = train_output[:, i+1]

    train_real_values[2*i+1, :] = output_data[:, i]
    train_real_values[2*i+2, :] = output_data[:, i+1]

for i in range(ntest-1):
    test_output_trange[2*i+1] = events[ntransient+ntrain+i]
    test_output_trange[2*i+2] = events[ntransient+ntrain+i]

    test_pooled_values[2*i+1, :] = test_pooled_states[:, i]
    test_pooled_values[2*i+2, :] = test_pooled_states[:, i+1]
    
    test_output_values[2*i+1, :] = test_output[:, i]
    test_output_values[2*i+2, :] = test_output[:, i+1]

    test_real_values[2*i+1, :] = output_data[:, ntrain+i]
    test_real_values[2*i+2, :] = output_data[:, ntrain+i+1]


before_tstart_idx = np.where(train_output_trange < tstart)[0][-1]
after_tend_idx = np.where(train_output_trange > tend)[0][1]
wanted_train_output_trange = train_output_trange[before_tstart_idx:after_tend_idx]
wanted_train_output_trange[0] = tstart
wanted_train_output_trange[-1] = tend
wanted_train_pooled_values = train_pooled_values[before_tstart_idx:after_tend_idx, :]
wanted_train_output_values = train_output_values[before_tstart_idx:after_tend_idx, :]
wanted_train_real_values = train_real_values[before_tstart_idx:after_tend_idx, :]



fig, axes = plt.subplots(3, 2, figsize=(4.4252, 5.0), sharex=True, sharey=False)

if not zoomed:
    im1 = axes[0, 0].imshow((Win.dot(wanted_train_input)), origin="lower",
                            extent=(wanted_train_trange[0], wanted_train_trange[-1], 1, N),
                            aspect="auto", vmin=0.0, vmax=0.75)
else:
    im1 = axes[0, 0].imshow((Win.dot(wanted_train_input)), origin="lower",
                            extent=(wanted_train_trange[0], wanted_train_trange[-1], 1, N),
                            aspect="auto", vmin=0.0, vmax=0.5)
axes[1, 0].plot(wanted_train_trange, (Win.dot(wanted_train_input))[random_cells, :].T)

centered_wanted_train_rnn_states = wanted_train_rnn_states - wanted_train_rnn_states.mean(axis=1, keepdims=True)

if not zoomed:
    im2 = axes[2, 0].imshow(centered_wanted_train_rnn_states, origin="lower",
                            extent=(wanted_train_trange[0], wanted_train_trange[-1], 1, N),
                            aspect="auto")
else:
    im2 = axes[2, 0].imshow(centered_wanted_train_rnn_states, origin="lower",
                            extent=(wanted_train_trange[0], wanted_train_trange[-1], 1, N),
                            aspect="auto", vmin=-0.1, vmax=0.1)

axes[0, 1].plot(wanted_train_trange, wanted_train_rnn_states[random_cells, :].T)
axes[1, 1].plot(wanted_train_output_trange, wanted_train_pooled_values[:, random_cells])
axes[2, 1].plot(wanted_train_output_trange, wanted_train_real_values[:, 0], label=r"$T_n$", color="tab:blue")
axes[2, 1].plot(wanted_train_output_trange, wanted_train_real_values[:, 1], label=r"$T_{n-1}$", color="tab:orange")

axes[2, 1].plot(wanted_train_output_trange, wanted_train_output_values[:, 0], ls=":", color="tab:blue")
axes[2, 1].plot(wanted_train_output_trange, wanted_train_output_values[:, 1], ls=":", color="tab:orange")

axes[0, 0].tick_params(axis="y", which="both", pad=1.5)
axes[1, 0].tick_params(axis="y", which="both", pad=1.5)
axes[2, 0].tick_params(axis="y", which="both", pad=1.5)
axes[0, 1].tick_params(axis="y", which="both", pad=1.5)
axes[1, 1].tick_params(axis="y", which="both", pad=1.5)
axes[2, 1].tick_params(axis="y", which="both", pad=1.5)

axes[0, 0].set_yticks([0, 125, 250])
axes[1, 0].set_yticks([0.0, 0.25, 0.5, 0.75])
# axes[1, 0].set_yticks([0.0, 0.25, 0.5])
axes[2, 0].set_yticks([0, 125, 250])
axes[0, 1].set_yticks([-0.5, 0.0, 0.5])
axes[1, 1].set_yticks([-5.0, 0.0, 5.0])
axes[2, 1].set_yticks([0.0, 10.0, 20.0, 30.0])

axes[2, 0].set_xticks([730.0, 745.0, 760.0])
if (zoomed):
    axes[2, 0].set_xticks([740.0, 743.0])

axes[0, 0].set_xlim(tstart, tend)
axes[1, 0].set_ylim(0.0, 0.75)
if zoomed:
    axes[1, 0].set_ylim(0.0, 0.5)
axes[1, 1].set_ylim(-6.0, 6.0)
axes[2, 1].set_ylim(0.0, 35.0)


# txt = axes[0, 0].text(0.05, 0.97, r"\textbf{A}", transform=axes[0, 0].transAxes, fontsize=12, fontweight='bold', va='top', ha='left', 
#                       color='black', path_effects=[patheffects.withStroke(linewidth=1,foreground="white")])
panel_label(axes[0, 0], "(a)", loc="tl", fontsize=12, pad=4)
# txt = axes[1, 0].text(0.05, 0.97, r"\textbf{B}", transform=axes[1, 0].transAxes, fontsize=12, fontweight='bold', va='top', ha='left', 
#                       color='black', path_effects=[patheffects.withStroke(linewidth=1,foreground="white")])
if not zoomed:
    panel_label(axes[1, 0], "(b)", loc="tr", fontsize=12, pad=4)
else:
    panel_label(axes[1, 0], "(b)", loc="tl", fontsize=12, pad=4)

# txt = axes[2, 0].text(0.05, 0.97, r"\textbf{C}", transform=axes[2, 0].transAxes, fontsize=12, fontweight='bold', va='top', ha='left', 
#                       color='black', path_effects=[patheffects.withStroke(linewidth=1,foreground="white")])
panel_label(axes[2, 0], "(c)", loc="tl", fontsize=12, pad=4)
# txt = axes[0, 1].text(0.05, 0.97, r"\textbf{D}", transform=axes[0, 1].transAxes, fontsize=12, fontweight='bold', va='top', ha='left', 
#                       color='black', path_effects=[patheffects.withStroke(linewidth=1,foreground="white")])
panel_label(axes[0, 1], "(d)", loc="tl", fontsize=12, pad=4)
# txt = axes[1, 1].text(0.05, 0.97, r"\textbf{E}", transform=axes[1, 1].transAxes, fontsize=12, fontweight='bold', va='top', ha='left', 
#                       color='black', path_effects=[patheffects.withStroke(linewidth=1,foreground="white")])
if not zoomed:
    panel_label(axes[1, 1], "(e)", loc="tr", fontsize=12, pad=4)
else:
    panel_label(axes[1, 1], "(e)", loc="tl", fontsize=12, pad=4)

# txt = axes[2, 1].text(0.05, 0.97, r"\textbf{F}", transform=axes[2, 1].transAxes, fontsize=12, fontweight='bold', va='top', ha='left', 
#                       color='black', path_effects=[patheffects.withStroke(linewidth=1,foreground="white")])
panel_label(axes[2, 1], "(f)", loc="tl", fontsize=12, pad=4)



axes[0, 0].set_ylabel("Neuron \#")
axes[1, 0].set_ylabel("Input signal")
axes[2, 0].set_ylabel("Neuron \#")
axes[2, 0].set_xlabel("Simulation time (s)")
axes[0, 1].set_ylabel("Neuron activity")
axes[1, 1].set_ylabel("Pooled activity")
axes[2, 1].set_ylabel("Time interval\nprediction (s)")
axes[2, 1].set_xlabel("Simulation time (s)")
axes[2, 1].legend()


fig.subplots_adjust(left=0.165, right=0.95, bottom=0.15, top=0.92, wspace=1.0)


ax_bbox = axes[0, 0].get_position()
ax_bounds = ax_bbox.bounds
x0 = ax_bounds[0]
y0 = ax_bounds[1]
w = ax_bounds[2]
h = ax_bounds[3]

cbar_ax = fig.add_axes([x0+w+0.01, y0, 0.015, h])
cbar = fig.colorbar(im1, cax=cbar_ax)
if not zoomed:
    cbar_ax.set_yticks([0.0, 0.25, 0.5, 0.75])
else:
    cbar_ax.set_yticks([0.0, 0.25, 0.5])
cbar_ax.tick_params(axis="y", which="both", length=3.0, pad=2)

ax_bbox = axes[2, 0].get_position()
ax_bounds = ax_bbox.bounds
x0 = ax_bounds[0]
y0 = ax_bounds[1]
w = ax_bounds[2]
h = ax_bounds[3]

cbar_ax = fig.add_axes([x0+w+0.01, y0, 0.015, h])
cbar = fig.colorbar(im2, cax=cbar_ax)
cbar_ax.tick_params(axis="y", which="both", length=3.0, pad=2)
if not zoomed:
    cbar_ax.set_yticks([-0.5, 0.0, 0.5])
    cbar_ax.set_yticklabels(["-0.5", "0.0", "0.5"])
else:
    cbar_ax.set_yticks([-0.1, 0.0, 0.1])
    cbar_ax.set_yticklabels(["-0.1", "0.0", "0.1"])


outdir = r"Fig S5-S6"
import os
os.makedirs(outdir, exist_ok=True)
fname = os.path.basename(os.path.normpath(outdir))
if not zoomed:
    fig.savefig(os.path.join(outdir, "Fig S5.png"), dpi=600, bbox_inches="tight")
    fig.savefig(os.path.join(outdir, "Fig S5.pdf"), bbox_inches="tight")
else:
    fig.savefig(os.path.join(outdir, "Fig S6.png"), dpi=600, bbox_inches="tight")
    fig.savefig(os.path.join(outdir, "Fig S6.pdf"), bbox_inches="tight")






plt.close(fig)

