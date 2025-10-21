# Figure S3

import matplotlib
import matplotlib.patches
import matplotlib.pyplot as plt
from matplotlib import patheffects
import numpy as np
import h5py
from scipy.stats import linregress

matplotlib.style.use("custom-style.mplstyle")

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

x = np.arange(1, n+1)
slope, intercept, r_value, p_value, std_err = linregress(x, np.log10(min_crlb_values))
print(f"Slope: {slope}, Intercept: {intercept}, R: {r_value}")

fig = plt.figure(figsize=(3.4252, 3.4252))
plt.plot(x, 10.0**(intercept + slope * x), "--", color="black", label="Linear fit")
plt.plot(x, min_crlb_values, "o", scaley="log10")
plt.yscale("log")
plt.ylabel("Minimum CRLB (s)")
plt.xlabel("Sequence length")
plt.xticks([1, 2, 3, 4, 5, 6])
plt.tight_layout()

outdir = r"Fig S3"
import os
os.makedirs(outdir, exist_ok=True)
fname = os.path.basename(os.path.normpath(outdir))
fig.savefig(os.path.join(outdir, fname + ".png"), dpi=600, bbox_inches="tight")
fig.savefig(os.path.join(outdir, fname + ".pdf"), bbox_inches="tight")
plt.close(fig)

