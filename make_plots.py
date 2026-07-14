import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.load("sweep_log.npz")
t = d["t"]
Fmin, Fmax = 2.0, 4.0

fig, axs = plt.subplots(3, 1, figsize=(7.5, 8.5), sharex=True)

# 1) along-board tracking
axs[0].plot(t, d["u_des"] * 1000, "k--", label="u desired (board frame)")
axs[0].plot(t, d["u_meas"] * 1000, "C0", label="u measured (board frame)")
axs[0].set_ylabel("along-surface position u [mm]")
axs[0].legend(loc="upper left", fontsize=8)
axs[0].set_title("Along-surface (moving-frame) tracking")

# 2) contact force vs target band
axs[1].axhspan(Fmin, Fmax, color="green", alpha=0.15, label="target band")
axs[1].plot(t, d["Fn"], "C3")
axs[1].set_ylabel("contact normal force Fn [N]")
axs[1].set_ylim(0, min(20, d["Fn"].max() * 1.05))
axs[1].legend(loc="upper right", fontsize=8)
axs[1].set_title("Contact-force regulation")

# 3) board tilt
axs[2].plot(t, np.degrees(d["phi"]), "C2")
axs[2].set_ylabel("board tilt phi [deg]")
axs[2].set_xlabel("time [s]")
axs[2].set_title("Board tilt (the moving target frame)")

for ax in axs:
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("tracking_force_tilt.png", dpi=160)

# separate: tracking error magnitude and task-space position error
fig2, ax = plt.subplots(figsize=(7.5, 3.2))
err_u_mm = (d["u_des"] - d["u_meas"]) * 1000
ax.plot(t, err_u_mm, "C0", label="along-surface error (u_des - u_meas) [mm]")
ax.plot(t, d["pos_err"] * 1000, "C1", alpha=0.7, label="2D task-space |error| [mm]")
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("time [s]")
ax.set_ylabel("error [mm]")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.set_title("Tracking error over time")
plt.tight_layout()
plt.savefig("tracking_error.png", dpi=160)

print("Saved tracking_force_tilt.png and tracking_error.png")
print(f"Steady-state (t>5s) mean Fn = {d['Fn'][t>5].mean():.2f} N, std = {d['Fn'][t>5].std():.2f} N")
print(f"Steady-state (t>5s) mean |u error| = {np.abs(err_u_mm[t>5]).mean():.1f} mm")
