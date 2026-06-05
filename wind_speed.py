import numpy as np
import matplotlib.pyplot as plt
import rasterio
import os
import csv

# ==========================================
# CONFIG
# ==========================================
TIF_PATH = r"C:\wind_field_project\outputs\S1_preprocessed.tif"
OUTPUT_DIR = r"C:\wind_field_project\outputs"

# ==========================================
# CMOD5.n
# ==========================================

def cmod5n_forward(v, phi, theta):
    C = [0.0,
         -0.6878, -0.7957,  0.3380, -0.1728,  0.0000,
          0.0040,  0.1103,  0.0159,  6.7329,  2.7713,
         -2.2885,  0.3540, -0.0989,  0.0064,  0.1892,
          0.0123,  0.0495,  0.0101,  0.3701,  0.4100,
          0.3543, -0.1906,  0.0131,  0.0086,  0.0021]

    THETM  = 40.0
    THETHR = 25.0
    ZPOW   = 1.6

    FI  = (theta - THETM) / THETHR
    A0  =  C[1] + C[2]*FI + C[3]*FI**2 + C[4]*FI**3
    A1  =  C[5] + C[6]*FI
    A2  =  C[7] + C[8]*FI
    GAM =  C[9] + C[10]*FI + C[11]*FI**2

    B0  = (10.0**(-0.1*(A0 + A1*v))) * (1.0 + A2*v)**ZPOW
    B1  = (C[12] + C[13]*FI + C[14]*FI**2) * v * np.exp(-v / max(abs(GAM), 0.01))
    B2_A = C[18] + C[19]*FI + C[20]*FI**2
    B2_B = C[21] + C[22]*FI
    B2   = B2_A * np.tanh(B2_B * v)

    PHI_RAD = np.radians(phi)
    F = (1.0 + B1*np.cos(PHI_RAD) + B2*np.cos(2.0*PHI_RAD))
    F = np.maximum(F, 0.001)

    return B0 * F**ZPOW


def speed_from_sigma0(sigma0_obs, phi, theta):
    if np.isnan(sigma0_obs) or sigma0_obs <= 0:
        return np.nan

    v = 7.0
    for _ in range(30):
        f  = cmod5n_forward(v, phi, theta)
        dv = 0.01
        df = (cmod5n_forward(v+dv, phi, theta) - f) / dv
        if abs(df) < 1e-12:
            break
        v_new = v - (f - sigma0_obs) / df
        v_new = np.clip(v_new, 0.5, 35.0)
        if abs(v_new - v) < 1e-4:
            v = v_new
            break
        v = v_new

    check = cmod5n_forward(v, phi, theta)
    if abs(check - sigma0_obs) / (sigma0_obs + 1e-10) < 0.05:
        return float(v)

    speeds = np.linspace(0.5, 35.0, 3000)
    sigmas = np.array([cmod5n_forward(vv, phi, theta) for vv in speeds])
    idx = np.argmin(np.abs(sigmas - sigma0_obs))
    return float(speeds[idx])


# ==========================================
# Sanity check
# ==========================================
test_s0 = cmod5n_forward(7.0, 90.0, 35.0)
test_v  = speed_from_sigma0(test_s0, 90.0, 35.0)
print(f"Sanity check: v=7.0 m/s → sigma0={test_s0:.5f} → recovered v={test_v:.2f} m/s")
print(f"  Test sigma0=0.051: {speed_from_sigma0(0.051, 90.0, 35.0):.2f} m/s")

# ==========================================
# Step 1 - Load SAR data
# ==========================================
print("\nStep 1: Loading SAR data...")
with rasterio.open(TIF_PATH) as src:
    sigma0_full = src.read(1).astype(np.float32)
    transform   = src.transform
    bounds      = src.bounds
print(f"  Shape: {sigma0_full.shape}")

# ==========================================
# Step 2 - Load Phase 3 outputs
# ==========================================
print("Step 2: Loading wind directions...")
directions    = np.load(os.path.join(OUTPUT_DIR, 'wind_directions.npy'))
positions_row = np.load(os.path.join(OUTPUT_DIR, 'positions_row.npy'))
positions_col = np.load(os.path.join(OUTPUT_DIR, 'positions_col.npy'))
lons          = np.load(os.path.join(OUTPUT_DIR, 'lons.npy'))
lats          = np.load(os.path.join(OUTPUT_DIR, 'lats.npy'))
print(f"  Loaded {len(directions)} direction points")

# Filter ocean only (east of coastline)
ocean_points  = lons > 79.5
directions    = directions[ocean_points]
positions_row = positions_row[ocean_points]
positions_col = positions_col[ocean_points]
lons          = lons[ocean_points]
lats          = lats[ocean_points]
print(f"  After ocean filter: {len(directions)} points remaining")

# ==========================================
# Step 3 - Extract sigma0 per patch
# ==========================================
print("Step 3: Extracting sigma0 values...")
PATCH_SIZE = 512
sigma0_vals    = []
incidence_angs = []

for r, c in zip(positions_row, positions_col):
    r0 = max(0, r - PATCH_SIZE//2)
    r1 = min(sigma0_full.shape[0], r + PATCH_SIZE//2)
    c0 = max(0, c - PATCH_SIZE//2)
    c1 = min(sigma0_full.shape[1], c + PATCH_SIZE//2)
    patch = sigma0_full[r0:r1, c0:c1]
    ocean = patch[(patch > 0.001) & (patch < 0.05)]
    if len(ocean) > 100:
        sigma0_vals.append(float(np.nanmean(ocean)))
    else:
        sigma0_vals.append(np.nan)
    col_frac = c / sigma0_full.shape[1]
    incidence_angs.append(30.0 + col_frac * 15.0)

sigma0_arr = np.array(sigma0_vals, dtype=np.float64)
inc_arr    = np.array(incidence_angs, dtype=np.float64)
valid_s0   = ~np.isnan(sigma0_arr)
print(f"  Valid patches : {np.sum(valid_s0)}")
print(f"  Mean sigma0   : {np.nanmean(sigma0_arr):.5f}  ({10*np.log10(np.nanmean(sigma0_arr)):.1f} dB)")

# ==========================================
# Step 4 - Run CMOD5.n inversion
# ==========================================
print("Step 4: Running CMOD5.n inversion...")
look_angle  = -12.0
wind_speeds = np.full(len(sigma0_arr), np.nan)

for i in range(len(sigma0_arr)):
    if not np.isnan(sigma0_arr[i]):
        phi_rel = float(directions[i]) - look_angle
        wind_speeds[i] = speed_from_sigma0(
            sigma0_arr[i], phi_rel, inc_arr[i]
        )

valid_ws = ~np.isnan(wind_speeds)
print(f"  Valid points : {np.sum(valid_ws)}")
if np.sum(valid_ws) > 0:
    print(f"  Mean speed   : {np.nanmean(wind_speeds):.2f} m/s")
    print(f"  Min  speed   : {np.nanmin(wind_speeds):.2f} m/s")
    print(f"  Max  speed   : {np.nanmax(wind_speeds):.2f} m/s")

# ==========================================
# Step 5 - Plot wind field
# ==========================================
print("Step 5: Plotting wind field...")
fig, ax = plt.subplots(figsize=(12, 10))

sigma0_db = 10 * np.log10(np.where(sigma0_full > 0, sigma0_full, np.nan))
ax.imshow(sigma0_db, cmap='gray', vmin=-25, vmax=0,
          extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
          aspect='auto', origin='upper')

if np.sum(valid_ws) > 0:
    spd = wind_speeds[valid_ws]
    u   = np.sin(np.radians(directions[valid_ws]))
    v   = np.cos(np.radians(directions[valid_ws]))
    sc  = ax.quiver(lons[valid_ws], lats[valid_ws], u, v,
                    spd, cmap='jet', scale=30, width=0.003,
                    headwidth=4, headlength=5, clim=[0, 15])
    plt.colorbar(sc, ax=ax, label='Wind Speed (m/s)', shrink=0.8)
    print(f"  Vectors plotted: {np.sum(valid_ws)}")
else:
    print("  WARNING: no valid vectors!")

ax.set_title('SAR Wind Field — Tamil Nadu Coast\n2024-03-29 | CMOD5.n Retrieval')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, 'wind_field.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"  Saved: {out_path}")

# ==========================================
# Step 6 - Save outputs
# ==========================================
np.save(os.path.join(OUTPUT_DIR, 'wind_speeds.npy'), wind_speeds)
csv_path = os.path.join(OUTPUT_DIR, 'wind_field.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['lon', 'lat', 'wind_speed_ms', 'wind_direction_deg'])
    for i in range(len(lons)):
        if not np.isnan(wind_speeds[i]):
            writer.writerow([f"{lons[i]:.4f}", f"{lats[i]:.4f}",
                             f"{wind_speeds[i]:.2f}", f"{directions[i]:.1f}"])
print(f"  CSV saved: {csv_path}")
plt.show()
print("\nPhase 4 Complete! Wind field retrieved.")