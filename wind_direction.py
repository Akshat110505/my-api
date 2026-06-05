import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import from_bounds
from scipy import ndimage
import os

# ==========================================
# CONFIG
# ==========================================
TIF_PATH = r"C:\wind_field_project\outputs\S1_preprocessed.tif"
OUTPUT_DIR = r"C:\wind_field_project\outputs"
PATCH_SIZE = 512
STEP_SIZE = 256

print("Step 1: Loading preprocessed SAR image...")
with rasterio.open(TIF_PATH) as src:
    data = src.read(1).astype(np.float32)
    transform = src.transform
    crs = src.crs
    bounds = src.bounds

print(f"  Image shape: {data.shape}")
print(f"  Value range: {np.nanmin(data):.4f} to {np.nanmax(data):.4f}")

# ==========================================
# Step 2 — Mask ocean pixels only
# ==========================================
print("Step 2: Masking ocean pixels...")
ocean_mask = (data > 0) & (data < 0.05)
print(f"  Ocean pixels: {np.sum(ocean_mask):,} out of {data.size:,}")
data_ocean = np.where(ocean_mask, data, np.nan)

# ==========================================
# Step 3 — FFT-based wind direction
# ==========================================
print("Step 3: Computing wind direction via FFT...")

rows, cols = data.shape
directions = []
positions_row = []
positions_col = []

patch_count = 0
valid_count = 0

for r in range(0, rows - PATCH_SIZE, STEP_SIZE):
    for c in range(0, cols - PATCH_SIZE, STEP_SIZE):
        patch = data_ocean[r:r+PATCH_SIZE, c:c+PATCH_SIZE]

        valid_pixels = np.sum(~np.isnan(patch))
        patch_count += 1
        if valid_pixels < 0.3 * PATCH_SIZE * PATCH_SIZE:
            continue

        patch_clean = np.where(np.isnan(patch), np.nanmean(patch), patch)

        window = np.outer(np.hanning(PATCH_SIZE), np.hanning(PATCH_SIZE))
        patch_windowed = patch_clean * window

        fft = np.fft.fft2(patch_windowed)
        fft_shift = np.fft.fftshift(fft)
        power = np.abs(fft_shift) ** 2

        center = PATCH_SIZE // 2
        power[center-5:center+5, center-5:center+5] = 0

        freq_min = PATCH_SIZE // 32
        freq_max = PATCH_SIZE // 4
        mask_freq = np.zeros_like(power)
        y, x = np.ogrid[-center:center, -center:center]
        dist = np.sqrt(x**2 + y**2)
        mask_freq[(dist >= freq_min) & (dist <= freq_max)] = 1
        power_filtered = power * mask_freq

        max_idx = np.unravel_index(np.argmax(power_filtered), power_filtered.shape)
        dy = max_idx[0] - center
        dx = max_idx[1] - center

        streak_angle = np.degrees(np.arctan2(dy, dx))
        wind_angle = streak_angle % 180

        directions.append(wind_angle)
        positions_row.append(r + PATCH_SIZE // 2)
        positions_col.append(c + PATCH_SIZE // 2)
        valid_count += 1

print(f"  Processed {patch_count} patches, {valid_count} valid ocean patches")

directions    = np.array(directions)
positions_row = np.array(positions_row)
positions_col = np.array(positions_col)

# ==========================================
# Step 4 — Convert pixel positions to lat/lon
# ==========================================
print("Step 4: Converting to geographic coordinates...")

lons = []
lats = []
for r, c in zip(positions_row, positions_col):
    lon, lat = transform * (c, r)
    lons.append(lon)
    lats.append(lat)

lons = np.array(lons)
lats = np.array(lats)

# ==========================================
# Filter: keep only ocean points (east of coastline)
# ==========================================
ocean_points  = lons > 79.5
directions    = directions[ocean_points]
positions_row = positions_row[ocean_points]
positions_col = positions_col[ocean_points]
lons          = lons[ocean_points]
lats          = lats[ocean_points]
print(f"  After ocean filter: {len(directions)} points remaining")

if len(directions) == 0:
    print("  WARNING: No ocean points found! Check longitude cutoff.")
    exit()

print(f"  Mean wind direction: {np.mean(directions):.1f}°")
print(f"  Std wind direction:  {np.std(directions):.1f}°")

# ==========================================
# Step 5 — Plot wind directions
# ==========================================
print("Step 5: Plotting wind directions...")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

ax1 = axes[0]
data_db = 10 * np.log10(np.where(data > 0, data, np.nan))
ax1.imshow(data_db, cmap='gray', vmin=-25, vmax=0,
           extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
           aspect='auto')

u = np.sin(np.radians(directions))
v = np.cos(np.radians(directions))
ax1.quiver(lons, lats, u, v,
           color='red', scale=30, width=0.003,
           headwidth=3, headlength=4)

ax1.set_title('Wind Direction from FFT\nSentinel-1 VV 2024-03-29')
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')

ax2 = axes[1]
ax2.hist(directions, bins=36, range=(0, 180),
         color='steelblue', edgecolor='white', linewidth=0.5)
ax2.set_xlabel('Wind Direction (degrees)')
ax2.set_ylabel('Count')
ax2.set_title('Wind Direction Distribution')
ax2.set_xticks([0, 45, 90, 135, 180])
ax2.set_xticklabels(['N', 'NE', 'E', 'SE', 'S'])

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, 'wind_direction.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"  Saved to {out_path}")

np.save(os.path.join(OUTPUT_DIR, 'wind_directions.npy'), directions)
np.save(os.path.join(OUTPUT_DIR, 'positions_row.npy'), positions_row)
np.save(os.path.join(OUTPUT_DIR, 'positions_col.npy'), positions_col)
np.save(os.path.join(OUTPUT_DIR, 'lons.npy'), lons)
np.save(os.path.join(OUTPUT_DIR, 'lats.npy'), lats)
print("  Direction data saved for Phase 4.")

plt.show()
print("\nPhase 3 Complete! Wind directions retrieved.")