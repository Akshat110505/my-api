import numpy as np
import matplotlib.pyplot as plt
import cdsapi
import netCDF4 as nc
import os
from scipy.interpolate import RegularGridInterpolator

# ==========================================
# CONFIG
# ==========================================
OUTPUT_DIR = r"C:\wind_field_project\outputs"
ERA5_FILE  = os.path.join(OUTPUT_DIR, 'era5_wind.nc')

# Bounding box — Tamil Nadu coast
LAT_MIN, LAT_MAX = 9.5,  12.0
LON_MIN, LON_MAX = 77.5, 80.5
DATE = '2024-03-29'
TIME = '00:00'   # SAR acquisition was ~00:32 UTC, closest ERA5 hour is 00:00

# ==========================================
# Step 1 - Download ERA5
# ==========================================
print("Step 1: Downloading ERA5 data...")
if not os.path.exists(ERA5_FILE):
    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': [
                '10m_u_component_of_wind',
                '10m_v_component_of_wind',
            ],
            'year':  '2024',
            'month': '03',
            'day':   '29',
            'time':  TIME,
            'area':  [LAT_MAX, LON_MIN, LAT_MIN, LON_MAX],
            'format': 'netcdf',
        },
        ERA5_FILE
    )
    print(f"  Downloaded: {ERA5_FILE}")
else:
    print(f"  Already exists: {ERA5_FILE}")

# ==========================================
# Step 2 - Load ERA5
# ==========================================
print("Step 2: Loading ERA5 data...")
ds = nc.Dataset(ERA5_FILE)
print(f"  Variables: {list(ds.variables.keys())}")

# Handle variable names (CDS sometimes uses u10/v10 or u/v)
u_key = 'u10' if 'u10' in ds.variables else 'u'
v_key = 'v10' if 'v10' in ds.variables else 'v'

era5_lats = ds.variables['latitude'][:]
era5_lons = ds.variables['longitude'][:]
era5_u    = ds.variables[u_key][0, :, :]   # first time step
era5_v    = ds.variables[v_key][0, :, :]

# Compute ERA5 wind speed and direction
era5_speed = np.sqrt(era5_u**2 + era5_v**2)
era5_dir   = (np.degrees(np.arctan2(era5_u, era5_v)) % 180)

print(f"  ERA5 grid: {era5_lats.shape} lats x {era5_lons.shape} lons")
print(f"  ERA5 speed range: {np.min(era5_speed):.2f} to {np.max(era5_speed):.2f} m/s")
print(f"  ERA5 mean speed : {np.mean(era5_speed):.2f} m/s")

# ==========================================
# Step 3 - Load SAR wind field
# ==========================================
print("Step 3: Loading SAR wind field...")
wind_speeds   = np.load(os.path.join(OUTPUT_DIR, 'wind_speeds.npy'))
directions    = np.load(os.path.join(OUTPUT_DIR, 'wind_directions.npy'))
lons          = np.load(os.path.join(OUTPUT_DIR, 'lons.npy'))
lats          = np.load(os.path.join(OUTPUT_DIR, 'lats.npy'))

valid = ~np.isnan(wind_speeds)
sar_speeds = wind_speeds[valid]
sar_lons   = lons[valid]
sar_lats   = lats[valid]
sar_dirs   = directions[valid]
print(f"  SAR valid points: {len(sar_speeds)}")

# ==========================================
# Step 4 - Interpolate ERA5 to SAR points
# ==========================================
print("Step 4: Interpolating ERA5 to SAR locations...")

# ERA5 lats may be descending — sort ascending for interpolator
if era5_lats[0] > era5_lats[-1]:
    era5_lats  = era5_lats[::-1]
    era5_speed = era5_speed[::-1, :]
    era5_dir   = era5_dir[::-1, :]

interp_speed = RegularGridInterpolator(
    (era5_lats, era5_lons), era5_speed, method='linear', bounds_error=False, fill_value=np.nan
)
interp_dir = RegularGridInterpolator(
    (era5_lats, era5_lons), era5_dir, method='linear', bounds_error=False, fill_value=np.nan
)

points = np.column_stack([sar_lats, sar_lons])
era5_speed_at_sar = interp_speed(points)
era5_dir_at_sar   = interp_dir(points)

# ==========================================
# Step 5 - Compute validation metrics
# ==========================================
print("Step 5: Computing validation metrics...")

mask = ~np.isnan(era5_speed_at_sar)
sar_s  = sar_speeds[mask]
era_s  = era5_speed_at_sar[mask]

rmse = np.sqrt(np.mean((sar_s - era_s)**2))
mae  = np.mean(np.abs(sar_s - era_s))
bias = np.mean(sar_s - era_s)
corr = np.corrcoef(sar_s, era_s)[0, 1]

print(f"\n  ========== VALIDATION RESULTS ==========")
print(f"  Points compared : {len(sar_s)}")
print(f"  SAR mean speed  : {np.mean(sar_s):.2f} m/s")
print(f"  ERA5 mean speed : {np.mean(era_s):.2f} m/s")
print(f"  RMSE            : {rmse:.2f} m/s")
print(f"  MAE             : {mae:.2f} m/s")
print(f"  Bias (SAR-ERA5) : {bias:.2f} m/s")
print(f"  Correlation     : {corr:.3f}")
print(f"  =========================================\n")

# ==========================================
# Step 6 - Plot
# ==========================================
print("Step 6: Plotting validation...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1 — Scatter
ax1 = axes[0]
ax1.scatter(era_s, sar_s, alpha=0.6, color='steelblue', edgecolors='white', linewidth=0.3)
lim = max(np.max(era_s), np.max(sar_s)) + 1
ax1.plot([0, lim], [0, lim], 'r--', linewidth=1.5, label='1:1 line')
ax1.set_xlabel('ERA5 Wind Speed (m/s)')
ax1.set_ylabel('SAR Wind Speed (m/s)')
ax1.set_title(f'SAR vs ERA5\nRMSE={rmse:.2f}, Bias={bias:.2f}, R={corr:.3f}')
ax1.legend()
ax1.set_xlim(0, lim)
ax1.set_ylim(0, lim)

# Plot 2 — ERA5 wind speed map
ax2 = axes[1]
im2 = ax2.pcolormesh(era5_lons, era5_lats, era5_speed,
                      cmap='jet', vmin=0, vmax=15)
plt.colorbar(im2, ax=ax2, label='Wind Speed (m/s)')
ax2.scatter(sar_lons, sar_lats, c=sar_speeds, cmap='jet',
            vmin=0, vmax=15, edgecolors='black', linewidth=0.5, s=30)
ax2.set_title('ERA5 Wind Speed\n(dots = SAR points)')
ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')

# Plot 3 — Difference map
ax3 = axes[2]
diff = sar_s - era5_speed_at_sar[mask]
sc3 = ax3.scatter(sar_lons[mask], sar_lats[mask], c=diff,
                   cmap='RdBu_r', vmin=-5, vmax=5, s=40,
                   edgecolors='black', linewidth=0.3)
plt.colorbar(sc3, ax=ax3, label='SAR - ERA5 (m/s)')
ax3.set_title('Difference: SAR − ERA5')
ax3.set_xlabel('Longitude')
ax3.set_ylabel('Latitude')

plt.suptitle('ERA5 Validation — Tamil Nadu Coast 2024-03-29', fontsize=13)
plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, 'era5_validation.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"  Saved: {out_path}")

plt.show()
print("\nPhase 5 Complete! ERA5 validation done.")