import os
import numpy as np
import matplotlib.pyplot as plt

# esa_snappy import
import esa_snappy
from esa_snappy import ProductIO, GPF, HashMap, jpy

# ==========================================
# CONFIG — update this path
# ==========================================
SAFE_PATH = r"C:\Users\Dell\Downloads\S1A_IW_GRDH_1SDV_20240329T003248_20240329T003313_053189_0671D8_2450.SAFE"
OUTPUT_DIR = r"C:\wind_field_project\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Step 1: Reading product...")
product = ProductIO.readProduct(SAFE_PATH)
print(f"  Product: {product.getName()}")
print(f"  Bands: {list(product.getBandNames())}")

# ==========================================
# Step 2 — Apply orbit file
# ==========================================
print("Step 2: Applying orbit file...")
params = HashMap()
params.put('orbitType', 'Sentinel Precise (Auto Download)')
params.put('polyDegree', '3')
params.put('continueOnFail', 'true')
orbit = GPF.createProduct('Apply-Orbit-File', params, product)
print("  Orbit file applied.")

# ==========================================
# Step 3 — Thermal noise removal
# ==========================================
print("Step 3: Removing thermal noise...")
params = HashMap()
params.put('removeThermalNoise', 'true')
denoised = GPF.createProduct('ThermalNoiseRemoval', params, orbit)
print("  Thermal noise removed.")

# ==========================================
# Step 4 — Calibration to Sigma0
# ==========================================
print("Step 4: Calibrating to Sigma0...")
params = HashMap()
params.put('outputSigmaBand', 'true')
params.put('outputGammaBand', 'false')
params.put('outputBetaBand', 'false')
params.put('outputImageInDb', 'false')   # linear scale for CMOD5
params.put('selectedPolarisations', 'VV')
calibrated = GPF.createProduct('Calibration', params, denoised)
print("  Calibration done.")

# ==========================================
# Step 5 — Speckle filter (Lee 5x5)
# ==========================================
print("Step 5: Applying speckle filter...")
params = HashMap()
params.put('filter', 'Lee')
params.put('filterSizeX', '5')
params.put('filterSizeY', '5')
filtered = GPF.createProduct('Speckle-Filter', params, calibrated)
print("  Speckle filter applied.")

# ==========================================
# Step 6 — Terrain correction
# ==========================================
print("Step 6: Terrain correction...")
params = HashMap()
params.put('demName', 'SRTM 3Sec')
params.put('imgResamplingMethod', 'BILINEAR_INTERPOLATION')
params.put('pixelSpacingInMeter', '40.0')
params.put('mapProjection', 'WGS84(DD)')
params.put('nodataValueAtSea', 'false')
corrected = GPF.createProduct('Terrain-Correction', params, filtered)
print("  Terrain correction done.")

# ==========================================
# Step 7 — Save output
# ==========================================
print("Step 7: Saving output...")
output_path = os.path.join(OUTPUT_DIR, "S1_preprocessed")
ProductIO.writeProduct(corrected, output_path, 'GeoTIFF')
print(f"  Saved to {output_path}.tif")

# ==========================================
# Step 8 — Quick plot to verify
# ==========================================
print("Step 8: Plotting...")
band = corrected.getBand('Sigma0_VV')
w = band.getRasterWidth()
h = band.getRasterHeight()
data = np.zeros(w * h, dtype=np.float32)
band.readPixels(0, 0, w, h, data)
data = data.reshape(h, w)

# Mask zeros, convert to dB for visualization
data[data <= 0] = np.nan
data_db = 10 * np.log10(data)

plt.figure(figsize=(10, 10))
plt.imshow(data_db, cmap='gray', vmin=-25, vmax=0)
plt.colorbar(label='Sigma0 VV (dB)')
plt.title('Sentinel-1 IW GRD VV - Preprocessed\n2024-03-29 Tamil Nadu Coast')
plt.savefig(os.path.join(OUTPUT_DIR, 'preprocessed_VV.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Done! Check outputs folder.")