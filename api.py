import numpy as np
import rasterio
import os
import json
import csv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from scipy.interpolate import RegularGridInterpolator

# ==========================================
# CONFIG
# ==========================================
OUTPUT_DIR = r"C:\wind_field_project\outputs"
TIF_PATH   = r"C:\wind_field_project\outputs\S1_preprocessed.tif"

app = FastAPI(
    title="SAR Wind Field API",
    description="Wind field estimation from Sentinel-1 SAR imagery — Tamil Nadu Coast",
    version="1.0.0"
)

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
        return None

    v = 7.0
    for _ in range(30):
        f  = cmod5n_forward(v, phi, theta)
        dv = 0.01
        df = (cmod5n_forward(v+dv, phi, theta) - f) / dv
        if abs(df) < 1e-12:
            break
        v_new = v - (f - sigma0_obs) / df
        v_new = float(np.clip(v_new, 0.5, 35.0))
        if abs(v_new - v) < 1e-4:
            v = v_new
            break
        v = v_new

    check = cmod5n_forward(v, phi, theta)
    if abs(check - sigma0_obs) / (sigma0_obs + 1e-10) < 0.05:
        return round(float(v), 2)

    speeds = np.linspace(0.5, 35.0, 3000)
    sigmas = np.array([cmod5n_forward(vv, phi, theta) for vv in speeds])
    idx = np.argmin(np.abs(sigmas - sigma0_obs))
    return round(float(speeds[idx]), 2)


# ==========================================
# Request / Response models
# ==========================================
class WindRequest(BaseModel):
    date: str                  # e.g. "2024-03-29"
    bbox: List[float]          # [lon_min, lat_min, lon_max, lat_max]

class WindVector(BaseModel):
    lon: float
    lat: float
    wind_speed_ms: float
    wind_direction_deg: float

class WindResponse(BaseModel):
    date: str
    bbox: List[float]
    points: int
    mean_speed_ms: float
    features: list             # GeoJSON features


# ==========================================
# Helper: load precomputed results
# ==========================================
def load_precomputed(bbox):
    lon_min, lat_min, lon_max, lat_max = bbox

    wind_speeds   = np.load(os.path.join(OUTPUT_DIR, 'wind_speeds.npy'))
    directions    = np.load(os.path.join(OUTPUT_DIR, 'wind_directions.npy'))
    lons          = np.load(os.path.join(OUTPUT_DIR, 'lons.npy'))
    lats          = np.load(os.path.join(OUTPUT_DIR, 'lats.npy'))

    # Filter by bbox and valid speeds
    mask = (
        ~np.isnan(wind_speeds) &
        (lons >= lon_min) & (lons <= lon_max) &
        (lats >= lat_min) & (lats <= lat_max)
    )

    return (
        lons[mask], lats[mask],
        wind_speeds[mask], directions[mask]
    )


def build_geojson(lons, lats, speeds, directions):
    features = []
    for lo, la, sp, di in zip(lons, lats, speeds, directions):
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [round(float(lo), 4), round(float(la), 4)]
            },
            "properties": {
                "wind_speed_ms":      round(float(sp), 2),
                "wind_direction_deg": round(float(di), 1)
            }
        })
    return features


# ==========================================
# Routes
# ==========================================

@app.get("/")
def root():
    return {
        "message": "SAR Wind Field API",
        "usage": "POST /wind-field with {date, bbox}",
        "example": {
            "date": "2024-03-29",
            "bbox": [79.5, 9.5, 80.5, 12.0]
        }
    }


@app.get("/health")
def health():
    files = ['wind_speeds.npy', 'wind_directions.npy', 'lons.npy', 'lats.npy']
    missing = [f for f in files if not os.path.exists(os.path.join(OUTPUT_DIR, f))]
    if missing:
        return {"status": "warning", "missing_files": missing}
    return {"status": "ok", "message": "All data files found"}


@app.post("/wind-field", response_model=WindResponse)
def get_wind_field(request: WindRequest):

    # Validate bbox
    if len(request.bbox) != 4:
        raise HTTPException(status_code=400, detail="bbox must be [lon_min, lat_min, lon_max, lat_max]")

    lon_min, lat_min, lon_max, lat_max = request.bbox
    if lon_min >= lon_max or lat_min >= lat_max:
        raise HTTPException(status_code=400, detail="Invalid bbox coordinates")

    # Check date matches available data
    if request.date != "2024-03-29":
        raise HTTPException(
            status_code=404,
            detail=f"No data available for {request.date}. Available: 2024-03-29"
        )

    # Load and filter data
    try:
        lons, lats, speeds, directions = load_precomputed(request.bbox)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Data file missing: {str(e)}")

    if len(speeds) == 0:
        raise HTTPException(
            status_code=404,
            detail="No wind data found in the specified bbox. Try [79.5, 9.5, 80.5, 12.0]"
        )

    features = build_geojson(lons, lats, speeds, directions)

    return WindResponse(
        date=request.date,
        bbox=request.bbox,
        points=len(speeds),
        mean_speed_ms=round(float(np.mean(speeds)), 2),
        features=features
    )


@app.get("/wind-field/summary")
def get_summary():
    try:
        wind_speeds = np.load(os.path.join(OUTPUT_DIR, 'wind_speeds.npy'))
        lons        = np.load(os.path.join(OUTPUT_DIR, 'lons.npy'))
        lats        = np.load(os.path.join(OUTPUT_DIR, 'lats.npy'))
        valid       = ~np.isnan(wind_speeds)
        return {
            "date": "2024-03-29",
            "total_points": int(np.sum(valid)),
            "mean_speed_ms": round(float(np.nanmean(wind_speeds)), 2),
            "min_speed_ms":  round(float(np.nanmin(wind_speeds[valid])), 2),
            "max_speed_ms":  round(float(np.nanmax(wind_speeds[valid])), 2),
            "lon_range": [round(float(np.min(lons)), 3), round(float(np.max(lons)), 3)],
            "lat_range": [round(float(np.min(lats)), 3), round(float(np.max(lats)), 3)],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))