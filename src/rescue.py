import os
import uuid
import numpy as np
import cv2
from datetime import datetime
from flask import Blueprint, request, jsonify
from skimage.metrics import structural_similarity as ssim

rescue_bp = Blueprint("rescue", __name__)

# ── Folder config ────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
REFERENCE_DIR   = os.path.join(BASE_DIR, "static", "reference")
OUTPUT_DIR      = os.path.join(BASE_DIR, "static", "rescue_output")
os.makedirs(REFERENCE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR,    exist_ok=True)

# ── SSIM / analysis parameters ───────────────────────────────────────────────
TARGET_W         = 640
TARGET_H         = 480
SSIM_WIN_SIZE    = 7

DIFF_THRESHOLD   = 85
MIN_CONTOUR_AREA = 1500
MAX_VICTIM_ZONES = 10

# ── Known regions (kept in sync with app.py) ─────────────────────────────────
REGIONS = {
    "kavalappara": "Kavalappara",
    "munnar":      "Munnar",
    "wayanad":     "Wayanad",
    "malappuram":  "Malappuram",
}

# ── In-memory SOS log (persists as long as the server is running) ────────────
sos_alerts = []


# ─────────────────────────────────────────────────────────────────────────────
# Core analysis function
# ─────────────────────────────────────────────────────────────────────────────

def _single_image_damage_mask(post_hsv: np.ndarray, post_gray: np.ndarray) -> np.ndarray:
    """
    Detect damage zones from a single post-disaster image using colour cues.

    Targets:
      • Mudslide / floodwater  – desaturated brown/tan/grey tones
      • Debris fields          – low-saturation mixed areas
      • Standing water         – dark, low-saturation blueish regions

    Returns a binary mask (uint8, 0/255) the same size as post_hsv.
    """
    h, w = post_hsv.shape[:2]
    H, S, V = cv2.split(post_hsv)

    mud_mask = cv2.inRange(post_hsv,
                           np.array([5,  35,  35], dtype=np.uint8),
                           np.array([28, 200, 210], dtype=np.uint8))

    debris_mask = cv2.inRange(post_hsv,
                               np.array([0,   0,  40], dtype=np.uint8),
                               np.array([180, 40, 200], dtype=np.uint8))

    water_mask = cv2.inRange(post_hsv,
                              np.array([90,  10, 20], dtype=np.uint8),
                              np.array([140, 80, 140], dtype=np.uint8))

    combined = cv2.bitwise_or(mud_mask, debris_mask)
    combined = cv2.bitwise_or(combined, water_mask)

    sky_zone  = np.zeros((h, w), dtype=np.uint8)
    sky_zone[:h // 4, :] = 255
    sky_pixels = cv2.inRange(post_hsv,
                              np.array([0,   0, 160], dtype=np.uint8),
                              np.array([180, 60, 255], dtype=np.uint8))
    sky_actual  = cv2.bitwise_and(sky_pixels, sky_zone)
    combined    = cv2.bitwise_and(combined, cv2.bitwise_not(sky_actual))

    veg_mask = cv2.inRange(post_hsv,
                            np.array([35, 40, 30], dtype=np.uint8),
                            np.array([90, 255, 255], dtype=np.uint8))
    combined = cv2.bitwise_and(combined, cv2.bitwise_not(veg_mask))

    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=3)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  kernel, iterations=2)

    return combined


def analyse_disaster_image(reference_path: str, uploaded_bytes: bytes, region_key: str):
    """
    Compare reference image with uploaded post-disaster image.

    Strategy
    --------
    • If SSIM ≥ 0.35  → standard pixel-diff (images are comparable scenes)
    • If SSIM  < 0.35  → reference mismatch detected; fall back to single-image
                         colour-based damage segmentation (flood/mud/debris)

    Returns
    -------
    dict  with keys:
        ssim_score        – float
        damage_percent    – float
        analysis_mode     – "ssim_diff" | "colour_segmentation"
        reference_warning – str or None
        victim_zones      – list of dicts
        output_image_url  – str
        error             – str or None
    """

    ref_raw = cv2.imread(reference_path)
    if ref_raw is None:
        return {"error": f"Reference image not found at {reference_path}"}

    arr      = np.frombuffer(uploaded_bytes, np.uint8)
    post_raw = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if post_raw is None:
        return {"error": "Could not decode uploaded image. Send a valid JPEG/PNG."}

    ref_resized  = cv2.resize(ref_raw,  (TARGET_W, TARGET_H))
    post_resized = cv2.resize(post_raw, (TARGET_W, TARGET_H))

    ref_gray  = cv2.cvtColor(ref_resized,  cv2.COLOR_BGR2GRAY)
    post_gray = cv2.cvtColor(post_resized, cv2.COLOR_BGR2GRAY)

    clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ref_eq    = clahe.apply(ref_gray)
    post_eq   = clahe.apply(post_gray)

    ssim_score, diff = ssim(
        ref_eq, post_eq,
        win_size=SSIM_WIN_SIZE,
        full=True,
        data_range=255,
    )

    SSIM_TRUST_THRESHOLD = 0.35

    reference_warning = None
    analysis_mode     = "ssim_diff"

    if ssim_score < SSIM_TRUST_THRESHOLD:
        analysis_mode     = "colour_segmentation"
        reference_warning = (
            f"Reference image SSIM is very low ({ssim_score:.3f}). "
            "The reference may be from a different angle or scene. "
            "Falling back to single-image colour-based damage detection. "
            "For best results, upload a matching pre-disaster reference via /rescue/set_reference."
        )
        post_hsv = cv2.cvtColor(post_resized, cv2.COLOR_BGR2HSV)
        thresh   = _single_image_damage_mask(post_hsv, post_gray)

    else:
        diff_clipped = np.clip(diff, 0, 1)
        diff_uint8   = ((1.0 - diff_clipped) * 255).astype(np.uint8)
        diff_uint8   = cv2.GaussianBlur(diff_uint8, (5, 5), 0)

        _, thresh = cv2.threshold(diff_uint8, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
        kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh    = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh    = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel, iterations=1)

    damage_percent = round(float(np.count_nonzero(thresh)) / thresh.size * 100, 2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours    = [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]
    contours    = sorted(contours, key=cv2.contourArea, reverse=True)[:MAX_VICTIM_ZONES]

    from app import REGIONS as APP_REGIONS
    region_cfg = APP_REGIONS.get(region_key, {})
    lat_start  = region_cfg.get("lat_start", 0)
    lat_end    = region_cfg.get("lat_end",   0)
    lon_start  = region_cfg.get("lon_start", 0)
    lon_end    = region_cfg.get("lon_end",   0)

    def pixel_to_latlon(cx_px, cy_px):
        lon = lon_start + (cx_px / TARGET_W) * (lon_end - lon_start)
        lat = lat_end   - (cy_px / TARGET_H) * (lat_end - lat_start)
        return round(lat, 6), round(lon, 6)

    edges     = cv2.Canny(post_gray, 60, 150)
    edges_dmg = cv2.bitwise_and(edges, edges, mask=thresh)
    heat      = cv2.GaussianBlur(edges_dmg.astype(np.float32), (31, 31), 0)

    annotated    = post_resized.copy()
    victim_zones = []
    MARKER       = 18
    victim_id    = 1

    for cnt in contours:
        area = int(cv2.contourArea(cnt))
        n_markers = max(1, min(5, area // 8000))

        zone_mask  = np.zeros(thresh.shape, dtype=np.uint8)
        cv2.drawContours(zone_mask, [cnt], -1, 255, thickness=cv2.FILLED)
        local_heat = heat.copy()
        local_heat[zone_mask == 0] = 0

        zone_heat_vals = heat[zone_mask > 0]
        max_heat       = float(zone_heat_vals.max()) if zone_heat_vals.size else 1.0

        placed = []

        for _ in range(n_markers):
            if local_heat.max() < 1.0:
                break

            _, _, _, peak_pt = cv2.minMaxLoc(local_heat)
            px, py = peak_pt

            if any(abs(px - ox) < MARKER * 2 and abs(py - oy) < MARKER * 2
                   for ox, oy in placed):
                cv2.circle(local_heat, (px, py), MARKER * 2, 0, -1)
                continue

            placed.append((px, py))
            confidence = round(min(float(local_heat[py, px]) / (max_heat + 1e-6), 1.0), 3)
            clat, clon = pixel_to_latlon(px, py)

            victim_zones.append({
                "id":         f"V{victim_id:02d}",
                "rank":       victim_id,
                "cx": px, "cy": py,
                "confidence": confidence,
                "centre_lat": clat,
                "centre_lon": clon,
            })

            if confidence > 0.6:
                colour = (0, 0, 255)
            elif confidence > 0.3:
                colour = (0, 200, 255)
            else:
                colour = (255, 220, 0)

            x1 = max(px - MARKER, 0);       y1 = max(py - MARKER, 0)
            x2 = min(px + MARKER, TARGET_W - 1); y2 = min(py + MARKER, TARGET_H - 1)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
            cv2.circle(annotated, (px, py), 3, colour, -1)
            cv2.line(annotated, (px - 7, py), (px + 7, py), colour, 1)
            cv2.line(annotated, (px, py - 7), (px, py + 7), colour, 1)
            cv2.putText(annotated, f"V{victim_id:02d}",
                        (x1, max(y1 - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)

            cv2.circle(local_heat, (px, py), MARKER * 3, 0, -1)
            victim_id += 1

    tint_colour  = (0, 160, 0) if analysis_mode == "colour_segmentation" else (0, 0, 180)
    tint_overlay = annotated.copy()
    tint_overlay[thresh > 0] = tint_colour
    annotated    = cv2.addWeighted(annotated, 0.88, tint_overlay, 0.12, 0)

    mode_tag = "CLR-SEG" if analysis_mode == "colour_segmentation" else "SSIM"
    bar      = np.zeros((30, TARGET_W, 3), dtype=np.uint8)
    cv2.putText(
        bar,
        f"[{mode_tag}] SSIM:{ssim_score:.3f} | Dmg:{damage_percent}% | Victims:{victim_id - 1}",
        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
    )
    annotated = np.vstack([bar, annotated])

    out_filename = f"{region_key}_{uuid.uuid4().hex[:8]}.jpg"
    out_path     = os.path.join(OUTPUT_DIR, out_filename)
    cv2.imwrite(out_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])

    return {
        "ssim_score":        round(float(ssim_score), 4),
        "damage_percent":    damage_percent,
        "analysis_mode":     analysis_mode,
        "reference_warning": reference_warning,
        "victim_zones":      victim_zones,
        "output_image_url":  f"/static/rescue_output/{out_filename}",
        "error":             None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@rescue_bp.route("/rescue/analyse", methods=["POST"])
def rescue_analyse():
    """
    POST  multipart/form-data
        file    – uploaded post-disaster image (required)
        region  – region key, e.g. 'wayanad'  (required)
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Use field name 'file'."}), 400

    region_key = request.form.get("region", "kavalappara").lower().strip()
    if region_key not in REGIONS:
        return jsonify({
            "error":     f"Unknown region '{region_key}'.",
            "available": list(REGIONS.keys()),
        }), 400

    uploaded_file = request.files["file"]
    if uploaded_file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    uploaded_bytes = uploaded_file.read()
    reference_path = os.path.join(REFERENCE_DIR, f"{region_key}.jpg")

    if not os.path.exists(reference_path):
        with open(reference_path, "wb") as f:
            f.write(uploaded_bytes)
        return jsonify({
            "message": (
                f"No reference image existed for '{region_key}'. "
                "The uploaded image has been saved as the new reference. "
                "Upload a post-disaster image to begin analysis."
            ),
            "reference_saved": reference_path,
        }), 201

    result = analyse_disaster_image(reference_path, uploaded_bytes, region_key)

    if result.get("error"):
        return jsonify(result), 422

    return jsonify(result), 200


@rescue_bp.route("/rescue/set_reference", methods=["POST"])
def set_reference():
    """
    Upload / replace the pre-disaster reference image for a region.
    POST  multipart/form-data
        file    – reference image
        region  – region key
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    region_key = request.form.get("region", "").lower().strip()
    if region_key not in REGIONS:
        return jsonify({"error": f"Unknown region '{region_key}'."}), 400

    data = request.files["file"].read()
    path = os.path.join(REFERENCE_DIR, f"{region_key}.jpg")
    with open(path, "wb") as f:
        f.write(data)

    return jsonify({"message": f"Reference image saved for '{region_key}'.", "path": path})


# ─────────────────────────────────────────────────────────────────────────────
# ── VICTIM SOS ENDPOINT ──────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

@rescue_bp.route("/rescue/sos", methods=["POST"])
def victim_sos():
    """
    Called when a victim presses the SOS button on the frontend.

    Expects JSON body:
    {
        "latitude":  <float>,
        "longitude": <float>,
        "accuracy":  <float, optional – GPS accuracy in metres>,
        "name":      <str, optional – victim's name or identifier>
    }

    Logs the alert to the terminal and stores it in the in-memory sos_alerts list.
    Returns a JSON acknowledgement so the frontend can confirm receipt.
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400

    latitude  = data.get("latitude")
    longitude = data.get("longitude")

    if latitude is None or longitude is None:
        return jsonify({"error": "Both 'latitude' and 'longitude' are required."}), 400

    # Build alert record
    alert_id  = f"SOS-{uuid.uuid4().hex[:6].upper()}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    accuracy  = data.get("accuracy", "N/A")
    name      = data.get("name", "Unknown")

    alert = {
        "alert_id":  alert_id,
        "timestamp": timestamp,
        "name":      name,
        "latitude":  latitude,
        "longitude": longitude,
        "accuracy":  accuracy,
        "maps_link": f"https://www.google.com/maps?q={latitude},{longitude}",
    }

    # Store in memory
    sos_alerts.append(alert)

    # ── Print to terminal ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("🚨  VICTIM SOS ALERT RECEIVED")
    print("=" * 60)
    print(f"  Alert ID  : {alert_id}")
    print(f"  Time      : {timestamp}")
    print(f"  Name      : {name}")
    print(f"  Latitude  : {latitude}")
    print(f"  Longitude : {longitude}")
    print(f"  Accuracy  : {accuracy} m")
    print(f"  Maps Link : {alert['maps_link']}")
    print("=" * 60 + "\n")

    return jsonify({
        "status":    "received",
        "alert_id":  alert_id,
        "message":   "Your SOS has been received. Help is on the way.",
        "maps_link": alert["maps_link"],
    }), 200


@rescue_bp.route("/rescue/sos/all", methods=["GET"])
def list_sos_alerts():
    """
    GET /rescue/sos/all
    Returns all SOS alerts received since the server started.
    Useful for the rescue dashboard to display active victims.
    """
    return jsonify({
        "total":  len(sos_alerts),
        "alerts": sos_alerts,
    }), 200