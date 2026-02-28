import os
import uuid
import numpy as np
import cv2
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
TARGET_W         = 640      # resize both images to this before comparison
TARGET_H         = 480
SSIM_WIN_SIZE    = 7        # must be odd; smaller = finer detail

# ── FIX 1: Raised threshold — pixels must differ significantly to count ──────
#    Old value of 45 was far too low; near-identical images triggered it.
#    80–100 catches only genuinely changed pixels (flood, rubble, etc.)
DIFF_THRESHOLD   = 85

# ── FIX 2: Raised min area — ignore small noise blobs ────────────────────────
#    300 px² picked up JPEG compression artefacts. 1500 px² is safer.
MIN_CONTOUR_AREA = 1500

MAX_VICTIM_ZONES = 10       # return top-N zones ranked by area

# ── Known regions (kept in sync with app.py) ─────────────────────────────────
REGIONS = {
    "kavalappara": "Kavalappara",
    "munnar":      "Munnar",
    "wayanad":     "Wayanad",
    "malappuram":  "Malappuram",
}


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

    # ── Mud / landslide: brown-orange hue, moderate saturation ───────────────
    #    HSV brown: H≈10-25, S≈40-180, V≈40-200
    mud_mask = cv2.inRange(post_hsv,
                           np.array([5,  35,  35], dtype=np.uint8),
                           np.array([28, 200, 210], dtype=np.uint8))

    # ── Debris / rubble: low saturation (desaturated), not too dark/bright ───
    #    Captures grey concrete, wooden debris, washed-out ground
    debris_mask = cv2.inRange(post_hsv,
                               np.array([0,   0,  40], dtype=np.uint8),
                               np.array([180, 40, 200], dtype=np.uint8))

    # ── Floodwater: dark desaturated blue-grey areas ──────────────────────────
    water_mask = cv2.inRange(post_hsv,
                              np.array([90,  10, 20], dtype=np.uint8),
                              np.array([140, 80, 140], dtype=np.uint8))

    # ── Combine all damage cues ───────────────────────────────────────────────
    combined = cv2.bitwise_or(mud_mask, debris_mask)
    combined = cv2.bitwise_or(combined, water_mask)

    # ── Remove sky (very bright, low-saturation top region) ──────────────────
    #    Mask out the top 25% of the image if it looks like sky/haze
    sky_zone  = np.zeros((h, w), dtype=np.uint8)
    sky_zone[:h // 4, :] = 255
    sky_pixels = cv2.inRange(post_hsv,
                              np.array([0,   0, 160], dtype=np.uint8),
                              np.array([180, 60, 255], dtype=np.uint8))
    sky_actual  = cv2.bitwise_and(sky_pixels, sky_zone)
    combined    = cv2.bitwise_and(combined, cv2.bitwise_not(sky_actual))

    # ── Remove vegetation (green hues) so healthy trees aren't flagged ───────
    veg_mask = cv2.inRange(post_hsv,
                            np.array([35, 40, 30], dtype=np.uint8),
                            np.array([90, 255, 255], dtype=np.uint8))
    combined = cv2.bitwise_and(combined, cv2.bitwise_not(veg_mask))

    # ── Morphological cleanup ─────────────────────────────────────────────────
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
        ssim_score        – float, global similarity (1 = identical)
        damage_percent    – float, % of image area flagged as damaged
        analysis_mode     – "ssim_diff" | "colour_segmentation"
        reference_warning – str or None (set when reference mismatch detected)
        victim_zones      – list of dicts
        output_image_url  – str
        error             – str or None
    """

    # ── Load reference ────────────────────────────────────────────────────────
    ref_raw = cv2.imread(reference_path)
    if ref_raw is None:
        return {"error": f"Reference image not found at {reference_path}"}

    # ── Decode uploaded bytes ─────────────────────────────────────────────────
    arr      = np.frombuffer(uploaded_bytes, np.uint8)
    post_raw = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if post_raw is None:
        return {"error": "Could not decode uploaded image. Send a valid JPEG/PNG."}

    # ── Resize both to a common size ──────────────────────────────────────────
    ref_resized  = cv2.resize(ref_raw,  (TARGET_W, TARGET_H))
    post_resized = cv2.resize(post_raw, (TARGET_W, TARGET_H))

    ref_gray  = cv2.cvtColor(ref_resized,  cv2.COLOR_BGR2GRAY)
    post_gray = cv2.cvtColor(post_resized, cv2.COLOR_BGR2GRAY)

    # ── CLAHE normalisation ───────────────────────────────────────────────────
    clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ref_eq    = clahe.apply(ref_gray)
    post_eq   = clahe.apply(post_gray)

    # ── Always compute SSIM (used for mode selection + reporting) ────────────
    ssim_score, diff = ssim(
        ref_eq, post_eq,
        win_size=SSIM_WIN_SIZE,
        full=True,
        data_range=255,
    )

    # ── DECIDE: SSIM-diff vs colour-segmentation ──────────────────────────────
    SSIM_TRUST_THRESHOLD = 0.35   # below this → images are too dissimilar to diff

    reference_warning = None
    analysis_mode     = "ssim_diff"

    if ssim_score < SSIM_TRUST_THRESHOLD:
        # ── Reference mismatch — use colour segmentation ──────────────────────
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
        # ── Good reference — standard SSIM pixel diff ─────────────────────────
        diff_clipped = np.clip(diff, 0, 1)
        diff_uint8   = ((1.0 - diff_clipped) * 255).astype(np.uint8)
        diff_uint8   = cv2.GaussianBlur(diff_uint8, (5, 5), 0)

        _, thresh = cv2.threshold(diff_uint8, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
        kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh    = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh    = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel, iterations=1)

    damage_percent = round(float(np.count_nonzero(thresh)) / thresh.size * 100, 2)

    # ── Find contours ─────────────────────────────────────────────────────────
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours    = [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]
    contours    = sorted(contours, key=cv2.contourArea, reverse=True)[:MAX_VICTIM_ZONES]

    # ── Geo-localise ──────────────────────────────────────────────────────────
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

    # ── Build victim likelihood heatmap ──────────────────────────────────────
    #    Edge density inside damage areas marks structural boundaries where
    #    survivors are most likely trapped (under debris, near walls, etc.)
    edges     = cv2.Canny(post_gray, 60, 150)
    edges_dmg = cv2.bitwise_and(edges, edges, mask=thresh)
    heat      = cv2.GaussianBlur(edges_dmg.astype(np.float32), (31, 31), 0)

    # ── Annotate: small target squares at victim hotspots ────────────────────
    annotated    = post_resized.copy()
    victim_zones = []
    MARKER       = 18          # half-side of small victim square (px)
    victim_id    = 1

    for cnt in contours:
        area = int(cv2.contourArea(cnt))

        # Number of markers scales with damage area (1–5 per zone)
        n_markers = max(1, min(5, area // 8000))

        # Isolate heat to this contour only
        zone_mask  = np.zeros(thresh.shape, dtype=np.uint8)
        cv2.drawContours(zone_mask, [cnt], -1, 255, thickness=cv2.FILLED)
        local_heat = heat.copy()
        local_heat[zone_mask == 0] = 0

        zone_heat_vals = heat[zone_mask > 0]
        max_heat       = float(zone_heat_vals.max()) if zone_heat_vals.size else 1.0

        placed = []   # track placed centres to avoid overlap

        for _ in range(n_markers):
            if local_heat.max() < 1.0:
                break

            _, _, _, peak_pt = cv2.minMaxLoc(local_heat)
            px, py = peak_pt

            # Skip if overlapping an existing marker
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

            # Colour by confidence
            if confidence > 0.6:
                colour = (0, 0, 255)       # red   – high
            elif confidence > 0.3:
                colour = (0, 200, 255)     # amber – medium
            else:
                colour = (255, 220, 0)     # cyan  – low

            x1 = max(px - MARKER, 0);       y1 = max(py - MARKER, 0)
            x2 = min(px + MARKER, TARGET_W - 1); y2 = min(py + MARKER, TARGET_H - 1)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)   # square
            cv2.circle(annotated, (px, py), 3, colour, -1)              # centre dot
            cv2.line(annotated, (px - 7, py), (px + 7, py), colour, 1) # crosshair H
            cv2.line(annotated, (px, py - 7), (px, py + 7), colour, 1) # crosshair V
            cv2.putText(annotated, f"V{victim_id:02d}",
                        (x1, max(y1 - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)

            cv2.circle(local_heat, (px, py), MARKER * 3, 0, -1)  # suppress peak
            victim_id += 1

    # ── Very light damage tint so markers remain readable ────────────────────
    tint_colour  = (0, 160, 0) if analysis_mode == "colour_segmentation" else (0, 0, 180)
    tint_overlay = annotated.copy()
    tint_overlay[thresh > 0] = tint_colour
    annotated    = cv2.addWeighted(annotated, 0.88, tint_overlay, 0.12, 0)

    # ── Header bar ────────────────────────────────────────────────────────────
    mode_tag = "CLR-SEG" if analysis_mode == "colour_segmentation" else "SSIM"
    bar      = np.zeros((30, TARGET_W, 3), dtype=np.uint8)
    cv2.putText(
        bar,
        f"[{mode_tag}] SSIM:{ssim_score:.3f} | Dmg:{damage_percent}% | Victims:{victim_id - 1}",
        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
    )
    annotated = np.vstack([bar, annotated])

    # ── Save ──────────────────────────────────────────────────────────────────
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

    # ── Fallback: if no reference exists, save the upload as the baseline ────
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