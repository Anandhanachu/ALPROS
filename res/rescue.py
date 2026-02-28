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
DIFF_THRESHOLD   = 45       # 0-255; pixels above this are "changed"
MIN_CONTOUR_AREA = 300      # ignore tiny noise blobs (px²)
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

def analyse_disaster_image(reference_path: str, uploaded_bytes: bytes, region_key: str):
    """
    Compare reference image with uploaded post-disaster image.

    Returns
    -------
    dict  with keys:
        ssim_score       – float, global similarity (1 = identical, 0 = totally different)
        damage_percent   – float, % of image area flagged as changed
        victim_zones     – list of dicts, each with:
                               id, rank, x, y, w, h, area_px, confidence,
                               centre_lat, centre_lon
        output_image_url – str, URL path to annotated image
        error            – str or None
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

    # ── SSIM diff ─────────────────────────────────────────────────────────────
    ssim_score, diff = ssim(
        ref_gray, post_gray,
        win_size=SSIM_WIN_SIZE,
        full=True,
        data_range=255,
    )
    # diff is in [-1, 1]; invert so damaged areas are bright
    diff_uint8 = (np.clip((1 - diff) / 2, 0, 1) * 255).astype(np.uint8)

    # ── Threshold + morphological cleanup ────────────────────────────────────
    _, thresh = cv2.threshold(diff_uint8, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh    = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh    = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel, iterations=1)

    damage_percent = round(float(np.count_nonzero(thresh)) / thresh.size * 100, 2)

    # ── Find contours (changed blobs) ────────────────────────────────────────
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours    = [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]
    contours    = sorted(contours, key=cv2.contourArea, reverse=True)[:MAX_VICTIM_ZONES]

    # ── Geo-localise each zone (pixel → lat/lon using region bounding box) ───
    from app import REGIONS as APP_REGIONS   # import from sibling file
    region_cfg = APP_REGIONS.get(region_key, {})
    lat_start  = region_cfg.get("lat_start", 0)
    lat_end    = region_cfg.get("lat_end",   0)
    lon_start  = region_cfg.get("lon_start", 0)
    lon_end    = region_cfg.get("lon_end",   0)
    lat_range  = lat_end  - lat_start
    lon_range  = lon_end  - lon_start

    def pixel_to_latlon(cx_px, cy_px):
        lon = lon_start + (cx_px / TARGET_W) * lon_range
        lat = lat_end   - (cy_px / TARGET_H) * lat_range   # y-axis flipped
        return round(lat, 6), round(lon, 6)

    # ── Annotate result image ─────────────────────────────────────────────────
    annotated    = post_resized.copy()
    victim_zones = []

    for rank, cnt in enumerate(contours, start=1):
        x, y, w, h = cv2.boundingRect(cnt)
        area        = int(cv2.contourArea(cnt))
        cx, cy      = x + w // 2, y + h // 2
        clat, clon  = pixel_to_latlon(cx, cy)

        # confidence: normalise area against image size (0–1)
        confidence = round(min(area / (TARGET_W * TARGET_H * 0.10), 1.0), 3)

        victim_zones.append({
            "id":         f"VZ{rank:02d}",
            "rank":       rank,
            "x":          x,  "y": y,  "w": w,  "h": h,
            "area_px":    area,
            "confidence": confidence,
            "centre_lat": clat,
            "centre_lon": clon,
        })

        # Colour: red = high confidence, yellow = medium, cyan = low
        if confidence > 0.6:
            colour = (0, 0, 255)
        elif confidence > 0.3:
            colour = (0, 200, 255)
        else:
            colour = (255, 220, 0)

        cv2.rectangle(annotated, (x, y), (x + w, y + h), colour, 2)
        label = f"VZ{rank:02d} {confidence * 100:.0f}%"
        cv2.putText(annotated, label, (x, max(y - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)

    # ── Legend / header bar ───────────────────────────────────────────────────
    bar = np.zeros((30, TARGET_W, 3), dtype=np.uint8)
    cv2.putText(
        bar,
        f"SSIM: {ssim_score:.3f}  |  Damage: {damage_percent}%  |  Zones: {len(victim_zones)}",
        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
    )
    annotated = np.vstack([bar, annotated])

    # ── Save annotated output ─────────────────────────────────────────────────
    out_filename = f"{region_key}_{uuid.uuid4().hex[:8]}.jpg"
    out_path     = os.path.join(OUTPUT_DIR, out_filename)
    cv2.imwrite(out_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])

    return {
        "ssim_score":       round(float(ssim_score), 4),
        "damage_percent":   damage_percent,
        "victim_zones":     victim_zones,
        "output_image_url": f"/static/rescue_output/{out_filename}",
        "error":            None,
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
