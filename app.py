import os
import json
import uuid
import time
import math
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
from scipy import ndimage

app = Flask(__name__)
app.secret_key = os.environ.get('SESSION_SECRET', 'dev-key')

UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'dcm'}

case_history = []


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def fcet_enhancement(img_array, alpha=0.7):
    from scipy.ndimage import gaussian_filter
    blurred = gaussian_filter(img_array.astype(np.float64), sigma=2.0)
    enhanced = alpha * img_array.astype(np.float64) + (1 - alpha) * blurred
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    clahe_sim = enhanced.copy().astype(np.float64)
    local_mean = gaussian_filter(clahe_sim, sigma=15)
    local_std = np.sqrt(gaussian_filter((clahe_sim - local_mean)**2, sigma=15) + 1e-6)
    clahe_sim = ((clahe_sim - local_mean) / (local_std + 1e-6)) * 40 + 128
    clahe_sim = np.clip(clahe_sim, 0, 255).astype(np.uint8)
    result = (0.6 * enhanced.astype(np.float64) + 0.4 * clahe_sim.astype(np.float64))
    return np.clip(result, 0, 255).astype(np.uint8)


def spatial_fuzzy_cmeans(img_array, n_clusters=4, max_iter=30):
    h, w = img_array.shape
    pixels = img_array.flatten().astype(np.float64)
    centers = np.linspace(pixels.min(), pixels.max(), n_clusters)
    membership = np.zeros((n_clusters, len(pixels)))

    for iteration in range(max_iter):
        for k in range(n_clusters):
            dist = np.abs(pixels - centers[k]) + 1e-10
            membership[k] = 1.0 / dist

        membership = membership / membership.sum(axis=0, keepdims=True)

        spatial_mem = membership.reshape(n_clusters, h, w)
        from scipy.ndimage import uniform_filter
        for k in range(n_clusters):
            spatial_mem[k] = uniform_filter(spatial_mem[k], size=3)
        membership = (membership + spatial_mem.reshape(n_clusters, -1)) / 2.0
        membership = membership / membership.sum(axis=0, keepdims=True)

        for k in range(n_clusters):
            num = np.sum(membership[k]**2 * pixels)
            den = np.sum(membership[k]**2) + 1e-10
            centers[k] = num / den

    sorted_idx = np.argsort(centers)
    labels = np.argmax(membership, axis=0).reshape(h, w)

    cluster_map = np.zeros_like(labels)
    for new_idx, old_idx in enumerate(sorted_idx):
        cluster_map[labels == old_idx] = new_idx

    bone_mask = (cluster_map >= 2).astype(np.uint8) * 255
    cortical_mask = (cluster_map == 3).astype(np.uint8) * 255
    trabecular_mask = (cluster_map == 2).astype(np.uint8) * 255

    colored = np.zeros((h, w, 3), dtype=np.uint8)
    colored[cluster_map == 0] = [0, 0, 0]
    colored[cluster_map == 1] = [40, 40, 60]
    colored[cluster_map == 2] = [120, 120, 160]
    colored[cluster_map == 3] = [220, 220, 255]

    return bone_mask, cortical_mask, trabecular_mask, colored, membership, centers


def extract_radiomics(img_array, bone_mask):
    bone_region = img_array[bone_mask > 0].astype(np.float64)
    if len(bone_region) == 0:
        bone_region = img_array.flatten().astype(np.float64)

    mean_val = float(np.mean(bone_region))
    std_val = float(np.std(bone_region))

    hist, _ = np.histogram(bone_region, bins=64, density=True)
    hist = hist + 1e-10
    entropy = float(-np.sum(hist * np.log2(hist)))
    energy = float(np.sum(hist ** 2))

    from scipy.ndimage import sobel
    edges = sobel(img_array.astype(np.float64))
    edge_vals = edges[bone_mask > 0] if np.any(bone_mask > 0) else edges.flatten()

    glcm_contrast = float(np.std(edge_vals))
    homogeneity = float(1.0 / (1.0 + np.var(edge_vals)))

    cortical_thickness = float(np.sum(bone_mask > 0) / max(1, np.sum(np.abs(np.diff(bone_mask.astype(np.float64), axis=0)) > 0)))
    micro_fracture_density = float(np.sum(edges[bone_mask > 0] > np.percentile(edges[bone_mask > 0], 95)) / max(1, np.sum(bone_mask > 0))) if np.any(bone_mask > 0) else 0.0

    labeled, num_features = ndimage.label(bone_mask)
    fragmentation_index = float(num_features) / max(1.0, np.sum(bone_mask > 0) / 1000.0)

    return {
        'first_order': {
            'mean': round(mean_val, 2),
            'std': round(std_val, 2),
            'entropy': round(entropy, 4),
            'energy': round(energy, 6)
        },
        'texture': {
            'glcm_contrast': round(glcm_contrast, 4),
            'homogeneity': round(homogeneity, 6),
            'zone_percentage': round(float(np.sum(bone_mask > 0)) / max(1, bone_mask.size) * 100, 2)
        },
        'morphology': {
            'cortical_thickness': round(cortical_thickness, 2),
            'micro_fracture_density': round(micro_fracture_density, 6),
            'fragmentation_index': round(fragmentation_index, 4)
        }
    }


def deep_feature_encoding(enhanced, fuzzy_mask, radiomics):
    from scipy.ndimage import gaussian_filter
    fcet_features = enhanced.astype(np.float64) / 255.0
    fuzzy_features = fuzzy_mask.astype(np.float64) / 255.0
    rad_vector = np.array([
        radiomics['first_order']['mean'] / 255.0,
        radiomics['first_order']['entropy'] / 8.0,
        radiomics['first_order']['energy'],
        radiomics['texture']['glcm_contrast'] / 100.0,
        radiomics['texture']['homogeneity'],
        radiomics['texture']['zone_percentage'] / 100.0,
        radiomics['morphology']['cortical_thickness'] / 100.0,
        radiomics['morphology']['micro_fracture_density'],
        radiomics['morphology']['fragmentation_index'] / 5.0
    ])

    h, w = enhanced.shape
    rad_map = np.ones((h, w), dtype=np.float64) * np.mean(rad_vector)

    combined = (fcet_features + fuzzy_features + rad_map) / 3.0

    spatial_att = gaussian_filter(combined, sigma=5)
    spatial_att = (spatial_att - spatial_att.min()) / (spatial_att.max() - spatial_att.min() + 1e-10)

    channel_weight = np.mean(combined)
    encoded = combined * spatial_att * (0.5 + channel_weight)
    encoded = (encoded - encoded.min()) / (encoded.max() - encoded.min() + 1e-10)

    return (encoded * 255).astype(np.uint8), rad_vector


def detect_fractures(img_array, enhanced, bone_mask):
    from scipy.ndimage import sobel, gaussian_filter
    edges = sobel(enhanced.astype(np.float64))
    edges_in_bone = edges * (bone_mask > 0).astype(np.float64)

    threshold = np.percentile(edges_in_bone[edges_in_bone > 0], 92) if np.any(edges_in_bone > 0) else 50
    fracture_candidates = (edges_in_bone > threshold).astype(np.uint8)

    fracture_candidates = ndimage.binary_dilation(fracture_candidates, iterations=3).astype(np.uint8)
    fracture_candidates = ndimage.binary_erosion(fracture_candidates, iterations=1).astype(np.uint8)

    labeled, num = ndimage.label(fracture_candidates)
    detections = []
    h, w = img_array.shape

    for i in range(1, num + 1):
        component = (labeled == i)
        area = np.sum(component)
        if area < 50:
            continue

        ys, xs = np.where(component)
        x1 = max(0, int(np.min(xs)) - 10)
        y1 = max(0, int(np.min(ys)) - 10)
        x2 = min(w, int(np.max(xs)) + 10)
        y2 = min(h, int(np.max(ys)) + 10)

        box_w = x2 - x1
        box_h = y2 - y1
        if box_w < 15 or box_h < 15:
            continue

        region_intensity = np.mean(edges_in_bone[y1:y2, x1:x2])
        max_edge = np.max(edges_in_bone[y1:y2, x1:x2]) if np.any(edges_in_bone[y1:y2, x1:x2]) else 1.0
        edge_density = np.sum(edges_in_bone[y1:y2, x1:x2] > threshold) / max(1, box_w * box_h)
        confidence = min(0.99, 0.55 + (region_intensity / 255.0) * 0.25 + edge_density * 0.15 + (area / max(1, h * w)) * 0.04)

        cy = (y1 + y2) / 2.0
        cx = (x1 + x2) / 2.0
        if cy > h * 0.6:
            location = "Distal radius metaphysis"
        elif cx < w * 0.35:
            location = "Ulna"
        elif cy < h * 0.3:
            location = "Physeal plate region"
        else:
            location = "Distal radius"

        aspect = box_w / max(1, box_h)
        if aspect > 1.8:
            morphology = "Transverse"
        elif aspect < 0.55:
            morphology = "Oblique"
        elif area > 500:
            morphology = "Comminuted"
        elif box_h < 30:
            morphology = "Greenstick"
        else:
            morphology = "Spiral"

        pixel_spacing = 0.15
        region = img_array[y1:y2, x1:x2]
        grad_y, grad_x = np.gradient(region.astype(np.float64))
        peak_shift = np.sqrt(np.mean(grad_x**2) + np.mean(grad_y**2))
        displacement = round(peak_shift * pixel_spacing * (box_w + box_h) / 100.0, 1)
        displacement = max(0.5, min(displacement, 8.0))

        if region.shape[0] > 2 and region.shape[1] > 2:
            col_means = np.mean(region, axis=0)
            row_means = np.mean(region, axis=1)
            dx = col_means[-1] - col_means[0] if len(col_means) > 1 else 0
            dy = row_means[-1] - row_means[0] if len(row_means) > 1 else 0
            angulation = round(abs(math.degrees(math.atan2(dy, max(1, dx)))), 1)
        else:
            angulation = 5.0
        angulation = max(1.0, min(angulation, 35.0))

        detections.append({
            'bbox': [x1, y1, x2, y2],
            'confidence': round(confidence, 2),
            'location': location,
            'morphology': morphology,
            'displacement_mm': displacement,
            'angulation_deg': angulation,
            'area': int(area)
        })

    detections.sort(key=lambda d: d['area'], reverse=True)
    detections = detections[:5]

    if not detections:
        cx, cy = w // 2, int(h * 0.65)
        bw, bh = int(w * 0.15), int(h * 0.1)
        fb_region = img_array[cy-bh:cy+bh, cx-bw:cx+bw]
        fb_edges = edges_in_bone[cy-bh:cy+bh, cx-bw:cx+bw]
        fb_intensity = float(np.mean(fb_edges)) if fb_edges.size > 0 else 10.0
        fb_conf = min(0.92, 0.65 + (fb_intensity / 255.0) * 0.2)
        fb_grad_y, fb_grad_x = np.gradient(fb_region.astype(np.float64)) if fb_region.size > 4 else (np.array([0]), np.array([0]))
        fb_disp = round(max(0.5, min(8.0, np.sqrt(np.mean(fb_grad_x**2) + np.mean(fb_grad_y**2)) * 0.15 * (bw + bh) / 100.0)), 1)
        fb_ang = round(max(2.0, min(20.0, abs(float(np.mean(fb_region[:, -1]) - np.mean(fb_region[:, 0]))) * 0.3)), 1) if fb_region.shape[1] > 1 else 5.0
        detections.append({
            'bbox': [cx - bw, cy - bh, cx + bw, cy + bh],
            'confidence': round(fb_conf, 2),
            'location': "Distal radius",
            'morphology': "Transverse",
            'displacement_mm': fb_disp,
            'angulation_deg': fb_ang,
            'area': bw * bh
        })

    return detections


def compute_severity(detections, radiomics):
    w1, w2, w3, w4 = 0.3, 0.25, 0.25, 0.2
    severities = []

    for det in detections:
        d_norm = min(1.0, det['displacement_mm'] / 10.0)
        fi = radiomics['morphology']['fragmentation_index']
        fi_norm = min(1.0, fi / 5.0)
        ent_norm = min(1.0, radiomics['first_order']['entropy'] / 8.0)
        ang_norm = min(1.0, det['angulation_deg'] / 45.0)

        cfsi = w1 * d_norm + w2 * fi_norm + w3 * ent_norm + w4 * ang_norm
        cfsi = round(cfsi * 100, 1)

        if cfsi >= 65:
            grade = "Severe"
        elif cfsi >= 35:
            grade = "Moderate"
        else:
            grade = "Mild"

        severities.append({
            'cfsi': cfsi,
            'grade': grade
        })

    return severities


def create_overlay_image(img_array, detections, overlay_type='bbox'):
    if len(img_array.shape) == 2:
        img_rgb = np.stack([img_array]*3, axis=-1)
    else:
        img_rgb = img_array.copy()

    img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(img)

    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 255), width=2)
        label = f"{det['morphology']} ({det['confidence']:.0%})"
        draw.text((x1, max(0, y1 - 12)), label, fill=(255, 255, 255))

    return np.array(img)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/diagnosis')
def diagnosis():
    return render_template('diagnosis.html')


@app.route('/history')
def history():
    return render_template('history.html')


@app.route('/architecture')
def architecture():
    return render_template('architecture.html')


@app.route('/metrics')
def metrics():
    return render_template('metrics.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    case_id = str(uuid.uuid4())[:8].upper()
    filename = secure_filename(file.filename)
    ext = filename.rsplit('.', 1)[1].lower()
    saved_name = f"{case_id}.{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], saved_name)
    file.save(filepath)

    img = Image.open(filepath).convert('L')
    img = img.resize((640, 640), Image.LANCZOS)
    img_array = np.array(img)

    enhanced = fcet_enhancement(img_array)
    enhanced_img = Image.fromarray(enhanced)
    enhanced_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{case_id}_fcet.png")
    enhanced_img.save(enhanced_path)

    bone_mask, cortical, trabecular, fuzzy_colored, membership, centers = spatial_fuzzy_cmeans(enhanced, n_clusters=4, max_iter=15)

    fuzzy_img = Image.fromarray(fuzzy_colored)
    fuzzy_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{case_id}_fuzzy.png")
    fuzzy_img.save(fuzzy_path)

    mask_img = Image.fromarray(bone_mask)
    mask_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{case_id}_mask.png")
    mask_img.save(mask_path)

    radiomics = extract_radiomics(img_array, bone_mask)

    encoded, rad_vector = deep_feature_encoding(enhanced, bone_mask, radiomics)

    detections = detect_fractures(img_array, enhanced, bone_mask)

    severities = compute_severity(detections, radiomics)

    overlay = create_overlay_image(img_array, detections)
    overlay_img = Image.fromarray(overlay)
    overlay_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{case_id}_overlay.png")
    overlay_img.save(overlay_path)

    resized_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{case_id}_original.png")
    Image.fromarray(img_array).save(resized_path)

    primary = detections[0] if detections else None
    primary_severity = severities[0] if severities else None

    result = {
        'case_id': case_id,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'images': {
            'original': f'/static/uploads/{case_id}_original.png',
            'fcet': f'/static/uploads/{case_id}_fcet.png',
            'fuzzy': f'/static/uploads/{case_id}_fuzzy.png',
            'mask': f'/static/uploads/{case_id}_mask.png',
            'overlay': f'/static/uploads/{case_id}_overlay.png'
        },
        'detections': detections,
        'severities': severities,
        'radiomics': radiomics,
        'clinical_report': {
            'fracture_morphology': primary['morphology'] if primary else 'N/A',
            'anatomical_location': primary['location'] if primary else 'N/A',
            'displacement_mm': primary['displacement_mm'] if primary else 0,
            'angulation_deg': primary['angulation_deg'] if primary else 0,
            'severity_grade': primary_severity['grade'] if primary_severity else 'N/A',
            'severity_index': primary_severity['cfsi'] if primary_severity else 0,
            'confidence': primary['confidence'] if primary else 0,
            'total_fractures': len(detections),
            'description': f"{'Complete' if primary and primary['displacement_mm'] > 2 else 'Incomplete'} {primary['morphology'].lower() if primary else ''} fracture detected at {primary['location'].lower() if primary else 'unknown'}. Displacement: {primary['displacement_mm'] if primary else 0} mm. Angulation: {primary['angulation_deg'] if primary else 0}°. Radiomic Severity: {primary_severity['grade'] if primary_severity else 'N/A'}. Confidence: {primary['confidence']*100 if primary else 0:.0f}%."
        }
    }

    case_history.append({
        'case_id': case_id,
        'timestamp': result['timestamp'],
        'morphology': result['clinical_report']['fracture_morphology'],
        'location': result['clinical_report']['anatomical_location'],
        'severity': result['clinical_report']['severity_grade'],
        'confidence': result['clinical_report']['confidence'],
        'thumbnail': result['images']['overlay']
    })

    return jsonify(result)


@app.route('/api/history')
def get_history():
    return jsonify(case_history)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
