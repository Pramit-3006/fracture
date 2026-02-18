# ARFCP — AI Radiographic Fracture Characterization Platform

## Overview
Medical imaging web application for AI-assisted radiographic fracture characterization. Features a black-white minimal clinical interface following radiology workstation design philosophy.

## Project Architecture
- **Backend**: Python 3.11 + Flask
- **Frontend**: HTML5 + CSS3 + Vanilla JavaScript
- **Image Processing**: NumPy, Pillow, SciPy
- **Design**: Monochrome clinical interface (pure black #000000 + white #FFFFFF)
- **Fonts**: IBM Plex Sans (primary), JetBrains Mono (metrics)

## Structure
```
app.py                  - Flask application with full pipeline
templates/
  base.html             - Base template with nav and footer
  index.html            - Landing page with upload zone
  diagnosis.html        - Diagnostic dashboard with split-panel layout
  history.html          - Case history table
  architecture.html     - System architecture documentation
  metrics.html          - Research metrics and formulas
static/
  css/style.css         - Monochrome clinical design system
  js/upload.js          - File upload and processing logic
  js/diagnosis.js       - Dashboard viewer and report rendering
  js/history.js         - Case history data loading
  uploads/              - Uploaded and processed images
```

## Technical Pipeline (8 Stages)
1. Input Acquisition (640x640 grayscale)
2. FCET Enhancement (contrast enhancement)
3. Spatial Fuzzy C-Means Segmentation (4 clusters)
4. Cellular Radiomics Extraction (9 features)
5. Deep Feature Encoding
6. Fracture Detection (edge-based with morphological analysis)
7. Orthopedic Fracture Analytics
8. Composite Fracture Severity Index (CFSI)

## Pages
- `/` - Landing page with drag-and-drop upload
- `/diagnosis` - Split-panel diagnostic dashboard
- `/history` - Case history table
- `/architecture` - 8-stage pipeline documentation
- `/metrics` - Research metrics and formulas

## Running
- Development: `python app.py` (port 5000)
- Production: `gunicorn --bind=0.0.0.0:5000 --reuse-port app:app`
