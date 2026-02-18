let currentData = null;
let currentView = 'overlay';
let zoomLevel = 1;

function init() {
    const stored = sessionStorage.getItem('analysisResult');
    if (!stored) {
        window.location.href = '/';
        return;
    }
    currentData = JSON.parse(stored);
    renderReport();
    switchView('overlay');
}

function switchView(view) {
    currentView = view;
    const img = document.getElementById('viewerImage');
    if (currentData && currentData.images[view]) {
        img.src = currentData.images[view];
    }
    document.querySelectorAll('.viewer-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.view === view);
    });
    resetZoom();
}

function zoomIn() {
    zoomLevel = Math.min(5, zoomLevel + 0.25);
    applyZoom();
}

function zoomOut() {
    zoomLevel = Math.max(0.5, zoomLevel - 0.25);
    applyZoom();
}

function resetZoom() {
    zoomLevel = 1;
    applyZoom();
}

function applyZoom() {
    const img = document.getElementById('viewerImage');
    img.style.transform = `scale(${zoomLevel})`;
}

function renderReport() {
    const r = currentData.clinical_report;

    document.getElementById('caseIdBadge').textContent = `CASE: ${currentData.case_id} — ${currentData.timestamp}`;
    document.getElementById('rMorphology').textContent = r.fracture_morphology;
    document.getElementById('rLocation').textContent = r.anatomical_location;
    document.getElementById('rDisplacement').textContent = `${r.displacement_mm} mm`;
    document.getElementById('rAngulation').textContent = `${r.angulation_deg}°`;
    document.getElementById('rSeverity').textContent = r.severity_index;
    document.getElementById('rConfidence').textContent = `${(r.confidence * 100).toFixed(0)}%`;
    document.getElementById('rTotal').textContent = r.total_fractures;

    const gradeEl = document.getElementById('rGrade');
    gradeEl.textContent = r.severity_grade;
    if (r.severity_grade === 'Severe') gradeEl.classList.add('severe');

    document.getElementById('rDescription').textContent = r.description;

    const detList = document.getElementById('detectionList');
    detList.innerHTML = '';
    currentData.detections.forEach((det, i) => {
        const sev = currentData.severities[i] || {};
        const item = document.createElement('div');
        item.className = 'detection-item';
        item.innerHTML = `
            <div class="detection-item-header">
                <span class="detection-item-title">${det.morphology} Fracture — ${det.location}</span>
                <span class="detection-item-conf">${(det.confidence * 100).toFixed(0)}%</span>
            </div>
            <div class="report-row">
                <span class="report-label">Displacement</span>
                <span class="report-value">${det.displacement_mm} mm</span>
            </div>
            <div class="report-row">
                <span class="report-label">Angulation</span>
                <span class="report-value">${det.angulation_deg}°</span>
            </div>
            <div class="report-row">
                <span class="report-label">CFSI</span>
                <span class="report-value">${sev.cfsi || '—'} (${sev.grade || '—'})</span>
            </div>
        `;
        detList.appendChild(item);
    });

    const grid = document.getElementById('radiomicsGrid');
    grid.innerHTML = '';
    const rad = currentData.radiomics;
    const items = [
        ['Mean', rad.first_order.mean],
        ['Entropy', rad.first_order.entropy],
        ['Energy', rad.first_order.energy],
        ['GLCM Contrast', rad.texture.glcm_contrast],
        ['Homogeneity', rad.texture.homogeneity],
        ['Zone %', rad.texture.zone_percentage + '%'],
        ['Cortical Thick.', rad.morphology.cortical_thickness],
        ['μ-Fracture Dens.', rad.morphology.micro_fracture_density],
        ['Frag. Index', rad.morphology.fragmentation_index]
    ];
    items.forEach(([label, value]) => {
        const el = document.createElement('div');
        el.className = 'radiomics-item';
        el.innerHTML = `
            <div class="radiomics-item-label">${label}</div>
            <div class="radiomics-item-value">${value}</div>
        `;
        grid.appendChild(el);
    });
}

let isDragging = false;
let startX, startY, scrollLeft, scrollTop;

const container = document.getElementById('imageContainer');

container.addEventListener('mousedown', (e) => {
    if (zoomLevel <= 1) return;
    isDragging = true;
    startX = e.pageX - container.offsetLeft;
    startY = e.pageY - container.offsetTop;
    scrollLeft = container.scrollLeft;
    scrollTop = container.scrollTop;
    container.style.cursor = 'grabbing';
});

container.addEventListener('mousemove', (e) => {
    if (!isDragging) return;
    e.preventDefault();
    const x = e.pageX - container.offsetLeft;
    const y = e.pageY - container.offsetTop;
    container.scrollLeft = scrollLeft - (x - startX);
    container.scrollTop = scrollTop - (y - startY);
});

document.addEventListener('mouseup', () => {
    isDragging = false;
    if (container) container.style.cursor = '';
});

init();
