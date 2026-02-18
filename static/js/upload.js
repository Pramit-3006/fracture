const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const fileName = document.getElementById('fileName');
const analyzeBtn = document.getElementById('analyzeBtn');
const processingOverlay = document.getElementById('processingOverlay');
const processingStage = document.getElementById('processingStage');

let selectedFile = null;

const stages = [
    'Stage 1 — Input Acquisition',
    'Stage 2 — FCET Enhancement',
    'Stage 3 — Spatial Fuzzy C-Means Segmentation',
    'Stage 4 — Cellular Radiomics Extraction',
    'Stage 5 — Deep Feature Encoding',
    'Stage 6 — Fracture Detection',
    'Stage 7 — Orthopedic Fracture Analytics',
    'Stage 8 — Composite Severity Index'
];

uploadZone.addEventListener('click', () => fileInput.click());

uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
});

uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('dragover');
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
        handleFile(fileInput.files[0]);
    }
});

function handleFile(file) {
    selectedFile = file;
    fileName.textContent = file.name;
    fileName.classList.add('visible');
    analyzeBtn.classList.add('visible');
    analyzeBtn.disabled = false;
}

analyzeBtn.addEventListener('click', () => {
    if (!selectedFile) return;
    analyzeBtn.disabled = true;
    processingOverlay.classList.add('active');

    let stageIdx = 0;
    const stageInterval = setInterval(() => {
        stageIdx++;
        if (stageIdx < stages.length) {
            processingStage.textContent = stages[stageIdx];
        }
    }, 800);

    const formData = new FormData();
    formData.append('file', selectedFile);

    fetch('/api/analyze', {
        method: 'POST',
        body: formData
    })
    .then(r => r.json())
    .then(data => {
        clearInterval(stageInterval);
        if (data.error) {
            processingOverlay.classList.remove('active');
            analyzeBtn.disabled = false;
            alert(data.error);
            return;
        }
        sessionStorage.setItem('analysisResult', JSON.stringify(data));
        window.location.href = '/diagnosis';
    })
    .catch(err => {
        clearInterval(stageInterval);
        processingOverlay.classList.remove('active');
        analyzeBtn.disabled = false;
        alert('Analysis failed. Please try again.');
    });
});
