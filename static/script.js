document.addEventListener('DOMContentLoaded', () => {

    let datasetMeta = {};

    const els = {
        uploadBtn: document.getElementById('upload-btn'),
        csvUpload: document.getElementById('csv-upload'),
        modelConfig: document.getElementById('model-config'),
        resultsContainer: document.getElementById('results-container'),
        loader: document.getElementById('loader'),

        // Display
        dataShape: document.getElementById('data-shape'),
        dataMissingSummary: document.getElementById('data-missing-summary'),
        dataTypes: document.getElementById('data-types'),
        dataDescribe: document.getElementById('data-describe'),
        correlationImg: document.getElementById('correlation-img'),

        // Config Inputs
        problemType: document.getElementById('problem-type'),
        targetSelect: document.getElementById('target'),
        testSize: document.getElementById('test-size'),
        testSizeVal: document.getElementById('test-size-val'),
        modelType: document.getElementById('model-type'),

        // Feature selector
        featureSelectorCard: document.getElementById('feature-selector-card'),
        featureCheckboxes: document.getElementById('feature-checkboxes'),

        // Execute
        runModelBtn: document.getElementById('run-model-btn'),
        evaluationResults: document.getElementById('evaluation-results'),
        evalContent: document.getElementById('eval-content')
    };

    els.testSize.addEventListener('input', (e) => {
        els.testSizeVal.textContent = e.target.value;
    });

    els.uploadBtn.addEventListener('click', async () => {
        const file = els.csvUpload.files[0];
        if (!file) return alert('Please select a CSV file first.');

        const formData = new FormData();
        formData.append('file', file);

        els.loader.classList.remove('hidden');

        try {
            const res = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();

            if (res.ok) {
                datasetMeta = data;
                renderDatasetOverview(data);
                populateConfigSelectors(data);
                els.modelConfig.classList.remove('hidden');
                els.resultsContainer.classList.remove('hidden');
            } else {
                alert(data.error);
            }
        } catch (err) {
            console.error(err);
            alert('Upload failed.');
        } finally {
            els.loader.classList.add('hidden');
        }
    });

    els.problemType.addEventListener('change', () => {
        updateTargetOptions();
        updateModelOptions();
    });

    els.targetSelect.addEventListener('change', () => {
        renderFeatureCheckboxes();
    });

    function renderDatasetOverview(data) {
        els.dataShape.textContent = `Rows: ${data.shape[0]}, Columns: ${data.shape[1]}`;

        let missingText = Object.entries(data.missing)
            .filter(([_, val]) => val > 0)
            .map(([col, val]) => `${col}: ${val}`)
            .join('\n');
        els.dataMissingSummary.textContent = missingText || "No missing values!";

        els.dataTypes.textContent = JSON.stringify(data.dtypes, null, 2);

        // Pretty print describe
        const descHTML = Object.entries(data.describe).map(([col, stats]) => {
            return `[${col}]\n` + Object.entries(stats).map(([k, v]) => `  ${k}: ${Number(v).toFixed(2)}`).join('\n');
        }).join('\n\n');
        els.dataDescribe.textContent = descHTML;

        if (data.corr_img) {
            els.correlationImg.src = data.corr_img;
        }
    }

    function populateConfigSelectors(data) {
        updateTargetOptions();
        updateModelOptions();
        renderFeatureCheckboxes();
    }

    function updateTargetOptions() {
        const type = els.problemType.value;
        const targets = type === 'Classification'
            ? datasetMeta.targets_classification
            : datasetMeta.targets_regression;

        els.targetSelect.innerHTML = '';
        targets.forEach(t => {
            const opt = document.createElement('option');
            opt.value = t;
            opt.textContent = t;
            els.targetSelect.appendChild(opt);
        });
        renderFeatureCheckboxes();
    }

    function updateModelOptions() {
        const type = els.problemType.value;
        const models = type === 'Classification'
            ? ['Gaussian', 'Multinomial', 'Bernoulli']
            : ['Linear Regression', 'Random Forest'];

        els.modelType.innerHTML = '';
        models.forEach(m => {
            const opt = document.createElement('option');
            opt.value = m;
            opt.textContent = m;
            els.modelType.appendChild(opt);
        });
    }

    function renderFeatureCheckboxes() {
        const target = els.targetSelect.value;
        els.featureCheckboxes.innerHTML = '';

        datasetMeta.all_columns.forEach(col => {
            if (col !== target) {
                const div = document.createElement('label');
                div.className = 'checkbox-item';
                div.innerHTML = `
                    <input type="checkbox" value="${col}" checked>
                    <span>${col}</span>
                `;
                els.featureCheckboxes.appendChild(div);
            }
        });

        if (datasetMeta.all_columns.length > 0) {
            els.featureSelectorCard.classList.remove('hidden');
        }
    }

    els.runModelBtn.addEventListener('click', async () => {
        const selectedFeatures = Array.from(
            els.featureCheckboxes.querySelectorAll('input:checked')
        ).map(cb => cb.value);

        if (selectedFeatures.length === 0) {
            return alert("Select at least one feature.");
        }

        const payload = {
            problem_type: els.problemType.value,
            target: els.targetSelect.value,
            features: selectedFeatures,
            test_size: parseFloat(els.testSize.value),
            model_type: els.modelType.value
        };

        els.loader.classList.remove('hidden');

        try {
            const res = await fetch('/api/run_model', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();

            if (res.ok) {
                renderEvaluation(data);
                els.evaluationResults.classList.remove('hidden');
                els.evaluationResults.scrollIntoView({ behavior: 'smooth' });
            } else {
                alert(data.error);
            }
        } catch (err) {
            console.error(err);
            alert("Model training failed.");
        } finally {
            els.loader.classList.add('hidden');
        }
    });

    function renderEvaluation(data) {
        els.evalContent.innerHTML = '';

        if (data.problem_type === 'Classification') {
            els.evalContent.innerHTML = `
                <div>
                    <div class="metric-card">
                        <h3>${(data.train_acc * 100).toFixed(2)}%</h3>
                        <p>Train Accuracy</p>
                    </div>
                    <div class="img-container mt-2">
                        <img src="${data.train_cm_img}" alt="Train CM"/>
                    </div>
                </div>
                <div>
                    <div class="metric-card">
                        <h3>${(data.test_acc * 100).toFixed(2)}%</h3>
                        <p>Test Accuracy</p>
                    </div>
                    <div class="img-container mt-2">
                        <img src="${data.test_cm_img}" alt="Test CM"/>
                    </div>
                </div>
            `;
        } else {
            els.evalContent.innerHTML = `
                <div>
                    <div class="metric-card mb-2">
                        <h3>${data.train_r2}</h3>
                        <p>Train R² Score</p>
                    </div>
                    <div class="metric-card">
                        <h3>${data.train_mse}</h3>
                        <p>Train MSE</p>
                    </div>
                </div>
                <div>
                    <div class="metric-card mb-2">
                        <h3>${data.test_r2}</h3>
                        <p>Test R² Score</p>
                    </div>
                    <div class="metric-card">
                        <h3>${data.test_mse}</h3>
                        <p>Test MSE</p>
                    </div>
                </div>
            `;
        }
    }
});
