fetch('/api/history')
    .then(r => r.json())
    .then(data => {
        const container = document.getElementById('historyContent');
        if (data.length === 0) return;

        let html = '<table class="history-table"><thead><tr>';
        html += '<th>Case ID</th><th>Timestamp</th><th>Morphology</th>';
        html += '<th>Location</th><th>Severity</th><th>Confidence</th>';
        html += '</tr></thead><tbody>';

        data.forEach(c => {
            html += `<tr>
                <td>${c.case_id}</td>
                <td>${c.timestamp}</td>
                <td>${c.morphology}</td>
                <td>${c.location}</td>
                <td>${c.severity}</td>
                <td>${(c.confidence * 100).toFixed(0)}%</td>
            </tr>`;
        });

        html += '</tbody></table>';
        container.innerHTML = html;
    });
