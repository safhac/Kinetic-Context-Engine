document.addEventListener('DOMContentLoaded', () => {
  const uploadBtn = document.getElementById('uploadBtn');
  const fileInput = document.getElementById('fileInput');
  const statusDiv = document.getElementById('status');
  const resultDiv = document.getElementById('result');

  const API_BASE = "http://localhost:8000";

  uploadBtn.addEventListener('click', async () => {
    const file = fileInput.files[0];
    const context = document.getElementById('contextSelect').value;
    if (!file) return alert("Select a file first.");

    statusDiv.innerHTML = '<div class="spinner"></div> Uploading to Backbone...';
    resultDiv.innerHTML = ''; // Clear previous results
    resultDiv.style.display = 'none';
    uploadBtn.disabled = true;

    const formData = new FormData();
    formData.append('file', file);
    formData.append('context', context);

    try {
      const res = await fetch(`${API_BASE}/ingest/upload`, {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Upload Failed");
      }

      const data = await res.json();
      const taskId = data.task_id;

      statusDiv.innerHTML = `<div class="spinner"></div> Processing (ID: ${taskId.substr(0, 8)})...`;
      listenForCompletion(taskId);

    } catch (e) {
      statusDiv.innerText = `Error: ${e.message}`;
      uploadBtn.disabled = false;
    }
  });

  function listenForCompletion(taskId) {
    const evtSource = new EventSource(`${API_BASE}/ingest/stream/${taskId}`);

    evtSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log("SSE Msg:", data);

      if (data.download_url) {
        renderWorkerLink(data);
      }

      // Final signal logic
      if (data.status === 'completed' && !data.worker_type) {
        evtSource.close();
        statusDiv.innerText = "Analysis Ready."; // This stops the spinner
        renderResult(data);
        uploadBtn.disabled = false;
      }
    };

    evtSource.onerror = (err) => {
      console.error("SSE Error:", err);
    };
  }

  function renderWorkerLink(data) {
    if (!data.download_url) return;
    resultDiv.style.display = 'block';

    // Ensure container exists
    let container = document.getElementById('vtt-container');
    if (!container) {
      container = document.createElement('div');
      container.id = 'vtt-container';
      container.className = 'download-section';
      resultDiv.appendChild(container);
    }

    // Prevent duplicates
    const workerId = `link-${data.worker_type}`;
    if (document.getElementById(workerId)) return;

    const link = document.createElement('a');
    link.id = workerId;
    link.href = `${API_BASE}${data.download_url}`;
    link.className = "download-btn";
    link.innerText = `📥 ${data.worker_type.toUpperCase()} VTT`;

    container.appendChild(link);
  }

  function renderResult(data) {
    resultDiv.style.display = 'block';

    let scoreInfo = document.getElementById('score-info');
    if (!scoreInfo) {
      scoreInfo = document.createElement('div');
      scoreInfo.id = 'score-info';
      resultDiv.prepend(scoreInfo);
    }

    scoreInfo.innerHTML = `
        <h3>Analysis Complete</h3>
        <p>Deception Score: <strong>${data.deception_score || 'N/A'}/100</strong></p>
        <hr>
    `;

    if (!document.getElementById('pdf-report') && data.report_url) {
      const reportLink = document.createElement('a');
      reportLink.id = 'pdf-report';
      reportLink.href = data.report_url;
      reportLink.className = 'report-link';
      reportLink.innerText = 'View Full PDF Report';
      resultDiv.appendChild(reportLink);
    }
  }
});