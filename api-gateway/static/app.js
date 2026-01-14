document.addEventListener('DOMContentLoaded', () => {
  const uploadBtn = document.getElementById('uploadBtn');
  const fileInput = document.getElementById('fileInput');
  const statusDiv = document.getElementById('status');
  const resultDiv = document.getElementById('result');

  uploadBtn.addEventListener('click', async () => {
    const file = fileInput.files[0];
    const context = document.getElementById('contextSelect').value;
    if (!file) return alert("Select a file first.");

    // 1. UI Reset
    statusDiv.innerHTML = '<div class="spinner"></div> Uploading to Backbone...';
    resultDiv.style.display = 'none';
    uploadBtn.disabled = true;

    // 2. Perform Upload
    const formData = new FormData();
    formData.append('file', file);
    formData.append('context', context);

    try {
      const res = await fetch('/ingest/upload', { method: 'POST', body: formData });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Upload Failed");
      }

      const data = await res.json();
      const taskId = data.task_id;

      // 3. Start SSE Listener
      statusDiv.innerHTML = `<div class="spinner"></div> Processing (ID: ${taskId.substr(0, 8)})...`;
      listenForCompletion(taskId);

    } catch (e) {
      statusDiv.innerText = `Error: ${e.message}`;
      uploadBtn.disabled = false;
    }
  });

  function listenForCompletion(taskId) {
    const evtSource = new EventSource(`/ingest/stream/${taskId}`);

    evtSource.onmessage = (event) => {
      console.log("SSE Msg:", event.data);
      const data = JSON.parse(event.data);

      if (data.status === 'completed') {
        evtSource.close();
        statusDiv.innerText = "Analysis Ready.";
        renderResult(data);
        uploadBtn.disabled = false;
      } else if (data.status === 'failed') {
        evtSource.close();
        statusDiv.innerText = "Processing Failed.";
        uploadBtn.disabled = false;
      }
    };

    evtSource.onerror = (err) => {
      console.error("SSE Error:", err);
    };
  }

  function renderResult(data) {
    resultDiv.style.display = 'block';

    // Create the download URLs for each worker result
    const taskId = data.task_id;
    const baseUrl = window.location.origin;

    resultDiv.innerHTML = `
            <h3>Analysis Complete</h3>
            <p>Deception Score: <strong>${data.score || 'N/A'}/100</strong></p>
            
            <div class="download-section" style="margin-top: 20px; display: flex; flex-direction: column; gap: 10px;">
                <a href="${baseUrl}/results/download/${taskId}/body" class="download-btn">📥 Download Body VTT</a>
                <a href="${baseUrl}/results/download/${taskId}/face" class="download-btn">📥 Download Face VTT</a>
                <a href="${baseUrl}/results/download/${taskId}/audio" class="download-btn">📥 Download Audio VTT</a>
            </div>

            <hr style="margin: 20px 0;">
            <a href="${data.report_url || '#'}" target="_blank" class="report-link">View Full PDF Report</a>
        `;
  }
});