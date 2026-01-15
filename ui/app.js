document.addEventListener('DOMContentLoaded', () => {
  const uploadBtn = document.getElementById('uploadBtn');
  const fileInput = document.getElementById('fileInput');
  const statusDiv = document.getElementById('status');
  const resultDiv = document.getElementById('result');

  // 1. Point to Nginx (Relative Path)
  const API_BASE = "/api";
  const MEDIA_BASE = "/media/results";

  uploadBtn.addEventListener('click', async () => {
    const file = fileInput.files[0];
    const context = document.getElementById('contextSelect').value;
    if (!file) return alert("Select a file first.");

    // UI Reset
    statusDiv.innerHTML = '<div class="spinner"></div> Uploading to KCE Brain...';
    resultDiv.innerHTML = '';
    resultDiv.style.display = 'none';
    uploadBtn.disabled = true;

    const formData = new FormData();
    formData.append('file', file);
    formData.append('context', context);

    try {
      // 2. Upload to Ingestor via Nginx
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

      statusDiv.innerHTML = `<div class="spinner"></div> 🧠 Brain Analysis in progress (ID: ${taskId.substr(0, 8)})...`;

      // 3. Start Polling for the VTT
      pollForResult(taskId);

    } catch (e) {
      statusDiv.innerText = `Error: ${e.message}`;
      statusDiv.style.color = "red";
      uploadBtn.disabled = false;
    }
  });

  async function pollForResult(taskId) {
    const vttUrl = `${MEDIA_BASE}/${taskId}.vtt`;
    let attempts = 0;
    const maxAttempts = 60; // Timeout after ~2 minutes

    const interval = setInterval(async () => {
      attempts++;
      try {
        // Check if Nginx can serve the file yet
        const res = await fetch(vttUrl, { method: 'HEAD' });

        if (res.ok) {
          // 4. Success! File exists.
          clearInterval(interval);
          displaySuccess(taskId, vttUrl);
        } else if (attempts >= maxAttempts) {
          clearInterval(interval);
          statusDiv.innerText = "Timeout: Analysis took too long.";
          uploadBtn.disabled = false;
        }
      } catch (err) {
        console.error("Polling error", err);
      }
    }, 2000); // Check every 2 seconds
  }

  function displaySuccess(taskId, vttUrl) {
    statusDiv.innerHTML = `✅ <strong>Analysis Complete!</strong>`;
    statusDiv.style.color = "#00ff00"; // Hacker Green

    resultDiv.style.display = 'block';
    resultDiv.innerHTML = `
      <p>The KCE Brain has generated the trinity context.</p>
      <a href="${vttUrl}" target="_blank" class="download-btn">Download VTT Subtitles</a>
      <br><br>
      <small>Task ID: ${taskId}</small>
    `;

    uploadBtn.disabled = false;
  }
});