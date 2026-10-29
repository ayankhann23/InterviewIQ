document.addEventListener('DOMContentLoaded', () => {
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('resume-file-input');
  const selectedBox = document.getElementById('file-selected-box');
  const fileNameDisplay = document.getElementById('file-selected-name');
  const fileSizeDisplay = document.getElementById('file-selected-size');
  const clearFileBtn = document.getElementById('btn-clear-file');
  const uploadForm = document.getElementById('resume-upload-form');
  const submitBtn = document.getElementById('btn-submit');
  const copyBtn = document.getElementById('btn-copy-text');
  const resumeRawText = document.getElementById('resume-raw-text');

  // Format file size
  function formatBytes(bytes, decimals = 1) {
    if (!bytes || bytes === 0) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
  }

  // Reset any previously valid selection before validating a new file.
  function resetFileSelection(clearInput = false) {
    if (clearInput && fileInput) fileInput.value = '';
    if (selectedBox) selectedBox.style.display = 'none';
    if (fileNameDisplay) fileNameDisplay.textContent = '';
    if (fileSizeDisplay) fileSizeDisplay.textContent = '';
    if (submitBtn) submitBtn.setAttribute('disabled', 'true');
  }

  // Handle file selection
  function handleFile(file) {
    if (!file) {
      resetFileSelection();
      return;
    }

    resetFileSelection();

    if (!file.name.toLowerCase().endsWith('.pdf')) {
      alert('Please select a valid PDF document.');
      if (fileInput) fileInput.value = '';
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      alert('File size exceeds the 10 MB limit. Please select a smaller PDF.');
      if (fileInput) fileInput.value = '';
      return;
    }


    fileNameDisplay.textContent = file.name;
    fileSizeDisplay.textContent = formatBytes(file.size);
    selectedBox.style.display = 'flex';
    submitBtn.removeAttribute('disabled');
  }

  if (dropzone && fileInput) {
    dropzone.addEventListener('click', () => {
      fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        handleFile(e.target.files[0]);
      }
    });

    ['dragenter', 'dragover'].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add('dragover');
      });
    });

    ['dragleave', 'drop'].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove('dragover');
      });
    });

    dropzone.addEventListener('drop', (e) => {
      const dt = e.dataTransfer;
      if (dt.files && dt.files[0]) {
        fileInput.files = dt.files;
        handleFile(dt.files[0]);
      }
    });
  }

  if (clearFileBtn && fileInput) {
    clearFileBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      resetFileSelection(true);
    });
  }

  // Loading indicator on form submit
  if (uploadForm && submitBtn) {
    uploadForm.addEventListener('submit', () => {
      submitBtn.setAttribute('disabled', 'true');
      submitBtn.innerHTML = `
        <svg style="animation: spin 1s linear infinite; width: 18px; height: 18px;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10" stroke-opacity="0.25"></circle>
          <path d="M12 2a10 10 0 0 1 10 10" stroke-linecap="round"></path>
        </svg>
        <span>Analyzing Resume...</span>
      `;
    });
  }

  // Copy resume text to clipboard
  if (copyBtn && resumeRawText) {
    copyBtn.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(resumeRawText.innerText);
        const originalHtml = copyBtn.innerHTML;
        copyBtn.innerHTML = `<span>✓ Copied!</span>`;
        setTimeout(() => {
          copyBtn.innerHTML = originalHtml;
        }, 2000);
      } catch (err) {
        console.error('Failed to copy text', err);
      }
    });
  }

  // Question category tabs filter
  const tabBtns = document.querySelectorAll('.interview-tab-btn');
  const questionCards = document.querySelectorAll('.question-card');

  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      tabBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const filter = btn.getAttribute('data-filter');

      questionCards.forEach(card => {
        const category = card.getAttribute('data-category');
        if (filter === 'all' || category === filter) {
          card.style.display = 'block';
        } else {
          card.style.display = 'none';
        }
      });
    });
  });

  // Practice box toggle
  const togglePracticeBtns = document.querySelectorAll('.practice-toggle-btn');
  togglePracticeBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-target');
      const targetBox = document.getElementById(targetId);
      if (targetBox) {
        const isHidden = targetBox.style.display === 'none' || targetBox.style.display === '';
        targetBox.style.display = isHidden ? 'block' : 'none';
        btn.innerHTML = isHidden 
          ? `<span>✕ Close Practice Space</span>` 
          : `<span>✏️ Practice Answering</span>`;
      }
    });
  });
});
