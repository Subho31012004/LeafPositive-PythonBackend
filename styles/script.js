document.addEventListener("DOMContentLoaded", () => {
  const dropZone = document.getElementById("dropZone");
  const fileInput = document.getElementById("fileInput");
  const imagePreview = document.getElementById("imagePreview");
  const previewContainer = document.querySelector(".preview-container");
  const uploadContainer = document.querySelector(".upload-container");
  const classifyBtn = document.querySelector(".classify-btn");
  const result = document.querySelector(".result");
  const diseaseType = document.querySelector(".disease-type");
  const diseaseIcon = document.querySelector(".disease-icon");
  const resultText = document.querySelector(".result-text");
  const tryAgainBtn = document.querySelector(".try-again-btn");
  const modelSelect = document.getElementById("modelSelect");
  const modelInfo = document.querySelector(".model-info");
  const modelName = document.querySelector(".model-name");
  const removeImageBtn = document.querySelector(".remove-image-btn");

  let currentFile = null;

  // Drag-and-drop handling
  ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
    dropZone.addEventListener(eventName, preventDefaults);
  });

  function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
  }

  ["dragenter", "dragover"].forEach((eventName) => {
    dropZone.addEventListener(eventName, () => {
      dropZone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropZone.addEventListener(eventName, () => {
      dropZone.classList.remove("dragover");
    });
  });

  dropZone.addEventListener("drop", (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;

    if (files.length > 0 && files[0].type.startsWith("image/")) {
      handleFile(files[0]);
    } else {
      alert("Please drop an image file");
    }
  });

  dropZone.addEventListener("click", () => {
    fileInput.click();
  });

  fileInput.addEventListener("change", (e) => {
    if (
      e.target.files.length > 0 &&
      e.target.files[0].type.startsWith("image/")
    ) {
      handleFile(e.target.files[0]);
    } else {
      alert("Please select an image file");
    }
  });

  function handleFile(file) {
    currentFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      imagePreview.src = e.target.result;
      dropZone.style.display = "none";
      previewContainer.classList.remove("hidden");
      previewContainer.classList.remove("removing");
      setTimeout(() => {
        removeImageBtn.style.transform = "scale(1)";
        removeImageBtn.style.opacity = "1";
      }, 300);
    };
    reader.readAsDataURL(file);
  }

  removeImageBtn.addEventListener("click", () => {
    previewContainer.classList.add("removing");
    setTimeout(() => {
      previewContainer.classList.add("hidden");
      dropZone.style.display = "block";
      currentFile = null;
      fileInput.value = "";
      removeImageBtn.style.transform = "scale(0)";
      removeImageBtn.style.opacity = "0";
    }, 300);
  });

  classifyBtn.addEventListener("click", async () => {
    if (!currentFile) {
      alert("Please select an image first");
      return;
    }

    classifyBtn.disabled = true;
    classifyBtn.textContent = "Analyzing...";

    try {
      const formData = new FormData();
      formData.append("image", currentFile);
      formData.append("model", modelSelect.value);

      const response = await fetch("http://localhost:5000/classify", {
        method: "POST",
        body: formData,
        mode: "cors",
      });

      if (!response.ok) throw new Error("Analysis failed");

      const data = await response.json();

      uploadContainer.classList.add("hidden");
      previewContainer.classList.add("hidden");
      result.classList.remove("hidden");
      result.classList.add("visible");

      // Add visible class to show the result animation
      diseaseIcon.classList.add("visible");
      resultText.classList.add("visible");
      modelInfo.classList.add("visible");

      diseaseType.textContent = `${data.category}`;
      resultText.innerHTML = `Disease Status: <strong>${data.category}</strong> <br> Confidence: <strong>${(data.confidence * 100).toFixed(2)}%</strong>`;
      diseaseIcon.className = `disease-icon visible ${
        data.category.toLowerCase() === "healthy" ? "healthy" : "diseased"
      }`;
      modelName.textContent = modelSelect.options[modelSelect.selectedIndex].text;
    } catch (error) {
      console.error("Error:", error);
      alert("Failed to analyze rice plant. Please try again.");
    } finally {
      classifyBtn.disabled = false;
      classifyBtn.textContent = "Analyze Plant";
    }
  });

  tryAgainBtn.addEventListener("click", () => {
    uploadContainer.classList.remove("hidden");
    dropZone.style.display = "block";
    previewContainer.classList.add("hidden");
    result.classList.remove("visible");
    diseaseIcon.className = "disease-icon";
    resultText.classList.remove("visible");
    modelInfo.classList.remove("visible");
    currentFile = null;
    removeImageBtn.style.transform = "scale(0)";
    removeImageBtn.style.opacity = "0";
    setTimeout(() => {
      result.classList.add("hidden");
      classifyBtn.disabled = false;
      classifyBtn.textContent = "Analyze Plant";
      fileInput.value = "";
    }, 500);
  });
});