// static/js/app.js

document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("upload-form");
    const fileInput = document.getElementById("image-input");
    const fileLabel = document.getElementById("file-label-text");
    const previewImg = document.getElementById("preview-image");
    const errorText = document.getElementById("error-text");
    const submitBtn = document.getElementById("submit-btn");
    const loadingText = document.getElementById("loading-text");

    if (!form || !fileInput) {
        // IDs not found: nothing to wire up
        return;
    }

    // When user selects a file
    fileInput.addEventListener("change", () => {
        errorText && (errorText.textContent = "");

        const file = fileInput.files[0];

        if (!file) {
            if (fileLabel) fileLabel.textContent = "Choose a CT image...";
            if (previewImg) {
                previewImg.src = "";
                previewImg.style.display = "none";
            }
            return;
        }

        // Update label text
        if (fileLabel) {
            fileLabel.textContent = file.name;
        }

        // Basic file type check
        const allowedTypes = ["image/png", "image/jpeg", "image/jpg"];
        if (!allowedTypes.includes(file.type)) {
            if (errorText) {
                errorText.textContent = "Please select a PNG or JPG image.";
            }
            fileInput.value = "";
            if (previewImg) {
                previewImg.src = "";
                previewImg.style.display = "none";
            }
            return;
        }

        // Show preview
        if (previewImg) {
            const reader = new FileReader();
            reader.onload = e => {
                previewImg.src = e.target.result;
                previewImg.style.display = "block";
            };
            reader.readAsDataURL(file);
        }
    });

    // On form submit
    form.addEventListener("submit", (e) => {
        const file = fileInput.files[0];

        if (!file) {
            e.preventDefault();
            if (errorText) {
                errorText.textContent = "Please choose an image before predicting.";
            }
            return;
        }

        // Show loading state
        if (submitBtn) {
            submitBtn.disabled = true;
            submitBtn.textContent = "Predicting...";
        }
        if (loadingText) {
            loadingText.style.display = "block";
        }
    });
});
