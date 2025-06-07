document.addEventListener('DOMContentLoaded', function () {
  const slider = document.getElementById("squareSlider");
  const sliderValue = document.getElementById("sliderValue");
  const backgroundImage = document.getElementById('backgroundImage');

  slider.addEventListener("input", (event) => {
    const value = event.target.value;
    const threshold = value / 100; // Convert the value to 0-1 range
    sliderValue.textContent = `${value}%`; // Display the slider value
    if (predictedLabel !== 'Normal' && !(backgroundImage.src.endsWith('/static/loading.gif'))) { // Only draw overlay if the label is not 'Normal'
      drawOverlay(threshold, gradientData);
    }
  });

  document.getElementById('fileInput').addEventListener('change', function () {
    handleFileSelect(this);
  });
});

// Function to update the classification text based on screen size
function updateClassificationText(predictedLabel) {
  const classificationText = document.querySelector('.prediction-text');

  // Check if the screen width is less than 400px
  if (window.matchMedia("(max-width: 400px)").matches) {
    // For smaller screens, display a different text
    classificationText.textContent = `${predictedLabel}`;
  } else {
    // For larger screens, display the normal text
    classificationText.textContent = `Predicted: ${predictedLabel}`;
  }
}

function toggleDropdown() {
  const dropdownOptions = document.getElementById('dropdownOptions');
  dropdownOptions.style.display = dropdownOptions.style.display === 'block' ? 'none' : 'block';
}

function resetState() {
  gradientData = [];
  predictedLabel = null;
  overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
}

// Close the dropdown if the user clicks outside of it
window.onclick = function(event) {
  if (!event.target.matches('.dropdown-button')) {
    const dropdownOptions = document.getElementById('dropdownOptions');
    if (dropdownOptions.style.display === 'block') {
      dropdownOptions.style.display = 'none';
    }
  }
}

document.getElementById("infoIcon").addEventListener("click", () => {
  alert("This application uses Guided Backpropagation to highlight key regions of a chest X-ray based on user-adjustable thresholds. This technique calculates gradients to determine which pixels most influence the model's prediction. By adjusting the threshold slider, users control the percentage of the maximum gradient that must be exceeded for a pixel to be highlighted. Lower percentages show broader, less critical regions, while higher percentages focus on the most relevant areas.");
});

// document.getElementById("infoIcon").addEventListener("click", () => {
//   const infoBox = document.getElementById("infoBox");
//   if (infoBox.style.display === "none" || infoBox.style.display === "") {
//     infoBox.style.display = "block";
//   } else {
//     infoBox.style.display = "none";
//   }
// });

// // Optional: Close the info box if the user clicks outside of it
// document.addEventListener("click", (event) => {
//   const infoBox = document.getElementById("infoBox");
//   const infoIcon = document.getElementById("infoIcon");
//   if (!infoBox.contains(event.target) && !infoIcon.contains(event.target)) {
//     infoBox.style.display = "none";
//   }
// });


function handleDropdownOption1() {
  if (window.innerWidth <= 1030) {
    toggleSidebar();
  }
  const fileInput = document.getElementById('fileInput');
  fileInput.value = ''; // Reset the value of the file input
  document.getElementById('fileInput').click();
}

function resizeCanvasToImage() {
  const backgroundImage = document.getElementById('backgroundImage');
  const overlayCanvas = document.getElementById("overlay");
  const rect = backgroundImage.getBoundingClientRect();
  overlayCanvas.width = rect.width;
  overlayCanvas.height = rect.height;
}

function drawOverlay(threshold, gradientData) {
  const overlayCanvas = document.getElementById("overlay");
  const overlayCtx = overlayCanvas.getContext("2d");

  const originalWidth = 224;
  const originalHeight = 224;

  console.log("drawOverlay called with threshold:", threshold);
  console.log("Gradient data in drawOverlay:", gradientData);

  if (!gradientData || gradientData.length === 0) {
    console.error("Gradient data is empty or undefined.");
    return;
  }


  const maxGradient = Math.max(...gradientData);
  console.log("maxGradient:", maxGradient);

  if (isNaN(maxGradient)) {
    console.error("maxGradient is NaN, indicating invalid gradientData.");
    return;
  }

  const thresholdValue = maxGradient * threshold;
  console.log("thresholdValue:", thresholdValue);

  const imageData = overlayCtx.createImageData(
    overlayCanvas.width,
    overlayCanvas.height
  );

  const width = overlayCanvas.width;
  const height = overlayCanvas.height;

  const leftLimit = Math.floor(width * 0.15);
  const rightLimit = Math.floor(width * 0.85);
  const topLimit = Math.floor(height * 0.15);
  const bottomLimit = Math.floor(height * 0.85);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const index = (y * width + x) * 4;

      if (x < leftLimit || x > rightLimit || y < topLimit || y > bottomLimit) {
        continue;
      }

      const gradientX = Math.floor((x / width) * originalWidth);
      const gradientY = Math.floor((y / height) * originalHeight);
      const gradientValue = gradientData[gradientY * originalWidth + gradientX];

      if (gradientValue > thresholdValue) {
        imageData.data[index] = 255; // Red
        imageData.data[index + 1] = 255; // Green
        imageData.data[index + 2] = 0; // Blue
        imageData.data[index + 3] = 128; // Alpha
      }
    }
  }

  overlayCtx.putImageData(imageData, 0, 0);
  console.log("Overlay updated");
}

function mapSliderValue(value) {
  const minValue = 0;
  const maxValue = 100;
  return minValue + (value / 100) * (maxValue - minValue);
}

function handleFileSelect(fileInput) {
  const backgroundImage = document.getElementById('backgroundImage');
  const overlayCanvas = document.getElementById('overlay');
  const overlayCtx = overlayCanvas.getContext("2d");
  const classificationText = document.querySelector('.prediction-text');
  const additionalText = document.querySelector('.additional-text');
  const sliderContainer = document.getElementById("sliderContainer");
  const slider = document.getElementById("squareSlider");
  const sliderValue = document.getElementById("sliderValue");
  const file = fileInput.files[0];

  overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  backgroundImage.src = '/static/loading.gif';

  if (file) {
    const dropdownButton = document.getElementById('dropdownButton');
    dropdownButton.innerText = file.name;

    const formData = new FormData();
    formData.append("file", file);
    
    backgroundImage.onload = function() {
      console.log("Sample image loaded successfully.");

      fetch("/upload", {
        method: "POST",
        body: formData,
      })
        .then((response) => {
          // Check if the response is successful
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          
          // Check if the response is JSON
          const contentType = response.headers.get("content-type");
          if (!contentType || !contentType.includes("application/json")) {
            throw new Error("Server returned non-JSON response");
          }
          
          return response.json();
        })
        .then((data) => {
          if (data.error) {
            console.error("Server error:", data.error);
            alert(`Error: ${data.error}`);
          } else {
            console.log("Raw data received from server:", data); // Log raw data
            console.log("File uploaded successfully:", data.filename);
            const newImageUrl = `/uploads/${data.filename}?${new Date().getTime()}`;
            console.log("New image URL:", newImageUrl);

            backgroundImage.src = newImageUrl;
            backgroundImage.onload = function() {
              console.log("Image loaded successfully.");
              resizeCanvasToImage();
              // The slider input event will handle the overlay drawing
              slider.value = 30; // Reset slider to default value
              slider.dispatchEvent(new Event('input')); // Trigger the input event
            };
            backgroundImage.onerror = function() {
              console.error("Failed to load image.");
            };

            // Fully flatten the nested array
            gradientData = data.predicted_label === 'Normal' ? [] : data.gradient_data.flat(Infinity);
            predictedLabel = data.predicted_label; // Store the predicted label

            console.log("Flattened Gradient data received:", gradientData);
            sliderValue.textContent = "30%"; // Update slider label to default value

            updateClassificationText(data.predicted_label);
            if (data.predicted_label == 'Covid') {
              additionalText.textContent = 'A chest X-ray indicative of COVID-19 often shows bilateral ground-glass opacities (GGOs), which are hazy, gray areas predominantly located in the peripheral and lower lung zones.';
              sliderContainer.style.display = "flex"; // Show slider container
            } else if (data.predicted_label == 'Normal') {
              additionalText.textContent = 'A normal chest X-ray should show clear, well-defined lung fields without any areas of abnormal opacity or shadowing. The lungs should appear dark, with the bronchial tree and blood vessels faintly visible.';
              gradientData = []; // Clear gradient data if the prediction is normal
              sliderContainer.style.display = "none"; // Hide slider container
            } else if (data.predicted_label == 'Pneumonia') {
              additionalText.textContent = 'A chest X-ray indicative of pneumonia typically reveals areas of consolidation, where lung tissue is visibly dense and white due to fluid or pus accumulation.';
              sliderContainer.style.display = "flex"; // Show slider container
            }
          }
        })
        .catch((error) => {
          console.error("Error:", error);
          
          // Reset UI state
          backgroundImage.src = '';
          const classificationText = document.querySelector('.prediction-text');
          const additionalText = document.querySelector('.additional-text');
          const sliderContainer = document.getElementById("sliderContainer");
          
          classificationText.textContent = 'Error';
          additionalText.textContent = 'Failed to process image. Please try again.';
          sliderContainer.style.display = "none";
          
          // Show user-friendly error message
          if (error.message.includes('HTTP error! status: 500')) {
            alert('Server error occurred. The external ML service may be unavailable. Please try again later.');
          } else if (error.message.includes('non-JSON response')) {
            alert('Server returned an unexpected response. Please try again.');
          } else {
            alert(`Upload failed: ${error.message}`);
          }
        });
    };

    backgroundImage.onerror = function() {
      console.error("Failed to load sample image.");
    };
  } else {
    alert("Please select an image file to upload.");
  }
}



function selectTestImage(imageName, folderName) {
  console.log('Selected image:', imageName);
  const dropdownButton = document.getElementById('dropdownButton');
  dropdownButton.innerText = imageName;
  const encodedFilePath = encodeURIComponent(imageName);
  const filePath = `${folderName}/${encodedFilePath}`;
  console.log('Fetching image from:', filePath);
  const backgroundImage = document.getElementById('backgroundImage');
  const overlayCanvas = document.getElementById('overlay');
  const overlayCtx = overlayCanvas.getContext("2d");
  const classificationText = document.querySelector('.prediction-text');
  const additionalText = document.querySelector('.additional-text');
  const sliderContainer = document.getElementById("sliderContainer");
  const slider = document.getElementById("squareSlider");
  const sliderValue = document.getElementById("sliderValue");

  overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  backgroundImage.src = '/static/loading.gif';    

  fetch(filePath)
    .then(response => response.blob())
    .then(blob => {
      const file = new File([blob], imageName, { type: blob.type });
      const formData = new FormData();
      formData.append('file', file);

      console.log('Uploading file...');
      fetch("/upload", {
        method: "POST",
        body: formData,
      })
        .then((response) => {
          // Check if the response is successful
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          
          // Check if the response is JSON
          const contentType = response.headers.get("content-type");
          if (!contentType || !contentType.includes("application/json")) {
            throw new Error("Server returned non-JSON response");
          }
          
          return response.json();
        })
        .then((data) => {
          if (data.error) {
            console.error("Server error:", data.error);
            alert(`Error: ${data.error}`);
          } else {
            console.log("Raw data received from server:", data);
            console.log("File uploaded successfully:", data.filename);
            const newImageUrl = `/uploads/${data.filename}?${new Date().getTime()}`;
            console.log("New image URL:", newImageUrl);

            backgroundImage.src = newImageUrl;
            backgroundImage.onload = function() {
              console.log("Image loaded successfully.");
              resizeCanvasToImage();
              // The slider input event will handle the overlay drawing
              slider.value = 30; // Reset slider to default value
              slider.dispatchEvent(new Event('input')); // Trigger the input event
            };
            backgroundImage.onerror = function() {
              console.error("Failed to load image.");
            };

            gradientData = data.predicted_label === 'Normal' ? [] : data.gradient_data.flat(Infinity);
            predictedLabel = data.predicted_label; // Store the predicted label

            console.log("Flattened Gradient data received:", gradientData);
            sliderValue.textContent = "30%"; // Update slider label to default value

            updateClassificationText(data.predicted_label);
            if (data.predicted_label == 'Covid') {
              additionalText.textContent = 'A chest X-ray indicative of COVID-19 often shows bilateral ground-glass opacities (GGOs), which are hazy, gray areas predominantly located in the peripheral and lower lung zones.';
              sliderContainer.style.display = "flex"; // Show slider container
            } else if (data.predicted_label == 'Normal') {
              additionalText.textContent = 'A normal chest X-ray should show clear, well-defined lung fields without any areas of abnormal opacity or shadowing. The lungs should appear dark, with the bronchial tree and blood vessels faintly visible.';
              gradientData = []; // Clear gradient data if the prediction is normal
              sliderContainer.style.display = "none"; // Hide slider container
            } else if (data.predicted_label == 'Pneumonia') {
              additionalText.textContent = 'A chest X-ray indicative of pneumonia typically reveals areas of consolidation, where lung tissue is visibly dense and white due to fluid or pus accumulation.';
              sliderContainer.style.display = "flex"; // Show slider container
            }
          }
        })
        .catch((error) => {
          console.error("Error:", error);
          
          // Reset UI state
          backgroundImage.src = '';
          const classificationText = document.querySelector('.prediction-text');
          const additionalText = document.querySelector('.additional-text');
          const sliderContainer = document.getElementById("sliderContainer");
          
          classificationText.textContent = 'Error';
          additionalText.textContent = 'Failed to process image. Please try again.';
          sliderContainer.style.display = "none";
          
          // Show user-friendly error message
          if (error.message.includes('HTTP error! status: 500')) {
            alert('Server error occurred. The external ML service may be unavailable. Please try again later.');
          } else if (error.message.includes('non-JSON response')) {
            alert('Server returned an unexpected response. Please try again.');
          } else {
            alert(`Upload failed: ${error.message}`);
          }
        });
    })
    .catch(error => console.error('Error fetching image:', error));
}


function handleDropdownOption2and3(folderName) {
  toggleDropdown();

  console.log('Fetching image list...');
  fetch(`/list_test_images?testFolder=${encodeURIComponent(folderName)}`)
    .then(response => {
      console.log('Response received from /list_test_images');
      return response.json();
    })
    .then(images => {
      console.log('Fetched images:', images);
      const fileDialog = document.getElementById('fileDialog');
      const fileListContainer = document.getElementById('fileListContainer');
      fileListContainer.innerHTML = ''; // Clear any existing content

      // Initialize arrays to hold the first 3 images for each category
      let cImages = [];
      let nImages = [];
      let pImages = [];
      let otherImages = [];

      // Organize the images into respective categories
      images.forEach(image => {
        if (image.startsWith('C') && cImages.length < 3) {
          cImages.push(image);
        } else if (image.startsWith('N') && nImages.length < 3) {
          nImages.push(image);
        } else if (image.startsWith('P') && pImages.length < 3) {
          pImages.push(image);
        } else {
          otherImages.push(image);
        }
      });

      // Combine the filtered images with the remaining ones, excluding the 9 chosen first
      const orderedImages = [...cImages, ...nImages, ...pImages, ...otherImages];

      if (orderedImages.length === 0) {
        console.log('No images found.');
      }

      // Add images to the list in the desired order
      orderedImages.forEach(image => {
        console.log('Adding image to list:', image);
        const imageItem = document.createElement('div');
        imageItem.textContent = image;
        imageItem.style.cursor = 'pointer';
        imageItem.style.margin = '5px 0';
        imageItem.addEventListener('click', () => {
          console.log('Image clicked:', image);

          const overlayCanvas = document.getElementById('overlay');
          const overlayCtx = overlayCanvas.getContext("2d");
          overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

          selectTestImage(image, folderName);
          fileDialog.close();
        });
        fileListContainer.appendChild(imageItem);
      });

      console.log('Displaying file dialog');
      fileDialog.showModal();
    })
    .catch(error => console.error('Error fetching image list:', error));
}


function closeDialog() {
  console.log('Closing dialog');
  document.getElementById('fileDialog').close();
}

function toggleSidebar() {
  const body = document.body;
  const sidebar = document.getElementById('sidebar');

  if (body.classList.contains('sidebar-hidden')) {
    body.classList.remove('sidebar-hidden');
    body.classList.add('sidebar-visible');
    sidebar.style.display = 'block';
    document.addEventListener('click', handleOutsideClick);
  } else {
    body.classList.remove('sidebar-visible');
    body.classList.add('sidebar-hidden');
    sidebar.style.display = 'none';
    document.removeEventListener('click', handleOutsideClick);
  }
}

function handleOutsideClick(event) {
  const sidebar = document.getElementById('sidebar');
  const hamburger = document.getElementById('hamburger');
  const isClickInsideSidebar = sidebar.contains(event.target);
  const isClickOnHamburger = hamburger.contains(event.target);

  if (!isClickInsideSidebar && !isClickOnHamburger) {
    document.body.classList.add('sidebar-hidden');
    document.body.classList.remove('sidebar-visible');
    sidebar.style.display = 'none';
    document.removeEventListener('click', handleOutsideClick);
  }
}

function checkSidebarInitialState() {
  const body = document.body;
  const sidebar = document.getElementById('sidebar');

  if (window.innerWidth > 1030) {
    body.classList.remove('sidebar-hidden');
    body.classList.add('sidebar-visible');
    sidebar.style.display = 'block';
    document.removeEventListener('click', handleOutsideClick); // No need for outside click listener when sidebar is auto-visible
  } else {
    body.classList.add('sidebar-hidden');
    body.classList.remove('sidebar-visible');
    sidebar.style.display = 'none';
    document.removeEventListener('click', handleOutsideClick); // Ensure listener is removed when sidebar is hidden
  }
}

window.addEventListener('resize', function() {
  checkSidebarInitialState();
});

document.addEventListener('DOMContentLoaded', function () {
  checkSidebarInitialState();
});

document.addEventListener('DOMContentLoaded', function() {
  const slider = document.getElementById('squareSlider');

  function updateSliderBackground() {
      const value = slider.value;
      const min = slider.min || 0;
      const max = slider.max || 100;
      const percentage = ((value - min) / (max - min)) * 100;

      // Set the background style inline to adjust the gradient based on thumb position
      slider.style.background = `linear-gradient(to right, #516287 ${percentage}%, #ccc ${percentage}%)`;
  }

  slider.addEventListener('input', updateSliderBackground);

  // Initialize the slider background on page load
  updateSliderBackground();
});

