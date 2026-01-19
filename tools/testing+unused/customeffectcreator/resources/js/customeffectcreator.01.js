const canvas = document.getElementById('imageCanvas');
    const ctx = canvas.getContext('2d');
    const coordinatesDiv = document.getElementById('coordinates');
    const clearBtn = document.getElementById('clearBtn');
    const saveBtn = document.getElementById('saveBtn');
    const imageUpload = document.getElementById('imageUpload');

    let image = new Image();
    let keypoints = [];
    let imageLoaded = false;

    // Define keypoints names for reference
    const keypointNames = ["Left Eye", "Right Eye", "Nose", "Top Center Mouth", "Left Mouth", "Right Mouth"];

    // Handle image upload
    imageUpload.addEventListener('change', function (event) {
      const file = event.target.files[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
          image.src = e.target.result;
        };
        reader.readAsDataURL(file);
      }
    });

    // Draw uploaded image on canvas
    image.onload = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
      imageLoaded = true;
    };

    // Handle click event on the canvas to define keypoints
    canvas.addEventListener('click', (event) => {
      if (!imageLoaded) return;

      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;

      if (keypoints.length < keypointNames.length) {
        keypoints.push({ name: keypointNames[keypoints.length], x, y });
        drawKeypoint(x, y, keypointNames[keypoints.length]);
        updateCoordinatesDisplay();
      } else {
        alert("All keypoints have been defined!");
      }
    });

    // Function to draw a keypoint on the canvas
    function drawKeypoint(x, y, name) {
      ctx.fillStyle = 'red';
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, 2 * Math.PI);
      ctx.fill();

      ctx.fillStyle = 'white';
      ctx.font = '14px Arial';
      ctx.fillText(name, x + 10, y - 10);
    }

    // Function to update the coordinates display
    function updateCoordinatesDisplay() {
      coordinatesDiv.innerHTML = keypoints.map(point => `${point.name}: (${point.x}, ${point.y})`).join('<br>');
    }

    // Clear keypoints and redraw the image
    clearBtn.addEventListener('click', () => {
      keypoints = [];
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
      coordinatesDiv.innerHTML = '';
    });

    // Save the keypoints (for later use in your effect creator)
    saveBtn.addEventListener('click', () => {
      if (keypoints.length !== keypointNames.length) {
        alert("Please define all keypoints before saving!");
        return;
      }

      const keypointsJSON = JSON.stringify(keypoints, null, 2);
      alert("Keypoints saved! You can now use them in your custom effects.\n" + keypointsJSON);

      // Optionally, you can send the keypoints to a server or save them locally.
    });
