const fileInput = document.getElementById("fileInput");
    const iframe = document.getElementById("preview");
    const deviceSelect = document.getElementById("device");
    const exportButton = document.getElementById("export-css");
    const addResizableButton = document.getElementById("add-resizable");
    const cssRules = [];

    // Load HTML File
    fileInput.addEventListener("change", (e) => {
      const file = e.target.files[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = () => {
          iframe.srcdoc = reader.result;
        };
        reader.readAsText(file);
      }
    });

    // Adjust Viewport for Devices
    deviceSelect.addEventListener("change", (e) => {
      const [width, height] = e.target.value.split("x");
      iframe.style.width = `${width}px`;
      iframe.style.height = `${height}px`;
      iframe.style.transform = `scale(${Math.min(1, window.innerWidth / width)})`;
    });

    // Add Draggable/Resizable Elements
    addResizableButton.addEventListener("click", () => {
      const iframeDocument = iframe.contentDocument || iframe.contentWindow.document;
      iframeDocument.body.querySelectorAll("*").forEach((el) => {
        if (!el.classList.contains("resizable")) {
          el.classList.add("resizable");
          el.style.position = "absolute";

          const resizeHandle = document.createElement("div");
          resizeHandle.className = "resize-handle bottom-right";
          resizeHandle.addEventListener("mousedown", (event) => initResize(event, el));
          el.appendChild(resizeHandle);

          el.addEventListener("mousedown", (event) => initDrag(event, el));
        }
      });
    });

    // Initialize Dragging
    function initDrag(event, element) {
      event.preventDefault();
      const startX = event.clientX;
      const startY = event.clientY;
      const startLeft = parseInt(window.getComputedStyle(element).left, 10) || 0;
      const startTop = parseInt(window.getComputedStyle(element).top, 10) || 0;

      function onMouseMove(e) {
        element.style.left = `${startLeft + e.clientX - startX}px`;
        element.style.top = `${startTop + e.clientY - startY}px`;
      }

      function onMouseUp() {
        document.removeEventListener("mousemove", onMouseMove);
        document.removeEventListener("mouseup", onMouseUp);
        cssRules.push(`#${element.id || element.tagName.toLowerCase()} { left: ${element.style.left}; top: ${element.style.top}; }`);
      }

      document.addEventListener("mousemove", onMouseMove);
      document.addEventListener("mouseup", onMouseUp);
    }

    // Initialize Resizing
    function initResize(event, element) {
      event.preventDefault();
      const startX = event.clientX;
      const startY = event.clientY;
      const startWidth = element.offsetWidth;
      const startHeight = element.offsetHeight;

      function onMouseMove(e) {
        element.style.width = `${startWidth + e.clientX - startX}px`;
        element.style.height = `${startHeight + e.clientY - startY}px`;
      }

      function onMouseUp() {
        document.removeEventListener("mousemove", onMouseMove);
        document.removeEventListener("mouseup", onMouseUp);
        cssRules.push(`#${element.id || element.tagName.toLowerCase()} { width: ${element.style.width}; height: ${element.style.height}; }`);
      }

      document.addEventListener("mousemove", onMouseMove);
      document.addEventListener("mouseup", onMouseUp);
    }

    // Export CSS
    exportButton.addEventListener("click", () => {
      const blob = new Blob([cssRules.join("\n")], { type: "text/css" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "styles.css";
      a.click();
    });
