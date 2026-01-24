// Fade loader out cleanly (and remove it from the DOM)
    window.addEventListener("load", () => {
      const loader = document.getElementById("pageLoader");
      if (!loader) return;
      loader.style.animation = "load-out 0.9s forwards";
      loader.style.webkitAnimation = "load-out 0.9s forwards";
      setTimeout(() => loader.remove(), 950);
    });

    // Footer year
    document.addEventListener("DOMContentLoaded", () => {
      const currentYear = new Date().getFullYear();
      const footerText = `&copy; 2019-${currentYear} Unlim8ted Studio Productions. All rights reserved.`;
      const el = document.getElementById("footer-text");
      if (el) el.innerHTML = footerText;
    });

    // Mobile nav
    function toggleMenu() {
      const navLinks = document.getElementById("navbarLinks");
      if (!navLinks) return;
      navLinks.classList.toggle("show");
    }

    // Mobile dropdown
    function toggleDropdown(e){
      // allow hover on desktop, click on mobile
      if (window.matchMedia("(max-width: 640px)").matches) {
        e.preventDefault();
        const dd = document.getElementById("moreDropdown");
        if (dd) dd.classList.toggle("open");
      }
    }

    // Close mobile menus when tapping outside
    document.addEventListener("click", (e) => {
      const nav = document.getElementById("navbarLinks");
      const dd = document.getElementById("moreDropdown");
      if (!nav || !dd) return;

      if (window.matchMedia("(max-width: 640px)").matches) {
        const clickedInsideNav = nav.contains(e.target) || e.target.classList?.contains("nav-toggle");
        if (!clickedInsideNav) nav.classList.remove("show");
        const clickedInsideDropdown = dd.contains(e.target);
        if (!clickedInsideDropdown) dd.classList.remove("open");
      }
    });

    // Custom cursor (desktop only)
    const cursor = document.getElementById("cursor");
    const isFinePointer = window.matchMedia("(pointer:fine)").matches;
    if (cursor && isFinePointer) {
      document.addEventListener("mousemove", (e) => {
        cursor.style.top = `${e.clientY}px`;
        cursor.style.left = `${e.clientX}px`;
      });
    }

    // Audio player
    let currentPodcast = null;
    let currentLabel = null;
    let isPlaying = false;
    const audio = new Audio();
    const playBtn = document.getElementById("playBtn");
    const npName = document.getElementById("npName");

    function playPodcast(fileName, label) {
      // If you click a “Coming soon” card, do nothing but still show label
      // (Optional: remove this if those files exist)
      if (!fileName || !fileName.endsWith(".wav")) {
        currentPodcast = null;
        currentLabel = label || "Coming soon";
        if (npName) npName.textContent = currentLabel;
        if (playBtn) playBtn.textContent = "Play";
        isPlaying = false;
        return;
      }

      audio.src = `https://assets.unlim8ted.com/podcasts/${encodeURIComponent(fileName)}`;
      audio.play().catch(() => {
        // Autoplay can be blocked; keep UI sane
        isPlaying = false;
        if (playBtn) playBtn.textContent = "Play";
      });

      currentPodcast = fileName;
      currentLabel = label || fileName;
      if (npName) npName.textContent = currentLabel;

      isPlaying = true;
      if (playBtn) playBtn.textContent = "Pause";
    }

    function togglePlayPause() {
      if (!audio.src) return;
      if (isPlaying) {
        audio.pause();
        isPlaying = false;
        if (playBtn) playBtn.textContent = "Play";
      } else {
        audio.play().catch(() => {});
        isPlaying = true;
        if (playBtn) playBtn.textContent = "Pause";
      }
    }

    function setVolume(value) {
      audio.volume = Math.max(0, Math.min(1, value / 100));
    }

    // Time bar
    audio.addEventListener("timeupdate", () => {
      if (!audio.duration || !isFinite(audio.duration)) return;
      const pct = (audio.currentTime / audio.duration) * 100;
      const fill = document.getElementById("timeBarFill");
      if (fill) fill.style.width = `${pct}%`;
    });

    function setPlaybackPosition(event) {
      if (!audio.duration || !isFinite(audio.duration)) return;
      const timeBar = event.currentTarget;
      const rect = timeBar.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const pct = Math.max(0, Math.min(1, x / rect.width));
      audio.currentTime = pct * audio.duration;
    }

    function showHoverLine(event) {
      const hoverLine = document.getElementById("timeBarHoverLine");
      if (!hoverLine) return;
      const timeBar = event.currentTarget;
      const rect = timeBar.getBoundingClientRect();
      const x = event.clientX - rect.left;
      hoverLine.style.left = `${Math.max(0, Math.min(rect.width, x))}px`;
      hoverLine.style.width = "2px";
    }

    function hideHoverLine() {
      const hoverLine = document.getElementById("timeBarHoverLine");
      if (!hoverLine) return;
      hoverLine.style.width = "0";
    }

    // Keep UI synced if audio ends
    audio.addEventListener("ended", () => {
      isPlaying = false;
      if (playBtn) playBtn.textContent = "Play";
    });
