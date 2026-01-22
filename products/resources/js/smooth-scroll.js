/* Drop-in Smooth Scroll (wheel + touch) */
(() => {
  const prefersReduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (prefersReduced) return;

  // If you have a custom scroll container, set it here; otherwise it uses the document.
  const container = document.scrollingElement || document.documentElement;

  let targetY = window.scrollY || 0;
  let currentY = targetY;
  let running = false;

  // Tuning
  const EASE = 0.085;          // lower = smoother/laggier, higher = snappier (0.06–0.14)
  const WHEEL_MULT = 1.0;      // 0.8–1.2
  const TOUCH_MULT = 1.0;      // 0.8–1.2
  const STOP_EPS = 0.5;        // px

  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const maxScroll = () => Math.max(0, container.scrollHeight - innerHeight);

  function tick() {
    running = true;

    const diff = targetY - currentY;
    currentY += diff * EASE;

    // snap when close
    if (Math.abs(diff) < STOP_EPS) {
      currentY = targetY;
    }

    window.scrollTo(0, currentY);

    // Keep anims that listen to scroll in sync
    // (If you already have onScroll -> requestAnimationFrame(update), this will trigger it naturally)
    // But some setups need a nudge:
    window.dispatchEvent(new Event("scroll"));

    if (currentY !== targetY) {
      requestAnimationFrame(tick);
    } else {
      running = false;
    }
  }

  function go(delta) {
    targetY = clamp(targetY + delta, 0, maxScroll());
    if (!running) requestAnimationFrame(tick);
  }

  // Wheel
  function onWheel(e) {
    // allow normal page zoom with ctrl/cmd + wheel
    if (e.ctrlKey || e.metaKey) return;

    e.preventDefault();
    go(e.deltaY * WHEEL_MULT);
  }

  // Touch
  let lastTouchY = 0;
  function onTouchStart(e) {
    if (!e.touches || !e.touches.length) return;
    lastTouchY = e.touches[0].clientY;
  }
  function onTouchMove(e) {
    if (!e.touches || !e.touches.length) return;
    e.preventDefault();
    const y = e.touches[0].clientY;
    const delta = (lastTouchY - y) * TOUCH_MULT;
    lastTouchY = y;
    go(delta);
  }

  // Keep targets aligned if user drags scrollbar, uses PgDn, etc.
  function syncToNative() {
    if (running) return;
    targetY = currentY = window.scrollY || 0;
  }

  addEventListener("wheel", onWheel, { passive: false });
  addEventListener("touchstart", onTouchStart, { passive: true });
  addEventListener("touchmove", onTouchMove, { passive: false });
  addEventListener("scroll", syncToNative, { passive: true });
  addEventListener("resize", () => {
    targetY = clamp(targetY, 0, maxScroll());
    currentY = clamp(currentY, 0, maxScroll());
  }, { passive: true });

  // Initialize
  targetY = currentY = window.scrollY || 0;
})();