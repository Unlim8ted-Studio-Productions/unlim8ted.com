(() => {
      const prefersReduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
      if (prefersReduced) return;

      const container = document.scrollingElement || document.documentElement;

      let targetY = window.scrollY || 0;
      let currentY = targetY;
      let running = false;

      const EASE = 0.085;
      const WHEEL_MULT = 1.0;
      const TOUCH_MULT = 1.0;
      const STOP_EPS = 0.5;

      const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
      const maxScroll = () => Math.max(0, container.scrollHeight - innerHeight);

      function tick() {
        running = true;

        const diff = targetY - currentY;
        currentY += diff * EASE;

        if (Math.abs(diff) < STOP_EPS) currentY = targetY;

        window.scrollTo(0, currentY);
        window.dispatchEvent(new Event("scroll"));

        if (currentY !== targetY) requestAnimationFrame(tick);
        else running = false;
      }

      function go(delta) {
        targetY = clamp(targetY + delta, 0, maxScroll());
        if (!running) requestAnimationFrame(tick);
      }

      function onWheel(e) {
        if (e.ctrlKey || e.metaKey) return;
        e.preventDefault();
        go(e.deltaY * WHEEL_MULT);
      }

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

      targetY = currentY = window.scrollY || 0;
    })();
