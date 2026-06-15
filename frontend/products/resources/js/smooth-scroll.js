/* Lightweight desktop-only wheel smoothing. Native touch scrolling stays untouched. */
(() => {
  const prefersReduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const coarsePointer = matchMedia("(pointer: coarse)").matches;
  if (prefersReduced || coarsePointer) return;

  const container = document.scrollingElement || document.documentElement;
  let targetY = window.scrollY || 0;
  let currentY = targetY;
  let running = false;

  const EASE = 0.14;
  const STOP_EPS = 0.5;
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const maxScroll = () => Math.max(0, container.scrollHeight - innerHeight);

  function tick() {
    running = true;
    const diff = targetY - currentY;
    currentY += diff * EASE;
    if (Math.abs(diff) < STOP_EPS) currentY = targetY;
    window.scrollTo(0, currentY);
    if (currentY !== targetY) requestAnimationFrame(tick);
    else running = false;
  }

  addEventListener("wheel", (event) => {
    if (event.ctrlKey || event.metaKey) return;
    event.preventDefault();
    targetY = clamp(targetY + event.deltaY, 0, maxScroll());
    if (!running) requestAnimationFrame(tick);
  }, { passive: false });

  addEventListener("scroll", () => {
    if (running) return;
    targetY = currentY = window.scrollY || 0;
  }, { passive: true });

  addEventListener("resize", () => {
    targetY = clamp(targetY, 0, maxScroll());
    currentY = clamp(currentY, 0, maxScroll());
  }, { passive: true });
})();
