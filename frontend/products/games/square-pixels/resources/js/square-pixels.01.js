// Smooth hero parallax (no background-size hacks)
  const bg = document.getElementById("heroBg");
  window.addEventListener("scroll", () => {
    const y = window.scrollY || 0;
    // small translate + slight scale for depth, clamped to avoid weirdness
    const t = Math.min(160, y * 0.25);
    const s = 1.05 + Math.min(0.08, y * 0.00008);
    bg.style.transform = `translateY(${t}px) scale(${s})`;
  }, { passive: true });

  // Footer year
  document.addEventListener("DOMContentLoaded", () => {
    const currentYear = new Date().getFullYear();
    document.getElementById("footer-text").innerHTML =
      `&copy; 2019-${currentYear} Unlim8ted Studio Productions. All rights reserved.`;
  });
