// Footer year
    document.addEventListener("DOMContentLoaded", () => {
      const y = new Date().getFullYear();
      document.getElementById("footer-text").textContent =
        `© 2019-${y} Unlim8ted Studio Productions. All rights reserved.`;
    });

    // Custom cursor (desktop only)
    const cursor = document.getElementById("cursor");
    if (window.matchMedia("(pointer:fine)").matches) {
      document.addEventListener("mousemove", (e) => {
        cursor.style.top = e.clientY + "px";
        cursor.style.left = e.clientX + "px";
      });
    }

    // Lightweight particle field (canvas, not 100 DOM nodes)
    const canvas = document.getElementById("bgFX");
    const ctx = canvas.getContext("2d", { alpha: true });

    let W = 0, H = 0, DPR = 1;
    function resize(){
      DPR = Math.min(2, window.devicePixelRatio || 1);
      W = canvas.width = Math.floor(window.innerWidth * DPR);
      H = canvas.height = Math.floor(window.innerHeight * DPR);
      canvas.style.width = window.innerWidth + "px";
      canvas.style.height = window.innerHeight + "px";
    }
    window.addEventListener("resize", resize);
    resize();

    const N = 90;
    const pts = Array.from({length:N}, () => ({
      x: Math.random()*W,
      y: Math.random()*H,
      r: (Math.random()*1.6 + 0.6) * DPR,
      vx: (Math.random()*0.18 - 0.09) * DPR,
      vy: (Math.random()*0.45 + 0.15) * DPR,
      a: Math.random()*0.35 + 0.15
    }));

    function tick(){
      ctx.clearRect(0,0,W,H);

      // subtle glow dots drifting downward
      for(const p of pts){
        p.x += p.vx;
        p.y += p.vy;

        if(p.y > H + 30*DPR) { p.y = -30*DPR; p.x = Math.random()*W; }
        if(p.x < -30*DPR) p.x = W + 30*DPR;
        if(p.x > W + 30*DPR) p.x = -30*DPR;

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI*2);
        ctx.fillStyle = `rgba(255,255,255,${p.a})`;
        ctx.fill();
      }

      requestAnimationFrame(tick);
    }

    // Respect reduced motion
    if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      tick();
    } else {
      canvas.style.display = "none";
    }
