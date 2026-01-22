// Footer year + last updated
    document.addEventListener("DOMContentLoaded", () => {
      const y = new Date().getFullYear();
      const ft = document.getElementById("footer-text");
      if (ft) ft.textContent = `© 2019-${y} Unlim8ted Studio Productions. All rights reserved.`;

      const lu = document.getElementById("lastUpdated");
      if (lu) {
        const d = new Date();
        lu.textContent = d.toLocaleDateString(undefined, { year:"numeric", month:"long", day:"numeric" });
      }
    });

    // Custom cursor (desktop only)
    const cursor = document.getElementById("cursor");
    if (window.matchMedia("(pointer:fine)").matches && cursor) {
      document.addEventListener("mousemove", (e) => {
        cursor.style.top = e.clientY + "px";
        cursor.style.left = e.clientX + "px";
      });
    }

    // Lightweight particle field (canvas)
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

    const N = 95;
    const pts = Array.from({length:N}, () => ({
      x: Math.random()*W,
      y: Math.random()*H,
      r: (Math.random()*1.8 + 0.6) * DPR,
      vx: (Math.random()*0.18 - 0.09) * DPR,
      vy: (Math.random()*0.45 + 0.15) * DPR,
      a: Math.random()*0.35 + 0.12
    }));

    function tick(){
      ctx.clearRect(0,0,W,H);
      for(const p of pts){
        p.x += p.vx; p.y += p.vy;
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

    if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      tick();
    } else {
      canvas.style.display = "none";
    }
