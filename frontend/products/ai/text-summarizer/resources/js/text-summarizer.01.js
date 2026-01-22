import { pipeline } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.6.0";

    let summarizer = null;
    let isBusy = false;

    const $ = (id) => document.getElementById(id);

    const input = $("inputText");
    const btnSum = $("summarize");
    const btnCopy = $("copy");
    const statusEl = $("status");
    const outEl = $("output");
    const fill = $("loadingFill");
    const footer = $("footer-text");

    function setStatus(msg){ statusEl.textContent = "Status: " + msg; }
    function setProgress(pct){ fill.style.width = Math.max(0, Math.min(100, pct)) + "%"; }

    function saveToCookies(key, value) {
      const v = encodeURIComponent(value).slice(0, 3500);
      document.cookie = `${key}=${v}; max-age=86400; path=/; SameSite=Lax`;
    }

    async function ensureModel(){
      if (summarizer) return;
      setStatus("Loading model (first run can take a bit)...");
      setProgress(35);
      summarizer = await pipeline("summarization", "Xenova/distilbart-cnn-6-6");
      setProgress(0);
      setStatus("Ready");
    }

    btnSum.addEventListener("click", async () => {
      if (isBusy) return;
      const text = (input.value || "").trim();
      if (!text) {
        setStatus("Please paste text to summarize.");
        return;
      }

      try{
        isBusy = true;
        btnSum.disabled = true;
        btnCopy.disabled = true;

        await ensureModel();

        setStatus("Summarizing...");
        setProgress(90);

        const result = await summarizer(text, { max_new_tokens: 120 });
        const summary = result?.[0]?.summary_text ?? String(result);

        outEl.textContent = summary;
        btnCopy.disabled = !summary;

        saveToCookies("summarize", summary);

        setProgress(0);
        setStatus("Ready");
      } catch (e){
        console.error(e);
        setProgress(0);
        setStatus("Error: " + (e?.message || "Summarization failed"));
      } finally{
        isBusy = false;
        btnSum.disabled = false;
      }
    });

    btnCopy.addEventListener("click", async () => {
      const t = outEl.textContent || "";
      if (!t) return;
      try{
        await navigator.clipboard.writeText(t);
        setStatus("Copied to clipboard.");
        setTimeout(() => setStatus("Ready"), 900);
      } catch {
        setStatus("Copy failed (browser blocked clipboard).");
      }
    });

    document.addEventListener("DOMContentLoaded", () => {
      const y = new Date().getFullYear();
      footer.textContent = `© 2019-${y} Unlim8ted Studio Productions. All rights reserved.`;
    });

    // Canvas particle background (lightweight)
    const canvas = $("bgFX");
    const ctx = canvas.getContext("2d", { alpha:true });

    let W=0,H=0,DPR=1;
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

    if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches) tick();
    else canvas.style.display = "none";
