import { pipeline } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.6.0";

    let classifier = null;
    let isBusy = false;

    const $ = (id) => document.getElementById(id);

    const input = $("inputText");
    const btnCls = $("classify");
    const btnCopy = $("copy");
    const statusEl = $("status");
    const fill = $("loadingFill");
    const footer = $("footer-text");

    const resultsWrap = $("results");
    const resultsBody = $("resultsBody");

    function setStatus(msg){ statusEl.textContent = "Status: " + msg; }
    function setProgress(pct){ fill.style.width = Math.max(0, Math.min(100, pct)) + "%"; }

    function saveToCookies(key, value) {
      const v = encodeURIComponent(value).slice(0, 3500);
      document.cookie = `${key}=${v}; max-age=86400; path=/; SameSite=Lax`;
    }

    function prettyLabel(label){
      return (label || "")
        .replace(/_/g, " ")
        .toLowerCase()
        .replace(/\b\w/g, (c) => c.toUpperCase());
    }

    function renderResults(items){
      resultsBody.innerHTML = "";
      for(const it of items){
        const label = prettyLabel(it.label);
        const scorePct = Math.max(0, Math.min(100, (it.score || 0) * 100));
        const row = document.createElement("div");
        row.className = "rowItem";
        row.innerHTML = `
          <div>
            <div class="label">${label}</div>
          </div>
          <div>
            <div class="meter"><div style="width:${scorePct.toFixed(2)}%"></div></div>
            <div class="pct">${scorePct.toFixed(2)}%</div>
          </div>
        `;
        resultsBody.appendChild(row);
      }
      resultsWrap.hidden = false;
    }

    async function ensureModel(){
      if (classifier) return;
      setStatus("Loading model (first run can take a bit)...");
      setProgress(35);
      classifier = await pipeline("text-classification", "Xenova/toxic-bert");
      setProgress(0);
      setStatus("Ready");
    }

    btnCls.addEventListener("click", async () => {
      if (isBusy) return;
      const text = (input.value || "").trim();
      if (!text) { setStatus("Please enter text to classify."); return; }

      try{
        isBusy = true;
        btnCls.disabled = true;
        btnCopy.disabled = true;
        resultsWrap.hidden = true;

        await ensureModel();

        setStatus("Classifying...");
        setProgress(90);

        const results = await classifier(text, { topk: null });

        // Sort by score desc for readability
        const sorted = [...results].sort((a,b) => (b.score||0) - (a.score||0));
        renderResults(sorted);

        // Copy format + cookie format (plain text)
        const plain = sorted
          .map(r => `${prettyLabel(r.label)}: ${(r.score*100).toFixed(2)}%`)
          .join("\n");

        btnCopy.disabled = !plain;
        saveToCookies("classify", plain);

        setProgress(0);
        setStatus("Ready");
      } catch (e){
        console.error(e);
        setProgress(0);
        setStatus("Error: " + (e?.message || "Classification failed"));
      } finally{
        isBusy = false;
        btnCls.disabled = false;
      }
    });

    btnCopy.addEventListener("click", async () => {
      // Collect from DOM so it matches what user sees
      const lines = [];
      resultsBody.querySelectorAll(".rowItem").forEach(row => {
        const label = row.querySelector(".label")?.textContent?.trim() || "";
        const pct = row.querySelector(".pct")?.textContent?.trim() || "";
        if (label && pct) lines.push(`${label}: ${pct}`);
      });
      const text = lines.join("\n");
      if (!text) return;

      try{
        await navigator.clipboard.writeText(text);
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

    // Canvas particle background
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
